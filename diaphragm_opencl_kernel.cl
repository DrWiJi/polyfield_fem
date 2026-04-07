/**
 * OpenCL iris computing kernel.
 * Forces: elasticity (nonlinear BOPET spring) + air resistance + external pressure.
 * Integration: RK4 (four stages of acceleration).
 *
 * Key formulas:
 * - Elasticity: spring_len = center_len (center-center), rest_len = 0.5*(size_me + size_nb);
 * strain = (center_len - rest_len)/rest_len; F_elastic = k_eff * (center_len - rest_len).
 * - Air→FE traction: discrete −∇p·V on the brick — Fx ≈ (p(−x face) − p(+x face))·A_x·sx/(2·dx_air),
 *   i.e. central ∂p/∂x with stencil spacing 2·dx_air, V = sx·sy·sz, A_x = sy·sz. Material acoustic_impedance
 *   and host air_coupling_gain scale that force (dimensionless).
 *
 * 3D quantity style: for velocity, face areas, pressure gradient and forces
 * double3 / vload3 / vstore3 and component-wise vector operations are used.
 *
 * Injection FE→air: lumped continuity ∂p/∂t ≈ (ρ c²/V_cell)·Q with Q = Σ_faces A(v_rel·n) (volumetric
 * flux rate from brick motion); acoustic_inject ∈ [0,1] blends source strength per material.
 * Air update: FDTD leapfrog p_tt = c^2 lap(p); missing neighbors use neutral Laplacian ghosts and a
 * bounded Sommerfeld-style damping term on ∂p/∂t for robust sparse-boundary stability.
 *
 * Required: cl_khr_fp64 (double precision).
 * Compilation: built into PyOpenCL when creating a program from source.*/
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define DOF_PER_ELEMENT 6
#define TINY 1e-20
#define FACE_DIRS 6
#define MATERIAL_PROPS_STRIDE 8
#define MAT_PROP_DENSITY 0
#define MAT_PROP_E_PARALLEL 1
#define MAT_PROP_E_PERP 2
#define MAT_PROP_POISSON 3
#define MAT_PROP_CD 4
#define MAT_PROP_ETA_VISC 5
/*Acoustic impedance for air->solid interface coupling, Pa·s/m.*/
#define MAT_PROP_ACOUSTIC_IMPEDANCE 6
/*Injection into air-field from v·n (sound source): only membrane/specified in material; 0 = does not radiate into the grid.*/
#define MAT_PROP_ACOUSTIC_INJECT 7
/* Air boundary kinds encoded in air_absorb (uchar per face). */
#define AIR_BC_INTERIOR 0
#define AIR_BC_OPEN 1
#define AIR_BC_RIGID 2
/*Material aliases (synchronized with Python layer)*/
#define MAT_MEMBRANE_ID 0
#define MAT_FOAM_VE3015_ID 1
#define MAT_SHEEPSKIN_LEATHER_ID 2
#define MAT_HUMAN_EAR_AVG_ID 3
#define MAT_SENSOR_ID 4
#define MAT_COTTON_WOOL_ID 5
#define LAW_SOLID_SPRING 0
#ifndef ENABLE_DEBUG
#define ENABLE_DEBUG 0
#endif
#ifndef EXCITATION_MODE
#define EXCITATION_MODE 0
#endif
#define EXCITATION_MODE_EXTERNAL 0
#define EXCITATION_MODE_EXTERNAL_FULL_OVERRIDE 1
#define EXCITATION_MODE_SECOND_ORDER_BOUNDARY_FULL_OVERRIDE 2
#define EXCITATION_MODE_EXTERNAL_VELOCITY_OVERRIDE 3
#define FORCE_SHAPE_IMPULSE 0
#define FORCE_SHAPE_UNIFORM 1
#define FORCE_SHAPE_SINE 2
#define FORCE_SHAPE_SQUARE 3
#define FORCE_SHAPE_CHIRP 4
#define FORCE_SHAPE_WHITE_NOISE 5
/*Debugging M_total: 6 (F_total,M_total) + 6*6 (force_dir, lever_dir) = 42*/
#define DEBUG_ELASTIC_SIZE 42
/*Trace: step, elastic(42), pos_me(6), vel_me(6), pos_mid(6), vel_mid(6), F(6), mass(6), acc(6), x_new(6), v_new(6), rx,ry,rz, center_len0, strain0, k_eff0, force_mag0, force_local0(3), lever0(3), M0(3)*/
#define TRACE_BUF_SIZE 127
#define TRACE_ELASTIC_EXTRA 20

typedef struct {
    int nx, ny, n_elements, n_dof;
    double dx, dy, thickness;
    double arm_x, arm_y;
    double k_axial_x, k_axial_y, k_shear, k_bending_x, k_bending_y;
    double stiffness_transition_center, stiffness_transition_width, stiffness_ratio;
    int use_nonlinear_stiffness;
    double rho_air, mu_air, Cd;
    double element_area;
    double element_mass, Ixx, Iyy, Izz;
    double dt;
    double pre_tension, k_soft, k_stiff, strain_transition, strain_width, k_bend;
    int debug_elem;
    int debug_step;
    int force_shape_id;
    int force_wave_enabled;
    uint force_noise_seed_u32;
    double force_amp, force_offset, force_freq_hz, force_freq_end_hz, force_phase_rad, force_dt, force_duration_s;
} Params;

inline int air_idx3d(int ix, int iy, int iz, int nx_air, int ny_air) {
    return iz * (nx_air * ny_air) + iy * nx_air + ix;
}

/*Vector operations: built-in dot, length, cross (OpenCL Geometric Functions).*/

/*3x3 matrix: rows row0, row1, row2*/
typedef struct { double3 row0, row1, row2; } double3x3;

inline double3 mat3x3_times_vec3(const double3x3* R, double3 v) {
    return (double3)(
        dot(R->row0, v),
        dot(R->row1, v),
        dot(R->row2, v)
    );
}

/*R = Rz(rz) * Ry(ry) * Rx(rx) - local -> global*/
inline void rotation_matrix_local_to_global(double rx, double ry, double rz, double3x3* R_out) {
    double cx = cos(rx), sx = sin(rx);
    double cy = cos(ry), sy = sin(ry);
    double cz = cos(rz), sz = sin(rz);
    R_out->row0 = (double3)(cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx);
    R_out->row1 = (double3)(sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx);
    R_out->row2 = (double3)(-sy, cy * sx, cy * cx);
}

inline void rotation_matrix_global_to_local(double rx, double ry, double rz, double3x3* R_T_out) {
    double cx = cos(rx), sx = sin(rx);
    double cy = cos(ry), sy = sin(ry);
    double cz = cos(rz), sz = sin(rz);
    R_T_out->row0 = (double3)(cz * cy, sz * cy, -sy);
    R_T_out->row1 = (double3)(cz * sy * sx - sz * cx, sz * sy * sx + cz * cx, cy * sx);
    R_T_out->row2 = (double3)(cz * sy * cx + sz * sx, sz * sy * cx - cz * sx, cy * cx);
}

inline double3 local_to_global(double rx, double ry, double rz, double3 v) {
    double3x3 R;
    rotation_matrix_local_to_global(rx, ry, rz, &R);
    return mat3x3_times_vec3(&R, v);
}

inline double3 global_to_local(double rx, double ry, double rz, double3 v) {
    double3x3 R_T;
    rotation_matrix_global_to_local(rx, ry, rz, &R_T);
    return mat3x3_times_vec3(&R_T, v);
}

inline double3 mounting_point_local(int direction, double3 arm) {
    if (direction == 0) return (double3)(arm.s0, 0.0, 0.0);
    if (direction == 1) return (double3)(-arm.s0, 0.0, 0.0);
    if (direction == 2) return (double3)(0.0, arm.s1, 0.0);
    if (direction == 3) return (double3)(0.0, -arm.s1, 0.0);
    if (direction == 4) return (double3)(0.0, 0.0, arm.s2);
    return (double3)(0.0, 0.0, -arm.s2);
}

inline double3 mounting_point_neighbor_local(int direction, double3 arm) {
    if (direction == 0) return (double3)(-arm.s0, 0.0, 0.0);
    if (direction == 1) return (double3)(arm.s0, 0.0, 0.0);
    if (direction == 2) return (double3)(0.0, -arm.s1, 0.0);
    if (direction == 3) return (double3)(0.0, arm.s1, 0.0);
    if (direction == 4) return (double3)(0.0, 0.0, -arm.s2);
    return (double3)(0.0, 0.0, arm.s2);
}

inline double3 face_normal(int direction) {
    if (direction == 0) return (double3)(1.0, 0.0, 0.0);
    if (direction == 1) return (double3)(-1.0, 0.0, 0.0);
    if (direction == 2) return (double3)(0.0, 1.0, 0.0);
    if (direction == 3) return (double3)(0.0, -1.0, 0.0);
    if (direction == 4) return (double3)(0.0, 0.0, 1.0);
    return (double3)(0.0, 0.0, -1.0);
}

inline double face_area_from_size(int direction, double3 elem_size) {
    if (direction < 2) return elem_size.s1 * elem_size.s2; /*±X: YZ-face*/
    if (direction < 4) return elem_size.s0 * elem_size.s2; /*±Y: XZ-face*/
    return elem_size.s0 * elem_size.s1;                    /*±Z: XY-face*/
}

/* Lumped FE→air volume flux rate (m³/s) for injection.
 * Full 6-face sum of A*(v_rel·n) cancels for rigid translation along the normal when both ±normal
 * faces are exterior (v_rel = -v_e on each, opposite n) — thin membranes then inject nothing from
 * the intended piston motion, while in-plane faces still get non-zero v_rel and produce spurious p.
 * If min(size)/max(size) <= AIR_INJECT_THIN_RATIO, use monopole A_face*v along the thinnest axis. */
#define AIR_INJECT_THIN_RATIO 0.36
inline double air_inject_dV_dot(
    double3 v_e,
    double3 size_e,
    const __global int* neighbors_e6,
    const __global double* velocity)
{
    double sx = size_e.s0;
    double sy = size_e.s1;
    double sz = size_e.s2;
    double smax = fmax(sx, fmax(sy, sz));
    double smin = fmin(sx, fmin(sy, sz));
    if (smin < TINY || smax < TINY)
        return 0.0;
    if (smin <= AIR_INJECT_THIN_RATIO * smax) {
        int thin = (sx <= sy && sx <= sz) ? 0 : ((sy <= sz) ? 1 : 2);
        double A_n = (thin == 0) ? sy * sz : ((thin == 1) ? sx * sz : sx * sy);
        double v_thin = (thin == 0) ? v_e.s0 : ((thin == 1) ? v_e.s1 : v_e.s2);
        return A_n * v_thin;
    }
    double dV_dot = 0.0;
    for (int d = 0; d < FACE_DIRS; d++) {
        int nb = neighbors_e6[d];
        double3 n_out = face_normal(d);
        double3 v_nb = (double3)(0.0, 0.0, 0.0);
        if (nb >= 0) {
            int nb_base = nb * DOF_PER_ELEMENT;
            v_nb = (double3)(velocity[nb_base + 0], velocity[nb_base + 1], velocity[nb_base + 2]);
        }
        double3 v_rel = (nb >= 0) ? (v_nb - v_e) : (-v_e);
        double v_n = dot(n_out, v_rel);
        double A_face = face_area_from_size(d, size_e);
        dV_dot += A_face * v_n;
    }
    return dV_dot;
}

inline double rest_length_from_size(int direction, double3 size_me, double3 size_nb) {
    if (direction < 2) return 0.5 * (size_me.s0 + size_nb.s0);
    if (direction < 4) return 0.5 * (size_me.s1 + size_nb.s1);
    return 0.5 * (size_me.s2 + size_nb.s2);
}

/*Effective rigidity against deformation (s-shaped transition soft↔hard). Anisotropy is taken into account by the calling code: k_soft_dir, k_stiff_dir are transmitted in the communication direction.*/
inline double nonlinear_stiffness(double strain, double k_soft, double k_stiff, double e0, double ew) {
    double arg = -(strain - e0) / (ew + TINY);
    double s = 1.0 / (1.0 + exp(arg));
    return k_soft * (1.0 - s) + k_stiff * s;
}

/*Vector from the center of the neighbor’s face to the center of the “me” face in the local SC of the current element*/
inline double3 face_to_face_vector(
    const double3 center_me_global,
    const double3 center_nb_global,
    double3 arm_me, double3 arm_nb, int direction)
{
    double3 mount_me_local = mounting_point_local(direction, arm_me);
    double3 mount_nb_local = mounting_point_neighbor_local(direction, arm_nb);

    /*Rotations are not taken into account: the local and global CS are the same.
     * Mounting points are simply offset relative to the centers of the elements.*/
    double3 P_me = center_me_global + mount_me_local;
    double3 P_nb = center_nb_global + mount_nb_local;
    double3 vec_global = P_me - P_nb;
    return vec_global;
}

inline void load_dof(const __global double* position, int elem, double* out) {
    int base = elem * DOF_PER_ELEMENT;
    for (int i = 0; i < DOF_PER_ELEMENT; i++)
        out[i] = position[base + i];
}

inline double material_prop(const __global double* material_props, uchar material_id, int prop_idx) {
    int base = ((int)material_id) * MATERIAL_PROPS_STRIDE;
    return material_props[base + prop_idx];
}

inline uchar interaction_law(const __global uchar* laws, int n_materials, uchar mat_i, uchar mat_j) {
    int idx = ((int)mat_i) * n_materials + (int)mat_j;
    return laws[idx];
}

inline int is_pretension_material(uchar material_id) {
    return (material_id == MAT_MEMBRANE_ID || material_id == MAT_SENSOR_ID);
}

/*Elasticity: 4 directions, the sum of forces and moments relative to the center of the element.
 * If debug_elastic != NULL and elem_idx == debug_elem, fills in debug_elastic (see above).
 * If trace_extra != NULL and elem_idx == debug_elem, writes [0..2]=rx,ry,rz; [3]=center_len0; [4]=strain0; [5]=k_eff0; [6]=force_mag0; [7..9]=force_local0; [10..12]=lever0; [13..15]=M0.*/
inline void add_force_elastic(
    double* F,
    const __global double* position,
    const __global double* velocity,
    const __global double* force_external,
    const __global int* boundary,
    const __global double* element_size_xyz,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* neighbors,
    const __global uchar* laws,
    int n_materials,
    const __constant Params* p,
    int elem_idx, int ix, int iy,
    const double* pos_me,
    const double* vel_me,
    double* debug_elastic,
    int debug_elem,
    double* trace_extra)
{
    double position_neighbor[DOF_PER_ELEMENT];
    double3 center = (double3)(pos_me[0], pos_me[1], pos_me[2]);
    double3 vel_vec = (double3)(vel_me[0], vel_me[1], vel_me[2]);
    double3 size_me = vload3(elem_idx, element_size_xyz);
    uchar material_id = material_index[elem_idx];
    double E_parallel = material_prop(material_props, material_id, MAT_PROP_E_PARALLEL);
    double E_perp = material_prop(material_props, material_id, MAT_PROP_E_PERP);
    double thickness_me = size_me.s2;
    double3 arm_me = 0.5 * size_me;

    double3 force_at_dir[FACE_DIRS], point_at_dir[FACE_DIRS];
    for (int i = 0; i < FACE_DIRS; i++) {
        force_at_dir[i] = (double3)(0.0, 0.0, 0.0);
        point_at_dir[i] = center;
    }

    for (int direction_index = 0; direction_index < FACE_DIRS; direction_index++) {
        int neighbor_index = neighbors[elem_idx * FACE_DIRS + direction_index];
        int has_neighbor = (neighbor_index >= 0 && neighbor_index < p->n_elements);
        uchar material_nb = has_neighbor ? material_index[neighbor_index] : material_id;
        uchar law = has_neighbor ? interaction_law(laws, n_materials, material_id, material_nb) : LAW_SOLID_SPRING;
        double3 mount_local = mounting_point_local(direction_index, arm_me);
        double3 lever_global = mount_local;
        double3 point_global = center + lever_global;
        point_at_dir[direction_index] = point_global;

        if (has_neighbor && law == LAW_SOLID_SPRING) {
            load_dof(position, neighbor_index, position_neighbor);
            double3 center_nb_global = (double3)(position_neighbor[0], position_neighbor[1], position_neighbor[2]);
            double vel_neighbor[DOF_PER_ELEMENT];
            load_dof(velocity, neighbor_index, vel_neighbor);
            double3 size_nb = vload3(neighbor_index, element_size_xyz);
            double3 arm_nb = 0.5 * size_nb;

            /*Center-center vector: used for both force direction and spring length.
             * Previously face-to-face (link_len) was used, which gave link_len=0 at rest when rest_len=dx,
             * i.e. deformation -1 and force zeroing - the elements did not have an elastic connection and flew away.*/
            double3 center_me_global = (double3)(pos_me[0], pos_me[1], pos_me[2]);
            double3 center_to_center_global = center_me_global - center_nb_global;
            double center_len = length(center_to_center_global);
            double3 direction_local_me = (center_len > TINY)
                ? (center_to_center_global / center_len)
                : face_normal(direction_index);

            double rest_len = rest_length_from_size(direction_index, size_me, size_nb);
            /*Deformation and force along center-to-center: spring_len=center_len, rest_len=center-to-center at rest.*/
            double strain = (rest_len > TINY) ? ((center_len - rest_len) / rest_len) : 0.0;

            /*Stiffness: directional anisotropy.
             * X/Y: same pattern as host (E_parallel along x, E_perp along y) with sheet thickness in z.
             * Z (faces 4,5): axial link along z — use E_perp through-thickness, area sx*sy, length sz (neighbor rest length still from rest_length_from_size).*/
            double k_axial_x = E_parallel * thickness_me * size_me.s1 / (size_me.s0 + TINY);
            double k_axial_y = E_perp * thickness_me * size_me.s0 / (size_me.s1 + TINY);
            double k_axial_z = E_perp * size_me.s0 * size_me.s1 / (size_me.s2 + TINY);
            double k_axial_dir = (direction_index < 2) ? k_axial_x : ((direction_index < 4) ? k_axial_y : k_axial_z);
            /*For materials with E≈0 (for example, air), we do not replace the stiffness with membrane p->k_*,
             * otherwise you get huge accelerations with a small mass and numerical expansion.*/
            double k_soft_dir = (k_axial_dir > TINY) ? (k_axial_dir / (p->stiffness_ratio + TINY)) : 0.0;
            double k_stiff_dir = (k_axial_dir > TINY) ? k_axial_dir : 0.0;
            double k_eff = nonlinear_stiffness(strain, k_soft_dir, k_stiff_dir, p->strain_transition, p->strain_width);
            double force_elastic = k_eff * (center_len - rest_len);
            double force_tension = 0.0;
            if (direction_index < 4 &&
                is_pretension_material(material_id) &&
                is_pretension_material(material_nb)) {
                double edge_length = (direction_index < 2) ? size_me.s1 : size_me.s0;
                force_tension = p->pre_tension * edge_length;
            }
            /*Bilateral spring: compression (negative force_mag) pushes back; no unilateral clip.*/
            double force_mag = force_elastic + force_tension;
            double3 force_local = -force_mag * direction_local_me;
            double eta_me = material_prop(material_props, material_id, MAT_PROP_ETA_VISC);
            double eta_nb = material_prop(material_props, material_nb, MAT_PROP_ETA_VISC);
            double eta_eff = 0.5 * (eta_me + eta_nb);
            if (eta_eff > 0.0) {
                double face_area_solid = face_area_from_size(direction_index, size_me);
                double c_link = eta_eff * face_area_solid / (rest_len + TINY);
                double3 vel_nb_vec = (double3)(vel_neighbor[0], vel_neighbor[1], vel_neighbor[2]);
                double3 v_rel = vel_vec - vel_nb_vec;
                double v_rel_n = dot(v_rel, direction_local_me);
                force_local += (-c_link * v_rel_n) * direction_local_me;
            }
            force_at_dir[direction_index] = force_local;

            #if ENABLE_DEBUG
            if (trace_extra != NULL && elem_idx == debug_elem && direction_index == 0) {
                trace_extra[3] = center_len; trace_extra[4] = strain; trace_extra[5] = k_eff; trace_extra[6] = force_mag;
                trace_extra[7] = force_local.s0; trace_extra[8] = force_local.s1; trace_extra[9] = force_local.s2;
                trace_extra[10] = lever_global.s0; trace_extra[11] = lever_global.s1; trace_extra[12] = lever_global.s2;
                trace_extra[16] = center_len; trace_extra[17] = rest_len;
                trace_extra[18] = force_tension;
            }
            #endif
        } else {
            /*For air-surface interface/interface: normal pressure + viscous resistance.*/
            double3 normal = face_normal(direction_index);
            double face_area = face_area_from_size(direction_index, size_me);
            double vn = dot(vel_vec, normal);  /*speed of FE relative to air along the normal*/
            if (has_neighbor) {
                double vel_nb[DOF_PER_ELEMENT];
                load_dof(velocity, neighbor_index, vel_nb);
                double3 vel_nb_vec = (double3)(vel_nb[0], vel_nb[1], vel_nb[2]);
                vn -= dot(vel_nb_vec, normal);
            }
            double a_eff = sqrt(face_area / 3.14159265358979);
            double v_abs = fabs(vn);
            double rho_eff = has_neighbor ? material_prop(material_props, material_nb, MAT_PROP_DENSITY) : p->rho_air;
            if (rho_eff < TINY) rho_eff = p->rho_air;
            double cd_me = material_prop(material_props, material_id, MAT_PROP_CD);
            double cd_nb = has_neighbor ? material_prop(material_props, material_nb, MAT_PROP_CD) : cd_me;
            double cd_eff = 0.5 * (cd_me + cd_nb);
            double Re = rho_eff * v_abs * (2.0 * a_eff) / (p->mu_air + TINY);
            double transition = 1.0 / (1.0 + exp(-(Re - 100.0) / 50.0));
            double c_linear = 6.0 * 3.14159265358979 * p->mu_air * a_eff;
            double c_quad = 0.5 * rho_eff * cd_eff * face_area * v_abs;
            double c_eff = (1.0 - transition) * c_linear + transition * c_quad;
            /*The absolute ATM term has been removed: the pressure comes from the air-field via add_air_pressure_to_force_external.*/
            force_at_dir[direction_index] = (-c_eff * vn) * normal;
        }
    }

    double3 F_total = (double3)(0.0, 0.0, 0.0);
    for (int i = 0; i < FACE_DIRS; i++)
        F_total += force_at_dir[i];
    F[0] += F_total.s0;
    F[1] += F_total.s1;
    F[2] += F_total.s2;
    
    #if ENABLE_DEBUG
    if (debug_elastic != NULL && elem_idx == debug_elem) {
        debug_elastic[0] = F_total.s0; debug_elastic[1] = F_total.s1; debug_elastic[2] = F_total.s2;
        debug_elastic[3] = 0.0; debug_elastic[4] = 0.0; debug_elastic[5] = 0.0; /*moments are disabled*/
        for (int i = 0; i < FACE_DIRS; i++) {
            double3 f = force_at_dir[i];
            double3 lever = point_at_dir[i] - center;
            debug_elastic[6 + i*6 + 0] = f.s0; debug_elastic[6 + i*6 + 1] = f.s1; debug_elastic[6 + i*6 + 2] = f.s2;
            debug_elastic[6 + i*6 + 3] = lever.s0; debug_elastic[6 + i*6 + 4] = lever.s1; debug_elastic[6 + i*6 + 5] = lever.s2;
        }
    }
    #endif
}

inline void add_force_external(double* F, const __global double* force_external, int base) {
    for (int d = 0; d < DOF_PER_ELEMENT; d++)
        F[d] += force_external[base + d];
}

inline uint mix_u32(uint x) {
    x ^= x >> 16;
    x *= (uint)0x7feb352dU;
    x ^= x >> 15;
    x *= (uint)0x846ca68bU;
    x ^= x >> 16;
    return x;
}

inline double kernel_force_wave_base(const __constant Params* p) {
    int step_idx = p->debug_step;
    if (step_idx < 0)
        step_idx = 0;
    double dt_wave = (p->force_dt > 0.0) ? p->force_dt : p->dt;
    if (dt_wave <= 0.0)
        dt_wave = 1e-30;
    double t = ((double)step_idx) * dt_wave;
    int shape = p->force_shape_id;
    if (shape == FORCE_SHAPE_UNIFORM)
        return 1.0;
    if (shape == FORCE_SHAPE_SINE)
        return sin(2.0 * 3.14159265358979323846 * p->force_freq_hz * t + p->force_phase_rad);
    if (shape == FORCE_SHAPE_SQUARE) {
        double s = sin(2.0 * 3.14159265358979323846 * p->force_freq_hz * t + p->force_phase_rad);
        return (s >= 0.0) ? 1.0 : -1.0;
    }
    if (shape == FORCE_SHAPE_CHIRP) {
        double f0 = p->force_freq_hz;
        double f1 = p->force_freq_end_hz;
        double T = (p->force_duration_s > 0.0) ? p->force_duration_s : dt_wave;
        T = fmax(T, 1e-30);
        double k = (f1 - f0) / T;
        double ph;
        if (t <= T) {
            ph = 2.0 * 3.14159265358979323846 * (f0 * t + 0.5 * k * t * t) + p->force_phase_rad;
        } else {
            double ph_T = 2.0 * 3.14159265358979323846 * (f0 * T + 0.5 * k * T * T) + p->force_phase_rad;
            ph = ph_T + 2.0 * 3.14159265358979323846 * f1 * (t - T);
        }
        return sin(ph);
    }
    if (shape == FORCE_SHAPE_WHITE_NOISE) {
        uint x = (uint)step_idx;
        x ^= p->force_noise_seed_u32;
        uint h = mix_u32(x);
        double u01 = ((double)h) * (1.0 / 4294967295.0);
        return 2.0 * u01 - 1.0;
    }
    return (step_idx == 0) ? 1.0 : 0.0;
}

inline double kernel_force_wave_pressure(const __constant Params* p, double host_pressure_pa) {
    if (p->force_wave_enabled == 0)
        return host_pressure_pa;
    /* base in [-1,1] (or 0/1 for impulse): excursion in [-force_amp/2, +force_amp/2], not full ±force_amp */
    return p->force_offset + 0.5 * p->force_amp * kernel_force_wave_base(p);
}

inline void add_force_external_generated(
    double* F,
    int elem_idx,
    double pressure_pa,
    const __global uchar* force_drive_mask,
    const __global uchar* force_drive_axis,
    const __global double* force_drive_area)
{
    if (force_drive_mask[elem_idx] == (uchar)0)
        return;
    int ax = (int)force_drive_axis[elem_idx];
    if (ax < 0 || ax > 2)
        return;
    F[ax] += pressure_pa * force_drive_area[elem_idx];
}

inline void add_force_air_external(double* F, const __global double* air_force_external, int base) {
    if (air_force_external != NULL) {
        for (int d = 0; d < DOF_PER_ELEMENT; d++)
            F[d] += air_force_external[base + d];
    }
}

inline void force_boundary_zero(double* F, int is_boundary) {
    if (is_boundary)
        for (int d = 0; d < DOF_PER_ELEMENT; d++) F[d] = 0.0;
}

/*We don’t use moments - we reset them to zero so that the corner DOFs do not receive acceleration.*/
inline void force_moments_zero(double* F) {
    F[3] = 0.0;
    F[4] = 0.0;
    F[5] = 0.0;
}

inline void get_mass_safe(double density, double3 size_me, double* mass_safe);

__kernel void diaphragm_rk4_acc(
    const __global double* position,
    const __global double* velocity,
    const __global double* force_external,
    double force_pressure_pa,
    const __global uchar* force_drive_mask,
    const __global uchar* force_drive_axis,
    const __global double* force_drive_area,
    const __global double* air_force_external,
    const __global int* boundary_mask_elements,
    const __global double* element_size_xyz,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* neighbors,
    const __global uchar* laws,
    int n_materials,
    __constant Params* params,
    __global double* acceleration_out,
    __global int* first_bad_elem,
    __global int* first_bad_meta,
    __global int* first_bad_neighbor_elem,
    __global int* first_bad_interface_dir,
    int acc_stage_id,
    int validate_finite)
{
    int elem_idx = get_global_id(0);
    __constant Params* p = params;
    if (elem_idx >= p->n_elements) return;

    int base = elem_idx * DOF_PER_ELEMENT;
    int ix = elem_idx % p->nx;
    int iy = elem_idx / p->nx;
    int is_boundary = boundary_mask_elements[elem_idx];
    double3 size_me = vload3(elem_idx, element_size_xyz);
    uchar material_id = material_index[elem_idx];
    double density = material_prop(material_props, material_id, MAT_PROP_DENSITY);

    double pos_me[DOF_PER_ELEMENT];
    double vel_me[DOF_PER_ELEMENT];
    load_dof(position, elem_idx, pos_me);
    load_dof(velocity, elem_idx, vel_me);

    double F[DOF_PER_ELEMENT] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    add_force_external(F, force_external, base);
    double drive_pressure_pa = kernel_force_wave_pressure(p, force_pressure_pa);
    add_force_external_generated(
        F,
        elem_idx,
        drive_pressure_pa,
        force_drive_mask,
        force_drive_axis,
        force_drive_area
    );
    add_force_air_external(F, air_force_external, base);
    add_force_elastic(
        F,
        position,
        velocity,
        force_external,
        boundary_mask_elements,
        element_size_xyz,
        material_index,
        material_props,
        neighbors,
        laws,
        n_materials,
        p,
        elem_idx,
        ix,
        iy,
        pos_me,
        vel_me,
        NULL,
        -1,
        NULL
    );
    force_boundary_zero(F, is_boundary);
    force_moments_zero(F);

    double mass_safe[DOF_PER_ELEMENT];
    get_mass_safe(density, size_me, mass_safe);

    for (int d = 0; d < DOF_PER_ELEMENT; d++) {
        if (is_boundary || d >= 3) {
            acceleration_out[base + d] = 0.0;
        } else {
            acceleration_out[base + d] = F[d] / mass_safe[d];
        }
    }

    /* NaN/Inf guard: record first offending element (atomic: one winner). */
    if (validate_finite && !is_boundary) {
        int reason = 0;
        for (int d = 0; d < 3; d++) {
            if (!isfinite(pos_me[d])) {
                reason |= (1 << (0 + d));      /* pos_me[d] */
            }
            if (!isfinite(vel_me[d])) {
                reason |= (1 << (3 + d));      /* vel_me[d] */
            }
            if (!isfinite(F[d])) {
                reason |= (1 << (6 + d));      /* F[d] */
            }
            if (!isfinite(acceleration_out[base + d])) {
                reason |= (1 << (9 + d));      /* a[d] */
            }
            if (!isfinite(air_force_external[base + d])) {
                reason |= (1 << (12 + d));     /* air_force_external[d] */
            }
        }

        if (reason != 0) {
            int bad_neighbor = -1;
            int interface_term_bad = 0;
            int interface_dir = -1;

            /* Best-effort origin hint: if a neighbor has non-finite state, store it. */
            uchar material_id = material_index[elem_idx];
            double cd_me = material_prop(material_props, material_id, MAT_PROP_CD);
            double3 size_me = vload3(elem_idx, element_size_xyz);
            double3 vel_vec = (double3)(vel_me[0], vel_me[1], vel_me[2]);

            /* Scan directions that use the interface (normal-pressure/viscous) branch. */
            for (int direction_index = 0; direction_index < FACE_DIRS; direction_index++) {
                int nb = neighbors[elem_idx * FACE_DIRS + direction_index];

                int has_neighbor = (nb >= 0 && nb < p->n_elements);
                if (has_neighbor) {
                    int nb_base = nb * DOF_PER_ELEMENT;
                    if (!isfinite(position[nb_base + 0]) || !isfinite(position[nb_base + 1]) || !isfinite(position[nb_base + 2]) ||
                        !isfinite(velocity[nb_base + 0]) || !isfinite(velocity[nb_base + 1]) || !isfinite(velocity[nb_base + 2])) {
                        bad_neighbor = nb;
                        break;
                    }
                }

                if (has_neighbor) {
                    uchar material_nb = material_index[nb];
                    uchar law = interaction_law(laws, n_materials, material_id, material_nb);
                    if (law == LAW_SOLID_SPRING) {
                        /* This direction uses the elastic branch, not the interface branch. */
                        continue;
                    }
                }

                double3 normal = face_normal(direction_index);
                double vn = dot(vel_vec, normal);
                if (has_neighbor) {
                    int nb_base = nb * DOF_PER_ELEMENT;
                    double3 v_nb_vec = (double3)(velocity[nb_base + 0], velocity[nb_base + 1], velocity[nb_base + 2]);
                    vn -= dot(v_nb_vec, normal);
                }

                double face_area = face_area_from_size(direction_index, size_me);
                double a_eff = sqrt(face_area / 3.14159265358979);
                double v_abs = fabs(vn);

                double rho_eff = has_neighbor ? material_prop(material_props, material_index[nb], MAT_PROP_DENSITY) : p->rho_air;
                if (rho_eff < TINY) rho_eff = p->rho_air;
                double cd_nb = has_neighbor ? material_prop(material_props, material_index[nb], MAT_PROP_CD) : cd_me;
                double cd_eff = 0.5 * (cd_me + cd_nb);

                double Re = rho_eff * v_abs * (2.0 * a_eff) / (p->mu_air + TINY);
                double transition = 1.0 / (1.0 + exp(-(Re - 100.0) / 50.0));
                double c_linear = 6.0 * 3.14159265358979 * p->mu_air * a_eff;
                double c_quad = 0.5 * rho_eff * cd_eff * face_area * v_abs;
                double c_eff = (1.0 - transition) * c_linear + transition * c_quad;
                double3 force_at_dir = (-c_eff * vn) * normal;

                if (!isfinite(vn) || !isfinite(Re) || !isfinite(transition) || !isfinite(c_eff) ||
                    !isfinite(force_at_dir.s0) || !isfinite(force_at_dir.s1) || !isfinite(force_at_dir.s2)) {
                    interface_term_bad = 1;
                    interface_dir = direction_index;
                    break;
                }
            }

            if (interface_term_bad) reason |= (1 << 15); /* interface term non-finite */
            int prev = atomic_cmpxchg(first_bad_elem, 0x7FFFFFFF, elem_idx);
            /* Winner sets meta once; other threads do nothing. */
            if (prev == 0x7FFFFFFF) {
                /* Pack: [15..0]=reason bits, [31..16]=stage id. */
                int meta = ((acc_stage_id & 0xFFFF) << 16) | (reason & 0xFFFF);
                first_bad_meta[0] = meta;
                first_bad_neighbor_elem[0] = bad_neighbor;
                first_bad_interface_dir[0] = interface_dir;
            }
        }
    }
}

inline void get_mass_safe(double density, double3 size_me, double* mass_safe) {
    double sx = size_me.s0;
    double sy = size_me.s1;
    double sz = size_me.s2;
    double volume = sx * sy * sz;
    double mass = density * volume;
    if (mass < 1e-18) mass = 1e-18;
    double Ixx = mass * (sy * sy + sz * sz) / 12.0;
    double Iyy = mass * (sx * sx + sz * sz) / 12.0;
    double Izz = mass * (sx * sx + sy * sy) / 12.0;
    mass_safe[0] = mass + 1e-18;
    mass_safe[1] = mass + 1e-18;
    mass_safe[2] = mass + 1e-18;
    mass_safe[3] = Ixx + 1e-18;
    mass_safe[4] = Iyy + 1e-18;
    mass_safe[5] = Izz + 1e-18;
}

/* Writes only Δp from FE motion; leapfrog uses p^n from p_curr and adds this once to p^{n+1}. */
__kernel void air_inject_reduce_to_pressure(
    const __global int* csr_offsets,                 /* [n_air+1] */
    const __global int* csr_indices,                 /* [nnz] elem ids */
    const __global double* csr_inject_sign,          /* [nnz] +1 / -1 bilateral sites */
    const __global double* velocity,                 /* [n_elements*6] */
    const __global int* neighbors,                   /* [n_elements*6] */
    const __global int* boundary_mask_elements,      /* [n_elements] */
    const __global double* element_size_xyz,         /* [n_elements*3] */
    const __global uchar* material_index,            /* [n_elements] */
    const __global double* material_props,           /* [n_materials*8] */
    int n_materials,
    int n_air,
    double rho_air,
    double c_sound,
    double dt,
    double dx_air,
    double dy_air,
    double dz_air,
    __global double* inject_dp_out)                  /* [n_air] */
{
    int cell = get_global_id(0);
    if (cell >= n_air) return;
    double s = 0.0;
    int beg = csr_offsets[cell];
    int end = csr_offsets[cell + 1];
    double cell_vol = fmax(dx_air * dy_air * dz_air, 1e-18);
    double bulk_modulus = fmax(rho_air * c_sound * c_sound, 1e-18);
    for (int k = beg; k < end; k++) {
        int e = csr_indices[k];
        if (boundary_mask_elements[e] != 0) continue;
        uchar mat_id = material_index[e];
        if ((int)mat_id >= n_materials) continue;
        double acoustic_inject = material_prop(material_props, mat_id, MAT_PROP_ACOUSTIC_INJECT);
        if (acoustic_inject <= 0.0) continue;
        int base = e * DOF_PER_ELEMENT;
        double3 v_e = (double3)(velocity[base + 0], velocity[base + 1], velocity[base + 2]);
        double3 size_e = vload3(e, element_size_xyz);
        double dV_dot = air_inject_dV_dot(v_e, size_e, neighbors + e * FACE_DIRS, velocity);
        double inj_s = csr_inject_sign[k];
        double dp = acoustic_inject * bulk_modulus * (dV_dot * dt / cell_vol) * inj_s;
        s += dp;
    }
    inject_dp_out[cell] = s;
}

__kernel void air_pressure_to_fe_force(
    const __global double* p_field,                  /* [n_air] */
    const __global int* air_map_6,                   /* [n_elements*6] */
    const __global int* air_elem_map,                /* [n_elements] */
    const __global double* element_size_xyz,         /* [n_elements*3] */
    const __global uchar* material_index,            /* [n_elements] */
    const __global uchar* fe_air_coupling_mask,      /* [n_elements], 1=allow air->FE traction */
    const __global double* material_props,           /* [n_materials*8] */
    int n_materials,
    int n_air,
    int n_elements,
    double air_pressure_fallback,
    double coupling_gain,
    double dx_air,
    double dy_air,
    double dz_air,
    __global double* air_force_external_out)         /* [n_elements*6] */
{
    int elem = get_global_id(0);
    if (elem >= n_elements) return;
    int base6 = elem * DOF_PER_ELEMENT;
    air_force_external_out[base6 + 0] = 0.0;
    air_force_external_out[base6 + 1] = 0.0;
    air_force_external_out[base6 + 2] = 0.0;
    air_force_external_out[base6 + 3] = 0.0;
    air_force_external_out[base6 + 4] = 0.0;
    air_force_external_out[base6 + 5] = 0.0;

    int c = air_elem_map[elem];
    double p_center = (c >= 0 && c < n_air) ? p_field[c] : air_pressure_fallback;
    int mbase = elem * FACE_DIRS;
    int ixp = air_map_6[mbase + 0], ixm = air_map_6[mbase + 1];
    int iyp = air_map_6[mbase + 2], iym = air_map_6[mbase + 3];
    int izp = air_map_6[mbase + 4], izm = air_map_6[mbase + 5];
    double pxp = (ixp >= 0 && ixp < n_air) ? p_field[ixp] : p_center;
    double pxm = (ixm >= 0 && ixm < n_air) ? p_field[ixm] : p_center;
    double pyp = (iyp >= 0 && iyp < n_air) ? p_field[iyp] : p_center;
    double pym = (iym >= 0 && iym < n_air) ? p_field[iym] : p_center;
    double pzp = (izp >= 0 && izp < n_air) ? p_field[izp] : p_center;
    double pzm = (izm >= 0 && izm < n_air) ? p_field[izm] : p_center;

    double3 sz = vload3(elem, element_size_xyz);
    double area_x = sz.s1 * sz.s2;
    double area_y = sz.s0 * sz.s2;
    double area_z = sz.s0 * sz.s1;
    if (fe_air_coupling_mask[elem] == (uchar)0) {
        return;
    }
    uchar mat_id = material_index[elem];
    double z0 = 1.2 * 343.0;
    double z_solid = ((int)mat_id < n_materials)
        ? fmax(0.0, material_prop(material_props, mat_id, MAT_PROP_ACOUSTIC_IMPEDANCE))
        : z0;
    (void)coupling_gain;
    /* impedance-consistent traction transfer (matched: 1, rigid limit: ~2, soft: ~0) */
    double scale = (2.0 * z_solid) / (z_solid + z0 + TINY);
    /* Fx ≈ −(∂p/∂x)·V with central difference across stencil spacing 2·dx_air, V = sx·sy·sz, area_x = sy·sz. */
    double inv_2dx = 1.0 / (2.0 * dx_air + TINY);
    double inv_2dy = 1.0 / (2.0 * dy_air + TINY);
    double inv_2dz = 1.0 / (2.0 * dz_air + TINY);
    double sx = sz.s0;
    double sy = sz.s1;
    double szz = sz.s2;
    air_force_external_out[base6 + 0] = scale * (pxm - pxp) * area_x * sx * inv_2dx;
    air_force_external_out[base6 + 1] = scale * (pym - pyp) * area_y * sy * inv_2dy;
    air_force_external_out[base6 + 2] = scale * (pzm - pzp) * area_z * szz * inv_2dz;
}

/*Preparing the state of the next stage RK4:
 * x_stage = x0 + alpha * dt * v_for_pos
 * v_stage = v0 + alpha * dt * a_for_vel*/
__kernel void diaphragm_rk4_stage_state(
    const __global double* position_0,
    const __global double* velocity_0,
    const __global double* velocity_for_position,
    const __global double* acceleration_for_velocity,
    const __global int* boundary_mask_elements,
    int n_elements,
    double dt,
    double alpha,
    __global double* position_stage,
    __global double* velocity_stage,
    __global double* velocity_stage_snapshot)
{
    int elem = get_global_id(0);
    if (elem >= n_elements) return;
    int base = elem * DOF_PER_ELEMENT;
    int is_boundary = boundary_mask_elements[elem];
    double a_dt = alpha * dt;

    for (int d = 0; d < 3; d++) {
        if (is_boundary) {
            position_stage[base + d] = position_0[base + d];
            velocity_stage[base + d] = velocity_0[base + d];
        } else {
            position_stage[base + d] = position_0[base + d] + a_dt * velocity_for_position[base + d];
            velocity_stage[base + d] = velocity_0[base + d] + a_dt * acceleration_for_velocity[base + d];
        }
    }
    for (int d = 3; d < DOF_PER_ELEMENT; d++) {
        position_stage[base + d] = position_0[base + d];
        velocity_stage[base + d] = velocity_0[base + d];
    }
    for (int d = 0; d < DOF_PER_ELEMENT; d++) {
        velocity_stage_snapshot[base + d] = velocity_stage[base + d];
    }
}

/*Final RK4 step for progressive DOFs.*/
__kernel void diaphragm_rk4_finalize(
    const __global double* position_0,
    const __global double* velocity_0,
    const __global double* velocity_k2,
    const __global double* velocity_k3,
    const __global double* velocity_k4,
    const __global double* acc_k1,
    const __global double* acc_k2,
    const __global double* acc_k3,
    const __global double* acc_k4,
    const __global int* boundary_mask_elements,
    int n_elements,
    double dt,
    __global double* position_out,
    __global double* velocity_out)
{
    int elem = get_global_id(0);
    if (elem >= n_elements) return;
    int base = elem * DOF_PER_ELEMENT;
    int is_boundary = boundary_mask_elements[elem];
    double w = dt / 6.0;

    for (int d = 0; d < 3; d++) {
        if (is_boundary) {
            position_out[base + d] = position_0[base + d];
            velocity_out[base + d] = velocity_0[base + d];
        } else {
            double kx1 = velocity_0[base + d];
            double kx2 = velocity_k2[base + d];
            double kx3 = velocity_k3[base + d];
            double kx4 = velocity_k4[base + d];
            position_out[base + d] = position_0[base + d] + w * (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4);
            velocity_out[base + d] = velocity_0[base + d] + w * (
                acc_k1[base + d] + 2.0 * acc_k2[base + d] + 2.0 * acc_k3[base + d] + acc_k4[base + d]
            );
        }
    }
    for (int d = 3; d < DOF_PER_ELEMENT; d++) {
        position_out[base + d] = position_0[base + d];
        velocity_out[base + d] = velocity_0[base + d];
    }
}

/* --- Acoustic air field: leapfrog p_tt = c^2 lap(p) with stable sparse-boundary damping --- */
/* FACE_DIRS order matches Python: +X,-X,+Y,-Y,+Z,-Z. */
/* air_absorb[6*i+d]: reserved; currently not used in this kernel. */

/* Mur first-order (nu = c*dt/h): p_ghost = p_i + ((nu-1)/(nu+1))*(p_i - p_inward). Better 1D absorbing
 * near CFL~1 than time-only Sommerfeld. If has_inward==0, fall back to Sommerfeld p_i - (h/c)*dp/dt.
 *
 * NOTE: For very sparse cells (many missing neighbors), directly injecting Sommerfeld ghosts into the
 * second-derivative stencil can destabilize leapfrog. In the main kernel we therefore keep Laplacian
 * ghosts neutral (Neumann-like p_ghost = p_i) and apply Sommerfeld as a bounded first-order damping term. */
inline double air_boundary_ghost_pressure(
    double p_i,
    double p_im1_i,
    double p_inward,
    int has_inward,
    double h_axis,
    double c_sound,
    double dt_safe)
{
    double pt = (p_i - p_im1_i) / dt_safe;
    /* First-order Sommerfeld (always stable in sign for outgoing linearization). */
    double sommerfeld = p_i - (h_axis / c_sound) * pt;
    if (!has_inward)
        return sommerfeld;
    double nu = c_sound * dt_safe / (h_axis + TINY);
    /* Mur coeff -> ±1 when nu -> 0 or +inf: amplifies roundoff / violates discrete CFL. Blend to Sommerfeld. */
    if (nu < 0.1 || nu > 0.95)
        return sommerfeld;
    double coeff = (nu - 1.0) / (nu + 1.0 + TINY);
    if (coeff > 0.82)
        coeff = 0.82;
    if (coeff < -0.82)
        coeff = -0.82;
    return p_i + coeff * (p_i - p_inward);
}

__kernel void air_acoustic_leapfrog_sommerfeld(
    __global const double* p_im1,
    __global const double* p_curr,
    __global const double* inject_dp,
    __global double* p_next,
    __global const int* air_nb,
    __global const uchar* air_absorb,
    int n_air,
    double dx,
    double dy,
    double dz,
    double c_sound,
    double dt)
{
    int i = get_global_id(0);
    if (i >= n_air) return;
    const int S = FACE_DIRS;
    int base = i * S;
    double dt_safe = dt;
    if (dt_safe < 1e-30) dt_safe = 1e-30;
    double c = c_sound;
    if (c < 1e-30) c = 1e-30;

    int jxp = air_nb[base + 0];
    int jxm = air_nb[base + 1];
    int jyp = air_nb[base + 2];
    int jym = air_nb[base + 3];
    int jzp = air_nb[base + 4];
    int jzm = air_nb[base + 5];

    double pxp, pxm, pyp, pym, pzp, pzm;
    double pi = p_curr[i];
    double pim = p_im1[i];
    (void)air_absorb;

    int ok_xp = (jxp >= 0 && jxp < n_air);
    int ok_xm = (jxm >= 0 && jxm < n_air);
    int ok_yp = (jyp >= 0 && jyp < n_air);
    int ok_ym = (jym >= 0 && jym < n_air);
    int ok_zp = (jzp >= 0 && jzp < n_air);
    int ok_zm = (jzm >= 0 && jzm < n_air);

    pxp = ok_xp ? p_curr[jxp] : pi;
    pxm = ok_xm ? p_curr[jxm] : pi;
    pyp = ok_yp ? p_curr[jyp] : pi;
    pym = ok_ym ? p_curr[jym] : pi;
    pzp = ok_zp ? p_curr[jzp] : pi;
    pzm = ok_zm ? p_curr[jzm] : pi;

    double inv_dx2 = 1.0 / (dx * dx + TINY);
    double inv_dy2 = 1.0 / (dy * dy + TINY);
    double inv_dz2 = 1.0 / (dz * dz + TINY);

    double lap =
        (pxp - 2.0 * pi + pxm) * inv_dx2 +
        (pyp - 2.0 * pi + pym) * inv_dy2 +
        (pzp - 2.0 * pi + pzm) * inv_dz2;

    int miss_xp = ok_xp ? 0 : 1;
    int miss_xm = ok_xm ? 0 : 1;
    int miss_yp = ok_yp ? 0 : 1;
    int miss_ym = ok_ym ? 0 : 1;
    int miss_zp = ok_zp ? 0 : 1;
    int miss_zm = ok_zm ? 0 : 1;
    double missing_inv_h =
        (double)(miss_xp + miss_xm) / (dx + TINY) +
        (double)(miss_yp + miss_ym) / (dy + TINY) +
        (double)(miss_zp + miss_zm) / (dz + TINY);
    double gamma = c * dt_safe * missing_inv_h;
    if (gamma > 0.95) gamma = 0.95;
    if (gamma < 0.0) gamma = 0.0;

    double c2 = c * c;
    double inj = inject_dp[i];
    p_next[i] = 2.0 * pi - pim + c2 * (dt * dt) * lap - gamma * (pi - pim) + inj;
}

/* Second-order pressure-only wave equation with boundary kinds from air_absorb:
 * - AIR_BC_OPEN: Sommerfeld/Mur-style radiating ghost + bounded damping
 * - AIR_BC_RIGID: Neumann-like ghost p_ghost = p_i
 * For backward compatibility, old 0/1 masks map naturally: 1=open, 0=rigid.
 */
__kernel void air_pressure_wave_second_order_bc(
    __global const double* p_im1,
    __global const double* p_curr,
    __global const double* inject_dp,
    __global double* p_next,
    __global const int* air_nb,
    __global const uchar* air_absorb,
    int n_air,
    double dx,
    double dy,
    double dz,
    double c_sound,
    double dt)
{
    int i = get_global_id(0);
    if (i >= n_air) return;
    const int S = FACE_DIRS;
    int base = i * S;
    double dt_safe = dt;
    if (dt_safe < 1e-30) dt_safe = 1e-30;
    double c = c_sound;
    if (c < 1e-30) c = 1e-30;

    int jxp = air_nb[base + 0];
    int jxm = air_nb[base + 1];
    int jyp = air_nb[base + 2];
    int jym = air_nb[base + 3];
    int jzp = air_nb[base + 4];
    int jzm = air_nb[base + 5];

    double pi = p_curr[i];
    double pim = p_im1[i];

    int ok_xp = (jxp >= 0 && jxp < n_air);
    int ok_xm = (jxm >= 0 && jxm < n_air);
    int ok_yp = (jyp >= 0 && jyp < n_air);
    int ok_ym = (jym >= 0 && jym < n_air);
    int ok_zp = (jzp >= 0 && jzp < n_air);
    int ok_zm = (jzm >= 0 && jzm < n_air);

    uchar bc_xp = air_absorb[base + 0];
    uchar bc_xm = air_absorb[base + 1];
    uchar bc_yp = air_absorb[base + 2];
    uchar bc_ym = air_absorb[base + 3];
    uchar bc_zp = air_absorb[base + 4];
    uchar bc_zm = air_absorb[base + 5];

    double pxp, pxm, pyp, pym, pzp, pzm;

    if (ok_xp) {
        pxp = p_curr[jxp];
    } else if (bc_xp == (uchar)AIR_BC_OPEN) {
        double p_in = ok_xm ? p_curr[jxm] : pi;
        pxp = air_boundary_ghost_pressure(pi, pim, p_in, ok_xm, dx, c, dt_safe);
    } else {
        pxp = pi;
    }
    if (ok_xm) {
        pxm = p_curr[jxm];
    } else if (bc_xm == (uchar)AIR_BC_OPEN) {
        double p_in = ok_xp ? p_curr[jxp] : pi;
        pxm = air_boundary_ghost_pressure(pi, pim, p_in, ok_xp, dx, c, dt_safe);
    } else {
        pxm = pi;
    }

    if (ok_yp) {
        pyp = p_curr[jyp];
    } else if (bc_yp == (uchar)AIR_BC_OPEN) {
        double p_in = ok_ym ? p_curr[jym] : pi;
        pyp = air_boundary_ghost_pressure(pi, pim, p_in, ok_ym, dy, c, dt_safe);
    } else {
        pyp = pi;
    }
    if (ok_ym) {
        pym = p_curr[jym];
    } else if (bc_ym == (uchar)AIR_BC_OPEN) {
        double p_in = ok_yp ? p_curr[jyp] : pi;
        pym = air_boundary_ghost_pressure(pi, pim, p_in, ok_yp, dy, c, dt_safe);
    } else {
        pym = pi;
    }

    if (ok_zp) {
        pzp = p_curr[jzp];
    } else if (bc_zp == (uchar)AIR_BC_OPEN) {
        double p_in = ok_zm ? p_curr[jzm] : pi;
        pzp = air_boundary_ghost_pressure(pi, pim, p_in, ok_zm, dz, c, dt_safe);
    } else {
        pzp = pi;
    }
    if (ok_zm) {
        pzm = p_curr[jzm];
    } else if (bc_zm == (uchar)AIR_BC_OPEN) {
        double p_in = ok_zp ? p_curr[jzp] : pi;
        pzm = air_boundary_ghost_pressure(pi, pim, p_in, ok_zp, dz, c, dt_safe);
    } else {
        pzm = pi;
    }

    double inv_dx2 = 1.0 / (dx * dx + TINY);
    double inv_dy2 = 1.0 / (dy * dy + TINY);
    double inv_dz2 = 1.0 / (dz * dz + TINY);

    double lap =
        (pxp - 2.0 * pi + pxm) * inv_dx2 +
        (pyp - 2.0 * pi + pym) * inv_dy2 +
        (pzp - 2.0 * pi + pzm) * inv_dz2;

    int open_x = ((!ok_xp && bc_xp == (uchar)AIR_BC_OPEN) ? 1 : 0) + ((!ok_xm && bc_xm == (uchar)AIR_BC_OPEN) ? 1 : 0);
    int open_y = ((!ok_yp && bc_yp == (uchar)AIR_BC_OPEN) ? 1 : 0) + ((!ok_ym && bc_ym == (uchar)AIR_BC_OPEN) ? 1 : 0);
    int open_z = ((!ok_zp && bc_zp == (uchar)AIR_BC_OPEN) ? 1 : 0) + ((!ok_zm && bc_zm == (uchar)AIR_BC_OPEN) ? 1 : 0);
    double open_inv_h =
        (double)open_x / (dx + TINY) +
        (double)open_y / (dy + TINY) +
        (double)open_z / (dz + TINY);
    double gamma = c * dt_safe * open_inv_h;
    if (gamma > 0.95) gamma = 0.95;
    if (gamma < 0.0) gamma = 0.0;

    double c2 = c * c;
    double inj = inject_dp[i];
    p_next[i] = 2.0 * pi - pim + c2 * (dt * dt) * lap - gamma * (pi - pim) + inj;
}

/* Packed FE state for MAT_SENSOR elements only: one work-item per sensor slot.
 * snapshot_out layout per slot k: [pos 0..5][vel 0..5] = 12 doubles (DOF_PER_ELEMENT each). */
__kernel void gather_sensor_fe_snapshot(
    const __global double* position,
    const __global double* velocity,
    const __global int* sensor_elem_indices,
    int n_sensor,
    int n_elements,
    __global double* snapshot_out)
{
    int k = get_global_id(0);
    if (k >= n_sensor) return;
    int elem = sensor_elem_indices[k];
    if (elem < 0 || elem >= n_elements) return;
    int base = elem * DOF_PER_ELEMENT;
    int out_base = k * (2 * DOF_PER_ELEMENT);
    for (int d = 0; d < DOF_PER_ELEMENT; d++) {
        snapshot_out[out_base + d] = position[base + d];
        snapshot_out[out_base + DOF_PER_ELEMENT + d] = velocity[base + d];
    }
}

/* Lexicographic voxel order (matches host reshape nz,ny,nx): idx = iz*(ny*nx) + iy*nx + ix. */
__kernel void gather_air_pressure_xy_history_slice(
    const __global double* p_curr,
    int nx,
    int ny,
    int nz,
    int iz_plane,
    __global double* slice_out)
{
    int n = nx * ny;
    int k = get_global_id(0);
    if (k >= n) return;
    int iy = k / nx;
    int ix = k % nx;
    if (iz_plane < 0) iz_plane = 0;
    if (iz_plane >= nz) iz_plane = nz - 1;
    int idx = iz_plane * (ny * nx) + iy * nx + ix;
    slice_out[k] = p_curr[idx];
}

__kernel void gather_air_pressure_xz_history_slice(
    const __global double* p_curr,
    int nx,
    int ny,
    int nz,
    int iy_plane,
    __global double* slice_out)
{
    int n = nz * nx;
    int k = get_global_id(0);
    if (k >= n) return;
    int iz = k / nx;
    int ix = k % nx;
    if (iy_plane < 0) iy_plane = 0;
    if (iy_plane >= ny) iy_plane = ny - 1;
    int idx = iz * (ny * nx) + iy_plane * nx + ix;
    slice_out[k] = p_curr[idx];
}
