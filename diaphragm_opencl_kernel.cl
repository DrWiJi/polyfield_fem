/**
 * OpenCL iris computing kernel.
 * Forces: elasticity (nonlinear BOPET spring) + air resistance + external pressure.
 * Integration: RK4 (four stages of acceleration).
 *
 * Key formulas:
 * - Elasticity: spring_len = center_len (center-center), rest_len = 0.5*(size_me + size_nb);
 * strain = (center_len - rest_len)/rest_len; F_elastic = k_eff * (center_len - rest_len).
 * - Air connection: injection p_drive = rho*c*v_n; idx_lo += -p_drive, idx_hi += +p_drive;
 * force F = (p_lo - p_hi) * A * n (reaction from an area with high pressure).
 *
 * 3D quantity style: for velocity, face areas, pressure gradient and forces
 * double3 / vload3 / vstore3 and component-wise vector operations are used.
 *
 * Injection FE→air: air_fe_inject_scatter (contribution to FE) + air_inject_reduce_to_pressure
 * (sum of CSR by cells) - without races with many CEs per cell.
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
/*Receiving pressure from the air-field: F ~ -grad(p)*V (microphone, passive layers).*/
#define MAT_PROP_COUPLING_RECV 6
/*Injection into air-field from v·n (sound source): only membrane/specified in material; 0 = does not radiate into the grid.*/
#define MAT_PROP_ACOUSTIC_INJECT 7
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

            /*Stiffness: directional anisotropy (x: 0.1 → k_axial_x; y: 2.3 → k_axial_y).*/
            double k_axial_x = E_parallel * thickness_me * size_me.s1 / (size_me.s0 + TINY);
            double k_axial_y = E_perp * thickness_me * size_me.s0 / (size_me.s1 + TINY);
            double k_axial_dir = (direction_index < 2) ? k_axial_x : k_axial_y;
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
            double force_mag = force_elastic + force_tension;
            if (force_mag < 0.0) force_mag = 0.0;
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
    const __global double* air_force_external,
    const __global int* boundary_mask_elements,
    const __global double* element_size_xyz,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* neighbors,
    const __global uchar* laws,
    int n_materials,
    __constant Params* params,
    __global double* acceleration_out)
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
    __global double* velocity_stage)
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

/* Air-field kernels removed. */