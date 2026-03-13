/**
 * OpenCL вычислительное ядро диафрагмы.
 * Силы: упругость (нелинейная пружина БОПЭТ) + сопротивление воздуха + внешнее давление.
 * Интегрирование: RK2 (два этапа).
 *
 * Ключевые формулы:
 * - Упругость: spring_len = center_len (центр-центр), rest_len = 0.5*(size_me + size_nb);
 *   strain = (center_len - rest_len)/rest_len; F_elastic = k_eff * (center_len - rest_len).
 * - Связь воздух: инжекция p_drive = rho*c*v_n; idx_lo += -p_drive, idx_hi += +p_drive;
 *   сила F = (p_lo - p_hi) * A * n (реакция от области с большим давлением).
 *
 * Требуется: cl_khr_fp64 (double precision).
 * Компиляция: встроена в PyOpenCL при создании программы из исходника.
 */
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define DOF_PER_ELEMENT 6
#define TINY 1e-20
#define FACE_DIRS 6
#define MATERIAL_PROPS_STRIDE 7
#define MAT_PROP_DENSITY 0
#define MAT_PROP_E_PARALLEL 1
#define MAT_PROP_E_PERP 2
#define MAT_PROP_POISSON 3
#define MAT_PROP_CD 4
#define MAT_PROP_ETA_VISC 5
#define MAT_PROP_COUPLING_GAIN 6
/* Алиасы материалов (синхронизированы с Python-слоем) */
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
/* Отладка M_total: 6 (F_total,M_total) + 6*6 (force_dir, lever_dir) = 42 */
#define DEBUG_ELASTIC_SIZE 42
/* Трассировка: шаг, elastic(42), pos_me(6), vel_me(6), pos_mid(6), vel_mid(6), F(6), mass(6), acc(6), x_new(6), v_new(6), rx,ry,rz, center_len0, strain0, k_eff0, force_mag0, force_local0(3), lever0(3), M0(3) */
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

/* Векторные операции: встроенные dot, length, cross (OpenCL Geometric Functions). */

/* Матрица 3x3: строки row0, row1, row2 */
typedef struct { double3 row0, row1, row2; } double3x3;

inline double3 mat3x3_times_vec3(const double3x3* R, double3 v) {
    return (double3)(
        dot(R->row0, v),
        dot(R->row1, v),
        dot(R->row2, v)
    );
}

/* R = Rz(rz) * Ry(ry) * Rx(rx) — локальная -> глобальная */
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
    if (direction < 2) return elem_size.s1 * elem_size.s2; /* ±X: YZ-грань */
    if (direction < 4) return elem_size.s0 * elem_size.s2; /* ±Y: XZ-грань */
    return elem_size.s0 * elem_size.s1;                    /* ±Z: XY-грань */
}

inline double rest_length_from_size(int direction, double3 size_me, double3 size_nb) {
    if (direction < 2) return 0.5 * (size_me.s0 + size_nb.s0);
    if (direction < 4) return 0.5 * (size_me.s1 + size_nb.s1);
    return 0.5 * (size_me.s2 + size_nb.s2);
}

/* Эффективная жёсткость от деформации (s-образный переход мягкий↔жёсткий). Анизотропия учитывается вызывающим кодом: передаются k_soft_dir, k_stiff_dir по направлению связи. */
inline double nonlinear_stiffness(double strain, double k_soft, double k_stiff, double e0, double ew) {
    double arg = -(strain - e0) / (ew + TINY);
    double s = 1.0 / (1.0 + exp(arg));
    return k_soft * (1.0 - s) + k_stiff * s;
}

/* Вектор от центра грани соседа до центра грани «меня» в локальной СК текущего элемента */
inline double3 face_to_face_vector(
    const double3 center_me_global,
    const double3 center_nb_global,
    double3 arm_me, double3 arm_nb, int direction)
{
    double3 mount_me_local = mounting_point_local(direction, arm_me);
    double3 mount_nb_local = mounting_point_neighbor_local(direction, arm_nb);

    /* Вращения не учитываются: локальная и глобальная СК совпадают.
     * Монтажные точки просто смещены относительно центров элементов. */
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

/* Упругость: 4 направления, сумма сил и моментов относительно центра элемента.
 * Если debug_elastic != NULL и elem_idx == debug_elem, заполняет debug_elastic (см. выше).
 * Если trace_extra != NULL и elem_idx == debug_elem, пишет [0..2]=rx,ry,rz; [3]=center_len0; [4]=strain0; [5]=k_eff0; [6]=force_mag0; [7..9]=force_local0; [10..12]=lever0; [13..15]=M0. */
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

            /* Вектор центр-центр: используется и для направления силы, и для длины пружины.
             * Ранее использовался face-to-face (link_len), что давало link_len=0 в покое при rest_len=dx,
             * т.е. деформация -1 и обнуление силы — элементы не имели упругой связи и улетали. */
            double3 center_me_global = (double3)(pos_me[0], pos_me[1], pos_me[2]);
            double3 center_to_center_global = center_me_global - center_nb_global;
            double center_len = length(center_to_center_global);
            double3 direction_local_me = (center_len > TINY)
                ? (center_to_center_global / center_len)
                : face_normal(direction_index);

            double rest_len = rest_length_from_size(direction_index, size_me, size_nb);
            /* Деформация и сила по center-to-center: spring_len=center_len, rest_len=center-to-center в покое. */
            double strain = (rest_len > TINY) ? ((center_len - rest_len) / rest_len) : 0.0;

            /* Жёсткость: анизотропия по направлению (x: 0,1 → k_axial_x; y: 2,3 → k_axial_y). */
            double k_axial_x = E_parallel * thickness_me * size_me.s1 / (size_me.s0 + TINY);
            double k_axial_y = E_perp * thickness_me * size_me.s0 / (size_me.s1 + TINY);
            double k_axial_dir = (direction_index < 2) ? k_axial_x : k_axial_y;
            /* Для материалов с E≈0 (например, воздух) не подменяем жёсткость мембранным p->k_*,
             * иначе получаются огромные ускорения при малой массе и численный разлёт. */
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
            /* Для границы/интерфейса воздух-поверхность: нормальное давление + вязкое сопротивление. */
            double3 normal = face_normal(direction_index);
            double face_area = face_area_from_size(direction_index, size_me);
            double vn = dot(vel_vec, normal);  /* скорость КЭ относительно воздуха по нормали */
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
            /* Абсолютный ATM-терм убран: давление приходит из air-field через add_air_pressure_to_force_external. */
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
        debug_elastic[3] = 0.0; debug_elastic[4] = 0.0; debug_elastic[5] = 0.0; /* моменты отключены */
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

inline void force_boundary_zero(double* F, int is_boundary) {
    if (is_boundary)
        for (int d = 0; d < DOF_PER_ELEMENT; d++) F[d] = 0.0;
}

/* Моменты не используем — обнуляем, чтобы угловые DOF не получали ускорение. */
inline void force_moments_zero(double* F) {
    F[3] = 0.0;
    F[4] = 0.0;
    F[5] = 0.0;
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

/* RK2 stage 1: только поступательные DOF (0,1,2). Угловые (3,4,5) не интегрируем — момент инерции мал, нестабильность. */
inline void integrate_rk2_stage1(const double* pos_me, const double* vel_me, const double* F,
    const __constant Params* p, const double* mass_safe, int is_boundary,
    __global double* position_mid, __global double* velocity_mid, int base)
{
    double half_dt = 0.5 * p->dt;
    for (int d = 0; d < 3; d++) {
        double acc = F[d] / mass_safe[d];
        double v_mid = vel_me[d] + acc * half_dt;
        double x_mid = pos_me[d] + v_mid * half_dt;
        velocity_mid[base + d] = v_mid;
        position_mid[base + d] = x_mid;
    }
    for (int d = 3; d < DOF_PER_ELEMENT; d++) {
        velocity_mid[base + d] = vel_me[d];
        position_mid[base + d] = pos_me[d];
    }
}

/* RK2 stage 2: только поступательные DOF. Угловые сохраняем без изменений. */
inline void integrate_rk2_stage2(const double* pos_me, const double* vel_me,
    const __global double* velocity_mid, const double* F,
    const __constant Params* p, const double* mass_safe, int is_boundary, int base,
    double* x_new, double* v_new)
{
    for (int d = 0; d < 3; d++) {
        double acc = F[d] / mass_safe[d];
        v_new[d] = vel_me[d] + acc * p->dt;
        x_new[d] = pos_me[d] + velocity_mid[base + d] * p->dt;
    }
    for (int d = 3; d < DOF_PER_ELEMENT; d++) {
        v_new[d] = vel_me[d];
        x_new[d] = pos_me[d];
    }
}

__kernel void diaphragm_rk2_stage1(
    const __global double* position,
    const __global double* velocity,
    const __global double* force_external,
    const __global int* boundary_mask_elements,
    const __global double* element_size_xyz,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* neighbors,
    const __global uchar* laws,
    int n_materials,
    __constant Params* params,
    __global double* position_mid,
    __global double* velocity_mid)
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

    double pos_me[DOF_PER_ELEMENT], vel_me[DOF_PER_ELEMENT];
    load_dof(position, elem_idx, pos_me);
    load_dof(velocity, elem_idx, vel_me);

    double F[DOF_PER_ELEMENT] = {0, 0, 0, 0, 0, 0};
    add_force_external(F, force_external, base);
    add_force_elastic(F, position, velocity, force_external, boundary_mask_elements, element_size_xyz, material_index, material_props, neighbors, laws, n_materials, p, elem_idx, ix, iy, pos_me, vel_me, NULL, -1, NULL);
    force_boundary_zero(F, is_boundary);
    force_moments_zero(F);

    double mass_safe[DOF_PER_ELEMENT];
    get_mass_safe(density, size_me, mass_safe);
    if (is_boundary) {
        for (int d = 0; d < DOF_PER_ELEMENT; d++) {
            position_mid[base + d] = pos_me[d];
            velocity_mid[base + d] = vel_me[d];
        }
    } else {
        integrate_rk2_stage1(pos_me, vel_me, F, p, mass_safe, is_boundary, position_mid, velocity_mid, base);
    }
}

__kernel void diaphragm_rk2_stage2(
    const __global double* position,
    const __global double* velocity,
    const __global double* position_mid,
    const __global double* velocity_mid,
    const __global double* force_external,
    const __global int* boundary_mask_elements,
    const __global double* element_size_xyz,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* neighbors,
    const __global uchar* laws,
    int n_materials,
    __constant Params* params,
    __global double* position_out,
    __global double* velocity_out,
    __global int* first_bad_elem,
    __global double* debug_buf)
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

    double pos_mid[DOF_PER_ELEMENT], vel_mid[DOF_PER_ELEMENT];
    load_dof(position_mid, elem_idx, pos_mid);
    load_dof(velocity_mid, elem_idx, vel_mid);

    double F[DOF_PER_ELEMENT] = {0, 0, 0, 0, 0, 0};
    add_force_external(F, force_external, base);
    #if ENABLE_DEBUG
    double debug_elastic[DEBUG_ELASTIC_SIZE];
    double trace_elastic[TRACE_ELASTIC_EXTRA];
    for (int i = 0; i < TRACE_ELASTIC_EXTRA; i++) trace_elastic[i] = 0.0;
    add_force_elastic(F, position_mid, velocity_mid, force_external, boundary_mask_elements, element_size_xyz, material_index, material_props, neighbors, laws, n_materials, p, elem_idx, ix, iy, pos_mid, vel_mid, debug_elastic, p->debug_elem, trace_elastic);
    #else
    add_force_elastic(F, position_mid, velocity_mid, force_external, boundary_mask_elements, element_size_xyz, material_index, material_props, neighbors, laws, n_materials, p, elem_idx, ix, iy, pos_mid, vel_mid, NULL, -1, NULL);
    #endif
    force_boundary_zero(F, is_boundary);
    force_moments_zero(F);

    double mass_safe[DOF_PER_ELEMENT];
    get_mass_safe(density, size_me, mass_safe);
    double pos_me[DOF_PER_ELEMENT], vel_me[DOF_PER_ELEMENT];
    load_dof(position, elem_idx, pos_me);
    load_dof(velocity, elem_idx, vel_me);

    double x_new[DOF_PER_ELEMENT], v_new[DOF_PER_ELEMENT];
    if (is_boundary) {
        for (int d = 0; d < DOF_PER_ELEMENT; d++) {
            x_new[d] = pos_me[d];
            v_new[d] = vel_me[d];
        }
    } else {
        integrate_rk2_stage2(pos_me, vel_me, velocity_mid, F, p, mass_safe, is_boundary, base, x_new, v_new);
    }

    #if ENABLE_DEBUG
    if (debug_buf != NULL && p->debug_elem >= 0 && elem_idx == p->debug_elem && p->debug_step < 250) {
        int o = 0;
        debug_buf[o++] = (double)p->debug_step;
        for (int i = 0; i < DEBUG_ELASTIC_SIZE; i++) debug_buf[o++] = debug_elastic[i];
        debug_buf[o++] = F[3]; debug_buf[o++] = F[4]; debug_buf[o++] = F[5];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = pos_me[d];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = vel_me[d];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = pos_mid[d];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = vel_mid[d];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = F[d];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = mass_safe[d];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = (mass_safe[d] > 1e-30) ? (F[d] / mass_safe[d]) : 0.0;
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = x_new[d];
        for (int d = 0; d < DOF_PER_ELEMENT; d++) debug_buf[o++] = v_new[d];
        for (int i = 0; i < TRACE_ELASTIC_EXTRA; i++) debug_buf[o++] = trace_elastic[i];
        debug_buf[o++] = p->Ixx; debug_buf[o++] = p->Iyy; debug_buf[o++] = p->Izz;
    }
    #endif

    /* Проверка на NaN/Inf и запись first_bad_elem (atomic: первый записавший побеждает) */
    #if ENABLE_DEBUG
    if (first_bad_elem) {
        int bad = 0;
        for (int d = 0; d < DOF_PER_ELEMENT; d++) {
            if (!isfinite(F[d]) || !isfinite(x_new[d]) || !isfinite(v_new[d])) { bad = 1; break; }
        }
        if (bad) {
            atomic_cmpxchg(first_bad_elem, 0x7FFFFFFF, elem_idx);
        }
    }
    #endif

    for (int d = 0; d < DOF_PER_ELEMENT; d++) {
        position_out[base + d] = x_new[d];
        velocity_out[base + d] = v_new[d];
    }
}

__kernel void air_step_3d(
    const __global double* p_prev,
    const __global double* p_curr,
    __global double* p_next,
    int nx_air,
    int ny_air,
    int nz_air,
    double dx_air,
    double dy_air,
    double dz_air,
    double dt,
    double c_air,
    double bulk_damping,
    double boundary_damping,
    int sponge_cells,
    double pressure_clip)
{
    int gid = get_global_id(0);
    int n_cells = nx_air * ny_air * nz_air;
    if (gid >= n_cells) return;

    int xy = nx_air * ny_air;
    int iz = gid / xy;
    int rem = gid - iz * xy;
    int iy = rem / nx_air;
    int ix = rem - iy * nx_air;

    int ixm = (ix > 0) ? ix - 1 : ix;
    int ixp = (ix + 1 < nx_air) ? ix + 1 : ix;
    int iym = (iy > 0) ? iy - 1 : iy;
    int iyp = (iy + 1 < ny_air) ? iy + 1 : iy;
    int izm = (iz > 0) ? iz - 1 : iz;
    int izp = (iz + 1 < nz_air) ? iz + 1 : iz;

    int idx_xm = air_idx3d(ixm, iy, iz, nx_air, ny_air);
    int idx_xp = air_idx3d(ixp, iy, iz, nx_air, ny_air);
    int idx_ym = air_idx3d(ix, iym, iz, nx_air, ny_air);
    int idx_yp = air_idx3d(ix, iyp, iz, nx_air, ny_air);
    int idx_zm = air_idx3d(ix, iy, izm, nx_air, ny_air);
    int idx_zp = air_idx3d(ix, iy, izp, nx_air, ny_air);

    double p0 = p_curr[gid];
    double d2x = (p_curr[idx_xp] - 2.0 * p0 + p_curr[idx_xm]) / (dx_air * dx_air + TINY);
    double d2y = (p_curr[idx_yp] - 2.0 * p0 + p_curr[idx_ym]) / (dy_air * dy_air + TINY);
    double d2z = (p_curr[idx_zp] - 2.0 * p0 + p_curr[idx_zm]) / (dz_air * dz_air + TINY);
    double lap = d2x + d2y + d2z;

    int dist_x = min(ix, nx_air - 1 - ix);
    int dist_y = min(iy, ny_air - 1 - iy);
    int dist_z = min(iz, nz_air - 1 - iz);
    int dist_edge = min(dist_x, min(dist_y, dist_z));
    double sponge = 0.0;
    if (sponge_cells > 0 && dist_edge < sponge_cells) {
        double r = ((double)(sponge_cells - dist_edge)) / (double)sponge_cells;
        sponge = boundary_damping * r * r;
    }
    double sigma = bulk_damping + sponge;

    double c2dt2 = (c_air * dt) * (c_air * dt);
    double a = 2.0 - sigma * dt;
    double b = 1.0 - sigma * dt;
    double p_new = a * p_curr[gid] - b * p_prev[gid] + c2dt2 * lap;
    // Радиационное граничное условие (Sommerfeld):
    // на границе приближённо задаём ∂p/∂t + c * ∂p/∂n = 0 (волна уходит наружу).
    // Это уменьшает отражения в режиме "открытого поля".
    if (ix == 0 || ix == nx_air - 1 || iy == 0 || iy == ny_air - 1 || iz == 0 || iz == nz_air - 1) {
        double flux = 0.0;
        int n_faces = 0;
        if (ix == 0) {
            flux += c_air * (p_curr[idx_xp] - p0) / (dx_air + TINY);
            n_faces++;
        } else if (ix == nx_air - 1) {
            flux += c_air * (p_curr[idx_xm] - p0) / (dx_air + TINY);
            n_faces++;
        }
        if (iy == 0) {
            flux += c_air * (p_curr[idx_yp] - p0) / (dy_air + TINY);
            n_faces++;
        } else if (iy == ny_air - 1) {
            flux += c_air * (p_curr[idx_ym] - p0) / (dy_air + TINY);
            n_faces++;
        }
        if (iz == 0) {
            flux += c_air * (p_curr[idx_zp] - p0) / (dz_air + TINY);
            n_faces++;
        } else if (iz == nz_air - 1) {
            flux += c_air * (p_curr[idx_zm] - p0) / (dz_air + TINY);
            n_faces++;
        }
        if (n_faces > 0) {
            // Устойчивая upwind-форма: p^{n+1} = p^n + dt * c * d_inward(p).
            // При CFL<=1 не вызывает разлёт и даёт выход волны в открытую границу.
            p_new = p0 + dt * (flux / (double)n_faces);
        }
    }
    // Модель стабильна без ограничения давления
    /*if (pressure_clip > 0.0) {
        if (p_new > pressure_clip) p_new = pressure_clip;
        if (p_new < -pressure_clip) p_new = -pressure_clip;
    }*/
    p_next[gid] = p_new;
}

__kernel void air_inject_membrane_velocity(
    __global double* elem_air_inject_delta_pair,
    const __global double* velocity_delta,
    const __global int* boundary_mask_elements,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* air_map_lo,
    const __global int* air_map_hi,
    const __global double* air_elem_normal_xyz,
    int n_elements,
    double rho_air,
    double c_air,
    double coupling_gain,
    double pressure_clip)
{
    int elem = get_global_id(0);
    if (elem >= n_elements) return;
    int out_base = elem * 2;
    elem_air_inject_delta_pair[out_base + 0] = 0.0;
    elem_air_inject_delta_pair[out_base + 1] = 0.0;
    if (boundary_mask_elements[elem] != 0) return;
    uchar material_id = material_index[elem];
    double coupling_gain_material = material_prop(material_props, material_id, MAT_PROP_COUPLING_GAIN);
    if (coupling_gain_material <= 0.0) return;

    int base = elem * DOF_PER_ELEMENT;
    double vx = velocity_delta[base + 0];
    double vy = velocity_delta[base + 1];
    double vz = velocity_delta[base + 2];
    double nx = air_elem_normal_xyz[elem * 3 + 0];
    double ny = air_elem_normal_xyz[elem * 3 + 1];
    double nz = air_elem_normal_xyz[elem * 3 + 2];
    double v_n = vx * nx + vy * ny + vz * nz;
    double p_drive = coupling_gain * coupling_gain_material * rho_air * c_air * v_n;
    // Защита от мгновенного перекачивания энергии при резких скачках скорости.
    double p_drive_limit = 0.25 * pressure_clip;
    if (p_drive > p_drive_limit) p_drive = p_drive_limit;
    if (p_drive < -p_drive_limit) p_drive = -p_drive_limit;
    int idx_lo = air_map_lo[elem];
    int idx_hi = air_map_hi[elem];
    if (idx_lo >= 0) elem_air_inject_delta_pair[out_base + 0] = -p_drive;
    if (idx_hi >= 0) elem_air_inject_delta_pair[out_base + 1] = +p_drive;
}

__kernel void air_inject_membrane_velocity_direct(
    __global double* p_next,
    const __global double* velocity_delta,
    const __global int* boundary_mask_elements,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* air_map_lo,
    const __global int* air_map_hi,
    const __global double* air_elem_normal_xyz,
    int n_elements,
    double rho_air,
    double c_air,
    double coupling_gain,
    double pressure_clip)
{
    int elem = get_global_id(0);
    if (elem >= n_elements) return;
    if (boundary_mask_elements[elem] != 0) return;
    uchar material_id = material_index[elem];
    double coupling_gain_material = material_prop(material_props, material_id, MAT_PROP_COUPLING_GAIN);
    if (coupling_gain_material <= 0.0) return;

    int base = elem * DOF_PER_ELEMENT;
    double vx = velocity_delta[base + 0];
    double vy = velocity_delta[base + 1];
    double vz = velocity_delta[base + 2];
    double nx = air_elem_normal_xyz[elem * 3 + 0];
    double ny = air_elem_normal_xyz[elem * 3 + 1];
    double nz = air_elem_normal_xyz[elem * 3 + 2];
    double v_n = vx * nx + vy * ny + vz * nz;
    double p_drive = coupling_gain * coupling_gain_material * rho_air * c_air * v_n;
    double p_drive_limit = 0.25 * pressure_clip;
    if (p_drive > p_drive_limit) p_drive = p_drive_limit;
    if (p_drive < -p_drive_limit) p_drive = -p_drive_limit;
    int idx_lo = air_map_lo[elem];
    int idx_hi = air_map_hi[elem];
    if (idx_lo >= 0) {
        double v = p_next[idx_lo] - p_drive;
        if (pressure_clip > 0.0) {
            if (v > pressure_clip) v = pressure_clip;
            if (v < -pressure_clip) v = -pressure_clip;
        }
        p_next[idx_lo] = v;
    }
    if (idx_hi >= 0) {
        double v = p_next[idx_hi] + p_drive;
        if (pressure_clip > 0.0) {
            if (v > pressure_clip) v = pressure_clip;
            if (v < -pressure_clip) v = -pressure_clip;
        }
        p_next[idx_hi] = v;
    }
}

__kernel void add_air_pressure_to_force_external(
    __global double* force_external,
    const __global double* p_curr,
    const __global int* boundary_mask_elements,
    const __global uchar* material_index,
    const __global double* material_props,
    const __global int* air_map_lo,
    const __global int* air_map_hi,
    const __global double* air_elem_normal_xyz,
    const __global double* air_elem_area,
    int n_elements)
{
    int elem = get_global_id(0);
    if (elem >= n_elements) return;
    if (boundary_mask_elements[elem] != 0) return;
    uchar material_id = material_index[elem];
    double coupling_gain_material = material_prop(material_props, material_id, MAT_PROP_COUPLING_GAIN);
    if (coupling_gain_material <= 0.0) return;

    int idx_lo = air_map_lo[elem];
    int idx_hi = air_map_hi[elem];
    if (idx_lo < 0 || idx_hi < 0) return;

    double p_lo = p_curr[idx_lo];
    double p_hi = p_curr[idx_hi];
    /* dp = p_lo - p_hi: сила направлена от области с большим давлением.
     * При p_hi > p_lo (сжатие в +n) реакция воздуха толкает мембрану в -n. */
    double dp = p_lo - p_hi;
    double area = air_elem_area[elem];
    double nx = air_elem_normal_xyz[elem * 3 + 0];
    double ny = air_elem_normal_xyz[elem * 3 + 1];
    double nz = air_elem_normal_xyz[elem * 3 + 2];
    double force_mag = coupling_gain_material * dp * area;
    int base = elem * DOF_PER_ELEMENT;
    force_external[base + 0] += force_mag * nx;
    force_external[base + 1] += force_mag * ny;
    force_external[base + 2] += force_mag * nz;
}
