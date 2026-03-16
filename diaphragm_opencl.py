# -*- coding: utf-8 -*-
"""
Модель диафрагмы с вычислительным ядром под OpenCL.

Один поток GPU = один конечный элемент: в одном ядре считаются
все силы (упругие от соседей, сопротивление воздуха, внешнее давление)
и выполняется шаг интегрирования RK2.

Требуется: PyOpenCL, OpenCL 1.2+ с поддержкой cl_khr_fp64 (double).
"""

from __future__ import annotations

import json
import os
import struct
import time
import numpy as np
from typing import Optional

# Критерии корректности для отладки (взрыв, 0 Гц, шум на спектрограмме)
MAX_UZ_UM_OK = 500.0       # max |uz| (µm): выше = "взрыв" модели
MIN_FREQ_HZ_OK = 1.0       # пик спектра ниже = "0 Гц", упругость не работает
MIN_PEAK_PROMINENCE = 2.0  # минимум (пик/среднее по спектру): ниже = шум, не осцилляция

try:
    from analytical_diaphragm import (
        analytical_natural_frequencies,
        natural_frequency_membrane_rect,
    )
except ImportError:
    analytical_natural_frequencies = None
    natural_frequency_membrane_rect = None

# Избегаем KeyError в PyOpenCL при ошибке сборки ядра (cache проверяет эту переменную)
if "PYOPENCL_CACHE_FAILURE_FATAL" not in os.environ:
    os.environ["PYOPENCL_CACHE_FAILURE_FATAL"] = "0"

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

try:
    import pyopencl as cl
except ImportError:
    cl = None

# Layout структуры Params (должен совпадать с diaphragm_opencl_kernel.cl).
# В C после int use_nonlinear_stiffness компилятор выравнивает следующий double на 8 байт — добавляет 4 байта padding.
# Без этого pre_tension и остальные поля после int читались бы ядром по неправильному смещению.
_PARAMS_FORMAT = (
    "i" * 4   # nx, ny, n_elements, n_dof
    + "d" * 5   # dx, dy, thickness, arm_x, arm_y
    + "d" * 5   # k_axial_x, k_axial_y, k_shear, k_bending_x, k_bending_y
    + "d" * 3   # stiffness_transition_center, width, ratio
    + "i"       # use_nonlinear_stiffness
    + "4x"      # padding до 8-байтной границы (как в C struct)
    + "d" * 8   # rho_air, mu_air, Cd, element_area, element_mass, Ixx, Iyy, Izz
    + "d" * 7   # dt, pre_tension, k_soft, k_stiff, strain_transition, strain_width, k_bend
    + "i" * 2   # debug_elem, debug_step
)

# Материалы:
# [density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_gain]
# eta_visc: эффективная вязкость материала (Па*с) для демпфирования в упругой связи.
# coupling_gain: коэффициент двусторонней связи с air-field (безразмерный, обычно 0..1).
_MATERIAL_PROPS_STRIDE = 7
FACE_DIRS = 6

# Законы взаимодействия материалов
LAW_SOLID_SPRING = np.uint8(0)

# Алиасы материалов (для читаемости и единых индексов)
MAT_MEMBRANE = np.uint8(0)
MAT_FOAM_VE3015 = np.uint8(1)
MAT_SHEEPSKIN_LEATHER = np.uint8(2)
MAT_HUMAN_EAR_AVG = np.uint8(3)
MAT_SENSOR = np.uint8(4)
MAT_COTTON_WOOL = np.uint8(5)


def _pack_params(
    nx: int, ny: int,
    n_elements: int,
    width: float, height: float, thickness: float,
    density: float, E_parallel: float, E_perp: float, poisson: float,
    use_nonlinear: bool,
    stiffness_transition_center: float, stiffness_transition_width: float, stiffness_ratio: float,
    rho_air: float, mu_air: float, Cd: float,
    dt: float,
    pre_tension: float,
    k_soft: float, k_stiff: float,
    strain_transition: float, strain_width: float,
    k_bend: float,
    debug_elem: int = -1,
    debug_step: int = 0,
) -> bytes:
    n_dof = n_elements * 6
    dx = width / nx
    dy = height / ny
    element_volume = dx * dy * thickness
    element_mass = density * element_volume
    Ixx = element_mass * (dy**2 + thickness**2) / 12.0
    Iyy = element_mass * (dx**2 + thickness**2) / 12.0
    Izz = element_mass * (dx**2 + dy**2) / 12.0
    element_area = dx * dy
    arm_x = dx / 2.0
    arm_y = dy / 2.0
    k_axial_x = E_parallel * thickness * dy / dx
    k_axial_y = E_perp * thickness * dx / dy
    k_shear = E_parallel * thickness / (2.0 * (1.0 + poisson))
    k_bending_x = E_parallel * thickness**3 * dy / (12.0 * dx**3)
    k_bending_y = E_perp * thickness**3 * dx / (12.0 * dy**3)

    return struct.pack(
        _PARAMS_FORMAT,
        nx, ny, n_elements, n_dof,
        dx, dy, thickness, arm_x, arm_y,
        k_axial_x, k_axial_y, k_shear, k_bending_x, k_bending_y,
        stiffness_transition_center, stiffness_transition_width, stiffness_ratio,
        1 if use_nonlinear else 0,
        rho_air, mu_air, Cd, element_area, element_mass, Ixx, Iyy, Izz,
        dt, pre_tension, k_soft, k_stiff, strain_transition, strain_width, k_bend,
        debug_elem, debug_step,
    )


# Смещения в трассе (должны совпадать с ядром TRACE_BUF_SIZE 127)
_TRACE_STEP = 0
_TRACE_ELASTIC = 1          # 42
_TRACE_M_FINAL = 43         # 3
_TRACE_POS_ME = 46          # 6
_TRACE_VEL_ME = 52          # 6
_TRACE_POS_MID = 58         # 6
_TRACE_VEL_MID = 64         # 6
_TRACE_F = 70               # 6
_TRACE_MASS = 76            # 6
_TRACE_ACC = 82             # 6
_TRACE_X_NEW = 88           # 6
_TRACE_V_NEW = 94           # 6
_TRACE_ELASTIC_EXTRA = 100  # 20: rx,ry,rz, link_len0, strain0, k_eff0, force_mag0, force_local0(3), lever0(3), M0(3), eff_len, rest_len
_TRACE_I = 120              # Ixx, Iyy, Izz


def _print_opencl_trace(buf: np.ndarray, debug_elem: int, step_idx: int) -> None:
    """Печать и валидация полной трассировки: интегрирование и упругость (в т.ч. вращения)."""
    n = min(buf.size, 127)
    if n < 123:
        print(f"\n--- Trace elem={debug_elem} step={step_idx}: буфер мал ({n}) ---")
        return
    step = int(buf[_TRACE_STEP])
    pos_me = buf[_TRACE_POS_ME : _TRACE_POS_ME + 6]
    vel_me = buf[_TRACE_VEL_ME : _TRACE_VEL_ME + 6]
    pos_mid = buf[_TRACE_POS_MID : _TRACE_POS_MID + 6]
    vel_mid = buf[_TRACE_VEL_MID : _TRACE_VEL_MID + 6]
    F = buf[_TRACE_F : _TRACE_F + 6]
    mass = buf[_TRACE_MASS : _TRACE_MASS + 6]
    acc = buf[_TRACE_ACC : _TRACE_ACC + 6]
    x_new = buf[_TRACE_X_NEW : _TRACE_X_NEW + 6]
    v_new = buf[_TRACE_V_NEW : _TRACE_V_NEW + 6]
    rx, ry, rz = buf[_TRACE_ELASTIC_EXTRA], buf[_TRACE_ELASTIC_EXTRA + 1], buf[_TRACE_ELASTIC_EXTRA + 2]
    link_len0 = buf[_TRACE_ELASTIC_EXTRA + 3]
    strain0 = buf[_TRACE_ELASTIC_EXTRA + 4]
    k_eff0 = buf[_TRACE_ELASTIC_EXTRA + 5]
    force_mag0 = buf[_TRACE_ELASTIC_EXTRA + 6]
    Ixx, Iyy, Izz = buf[_TRACE_I], buf[_TRACE_I + 1], buf[_TRACE_I + 2]
    M_el = buf[4:7]
    F_el = buf[1:4]

    finite = np.all(np.isfinite(buf[1:n]))
    rot_ok = np.isfinite(rx) and np.isfinite(ry) and np.isfinite(rz)
    rot_small = abs(rx) < 1.0 and abs(ry) < 1.0 and abs(rz) < 1.0
    ang_acc = np.array([F[3] / (Ixx + 1e-30), F[4] / (Iyy + 1e-30), F[5] / (Izz + 1e-30)])
    ang_acc_ok = np.all(np.isfinite(ang_acc)) and np.all(np.abs(ang_acc) < 1e12)

    issues = []
    if not finite:
        issues.append("не все значения finite")
    if not rot_ok:
        issues.append("rx,ry,rz не finite")
    if not rot_small:
        issues.append("вращения выходят из [-1,1] рад")
    if not ang_acc_ok:
        issues.append("угловое ускорение неадекватно")
    if not np.all(np.isfinite(x_new)):
        issues.append("x_new содержит NaN/Inf")
    if not np.all(np.isfinite(v_new)):
        issues.append("v_new содержит NaN/Inf")
    if step_idx == 0 and abs(pos_me[0]) < 1e-10 and abs(pos_me[1]) < 1e-10:
        issues.append("pos_me (x,y)=0 на шаге 0 — проверьте инициализацию сетки (rest positions)")
    status = " [ОШИБКА: " + "; ".join(issues) + "]" if issues else ""

    print(f"\n--- Trace elem={debug_elem} step={step_idx} (kernel step={step}){status} ---")
    print("  Интегрирование (RK2 stage2):")
    print(f"    pos_me   = {pos_me}  (x,y,z,rx,ry,rz)")
    print(f"    vel_me   = {vel_me}")
    print(f"    pos_mid  = {pos_mid}")
    print(f"    vel_mid  = {vel_mid}")
    print(f"    F        = {F}  (Fx,Fy,Fz, Mx,My,Mz)")
    print(f"    mass     = {mass}")
    print(f"    acc      = {acc}  (a = F/mass)")
    print(f"    x_new    = {x_new}")
    print(f"    v_new    = {v_new}")
    print("  Вращения (pos_mid[3:6], используются в упругости):")
    print(f"    rx={rx:.6e} ry={ry:.6e} rz={rz:.6e}")
    print("  Упругость dir0: center_len, strain, k_eff, force_mag:")
    print(f"    {link_len0:.6e} {strain0:.6e} {k_eff0:.6e} {force_mag0:.6e}")
    print("  Моменты: M_elastic_total =", M_el, "  F_elastic =", F_el)
    print("  Угловое ускорение (M/I):", ang_acc)
    print("  Моменты инерции Ixx,Iyy,Izz:", Ixx, Iyy, Izz)
    if buf.size > _TRACE_ELASTIC_EXTRA + 18:
        pre_tension_kernel = buf[_TRACE_ELASTIC_EXTRA + 18]
        print("  pre_tension (прочитано в ядре):", pre_tension_kernel)
    print()


# ---------------------------------------------------------------------------
# Класс модели (OpenCL)
# ---------------------------------------------------------------------------
class PlanarDiaphragmOpenCL:
    """
    Модель диафрагмы с вычислительным ядром под OpenCL.
    Один шаг = два ядра (RK2: stage1, stage2).

    Упругость: нелинейная пружина БОПЭТ между соседними гранями элементов.
    Интегрирование: Рунге–Кутта 2-го порядка (RK2).
    """

    def __init__(
        self,
        width_mm: float = 48.0,
        height_mm: float = 63.0,
        nx: int = 24,
        ny: int = 32,
        thickness_mm: float = 0.012,
        density_kg_m3: float = 1380.0,
        E_parallel_gpa: float = 5.0,
        E_perp_gpa: float = 3.5,
        poisson: float = 0.3,
        use_nonlinear_stiffness: bool = True,
        stiffness_transition_center: float = 0.002,
        stiffness_transition_width: float = 0.001,
        stiffness_ratio: float = 20.0,
        rho_air: float = 1.2,
        mu_air: float = 1.81e-5,
        Cd: float = 1.0,
        air_sound_speed_m_s: float = 343.0,
        air_padding_mm: float | None = None,
        air_grid_step_mm: float | None = None,
        air_boundary_damping: float = 600.0,
        air_coupling_gain: float = 0.05,
        air_inject_mode: str = "reduce",
        air_bulk_damping: float = 120.0,
        air_pressure_clip_pa: float = 2.0e4,
        pre_tension_N_per_m: float = 10.0,
        k_soft: float | None = None,
        k_stiff: float | None = None,
        strain_transition: float = 0.002,
        strain_width: float = 0.0005,
        k_bend: float | None = None,
        platform_index: int = 0,
        device_index: int = 0,
        kernel_debug: bool = False,
        material_props: np.ndarray | None = None,
    ) -> None:
        self.width = width_mm * 1e-3
        self.height = height_mm * 1e-3
        self.thickness = thickness_mm * 1e-3
        self.nx = nx
        self.ny = ny
        self.n_membrane_elements = nx * ny
        self.n_layers_total = 1
        self.n_elements = self.n_membrane_elements * self.n_layers_total
        self.dof_per_element = 6
        self.n_dof = self.n_elements * self.dof_per_element
        self._topology_is_rect_grid = True
        self._visual_shape = (self.ny, self.nx)
        self._visual_element_indices = np.arange(self.n_elements, dtype=np.int32)
        self.visualization_enabled = True

        self.density = density_kg_m3
        self.E_parallel = E_parallel_gpa * 1e9
        self.E_perp = E_perp_gpa * 1e9
        self.poisson = poisson
        self.use_nonlinear_stiffness = use_nonlinear_stiffness
        self.stiffness_transition_center = stiffness_transition_center
        self.stiffness_transition_width = stiffness_transition_width
        self.stiffness_ratio = stiffness_ratio
        self.rho_air = rho_air
        self.mu_air = mu_air
        self.Cd = Cd
        self.air_sound_speed = air_sound_speed_m_s
        self.air_padding = (air_padding_mm * 1e-3) if air_padding_mm is not None else None
        self.air_grid_step = (air_grid_step_mm * 1e-3) if air_grid_step_mm is not None else None
        self.air_boundary_damping = air_boundary_damping
        self.air_coupling_gain = air_coupling_gain
        mode = str(air_inject_mode).strip().lower()
        if mode not in ("reduce", "direct"):
            raise ValueError("air_inject_mode должен быть 'reduce' или 'direct'")
        self.air_inject_mode = mode
        self.air_inject_use_reduce = (mode == "reduce")
        self.air_bulk_damping = air_bulk_damping
        self.air_pressure_clip_pa = air_pressure_clip_pa
        self.air_cfl_safety = 0.45

        self.pre_tension = pre_tension_N_per_m
        dx = self.width / nx
        dy = self.height / ny
        dz = self.thickness
        self.element_size_xyz = np.empty((self.n_elements, 3), dtype=np.float64)
        self.element_size_xyz[:, 0] = dx
        self.element_size_xyz[:, 1] = dy
        self.element_size_xyz[:, 2] = dz
        k_axial_x = self.E_parallel * self.thickness * dy / dx
        k_axial_y = self.E_perp * self.thickness * dx / dy
        self.k_soft = k_soft if k_soft is not None else (k_axial_x + k_axial_y) / 2 / stiffness_ratio
        self.k_stiff = k_stiff if k_stiff is not None else (k_axial_x + k_axial_y) / 2
        self.strain_transition = strain_transition
        self.strain_width = strain_width
        k_bend_base = self.E_parallel * self.thickness**3 / 12.0
        self.k_bend = k_bend if k_bend is not None else k_bend_base * (dy / dx + dx / dy) / 2

        if material_props is not None:
            props = np.asarray(material_props, dtype=np.float64)
            if props.ndim != 2 or props.shape[1] not in (5, 6, _MATERIAL_PROPS_STRIDE):
                raise ValueError("material_props должен иметь shape [n_materials, 5], [n_materials, 6] или [n_materials, 7]")
            if props.shape[1] == 5:
                props = np.hstack((props, np.zeros((props.shape[0], 1)), np.ones((props.shape[0], 1))))
            elif props.shape[1] == 6:
                props = np.hstack((props, np.ones((props.shape[0], 1))))
            if props.shape[0] < 1:
                raise ValueError("material_props должен содержать минимум 1 материал")
            self.material_props = props
        else:
            self.material_props = self._build_default_material_library()
        self.material_id_map = {
            "membrane": int(MAT_MEMBRANE),
            "foam_ve3015": int(MAT_FOAM_VE3015),
            "sheepskin_leather": int(MAT_SHEEPSKIN_LEATHER),
            "human_ear_avg": int(MAT_HUMAN_EAR_AVG),
            "sensor": int(MAT_SENSOR),
            "cotton_wool": int(MAT_COTTON_WOOL),
        }

        self.material_index = np.full(self.n_elements, MAT_SENSOR, dtype=np.uint8)
        self._sensor_mask = np.zeros(self.n_elements, dtype=bool)
        self._update_sensor_mask()
        self.membrane_mask = np.zeros(self.n_elements, dtype=np.int32)
        self.membrane_mask[: self.n_membrane_elements] = 1
        self.laws = np.full(
            (self.material_props.shape[0], self.material_props.shape[0]),
            LAW_SOLID_SPRING,
            dtype=np.uint8,
        )
        flat_topology = self.generate_planar_membrane_topology(
            plane="xy",
            thickness_m=self.thickness,
            size_u_m=self.width,
            size_v_m=self.height,
        )
        self.neighbors = flat_topology["neighbors"]
        self.boundary_mask_elements = flat_topology["boundary_mask_elements"]
        self.position = np.zeros(self.n_dof, dtype=np.float64)
        self.position[0::self.dof_per_element] = flat_topology["element_position_xyz"][:, 0]
        self.position[1::self.dof_per_element] = flat_topology["element_position_xyz"][:, 1]
        self.position[2::self.dof_per_element] = flat_topology["element_position_xyz"][:, 2]
        self.element_size_xyz = flat_topology["element_size_xyz"]
        self._configure_air_field_grid()
        # Элементы, лежащие в базовой плоскости Z=0 в опорной конфигурации.
        self._z0_elements_mask = np.isclose(
            self.position[2 :: self.dof_per_element], 0.0, atol=1e-12
        )
        self.velocity = np.zeros(self.n_dof, dtype=np.float64)
        self._velocity_prev = np.zeros(self.n_dof, dtype=np.float64)
        self._velocity_delta = np.zeros(self.n_dof, dtype=np.float64)
        self.force_external = np.zeros(self.n_dof, dtype=np.float64)
        self._update_center_index()

        self.history_disp_center: list[float] = []
        self._record_history = False
        self.history_disp_all: list[np.ndarray] = []
        self.history_air_center_xz: list[np.ndarray] = []
        self.kernel_debug = bool(kernel_debug)

        if cl is None:
            raise RuntimeError("PyOpenCL не установлен. Установите: pip install pyopencl")

        # Платформа и устройство
        platforms = cl.get_platforms()
        if platform_index >= len(platforms):
            platform_index = 0
        platform = platforms[platform_index]
        devices = platform.get_devices(cl.device_type.GPU)
        if not devices and platform.get_devices():
            devices = platform.get_devices()
        if not devices:
            raise RuntimeError("OpenCL: не найдено подходящее устройство")
        if device_index >= len(devices):
            device_index = 0
        device = devices[device_index]
        self.ctx = cl.Context([device])
        self.queue = cl.CommandQueue(self.ctx)

        # Сборка программы (ядро из файла рядом со скриптом)
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diaphragm_opencl_kernel.cl")
        if not os.path.isfile(kernel_path):
            raise FileNotFoundError(f"Файл ядра не найден: {kernel_path}")
        with open(kernel_path, "r", encoding="utf-8") as f:
            kernel_src = f.read()

        # Без опций компилятора: часть драйверов (в т.ч. на Windows) даёт INVALID_COMPILER_OPTIONS на -cl-std / -cl-khr-fp64.
        # Double включается в ядре через #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        build_options = ["-DENABLE_DEBUG=1"] if self.kernel_debug else ["-DENABLE_DEBUG=0"]
        try:
            self.prg = cl.Program(self.ctx, kernel_src).build(options=build_options)
        except cl.RuntimeError as e:
            raise RuntimeError(f"OpenCL сборка ядра не удалась: {e}") from e

        self._kernel_stage1 = self.prg.diaphragm_rk2_stage1
        self._kernel_stage2 = self.prg.diaphragm_rk2_stage2
        self._kernel_air_step = self.prg.air_step_3d
        self._kernel_air_inject_reduce = self.prg.air_inject_membrane_velocity
        self._kernel_air_inject_direct = self.prg.air_inject_membrane_velocity_direct
        self._kernel_air_to_force = self.prg.add_air_pressure_to_force_external

        # Буферы (создаём один раз, переиспользуем)
        mf = cl.mem_flags
        self._buf_position = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_dof * 8)
        self._buf_velocity = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_dof * 8)
        self._buf_velocity_delta = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_dof * 8)
        self._buf_force_external = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_dof * 8)
        self._buf_boundary = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements * 4)
        self._buf_element_size = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements * 3 * 8)
        self._buf_material_index = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements)
        self._buf_material_props = cl.Buffer(
            self.ctx, mf.READ_ONLY, size=self.material_props.size * 8
        )
        self._buf_neighbors = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements * FACE_DIRS * 4)
        self._buf_laws = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.laws.size)
        self._buf_position_mid = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_dof * 8)
        self._buf_velocity_mid = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_dof * 8)
        self._buf_first_bad = cl.Buffer(self.ctx, mf.READ_WRITE, size=4)
        self._DEBUG_BUF_DOUBLES = 127  # трассировка: step, elastic(42), pos/vel/F/acc/x_new/v_new, trace_elastic(20), I
        self._buf_debug = cl.Buffer(self.ctx, mf.READ_WRITE, size=self._DEBUG_BUF_DOUBLES * 8)
        self._allocate_air_buffers()

        # Размер группы (local size) и глобальный размер
        self._local_size = min(256, self._get_max_work_group_size())
        self._global_size = ((self.n_elements + self._local_size - 1) // self._local_size) * self._local_size
        self._air_global_size = ((self.n_air_cells + self._local_size - 1) // self._local_size) * self._local_size

        backend = f"OpenCL ({platform.name}, {device.name})"
        print(
            f"PlanarDiaphragmOpenCL: {width_mm}x{height_mm} mm, {nx}x{ny}, membrane={self.n_membrane_elements}, "
            f"DOF={self.n_dof}, air_grid={self.nx_air}x{self.ny_air}x{self.nz_air}, backend={backend}, kernel_debug={self.kernel_debug}"
        )

    def _get_max_work_group_size(self) -> int:
        try:
            return self._kernel_stage1.get_work_group_info(
                cl.kernel_work_group_info.WORK_GROUP_SIZE, self.ctx.devices[0]
            )
        except Exception:
            return 256

    def _update_center_index(self) -> None:
        """
        Обновляет индекс центрального КЭ (для истории и диагностики).

        Приоритет:
        1) среди КЭ материала MAT_SENSOR;
        2) внутри одного Z-слоя (ближайшего к медиане Z сенсорных КЭ);
        3) ближайший к центру слоя в плоскости XY.
        Если сенсорных КЭ нет — fallback на геометрический центр всех КЭ.
        """
        xyz = self.position.reshape(self.n_elements, self.dof_per_element)[:, :3]
        sensor_idx = np.flatnonzero(self.material_index == MAT_SENSOR)
        if sensor_idx.size > 0:
            z_sensor = xyz[sensor_idx, 2]
            z_med = float(np.median(z_sensor))
            z_layer = float(z_sensor[int(np.argmin(np.abs(z_sensor - z_med)))])
            in_layer_mask = np.isclose(z_sensor, z_layer, atol=1e-12, rtol=1e-9)
            layer_idx = sensor_idx[in_layer_mask]
            if layer_idx.size == 0:
                layer_idx = sensor_idx
            xy_layer = xyz[layer_idx, :2]
            xy_center = np.mean(xy_layer, axis=0)
            d2 = np.sum((xy_layer - xy_center) ** 2, axis=1)
            center_idx = int(layer_idx[int(np.argmin(d2))])
        else:
            center = np.mean(xyz, axis=0)
            center_idx = int(np.argmin(np.sum((xyz - center) ** 2, axis=1)))
        self.center_idx = center_idx
        self.center_dof = self.center_idx * self.dof_per_element + 2

    def _sync_visualization_flag(self) -> None:
        """Отключает визуализацию, если топология не 2D-прямоугольная."""
        self.visualization_enabled = bool(
            self._topology_is_rect_grid
            and self._visual_shape is not None
            and self._visual_element_indices is not None
            and self._visual_element_indices.size == (self._visual_shape[0] * self._visual_shape[1])
        )

    def _allocate_air_buffers(self) -> None:
        """(Пере)создаёт буферы воздуха и карты связи КЭ <-> air-grid."""
        mf = cl.mem_flags
        self._buf_air_prev = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_air_cells * 8)
        self._buf_air_curr = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_air_cells * 8)
        self._buf_air_next = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_air_cells * 8)
        self._buf_air_inject_delta_pair = cl.Buffer(self.ctx, mf.READ_WRITE, size=self.n_elements * 2 * 8)
        self._buf_air_map_lo = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements * 4)
        self._buf_air_map_hi = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements * 4)
        self._buf_air_elem_normal = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements * 3 * 8)
        self._buf_air_elem_area = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.n_elements * 8)
        self._air_inject_delta_pair = np.zeros((self.n_elements, 2), dtype=np.float64)
        cl.enqueue_copy(self.queue, self._buf_air_map_lo, self.air_map_lo)
        cl.enqueue_copy(self.queue, self._buf_air_map_hi, self.air_map_hi)
        cl.enqueue_copy(self.queue, self._buf_air_elem_normal, self.air_elem_normal)
        cl.enqueue_copy(self.queue, self._buf_air_elem_area, self.air_elem_area)
        self._reset_air_field()
    
    def generate_planar_membrane_topology(
        self,
        plane: str,
        thickness_m: float,
        size_u_m: float,
        size_v_m: float,
    ) -> dict[str, np.ndarray]:
        """
        Генерация однослойной плоской мембраны на регулярной сетке nx x ny.

        plane:
            Плоскость построения: "xy", "xz" или "yz".
        thickness_m:
            Толщина единственного слоя (м).
        size_u_m, size_v_m:
            Размеры по двум осям выбранной плоскости (м).

        Возвращает словарь:
        - element_position_xyz: [n_elements, 3]
        - element_size_xyz: [n_elements, 3]
        - neighbors: [n_elements, FACE_DIRS]
        - boundary_mask_elements: [n_elements]
        """
        pl = str(plane).strip().lower()
        if pl == "xy":
            axis_u, axis_v, axis_n = 0, 1, 2
        elif pl == "xz":
            axis_u, axis_v, axis_n = 0, 2, 1
        elif pl == "yz":
            axis_u, axis_v, axis_n = 1, 2, 0
        else:
            raise ValueError("plane должен быть одним из: 'xy', 'xz', 'yz'")
        if thickness_m <= 0.0:
            raise ValueError("thickness_m должен быть > 0")
        if size_u_m <= 0.0 or size_v_m <= 0.0:
            raise ValueError("size_u_m и size_v_m должны быть > 0")

        du = float(size_u_m) / float(self.nx)
        dv = float(size_v_m) / float(self.ny)
        n = self.n_elements
        pos = np.zeros((n, 3), dtype=np.float64)
        size = np.zeros((n, 3), dtype=np.float64)
        neighbors = np.full((n, FACE_DIRS), -1, dtype=np.int32)
        boundary = np.zeros(n, dtype=np.int32)

        for elem in range(self.n_membrane_elements):
            ix = elem % self.nx
            iy = elem // self.nx
            pos[elem, axis_u] = (ix + 0.5) * du
            pos[elem, axis_v] = (iy + 0.5) * dv
            pos[elem, axis_n] = 0.0
            size[elem, axis_u] = du
            size[elem, axis_v] = dv
            size[elem, axis_n] = float(thickness_m)
            if ix + 1 < self.nx:
                neighbors[elem, 0] = elem + 1
            if ix - 1 >= 0:
                neighbors[elem, 1] = elem - 1
            if iy + 1 < self.ny:
                neighbors[elem, 2] = elem + self.nx
            if iy - 1 >= 0:
                neighbors[elem, 3] = elem - self.nx
            if ix == 0 or ix == self.nx - 1 or iy == 0 or iy == self.ny - 1:
                boundary[elem] = 1

        return {
            "element_position_xyz": pos,
            "element_size_xyz": size,
            "neighbors": neighbors,
            "boundary_mask_elements": boundary,
        }

    def _set_rest_position(self) -> None:
        """Сброс позиции к базовой плоской мембране в плоскости XY."""
        topo = self.generate_planar_membrane_topology(
            plane="xy",
            thickness_m=self.thickness,
            size_u_m=self.width,
            size_v_m=self.height,
        )
        self.position[0::self.dof_per_element] = topo["element_position_xyz"][:, 0]
        self.position[1::self.dof_per_element] = topo["element_position_xyz"][:, 1]
        self.position[2::self.dof_per_element] = topo["element_position_xyz"][:, 2]

    def _build_neighbors_topology(self) -> np.ndarray:
        """
        Универсальная топология связей [n_elements, FACE_DIRS].
        Порядок направлений: +X, -X, +Y, -Y, +Z, -Z.
        Воздушные КЭ отключены: строится только мембранная XY-сетка.
        """
        return self.generate_planar_membrane_topology(
            plane="xy",
            thickness_m=self.thickness,
            size_u_m=self.width,
            size_v_m=self.height,
        )["neighbors"]

    def _build_default_material_library(self) -> np.ndarray:
        """
        Встроенная библиотека материалов
        [density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_gain].
        Значения для foam/leather/ear ориентировочные, пригодны как стартовые FE-параметры.
        """
        # Поролон с эффектом памяти (VE3015, ориентир по вязкоэластичным PU-foam диапазонам).
        foam_density = 55.0
        foam_E_parallel = 0.08e6
        foam_E_perp = 0.05e6
        foam_poisson = 0.30
        foam_Cd = 1.20
        foam_eta_visc = 150.0
        foam_coupling_gain = 0.25

        # Овечья кожа (ориентир по FE-моделям leather: E ~ 10 MPa, rho ~ 998 кг/м^3).
        leather_density = 998.0
        leather_E_parallel = 10.0e6
        leather_E_perp = 7.0e6
        leather_poisson = 0.40
        leather_Cd = 1.05
        leather_eta_visc = 12.0
        leather_coupling_gain = 0.60

        # Ухо человека (усреднённо, без разделения тканей; ориентир по auricular cartilage E ~ 1.4..2.1 MPa).
        ear_density = 1080.0
        ear_E_parallel = 1.80e6
        ear_E_perp = 1.50e6
        ear_poisson = 0.45
        ear_Cd = 1.10
        ear_eta_visc = 20.0
        ear_coupling_gain = 0.50

        # Сенсор (микрофон): временно используем параметры ПЭТ-плёнки как у мембраны.
        sensor_density = self.density
        sensor_E_parallel = self.E_parallel
        sensor_E_perp = self.E_perp
        sensor_poisson = self.poisson
        sensor_Cd = self.Cd
        sensor_eta_visc = 0.8
        membrane_eta_visc = 0.8
        sensor_coupling_gain = 1.00
        membrane_coupling_gain = 0.90

        # Хлопковая вата (приближённые эффективные параметры для мягкого пористого наполнителя).
        cotton_density = 250.0
        cotton_E_parallel = 0.03e6
        cotton_E_perp = 0.02e6
        cotton_poisson = 0.20
        cotton_Cd = 1.35
        cotton_eta_visc = 220.0
        cotton_coupling_gain = 0.3

        return np.array(
            [
                [self.density, self.E_parallel, self.E_perp, self.poisson, self.Cd, membrane_eta_visc, membrane_coupling_gain],
                [foam_density, foam_E_parallel, foam_E_perp, foam_poisson, foam_Cd, foam_eta_visc, foam_coupling_gain],
                [leather_density, leather_E_parallel, leather_E_perp, leather_poisson, leather_Cd, leather_eta_visc, leather_coupling_gain],
                [ear_density, ear_E_parallel, ear_E_perp, ear_poisson, ear_Cd, ear_eta_visc, ear_coupling_gain],
                [sensor_density, sensor_E_parallel, sensor_E_perp, sensor_poisson, sensor_Cd, sensor_eta_visc, sensor_coupling_gain],
                [cotton_density, cotton_E_parallel, cotton_E_perp, cotton_poisson, cotton_Cd, cotton_eta_visc, cotton_coupling_gain],
            ],
            dtype=np.float64,
        )

    def _update_sensor_mask(self) -> None:
        self._sensor_mask = (self.material_index == MAT_SENSOR)

    def _configure_air_field_grid(self) -> None:
        """
        Отдельное 3D акустическое поле:
        - генерируется после построения базовой КЭ-модели;
        - границы вычисляются по крайним координатам КЭ с учетом половины размеров элементов;
        - добавляется отступ ~10 мм как компромисс между отражениями и объёмом.
        """
        if self.air_grid_step is not None:
            self.dx_air = float(self.air_grid_step)
            self.dy_air = float(self.air_grid_step)
            self.dz_air = float(self.air_grid_step)
        else:
            self.dx_air = float(np.mean(self.element_size_xyz[:, 0]))
            self.dy_air = float(np.mean(self.element_size_xyz[:, 1]))
            self.dz_air = float(max(self.dx_air, self.dy_air))
        pad = float(self.air_padding) if self.air_padding is not None else 0.01

        x = self.position[0 : self.n_elements * self.dof_per_element : self.dof_per_element]
        y = self.position[1 : self.n_elements * self.dof_per_element : self.dof_per_element]
        z = self.position[2 : self.n_elements * self.dof_per_element : self.dof_per_element]
        sx = self.element_size_xyz[: self.n_elements, 0]
        sy = self.element_size_xyz[: self.n_elements, 1]
        sz = self.element_size_xyz[: self.n_elements, 2]

        x_min_elem = float(np.min(x - 0.5 * sx))
        x_max_elem = float(np.max(x + 0.5 * sx))
        y_min_elem = float(np.min(y - 0.5 * sy))
        y_max_elem = float(np.max(y + 0.5 * sy))
        z_min_elem = float(np.min(z - 0.5 * sz))
        z_max_elem = float(np.max(z + 0.5 * sz))
        z_plane = float(np.mean(z))

        self.air_origin_x = x_min_elem - pad
        self.air_origin_y = y_min_elem - pad
        self.air_origin_z = z_min_elem - pad
        x_max_air = x_max_elem + pad
        y_max_air = y_max_elem + pad
        z_max_air = z_max_elem + pad

        self.nx_air = int(np.ceil((x_max_air - self.air_origin_x) / self.dx_air)) + 1
        self.ny_air = int(np.ceil((y_max_air - self.air_origin_y) / self.dy_air)) + 1
        self.nz_air = int(np.ceil((z_max_air - self.air_origin_z) / self.dz_air)) + 1
        if self._topology_is_rect_grid:
            self.nx_air = max(self.nx_air, self.nx + 3)
            self.ny_air = max(self.ny_air, self.ny + 3)
        else:
            min_xy = int(np.ceil(np.sqrt(self.n_elements))) + 2
            self.nx_air = max(self.nx_air, min_xy)
            self.ny_air = max(self.ny_air, min_xy)
        self.nz_air = max(self.nz_air, 5)
        self.n_air_cells = int(self.nx_air * self.ny_air * self.nz_air)
        self.air_z0 = int(np.round((z_plane - self.air_origin_z) / self.dz_air))
        self.air_z0 = int(np.clip(self.air_z0, 1, self.nz_air - 2))
        sponge_x = max(1, int(np.ceil(pad / self.dx_air)))
        sponge_y = max(1, int(np.ceil(pad / self.dy_air)))
        sponge_z = max(1, int(np.ceil(pad / self.dz_air)))
        self.air_sponge_cells = max(2, min(sponge_x, sponge_y, sponge_z) // 2)

        self.air_map_lo = np.full(self.n_elements, -1, dtype=np.int32)
        self.air_map_hi = np.full(self.n_elements, -1, dtype=np.int32)
        self.air_elem_normal = np.zeros((self.n_elements, 3), dtype=np.float64)
        self.air_elem_area = np.zeros(self.n_elements, dtype=np.float64)
        for elem in range(self.n_elements):
            base = elem * self.dof_per_element
            x_e = self.position[base + 0]
            y_e = self.position[base + 1]
            z_e = self.position[base + 2]
            ax = int(np.round((x_e - self.air_origin_x) / self.dx_air))
            ay = int(np.round((y_e - self.air_origin_y) / self.dy_air))
            az = int(np.round((z_e - self.air_origin_z) / self.dz_air))
            ax = int(np.clip(ax, 0, self.nx_air - 1))
            ay = int(np.clip(ay, 0, self.ny_air - 1))
            az = int(np.clip(az, 0, self.nz_air - 1))
            size = self.element_size_xyz[elem]
            normal = np.zeros(3, dtype=np.float64)
            normal[2] = 1.0
            self.air_elem_normal[elem] = normal
            self.air_elem_area[elem] = float(size[0] * size[1])
            ix_lo, iy_lo, iz_lo = ax, ay, az
            ix_hi, iy_hi, iz_hi = ax, ay, az
            iz_lo = max(0, az - 1)
            iz_hi = min(self.nz_air - 1, az + 1)
            idx_lo = iz_lo * (self.nx_air * self.ny_air) + iy_lo * self.nx_air + ix_lo
            idx_hi = iz_hi * (self.nx_air * self.ny_air) + iy_hi * self.nx_air + ix_hi
            self.air_map_lo[elem] = idx_lo
            self.air_map_hi[elem] = idx_hi
        self.air_pressure_prev = np.zeros(self.n_air_cells, dtype=np.float64)
        self.air_pressure_curr = np.zeros(self.n_air_cells, dtype=np.float64)
        self.air_pressure_next = np.zeros(self.n_air_cells, dtype=np.float64)

    def _update_air_coupling_geometry_from_motion(self) -> None:
        """
        Обновляет направление coupling и эффективную площадь КЭ по направлению движения.

        Направление берется из velocity_delta, fallback -> velocity, затем -> предыдущая нормаль.
        Эффективная площадь — проекция AABB КЭ на плоскость, перпендикулярную направлению:
        A_eff = |nx|*sy*sz + |ny|*sx*sz + |nz|*sx*sy.
        """
        n = self.n_elements
        pos_xyz = self.position.reshape(n, self.dof_per_element)[:, :3]
        vel_xyz = self.velocity.reshape(n, self.dof_per_element)[:, :3]
        vel_delta_xyz = self._velocity_delta.reshape(n, self.dof_per_element)[:, :3]
        prev_normals = self.air_elem_normal.copy()

        motion = vel_delta_xyz.copy()
        motion_norm = np.linalg.norm(motion, axis=1)
        weak = motion_norm < 1e-12
        if np.any(weak):
            motion[weak] = vel_xyz[weak]
            motion_norm = np.linalg.norm(motion, axis=1)
            weak = motion_norm < 1e-12
        if np.any(weak):
            motion[weak] = prev_normals[weak]
            motion_norm = np.linalg.norm(motion, axis=1)
            weak = motion_norm < 1e-12
        if np.any(weak):
            motion[weak] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            motion_norm = np.linalg.norm(motion, axis=1)

        normals = motion / (motion_norm[:, None] + 1e-30)
        bad = ~np.all(np.isfinite(normals), axis=1)
        if np.any(bad):
            normals[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        size = self.element_size_xyz
        area_eff = (
            np.abs(normals[:, 0]) * size[:, 1] * size[:, 2]
            + np.abs(normals[:, 1]) * size[:, 0] * size[:, 2]
            + np.abs(normals[:, 2]) * size[:, 0] * size[:, 1]
        )
        np.maximum(area_eff, 1e-18, out=area_eff)

        ax = np.rint((pos_xyz[:, 0] - self.air_origin_x) / self.dx_air).astype(np.int32)
        ay = np.rint((pos_xyz[:, 1] - self.air_origin_y) / self.dy_air).astype(np.int32)
        az = np.rint((pos_xyz[:, 2] - self.air_origin_z) / self.dz_air).astype(np.int32)
        np.clip(ax, 0, self.nx_air - 1, out=ax)
        np.clip(ay, 0, self.ny_air - 1, out=ay)
        np.clip(az, 0, self.nz_air - 1, out=az)

        step_xyz = np.zeros((n, 3), dtype=np.int32)
        strong = np.abs(normals) >= 1e-6
        step_xyz[strong] = np.sign(normals[strong]).astype(np.int32)
        all_zero = np.all(step_xyz == 0, axis=1)
        if np.any(all_zero):
            idx_max = np.argmax(np.abs(normals[all_zero]), axis=1)
            rows = np.flatnonzero(all_zero)
            signs = np.sign(normals[rows, idx_max]).astype(np.int32)
            signs = np.where(signs == 0, 1, signs)
            step_xyz[rows, idx_max] = signs

        ix_lo = np.clip(ax - step_xyz[:, 0], 0, self.nx_air - 1)
        iy_lo = np.clip(ay - step_xyz[:, 1], 0, self.ny_air - 1)
        iz_lo = np.clip(az - step_xyz[:, 2], 0, self.nz_air - 1)
        ix_hi = np.clip(ax + step_xyz[:, 0], 0, self.nx_air - 1)
        iy_hi = np.clip(ay + step_xyz[:, 1], 0, self.ny_air - 1)
        iz_hi = np.clip(az + step_xyz[:, 2], 0, self.nz_air - 1)

        self.air_elem_normal = normals.astype(np.float64, copy=False)
        self.air_elem_area = area_eff.astype(np.float64, copy=False)
        self.air_map_lo = (iz_lo * (self.nx_air * self.ny_air) + iy_lo * self.nx_air + ix_lo).astype(np.int32)
        self.air_map_hi = (iz_hi * (self.nx_air * self.ny_air) + iy_hi * self.nx_air + ix_hi).astype(np.int32)

        cl.enqueue_copy(self.queue, self._buf_air_map_lo, self.air_map_lo)
        cl.enqueue_copy(self.queue, self._buf_air_map_hi, self.air_map_hi)
        cl.enqueue_copy(self.queue, self._buf_air_elem_normal, self.air_elem_normal)
        cl.enqueue_copy(self.queue, self._buf_air_elem_area, self.air_elem_area)

    def _reset_air_field(self) -> None:
        self.air_pressure_prev.fill(0.0)
        self.air_pressure_curr.fill(0.0)
        self.air_pressure_next.fill(0.0)
        if hasattr(self, "_air_inject_delta_pair"):
            self._air_inject_delta_pair.fill(0.0)
        cl.enqueue_copy(self.queue, self._buf_air_prev, self.air_pressure_prev)
        cl.enqueue_copy(self.queue, self._buf_air_curr, self.air_pressure_curr)
        cl.enqueue_copy(self.queue, self._buf_air_next, self.air_pressure_next)
        if hasattr(self, "_buf_air_inject_delta_pair"):
            cl.enqueue_copy(self.queue, self._buf_air_inject_delta_pair, self._air_inject_delta_pair)

    def _reduce_air_injection_from_elements(self) -> None:
        """
        Reduce по промежуточному буферу инжекции:
        каждый КЭ пишет пару вкладов (lo/hi), затем CPU аккумулирует их в air_pressure_next.
        Это устраняет гонки записи множества КЭ в одни и те же ячейки air-grid.
        """
        cl.enqueue_copy(self.queue, self._air_inject_delta_pair, self._buf_air_inject_delta_pair)
        cl.enqueue_copy(self.queue, self.air_pressure_next, self._buf_air_next)
        self.queue.finish()

        idx_lo = self.air_map_lo
        idx_hi = self.air_map_hi
        d_lo = self._air_inject_delta_pair[:, 0]
        d_hi = self._air_inject_delta_pair[:, 1]

        valid_lo = idx_lo >= 0
        valid_hi = idx_hi >= 0
        if np.any(valid_lo):
            np.add.at(self.air_pressure_next, idx_lo[valid_lo], d_lo[valid_lo])
        if np.any(valid_hi):
            np.add.at(self.air_pressure_next, idx_hi[valid_hi], d_hi[valid_hi])

        clip = float(self.air_pressure_clip_pa)
        if clip > 0.0:
            np.clip(self.air_pressure_next, -clip, clip, out=self.air_pressure_next)

        cl.enqueue_copy(self.queue, self._buf_air_next, self.air_pressure_next)
        self.queue.finish()

    def _get_air_substeps(self, dt: float) -> tuple[int, float]:
        inv_h2 = (
            1.0 / (self.dx_air * self.dx_air + 1e-30)
            + 1.0 / (self.dy_air * self.dy_air + 1e-30)
            + 1.0 / (self.dz_air * self.dz_air + 1e-30)
        )
        dt_max = self.air_cfl_safety / (self.air_sound_speed * np.sqrt(inv_h2) + 1e-30)
        n_sub = max(1, int(np.ceil(dt / dt_max)))
        return n_sub, dt / n_sub

    def _air_center_xz_slice_from_flat(self, flat: np.ndarray) -> np.ndarray:
        """
        Центральный срез X-Z (перпендикулярно мембране) через середину по Y.
        Возвращает массив shape [nz_air, nx_air].
        """
        p3 = flat.reshape(self.nz_air, self.ny_air, self.nx_air)
        y_mid = self.ny_air // 2
        return p3[:, y_mid, :].copy()

    def _params_bytes(self, dt: float, step_idx: int = 0, debug_elem: int = -1) -> bytes:
        return _pack_params(
            self.nx, self.ny,
            self.n_elements,
            self.width, self.height, self.thickness,
            self.density, self.E_parallel, self.E_perp, self.poisson,
            self.use_nonlinear_stiffness,
            self.stiffness_transition_center,
            self.stiffness_transition_width,
            self.stiffness_ratio,
            self.rho_air, self.mu_air, self.Cd,
            dt,
            self.pre_tension,
            self.k_soft, self.k_stiff,
            self.strain_transition, self.strain_width,
            self.k_bend,
            debug_elem,
            step_idx,
        )

    def _build_force_external(self, pressure_pa: float | np.ndarray) -> None:
        self.force_external.fill(0.0)
        z0_indices = np.flatnonzero(self._z0_elements_mask)
        n_z0 = z0_indices.size
        if np.isscalar(pressure_pa):
            p = np.full(n_z0, float(pressure_pa), dtype=np.float64)
        else:
            p = np.asarray(pressure_pa, dtype=np.float64).ravel()
            if p.size == 1:
                p = np.full(n_z0, float(p[0]), dtype=np.float64)
            elif p.size == self.n_elements:
                p = p[z0_indices]
            elif p.size != n_z0:
                raise ValueError("pressure_pa должен иметь размер n_z0_elements, n_elements или 1")
        area = self.element_size_xyz[:, 0] * self.element_size_xyz[:, 1]
        for local_idx, elem in enumerate(z0_indices):
            if self.boundary_mask_elements[elem] == 0:
                self.force_external[elem * 6 + 2] = p[local_idx] * area[elem]

    def set_material_library(self, material_props: np.ndarray) -> None:
        """
        Обновляет таблицу материалов.
        Формат строки:
        [density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_gain].
        Для обратной совместимости допускаются форматы:
        [n_materials, 5] -> eta_visc=0, coupling_gain=1;
        [n_materials, 6] -> coupling_gain=1.
        """
        props = np.asarray(material_props, dtype=np.float64)
        if props.ndim != 2 or props.shape[1] not in (5, 6, _MATERIAL_PROPS_STRIDE):
            raise ValueError("material_props должен иметь shape [n_materials, 5], [n_materials, 6] или [n_materials, 7]")
        if props.shape[1] == 5:
            props = np.hstack(
                (
                    props,
                    np.zeros((props.shape[0], 1), dtype=np.float64),
                    np.ones((props.shape[0], 1), dtype=np.float64),
                )
            )
        elif props.shape[1] == 6:
            props = np.hstack((props, np.ones((props.shape[0], 1), dtype=np.float64)))
        if props.shape[0] < 1:
            raise ValueError("material_props должен содержать минимум 1 материал")
        if np.any(props[:, 6] < 0.0):
            raise ValueError("coupling_gain должен быть >= 0")
        if self.material_index.size > 0 and int(self.material_index.max()) >= int(props.shape[0]):
            raise ValueError("Текущий material_index содержит индексы вне новой библиотеки материалов")
        self.material_props = props.copy()
        n_materials = self.material_props.shape[0]
        if self.laws.shape != (n_materials, n_materials):
            self.laws = np.full((n_materials, n_materials), LAW_SOLID_SPRING, dtype=np.uint8)
        # Пересоздаём буфер, т.к. размер библиотеки может измениться.
        if hasattr(self, "ctx"):
            mf = cl.mem_flags
            self._buf_material_props = cl.Buffer(
                self.ctx, mf.READ_ONLY, size=self.material_props.size * 8
            )
            self._buf_laws = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.laws.size)

    def set_element_material_index(self, material_index: np.ndarray) -> None:
        """
        Назначает индекс материала для каждого КЭ (uint8).
        """
        idx = np.asarray(material_index, dtype=np.uint8).ravel()
        if idx.size != self.n_elements:
            raise ValueError("material_index должен иметь размер n_elements")
        if self.material_props.shape[0] > 255:
            raise ValueError("Количество материалов > 255 не поддерживается uint8 индексом")
        if idx.size > 0 and int(idx.max()) >= int(self.material_props.shape[0]):
            raise ValueError("В material_index есть индексы вне диапазона material_props")
        self.material_index = idx.copy()
        self._update_sensor_mask()

    def set_material_laws(self, laws: np.ndarray) -> None:
        """
        Матрица законов взаимодействия [n_materials, n_materials], dtype uint8.
        laws[i, j] задаёт закон для связи (материал i, материал j).
        """
        arr = np.asarray(laws, dtype=np.uint8)
        n_materials = self.material_props.shape[0]
        if arr.shape != (n_materials, n_materials):
            raise ValueError(f"laws должен иметь shape [{n_materials}, {n_materials}]")
        self.laws = arr.copy()
        if hasattr(self, "ctx"):
            mf = cl.mem_flags
            self._buf_laws = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.laws.size)

    def set_neighbors_topology(self, neighbors: np.ndarray) -> None:
        """
        Явно задаёт универсальную топологию соседей [n_elements, FACE_DIRS].
        Для отсутствующего соседа используйте -1.
        """
        arr = np.asarray(neighbors, dtype=np.int32)
        if arr.shape != (self.n_elements, FACE_DIRS):
            raise ValueError(f"neighbors должен иметь shape [{self.n_elements}, {FACE_DIRS}]")
        self.neighbors = arr.copy()

    def set_boundary_mask(self, boundary_mask_elements: np.ndarray) -> None:
        """
        Задаёт маску фиксированных КЭ (1 = фиксирован, 0 = свободен).
        Нужно для смешанных сцен (мембрана + другие материалы), где фиксировать следует не все края.
        """
        arr = np.asarray(boundary_mask_elements, dtype=np.int32).ravel()
        if arr.size != self.n_elements:
            raise ValueError("boundary_mask_elements должен иметь размер n_elements")
        arr = np.where(arr != 0, 1, 0).astype(np.int32)
        self.boundary_mask_elements = arr

    @staticmethod
    def _validate_topology_payload(
        element_position_xyz: np.ndarray,
        element_size_xyz: np.ndarray,
        neighbors: np.ndarray,
        material_index: np.ndarray | None = None,
        boundary_mask_elements: np.ndarray | None = None,
        element_active_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pos = np.asarray(element_position_xyz, dtype=np.float64)
        size = np.asarray(element_size_xyz, dtype=np.float64)
        nbh = np.asarray(neighbors, dtype=np.int32)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("element_position_xyz должен иметь shape [n, 3]")
        if size.shape != pos.shape:
            raise ValueError("element_size_xyz должен иметь shape [n, 3]")
        if nbh.shape != (pos.shape[0], FACE_DIRS):
            raise ValueError(f"neighbors должен иметь shape [n, {FACE_DIRS}]")
        if np.any(size <= 0.0):
            raise ValueError("element_size_xyz должен содержать только положительные размеры")
        n = pos.shape[0]
        if np.any((nbh < -1) | (nbh >= n)):
            raise ValueError("neighbors содержит индексы вне диапазона [-1, n)")

        if material_index is None:
            mat = np.full(n, int(MAT_SENSOR), dtype=np.uint8)
        else:
            mat = np.asarray(material_index, dtype=np.uint8).ravel()
            if mat.size != n:
                raise ValueError("material_index должен иметь размер n")

        if boundary_mask_elements is None:
            bnd = np.zeros(n, dtype=np.int32)
        else:
            bnd = np.asarray(boundary_mask_elements, dtype=np.int32).ravel()
            if bnd.size != n:
                raise ValueError("boundary_mask_elements должен иметь размер n")
            bnd = np.where(bnd != 0, 1, 0).astype(np.int32)

        if element_active_mask is None:
            active = np.ones(n, dtype=bool)
        else:
            active = np.asarray(element_active_mask, dtype=bool).ravel()
            if active.size != n:
                raise ValueError("element_active_mask должен иметь размер n")

        return pos, size, nbh, mat, bnd, active

    @staticmethod
    def _aabb_overlap(
        p1: np.ndarray, s1: np.ndarray, p2: np.ndarray, s2: np.ndarray, tol: float
    ) -> bool:
        h1 = 0.5 * s1
        h2 = 0.5 * s2
        for axis in range(3):
            if (p1[axis] + h1[axis]) < (p2[axis] - h2[axis] - tol):
                return False
            if (p2[axis] + h2[axis]) < (p1[axis] - h1[axis] - tol):
                return False
        return True

    @classmethod
    def merge_topologies(
        cls,
        primary: dict[str, np.ndarray],
        secondary: dict[str, np.ndarray],
        primary_is_main: bool = True,
        overlap_tol_m: float = 1e-12,
    ) -> dict[str, np.ndarray]:
        """
        Объединяет две топологии КЭ.

        На пересечениях (AABB-overlap) остаются элементы "главной" топологии.
        Возвращает словарь с ключами:
        - element_position_xyz, element_size_xyz, neighbors,
        - material_index, boundary_mask_elements.
        """
        p_pos, p_size, p_nbh, p_mat, p_bnd, p_active = cls._validate_topology_payload(
            primary["element_position_xyz"],
            primary["element_size_xyz"],
            primary["neighbors"],
            primary.get("material_index"),
            primary.get("boundary_mask_elements"),
            primary.get("element_active_mask"),
        )
        s_pos, s_size, s_nbh, s_mat, s_bnd, s_active = cls._validate_topology_payload(
            secondary["element_position_xyz"],
            secondary["element_size_xyz"],
            secondary["neighbors"],
            secondary.get("material_index"),
            secondary.get("boundary_mask_elements"),
            secondary.get("element_active_mask"),
        )

        if overlap_tol_m < 0.0:
            raise ValueError("overlap_tol_m должен быть >= 0")

        if primary_is_main:
            main_pos, main_size, main_nbh, main_mat, main_bnd, main_active = (
                p_pos, p_size, p_nbh, p_mat, p_bnd, p_active
            )
            aux_pos, aux_size, aux_nbh, aux_mat, aux_bnd, aux_active = (
                s_pos, s_size, s_nbh, s_mat, s_bnd, s_active
            )
        else:
            main_pos, main_size, main_nbh, main_mat, main_bnd, main_active = (
                s_pos, s_size, s_nbh, s_mat, s_bnd, s_active
            )
            aux_pos, aux_size, aux_nbh, aux_mat, aux_bnd, aux_active = (
                p_pos, p_size, p_nbh, p_mat, p_bnd, p_active
            )

        main_idx = np.flatnonzero(main_active)
        aux_idx = np.flatnonzero(aux_active)
        keep_aux = np.ones(aux_idx.size, dtype=bool)
        for i_main in main_idx:
            pm = main_pos[i_main]
            sm = main_size[i_main]
            for j, i_aux in enumerate(aux_idx):
                if not keep_aux[j]:
                    continue
                if cls._aabb_overlap(pm, sm, aux_pos[i_aux], aux_size[i_aux], overlap_tol_m):
                    keep_aux[j] = False

        aux_idx_kept = aux_idx[keep_aux]
        out_n = int(main_idx.size + aux_idx_kept.size)
        out_pos = np.zeros((out_n, 3), dtype=np.float64)
        out_size = np.zeros((out_n, 3), dtype=np.float64)
        out_nbh = np.full((out_n, FACE_DIRS), -1, dtype=np.int32)
        out_mat = np.zeros(out_n, dtype=np.uint8)
        out_bnd = np.zeros(out_n, dtype=np.int32)

        # Индексы результата: сначала main, затем оставшиеся aux.
        src_entries: list[tuple[str, int]] = [("main", int(i)) for i in main_idx]
        src_entries += [("aux", int(i)) for i in aux_idx_kept]
        map_main = {int(src_i): out_i for out_i, (src, src_i) in enumerate(src_entries) if src == "main"}
        map_aux = {int(src_i): out_i for out_i, (src, src_i) in enumerate(src_entries) if src == "aux"}

        for out_i, (src, src_i) in enumerate(src_entries):
            if src == "main":
                out_pos[out_i] = main_pos[src_i]
                out_size[out_i] = main_size[src_i]
                out_mat[out_i] = main_mat[src_i]
                out_bnd[out_i] = main_bnd[src_i]
                src_nbh = main_nbh[src_i]
                idx_map = map_main
            else:
                out_pos[out_i] = aux_pos[src_i]
                out_size[out_i] = aux_size[src_i]
                out_mat[out_i] = aux_mat[src_i]
                out_bnd[out_i] = aux_bnd[src_i]
                src_nbh = aux_nbh[src_i]
                idx_map = map_aux

            for d in range(FACE_DIRS):
                n_old = int(src_nbh[d])
                out_nbh[out_i, d] = idx_map.get(n_old, -1)

        return {
            "element_position_xyz": out_pos,
            "element_size_xyz": out_size,
            "neighbors": out_nbh,
            "material_index": out_mat,
            "boundary_mask_elements": out_bnd,
        }

    def set_merged_topologies(
        self,
        primary: dict[str, np.ndarray],
        secondary: dict[str, np.ndarray],
        primary_is_main: bool = True,
        overlap_tol_m: float = 1e-12,
        visual_shape: tuple[int, int] | None = None,
        preserve_velocity: bool = False,
        rebuild_air: bool = True,
        air_grid_step_mm: float | None = None,
        air_padding_mm: float | None = None,
    ) -> None:
        """
        Сливает две топологии и применяет результат к текущей модели.

        Важно: число КЭ после слияния должно совпасть с self.n_elements.
        """
        merged = self.merge_topologies(
            primary=primary,
            secondary=secondary,
            primary_is_main=primary_is_main,
            overlap_tol_m=overlap_tol_m,
        )
        if merged["element_position_xyz"].shape[0] != self.n_elements:
            raise ValueError(
                "После merge число КЭ не совпадает с текущей моделью. "
                "Пересоздайте модель с нужным n_elements или подайте топологии с подходящим размером."
            )
        self.set_custom_topology(
            element_position_xyz=merged["element_position_xyz"],
            element_size_xyz=merged["element_size_xyz"],
            neighbors=merged["neighbors"],
            material_index=merged["material_index"],
            boundary_mask_elements=merged["boundary_mask_elements"],
            visual_shape=visual_shape,
            preserve_velocity=preserve_velocity,
            rebuild_air=rebuild_air,
            air_grid_step_mm=air_grid_step_mm,
            air_padding_mm=air_padding_mm,
        )

    def rebuild_air_field(
        self,
        air_grid_step_mm: float | None = None,
        air_padding_mm: float | None = None,
    ) -> None:
        """
        Перестраивает 3D сетку акустического поля по текущей топологии КЭ.

        air_grid_step_mm:
            Шаг сетки воздуха в мм. Если None, используется текущий режим (авто/ранее заданный).
        air_padding_mm:
            Отступ воздуха вокруг геометрии КЭ в мм. Если None, используется текущий.
        """
        if air_grid_step_mm is not None:
            if air_grid_step_mm <= 0.0:
                raise ValueError("air_grid_step_mm должен быть > 0")
            self.air_grid_step = float(air_grid_step_mm) * 1e-3
        if air_padding_mm is not None:
            if air_padding_mm < 0.0:
                raise ValueError("air_padding_mm должен быть >= 0")
            self.air_padding = float(air_padding_mm) * 1e-3

        self._configure_air_field_grid()
        if hasattr(self, "ctx"):
            self._allocate_air_buffers()
            # После перестройки air-сетки число ячеек меняется, поэтому
            # launch size для air-kernel нужно пересчитать.
            self._global_size = ((self.n_elements + self._local_size - 1) // self._local_size) * self._local_size
            self._air_global_size = ((self.n_air_cells + self._local_size - 1) // self._local_size) * self._local_size

    def set_custom_topology(
        self,
        element_position_xyz: np.ndarray,
        element_size_xyz: np.ndarray,
        neighbors: np.ndarray,
        material_index: np.ndarray | None = None,
        boundary_mask_elements: np.ndarray | None = None,
        visual_shape: tuple[int, int] | None = None,
        preserve_velocity: bool = False,
        rebuild_air: bool = True,
        air_grid_step_mm: float | None = None,
        air_padding_mm: float | None = None,
    ) -> None:
        """
        Полностью задаёт топологию КЭ извне (для всех n_elements).

        Ограничение: количество КЭ не меняется (должно быть равно текущему self.n_elements).
        """
        pos = np.asarray(element_position_xyz, dtype=np.float64)
        size = np.asarray(element_size_xyz, dtype=np.float64)
        nbh = np.asarray(neighbors, dtype=np.int32)
        if pos.shape != (self.n_elements, 3):
            raise ValueError(f"element_position_xyz должен иметь shape [{self.n_elements}, 3]")
        if size.shape != (self.n_elements, 3):
            raise ValueError(f"element_size_xyz должен иметь shape [{self.n_elements}, 3]")
        if nbh.shape != (self.n_elements, FACE_DIRS):
            raise ValueError(f"neighbors должен иметь shape [{self.n_elements}, {FACE_DIRS}]")
        if np.any(size <= 0.0):
            raise ValueError("element_size_xyz должен содержать только положительные размеры")
        if np.any((nbh < -1) | (nbh >= self.n_elements)):
            raise ValueError("neighbors содержит индексы вне диапазона [-1, n_elements)")

        self.position[0::self.dof_per_element] = pos[:, 0]
        self.position[1::self.dof_per_element] = pos[:, 1]
        self.position[2::self.dof_per_element] = pos[:, 2]
        self.element_size_xyz = size.copy()
        self.membrane_mask = np.ones(self.n_elements, dtype=np.int32)
        self.set_neighbors_topology(nbh)

        if material_index is not None:
            self.set_element_material_index(material_index)
        if boundary_mask_elements is not None:
            self.set_boundary_mask(boundary_mask_elements)

        if not preserve_velocity:
            self.velocity.fill(0.0)
            self._velocity_prev.fill(0.0)
            self._velocity_delta.fill(0.0)

        self._z0_elements_mask = np.isclose(
            self.position[2 :: self.dof_per_element], 0.0, atol=1e-12
        )
        if visual_shape is not None:
            if len(visual_shape) != 2:
                raise ValueError("visual_shape должен быть tuple (ny, nx)")
            v_ny, v_nx = int(visual_shape[0]), int(visual_shape[1])
            sensor_idx = np.flatnonzero(self.material_index == MAT_SENSOR).astype(np.int32)
            if v_ny <= 0 or v_nx <= 0 or v_ny * v_nx != sensor_idx.size:
                raise ValueError("visual_shape должен содержать положительные размеры и ny*nx == n_sensor_elements (MAT_SENSOR)")
            self._topology_is_rect_grid = True
            self._visual_shape = (v_ny, v_nx)
            self._visual_element_indices = sensor_idx
        else:
            self._topology_is_rect_grid = False
            self._visual_shape = None
            self._visual_element_indices = None
        self._sync_visualization_flag()
        self._update_center_index()

        if rebuild_air:
            self.rebuild_air_field(
                air_grid_step_mm=air_grid_step_mm,
                air_padding_mm=air_padding_mm,
            )

    def compute_air_force_center(self) -> float:
        if self.boundary_mask_elements[self.center_idx] != 0:
            return 0.0
        cl.enqueue_copy(self.queue, self.air_pressure_curr, self._buf_air_curr)
        self.queue.finish()
        idx_lo = int(self.air_map_lo[self.center_idx])
        idx_hi = int(self.air_map_hi[self.center_idx])
        if idx_lo < 0 or idx_hi < 0:
            return 0.0
        dp = float(self.air_pressure_curr[idx_hi] - self.air_pressure_curr[idx_lo])
        area = float(self.element_size_xyz[self.center_idx, 0] * self.element_size_xyz[self.center_idx, 1])
        return dp * area

    def step(
        self,
        dt: float,
        pressure_pa: float | np.ndarray,
        step_idx: int = -1,
        first_bad_elem_out: Optional[np.ndarray] = None,
        debug_elem: int = -1,
        debug_buf_out: Optional[np.ndarray] = None,
        debug_silent: bool = False,
    ) -> None:
        self._build_force_external(pressure_pa)
        use_debug = self.kernel_debug and (debug_elem >= 0)
        use_validate = self.kernel_debug and (first_bad_elem_out is not None)
        params_bytes = self._params_bytes(dt, step_idx=max(0, step_idx), debug_elem=debug_elem if use_debug else -1)

        # Буфер параметров (constant) — каждый шаг новый
        mf = cl.mem_flags
        buf_params = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=params_bytes)

        # Копируем входные данные на устройство
        cl.enqueue_copy(self.queue, self._buf_position, self.position)
        cl.enqueue_copy(self.queue, self._buf_velocity, self.velocity)
        cl.enqueue_copy(self.queue, self._buf_velocity_delta, self._velocity_delta)
        cl.enqueue_copy(self.queue, self._buf_force_external, self.force_external)
        cl.enqueue_copy(self.queue, self._buf_boundary, self.boundary_mask_elements)
        cl.enqueue_copy(self.queue, self._buf_element_size, self.element_size_xyz)
        cl.enqueue_copy(self.queue, self._buf_material_index, self.material_index)
        cl.enqueue_copy(self.queue, self._buf_material_props, self.material_props)
        cl.enqueue_copy(self.queue, self._buf_neighbors, self.neighbors)
        cl.enqueue_copy(self.queue, self._buf_laws, self.laws)
        self._update_air_coupling_geometry_from_motion()

        # Шаг 3D поля воздуха (с CFL-субшагами) + инжекция от скорости КЭ
        n_air_substeps, dt_air = self._get_air_substeps(dt)
        coupling_gain_sub = self.air_coupling_gain / n_air_substeps
        for _ in range(n_air_substeps):
            self._kernel_air_step.set_args(
                self._buf_air_prev,
                self._buf_air_curr,
                self._buf_air_next,
                np.int32(self.nx_air),
                np.int32(self.ny_air),
                np.int32(self.nz_air),
                np.float64(self.dx_air),
                np.float64(self.dy_air),
                np.float64(self.dz_air),
                np.float64(dt_air),
                np.float64(self.air_sound_speed),
                np.float64(self.air_bulk_damping),
                np.float64(self.air_boundary_damping),
                np.int32(self.air_sponge_cells),
                np.float64(self.air_pressure_clip_pa),
            )
            cl.enqueue_nd_range_kernel(
                self.queue, self._kernel_air_step,
                (self._air_global_size,), (self._local_size,),
            )
            self._kernel_air_inject_reduce.set_args(
                self._buf_air_inject_delta_pair,
                self._buf_velocity_delta,
                self._buf_boundary,
                self._buf_material_index,
                self._buf_material_props,
                self._buf_air_map_lo,
                self._buf_air_map_hi,
                self._buf_air_elem_normal,
                np.int32(self.n_elements),
                np.float64(self.rho_air),
                np.float64(self.air_sound_speed),
                np.float64(coupling_gain_sub),
                np.float64(self.air_pressure_clip_pa),
            ) if self.air_inject_use_reduce else self._kernel_air_inject_direct.set_args(
                self._buf_air_next,
                self._buf_velocity_delta,
                self._buf_boundary,
                self._buf_material_index,
                self._buf_material_props,
                self._buf_air_map_lo,
                self._buf_air_map_hi,
                self._buf_air_elem_normal,
                np.int32(self.n_elements),
                np.float64(self.rho_air),
                np.float64(self.air_sound_speed),
                np.float64(coupling_gain_sub),
                np.float64(self.air_pressure_clip_pa),
            )
            cl.enqueue_nd_range_kernel(
                self.queue,
                self._kernel_air_inject_reduce if self.air_inject_use_reduce else self._kernel_air_inject_direct,
                (self._global_size,),
                (self._local_size,),
            )
            if self.air_inject_use_reduce:
                self._reduce_air_injection_from_elements()
            self._buf_air_prev, self._buf_air_curr, self._buf_air_next = (
                self._buf_air_curr,
                self._buf_air_next,
                self._buf_air_prev,
            )
        # Давление воздуха -> сила на КЭ
        self._kernel_air_to_force.set_args(
            self._buf_force_external,
            self._buf_air_curr,
            self._buf_boundary,
            self._buf_material_index,
            self._buf_material_props,
            self._buf_air_map_lo,
            self._buf_air_map_hi,
            self._buf_air_elem_normal,
            self._buf_air_elem_area,
            np.int32(self.n_elements),
        )
        cl.enqueue_nd_range_kernel(
            self.queue, self._kernel_air_to_force,
            (self._global_size,), (self._local_size,),
        )

        if use_validate:
            first_bad_init = np.array([0x7FFFFFFF], dtype=np.int32)
            cl.enqueue_copy(self.queue, self._buf_first_bad, first_bad_init)

        # Stage 1
        self._kernel_stage1.set_args(
            self._buf_position,
            self._buf_velocity,
            self._buf_force_external,
            self._buf_boundary,
            self._buf_element_size,
            self._buf_material_index,
            self._buf_material_props,
            self._buf_neighbors,
            self._buf_laws,
            np.int32(self.material_props.shape[0]),
            buf_params,
            self._buf_position_mid,
            self._buf_velocity_mid,
        )
        cl.enqueue_nd_range_kernel(
            self.queue, self._kernel_stage1,
            (self._global_size,), (self._local_size,),
        )

        # Stage 2 (результат пишем в position/velocity)
        self._kernel_stage2.set_args(
            self._buf_position,
            self._buf_velocity,
            self._buf_position_mid,
            self._buf_velocity_mid,
            self._buf_force_external,
            self._buf_boundary,
            self._buf_element_size,
            self._buf_material_index,
            self._buf_material_props,
            self._buf_neighbors,
            self._buf_laws,
            np.int32(self.material_props.shape[0]),
            buf_params,
            self._buf_position,
            self._buf_velocity,
            self._buf_first_bad,
            self._buf_debug,
        )
        cl.enqueue_nd_range_kernel(
            self.queue, self._kernel_stage2,
            (self._global_size,), (self._local_size,),
        )

        self.queue.finish()

        # Копируем результат обратно
        cl.enqueue_copy(self.queue, self.position, self._buf_position)
        cl.enqueue_copy(self.queue, self.velocity, self._buf_velocity)
        self._velocity_delta[:] = self.velocity - self._velocity_prev
        self._velocity_prev[:] = self.velocity
        if use_validate:
            cl.enqueue_copy(self.queue, first_bad_elem_out, self._buf_first_bad)
        if use_debug and debug_buf_out is not None:
            cl.enqueue_copy(self.queue, debug_buf_out, self._buf_debug)
        self.queue.finish()
        if first_bad_elem_out is not None and not self.kernel_debug:
            first_bad_elem_out[0] = -1
        if use_validate and first_bad_elem_out[0] >= self.n_elements:
            first_bad_elem_out[0] = -1
        if use_debug and debug_buf_out is not None and not debug_silent:
            _print_opencl_trace(debug_buf_out, debug_elem, step_idx)

        self.history_disp_center.append(float(self.position[self.center_dof]))
        if self._record_history:
            uz_all = self.position[2 : self.n_elements * 6 : 6]
            if (
                self._topology_is_rect_grid
                and self._visual_shape is not None
                and self._visual_element_indices is not None
            ):
                frame = uz_all[self._visual_element_indices].reshape(self._visual_shape).copy()
            else:
                sensor_disp = np.full(self.n_elements, np.nan, dtype=np.float64)
                if np.any(self._sensor_mask):
                    sensor_disp[self._sensor_mask] = uz_all[self._sensor_mask]
                frame = sensor_disp.copy()
            self.history_disp_all.append(frame)
            cl.enqueue_copy(self.queue, self.air_pressure_curr, self._buf_air_curr)
            self.queue.finish()
            self.history_air_center_xz.append(self._air_center_xz_slice_from_flat(self.air_pressure_curr))

    def simulate(
        self,
        pressure_profile: np.ndarray,
        dt: float,
        record_history: bool = False,
        check_air_resistance: bool = False,
        validate_steps: bool = True,
        reset_state: bool = True,
        show_progress: bool = True,
        progress_every_pct: float = 5.0,
    ) -> np.ndarray:
        """
        reset_state: если True, обнуляет position/velocity в начале.
        Если False, использует текущие position/velocity (нужно для теста собственной частоты:
        get_numerical_natural_frequency задаёт rest position и импульс, без reset_state они не затираются).
        Раньше затирание давало старт из (0,0,0) для всех элементов -> «взрыв» и 0 Гц на спектрограмме.
        """
        if pressure_profile.ndim not in (1, 2):
            raise ValueError("pressure_profile должен быть 1D [n_steps] или 2D [n_steps, n_elements]")
        n_steps = pressure_profile.shape[0]
        if progress_every_pct <= 0.0:
            raise ValueError("progress_every_pct должен быть > 0")
        if reset_state:
            self.velocity.fill(0.0)
            self._velocity_prev.fill(0.0)
            self._velocity_delta.fill(0.0)
            self._reset_air_field()
        self.force_external.fill(0.0)
        self.history_disp_center = []
        self.history_disp_all = []
        self.history_air_center_xz = []
        self._record_history = record_history
        self._last_max_uz_um = 0.0
        if validate_steps and not self.kernel_debug:
            print("  [validate] Пропуск: проверка NaN/Inf в ядре доступна только при kernel_debug=True.")
            validate_steps = False

        first_bad = np.array([-1], dtype=np.int32) if validate_steps else None
        self._first_bad_elem = -1
        self._first_bad_step = -1
        report_step = max(1, int(np.ceil(n_steps * progress_every_pct / 100.0))) if n_steps > 0 else 1
        next_report = report_step
        sim_t0 = time.perf_counter()
        executed_steps = 0
        if show_progress and n_steps > 0:
            print(f"[simulate] start: steps={n_steps}, dt={dt:.3e} s")
        for step_idx in range(n_steps):
            if pressure_profile.ndim == 1:
                p = pressure_profile[step_idx]
            else:
                p = pressure_profile[step_idx]
            self.step(dt, p, step_idx=step_idx, first_bad_elem_out=first_bad)
            executed_steps = step_idx + 1
            if show_progress and (executed_steps >= next_report or executed_steps == n_steps):
                elapsed = time.perf_counter() - sim_t0
                progress = 100.0 * executed_steps / max(n_steps, 1)
                eta = elapsed * (n_steps - executed_steps) / max(executed_steps, 1)
                print(
                    f"[simulate] {executed_steps}/{n_steps} ({progress:5.1f}%) "
                    f"elapsed={elapsed:7.2f}s eta={eta:7.2f}s"
                )
                next_report += report_step
            # Критерий "взрыв": max |uz| по всем элементам (µm)
            uz_all = self.position[2 : self.n_elements * 6 : 6]
            if uz_all.size > 0 and np.all(np.isfinite(uz_all)):
                max_uz_um = float(np.max(np.abs(uz_all)) * 1e6)
                if max_uz_um > self._last_max_uz_um:
                    self._last_max_uz_um = max_uz_um
            if validate_steps and first_bad is not None and first_bad[0] >= 0:
                elem = int(first_bad[0])
                self._first_bad_elem = elem
                self._first_bad_step = step_idx
                ix, iy = elem % self.nx, elem // self.nx
                pos = self.position[elem * 6 : elem * 6 + 6]
                vel = self.velocity[elem * 6 : elem * 6 + 6]
                print(f"  [validate] NaN/Inf на шаге {step_idx}, элемент {elem} (ix={ix}, iy={iy})")
                print(f"    position: {pos}")
                print(f"    velocity: {vel}")
                break
            if check_air_resistance and step_idx % 100 == 0 and step_idx < 500:
                v_z = float(self.velocity[self.center_dof])
                f_air = self.compute_air_force_center()
                print(f"  step {step_idx}: v_center={v_z:.2e} m/s, F_air={f_air:.2e} N")
        if check_air_resistance:
            f_end = self.compute_air_force_center()
            print(f"  (end) F_air_center={f_end:.2e} N")
        if show_progress and n_steps > 0:
            elapsed_total = time.perf_counter() - sim_t0
            print(f"[simulate] done: {executed_steps}/{n_steps} steps in {elapsed_total:.2f}s")

        return np.asarray(self.history_disp_center, dtype=np.float64)

    def get_numerical_natural_frequency(
        self,
        dt: float = 1e-7,
        duration: float = 0.01,
        impulse_velocity_z: float = 0.01,
        freq_min_hz: float = 1.0,
        freq_max_hz: float = 50_000.0,
        refine_peak: bool = True,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Возбуждение начальной скоростью центра, запись u_center(t), FFT и пик спектра.

        impulse_velocity_z — начальная скорость по z центрального элемента (м/с).
        refine_peak: если True, уточнение частоты пика параболической интерполяцией (суб-бин).
        Разрешение по частоте FFT: df = 1/duration; при малом duration пик квантуется по бинам.
        Возвращает (f_peak_Hz, freq_axis, magnitude_spectrum).
        """
        self._set_rest_position()
        self.velocity.fill(0.0)
        self.velocity[self.center_dof] = impulse_velocity_z
        n_steps = int(round(duration / dt))
        pressure = np.zeros(n_steps, dtype=np.float64)
        hist = self.simulate(
            pressure, dt=dt,
            record_history=False, check_air_resistance=False, validate_steps=False,
            reset_state=False,
        )
        if hist.size < 4:
            return np.nan, np.array([]), np.array([])
        freq = np.fft.rfftfreq(hist.size, dt)
        spec = np.abs(np.fft.rfft(hist))
        mask = (freq >= freq_min_hz) & (freq <= freq_max_hz)
        if not np.any(mask):
            return np.nan, freq, spec
        idx_in_masked = np.argmax(spec[mask])
        idx = np.where(mask)[0][idx_in_masked]
        f_peak = float(freq[idx])
        if refine_peak and idx > 0 and idx < len(freq) - 1:
            # Параболическая интерполяция для суб-бин оценки частоты пика
            y0, y1, y2 = spec[idx - 1], spec[idx], spec[idx + 1]
            denom = y0 - 2 * y1 + y2
            if abs(denom) > 1e-30:
                delta = 0.5 * (y0 - y2) / denom
                delta = np.clip(delta, -0.5, 0.5)
                df_bin = freq[1] - freq[0] if len(freq) > 1 else 1.0 / (hist.size * dt)
                f_peak = float(freq[idx] + delta * df_bin)
        return f_peak, freq, spec

    def plot_time_and_spectrum(
        self,
        history: np.ndarray | None = None,
        dt: float = 1e-6,
        max_freq_hz: float = 20_000.0,
    ) -> None:
        hist = np.asarray(history if history is not None else self.history_disp_center, dtype=float)
        if hist.size == 0:
            print("Визуализация пропущена: нет истории. Запустите simulate() перед визуализацией.")
            return
        t = np.arange(len(hist)) * dt * 1e3

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(t, hist * 1e6)
        ax1.set_xlabel("Time, ms")
        ax1.set_ylabel("Center displacement, um")
        ax1.set_title("Diaphragm center displacement (OpenCL)")
        ax1.grid(True, alpha=0.3)

        if len(hist) > 4:
            freq = np.fft.fftfreq(len(hist), dt)
            spec = np.fft.fft(hist)
            mask = (freq > 0) & (freq <= max_freq_hz)
            f_plot = freq[mask]
            amp = np.abs(spec[mask])
            amp_norm = amp / (np.max(amp) + 1e-30)
            ax2.loglog(f_plot, np.maximum(amp_norm, 1e-10))
        ax2.set_xlim(1.0, max_freq_hz)
        ax2.set_xlabel("Frequency, Hz")
        ax2.set_ylabel("Amplitude (norm.)")
        ax2.set_title("Spectrum")
        ax2.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.show()

    def plot_displacement_map(self, scale_um: bool = True) -> None:
        if not self.visualization_enabled:
            print(
                "Визуализация отключена: топология не поддерживает 2D карту. "
                "Передайте visual_shape в set_custom_topology()."
            )
            return
        if self._visual_element_indices is None:
            print("Визуализация отключена: не задан список визуализируемых элементов.")
            return
        uz_all = self.position[2 : self.n_elements * 6 : 6]
        disp_map = uz_all[self._visual_element_indices].reshape(self._visual_shape).astype(float)
        if scale_um:
            disp_map *= 1e6
        extent = [0.0, self.width * 1e3, 0.0, self.height * 1e3]
        plt.figure(figsize=(5, 5))
        im = plt.imshow(
            disp_map,
            cmap="RdBu",
            origin="lower",
            extent=extent,
            aspect="auto",
        )
        plt.xlabel("X, mm")
        plt.ylabel("Y, mm")
        unit = "um" if scale_um else "m"
        plt.title(f"Displacement uz ({unit})")
        plt.colorbar(im, label=f"uz, {unit}")
        plt.tight_layout()
        plt.show()

    def animate(
        self,
        history_disp_all: list[np.ndarray] | None = None,
        dt: float = 1e-6,
        skip: int = 1,
        interval_ms: int = 50,
        scale_um: bool = True,
        cmap: str = "RdBu",
    ) -> FuncAnimation | None:
        if not self.visualization_enabled:
            print(
                "Визуализация отключена: топология не поддерживает 2D анимацию. "
                "Передайте visual_shape в set_custom_topology()."
            )
            return None
        frames = history_disp_all if history_disp_all is not None else self.history_disp_all
        if not frames:
            raise ValueError(
                "No animation data. Run simulate(record_history=True) or pass history_disp_all."
            )
        if np.asarray(frames[0]).ndim != 2:
            print(
                "Визуализация отключена: получены не-2D кадры. "
                "Передайте visual_shape в set_custom_topology()."
            )
            return None
        extent = [0.0, self.width * 1e3, 0.0, self.height * 1e3]
        scale = 1e6 if scale_um else 1.0
        vmin = min(np.min(f) for f in frames) * scale
        vmax = max(np.max(f) for f in frames) * scale
        vabs = max(abs(vmin), abs(vmax), 1e-12)
        vmin, vmax = -vabs, vabs

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            frames[0] * scale,
            cmap=cmap,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("X, mm")
        ax.set_ylabel("Y, mm")
        unit = "um" if scale_um else "m"
        ax.set_title("t = 0.00 ms")
        plt.colorbar(im, ax=ax, label=f"uz, {unit}")
        indices = list(range(0, len(frames), skip))

        def update(i: int) -> tuple:
            idx = indices[i]
            im.set_data(frames[idx] * scale)
            ax.set_title(f"t = {idx * dt * 1e3:.2f} ms")
            return (im,)

        ani = FuncAnimation(fig, update, frames=len(indices), interval=interval_ms, repeat=True)
        plt.tight_layout()
        return ani

    def animate_air_pressure_center_plane(
        self,
        history_air_center_xz: list[np.ndarray] | None = None,
        dt: float = 1e-6,
        skip: int = 1,
        interval_ms: int = 50,
        cmap: str = "RdBu",
        symmetric: bool = True,
    ) -> FuncAnimation | None:
        frames = history_air_center_xz if history_air_center_xz is not None else self.history_air_center_xz
        if not frames:
            print(
                "Визуализация пропущена: нет данных air-slice. "
                "Запустите simulate(record_history=True) или передайте history_air_center_xz."
            )
            return None

        x0_mm = (self.air_origin_x - 0.5 * self.dx_air) * 1e3
        z0_mm = (self.air_origin_z - 0.5 * self.dz_air) * 1e3
        x1_mm = (self.air_origin_x + (self.nx_air - 0.5) * self.dx_air) * 1e3
        z1_mm = (self.air_origin_z + (self.nz_air - 0.5) * self.dz_air) * 1e3
        extent = [x0_mm, x1_mm, z0_mm, z1_mm]

        vmin = min(float(np.min(f)) for f in frames)
        vmax = max(float(np.max(f)) for f in frames)
        if symmetric:
            vabs = max(abs(vmin), abs(vmax), 1e-12)
            vmin, vmax = -vabs, vabs
        else:
            if abs(vmax - vmin) < 1e-12:
                vmax = vmin + 1e-12

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(
            frames[0],
            cmap=cmap,
            origin="lower",
            extent=extent,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("X, mm")
        ax.set_ylabel("Z, mm")
        ax.set_title("Air pressure center slice (X-Z), t = 0.00 ms")
        plt.colorbar(im, ax=ax, label="p, Pa")
        indices = list(range(0, len(frames), skip))

        def update(i: int) -> tuple:
            idx = indices[i]
            im.set_data(frames[idx])
            ax.set_title(f"Air pressure center slice (X-Z), t = {idx * dt * 1e3:.2f} ms")
            return (im,)

        ani = FuncAnimation(fig, update, frames=len(indices), interval=interval_ms, repeat=True)
        plt.tight_layout()
        return ani


def _spectrum_peak_prominence(freq: np.ndarray, spec: np.ndarray, freq_max_hz: float = 50_000.0) -> float:
    """Выделенность пика: max(spec)/mean(spec) по положительным частотам до freq_max_hz."""
    mask = (freq > 0) & (freq <= freq_max_hz)
    if not np.any(mask):
        return 0.0
    s = np.asarray(spec[mask], dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    return float(np.max(s) / (np.mean(s) + 1e-40))


def validate_natural_frequencies(
    model: "PlanarDiaphragmOpenCL",
    dt: float = 2e-7,
    duration: float = 0.02,
    impulse_velocity_z: float = 0.01,
) -> dict[str, float]:
    """
    Сравнение собственной частоты численной модели с аналитической (мембрана).

    Алгоритм: покой + импульс скорости по z в центре → simulate() → u_center(t) →
    FFT → поиск пика в [1, 50k] Гц; частота пика уточняется параболической интерполяцией.
    Разрешение FFT: df = 1/duration (duration=0.02 с → df=50 Гц; при 0.005 с было 200 Гц —
    пик квантовался по бинам 0, 200, 400… Гц, из-за чего численная f не реагировала на натяжение).
    Возвращает словарь: numerical_f11_Hz, membrane_f11_Hz, err_membrane_pct, ...
    """
    out: dict[str, float] = {}
    if analytical_natural_frequencies is None:
        print("validate_natural_frequencies: модуль analytical_diaphragm не найден, пропуск.")
        return out

    if model.kernel_debug:
        # Проверка: что упаковано в буфер и что реально читает ядро (один шаг без вывода трассировки)
        dt_check = 2e-7
        params_bytes = model._params_bytes(dt_check, 0, -1)
        unpacked = struct.unpack(_PARAMS_FORMAT, params_bytes)
        pre_tension_packed = unpacked[27]
        model._set_rest_position()
        model.velocity.fill(0.0)
        debug_buf = np.zeros(model._DEBUG_BUF_DOUBLES, dtype=np.float64)
        model.step(dt_check, 0.0, step_idx=0, debug_elem=0, debug_buf_out=debug_buf, debug_silent=True)
        pre_tension_in_kernel = float(debug_buf[_TRACE_ELASTIC_EXTRA + 18]) if debug_buf.size > _TRACE_ELASTIC_EXTRA + 18 else float("nan")
        print("\n--- Проверка передачи pre_tension в ядро ---")
        print(f"  Python (model.pre_tension):     {model.pre_tension} Н/м")
        print(f"  Упаковано в буфер:             {pre_tension_packed} Н/м")
        print(f"  Прочитано в ядре (trace):      {pre_tension_in_kernel} Н/м")
        if abs(pre_tension_in_kernel - model.pre_tension) > 0.01:
            print("  ВНИМАНИЕ: ядро получает другое значение — возможен неверный layout структуры Params (выравнивание OpenCL).")
        else:
            print("  Передача в ядро корректна. Если численная f не меняется с T — динамика задаётся жёсткостью сетки (k_eff), а не только натяжением.")
        print()
    else:
        print("\n--- Проверка передачи pre_tension в ядро ---")
        print("  Пропуск: трассировка ядра отключена (kernel_debug=False).")
        print()

    analytical = analytical_natural_frequencies(
        model.width, model.height, model.thickness,
        model.density, model.E_parallel, model.poisson,
        model.pre_tension,
    )
    f_mem = analytical["membrane_f11_Hz"]

    f_num, freq_axis, spec = model.get_numerical_natural_frequency(
        dt=dt, duration=duration, impulse_velocity_z=impulse_velocity_z, refine_peak=True,
    )
    df_fft = 1.0 / duration
    max_uz_um = getattr(model, "_last_max_uz_um", 0.0)
    prominence = _spectrum_peak_prominence(freq_axis, spec) if spec.size > 0 else 0.0

    is_explosion = max_uz_um > MAX_UZ_UM_OK
    is_zero_hz = not (np.isfinite(f_num) and f_num >= MIN_FREQ_HZ_OK)
    is_noisy = prominence < MIN_PEAK_PROMINENCE if prominence == prominence else True

    out["numerical_f11_Hz"] = f_num
    out["membrane_f11_Hz"] = f_mem
    out["max_uz_um"] = max_uz_um
    out["peak_prominence"] = prominence
    out["is_explosion"] = float(is_explosion)
    out["is_zero_hz"] = float(is_zero_hz)
    out["is_noisy"] = float(is_noisy)

    print("\n--- Валидация: собственная частота (мембрана, мода 1,1) ---")
    print(f"  Натяжение: pre_tension (числ.) = T (аналит.) = {model.pre_tension:.2f} Н/м")
    print(f"  Численная: duration={duration} s, разрешение FFT df = {df_fft:.1f} Hz, пик уточнён параболой.")
    print(f"  Аналитика (мембрана, натяжение T): f = {f_mem:.2f} Hz")
    print(f"  Численная (пик FFT центра):        f = {f_num:.2f} Hz")

    if np.isfinite(f_num) and np.isfinite(f_mem) and f_mem > 0:
        err_mem = 100.0 * (f_num - f_mem) / f_mem
        out["err_membrane_pct"] = err_mem
        print(f"  Относительная ошибка:              {err_mem:+.1f} %")
    else:
        out["err_membrane_pct"] = np.nan

    print("\n--- Критерии корректности (отладка взрыва / 0 Гц / шума) ---")
    print(f"  max |uz| по сетке:     {max_uz_um:.2f} µm  (порог OK: <={MAX_UZ_UM_OK:.0f} µm)")
    print(f"  Выделенность пика:    {prominence:.2f}  (порог OK: >={MIN_PEAK_PROMINENCE:.1f})")
    print(f"  Взрыв (uz > порог):   {'ДА — неадекватные смещения' if is_explosion else 'нет'}")
    print(f"  0 Гц (пик < 1 Гц):    {'ДА — упругость не держит' if is_zero_hz else 'нет'}")
    print(f"  Шум (пик не выделен): {'ДА — спектр шумовой' if is_noisy else 'нет'}")
    if is_explosion or is_zero_hz or is_noisy:
        print("  Причины взрыва/0 Гц/шума: 1) simulate() затирал position/velocity (исправлено: reset_state=False).")
        print("  2) Граница фиксируется в ядре (position/velocity не обновляются). 3) При необходимости уменьшите impulse_velocity_z или dt.")
    print()
    return out


def _parse_cli_args(argv: list[str]):
    """Единый парсер CLI-аргументов запуска."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plot", action="store_true", dest="no_plot")
    parser.add_argument("--uniform", action="store_true", dest="uniform_pressure")
    parser.add_argument("--debug", action="store_true", dest="debug_m_total")
    parser.add_argument("--validate", action="store_true", dest="do_validate")
    parser.add_argument(
        "--force-shape",
        choices=("impulse", "uniform", "sine", "square", "chirp"),
        default="impulse",
        dest="force_shape",
        help="Форма внешнего давления: impulse|uniform|sine|square|chirp",
    )
    parser.add_argument(
        "--force-amplitude",
        type=float,
        default=10.0,
        dest="force_amplitude",
        help="Амплитуда давления, Па (для impulse: величина первого шага)",
    )
    parser.add_argument(
        "--force-offset",
        type=float,
        default=0.0,
        dest="force_offset",
        help="Постоянная составляющая давления, Па",
    )
    parser.add_argument(
        "--force-freq",
        type=float,
        default=1000.0,
        dest="force_freq",
        help="Частота (Гц) для sine/square и стартовая частота для chirp",
    )
    parser.add_argument(
        "--force-freq-end",
        type=float,
        default=5000.0,
        dest="force_freq_end",
        help="Конечная частота (Гц) для chirp",
    )
    parser.add_argument(
        "--force-phase-deg",
        type=float,
        default=0.0,
        dest="force_phase_deg",
        help="Начальная фаза (градусы) для sine/square/chirp",
    )
    parser.add_argument("--pre-tension", "--pre_tension", type=float, default=10.0, dest="pre_tension")
    parser.add_argument("--dt", type=float, default=1e-6)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument(
        "--air-grid-step-mm",
        "--air_grid_step_mm",
        type=float,
        default=None,
        dest="air_grid_step_mm",
        help="Шаг сетки акустического поля воздуха, мм (если не задан, берётся из шага КЭ мембраны).",
    )
    parser.add_argument(
        "--air-inject-mode",
        choices=("reduce", "direct"),
        default="reduce",
        dest="air_inject_mode",
        help="Режим инжекции энергии КЭ->air: reduce (через промежуточный буфер) или direct (прямая запись).",
    )
    parser.add_argument(
        "--material-library-file",
        type=str,
        default=None,
        dest="material_library_file",
        help="Путь к JSON-файлу библиотеки материалов [[density,E_parallel,E_perp,poisson,Cd,eta_visc,coupling_gain], ...].",
    )
    args, _ = parser.parse_known_args(argv[1:])
    return args


def _load_material_library_from_file(path: str) -> np.ndarray:
    """
    Загружает библиотеку материалов из JSON-файла.
    Формат: массив строк [[density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_gain], ...]
    или объект {"materials": [{"density": ..., "E_parallel": ..., ...}, ...]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if "rows" in data:
            rows = data["rows"]
        elif "materials" in data:
            mats = data["materials"]
            rows = []
            for m in mats:
                if isinstance(m, (list, tuple)):
                    rows.append(list(m))
                else:
                    rows.append([
                        float(m.get("density", 1000)),
                        float(m.get("E_parallel", 1e9)),
                        float(m.get("E_perp", 1e9)),
                        float(m.get("poisson", 0.3)),
                        float(m.get("Cd", 1.0)),
                        float(m.get("eta_visc", 1.0)),
                        float(m.get("coupling_gain", 0.5)),
                    ])
    if rows is None or len(rows) == 0:
        raise ValueError(f"Файл {path}: не найден массив материалов (rows или materials)")
    return np.array(rows, dtype=np.float64)


def _build_test_topology_with_cotton_layer(
    model: "PlanarDiaphragmOpenCL",
    gap_mm: float = 1.0,
) -> dict[str, object]:
    """
    Генерирует тестовую топологию из двух параллельных слоёв:
    1) сенсорный слой, 2) слой ваты.

    Ограничения/соглашения:
    - общее число КЭ не меняется (используется model.n_elements);
    - связи только внутренние в пределах каждого слоя (+X, -X, +Y, -Y);
    - фиксация по периметру каждого слоя;
    - слой ваты смещён на gap_mm от сенсорного слоя по +Z (зазор между поверхностями).
    """
    n_total = int(model.n_elements)
    if n_total < 2 or (n_total % 2) != 0:
        raise ValueError("Для тестовой 2-слойной топологии требуется чётное n_elements >= 2")
    n_layer = n_total // 2
    target_ratio = float(model.width / model.height) if model.height > 0.0 else 1.0

    # Подбираем (nx_layer, ny_layer), чтобы:
    # 1) nx_layer * ny_layer == n_layer
    # 2) форма была близка геометрии модели (почти квадратные КЭ).
    best_nx = 1
    best_ny = n_layer
    best_score = float("inf")
    for nx_layer in range(1, int(np.sqrt(n_layer)) + 1):
        if (n_layer % nx_layer) != 0:
            continue
        ny_layer = n_layer // nx_layer
        ratio = float(nx_layer / ny_layer)
        score = abs(np.log((ratio + 1e-12) / (target_ratio + 1e-12)))
        if score < best_score:
            best_score = score
            best_nx = nx_layer
            best_ny = ny_layer

    # Гарантируем nx >= ny для более ожидаемой ориентации сетки.
    nx_layer, ny_layer = (best_nx, best_ny) if best_nx >= best_ny else (best_ny, best_nx)
    if nx_layer * ny_layer != n_layer:
        raise ValueError("Не удалось построить регулярную сетку для слоя")

    sx = float(model.width / nx_layer)
    sy = float(model.height / ny_layer)
    # "Кубические или близко": выбираем толщину по меньшему шагу в плоскости.
    sz = float(min(sx, sy))
    gap = float(gap_mm) * 1e-2

    pos = np.zeros((n_total, 3), dtype=np.float64)
    size = np.zeros((n_total, 3), dtype=np.float64)
    neighbors = np.full((n_total, FACE_DIRS), -1, dtype=np.int32)
    material_index = np.full(n_total, MAT_SENSOR, dtype=np.uint8)
    boundary_mask = np.zeros(n_total, dtype=np.int32)

    z_mem_center = 0.0
    z_cotton_center = z_mem_center + sz + gap

    def idx_local(i: int, j: int) -> int:
        return j * nx_layer + i

    for layer in range(2):
        layer_offset = layer * n_layer
        zc = z_mem_center if layer == 0 else z_cotton_center
        mat_id = MAT_SENSOR if layer == 0 else MAT_COTTON_WOOL
        for j in range(ny_layer):
            for i in range(nx_layer):
                local = idx_local(i, j)
                idx = layer_offset + local
                x = (i + 0.5) * sx - 0.5 * model.width
                y = (j + 0.5) * sy - 0.5 * model.height
                pos[idx, 0] = x
                pos[idx, 1] = y
                pos[idx, 2] = zc
                size[idx, 0] = sx
                size[idx, 1] = sy
                # membrane thickness is 10 microns, cotton thickness is sz microns
                size[idx, 2] = 10e-6 if mat_id == MAT_SENSOR else sz
                material_index[idx] = mat_id

                # Внутренние связи только в своей плоскости.
                if i + 1 < nx_layer:
                    neighbors[idx, 0] = layer_offset + idx_local(i + 1, j)  # +X
                if i - 1 >= 0:
                    neighbors[idx, 1] = layer_offset + idx_local(i - 1, j)  # -X
                if j + 1 < ny_layer:
                    neighbors[idx, 2] = layer_offset + idx_local(i, j + 1)  # +Y
                if j - 1 >= 0:
                    neighbors[idx, 3] = layer_offset + idx_local(i, j - 1)  # -Y
                # ±Z намеренно без связей.

                if i == 0 or i == nx_layer - 1 or j == 0 or j == ny_layer - 1:
                    boundary_mask[idx] = 1

    return {
        "element_position_xyz": pos,
        "element_size_xyz": size,
        "neighbors": neighbors,
        "material_index": material_index,
        "boundary_mask_elements": boundary_mask,
        "visual_shape": (ny_layer, nx_layer),
    }

def run_cli_simulation(parsed_args) -> tuple["PlanarDiaphragmOpenCL", np.ndarray]:
    """Вспомогательная функция: выполняет симуляцию по CLI-аргументам.

    Принимает объект, возвращаемый _parse_cli_args(), создаёт модель,
    выполняет simulate() и возвращает (model, hist_center).
    """
    args = parsed_args
    no_plot = args.no_plot
    uniform_pressure = args.uniform_pressure
    debug_m_total = args.debug_m_total
    do_validate = args.do_validate
    pre_tension = float(args.pre_tension)
    # Переменная окружения переопределяет аргументы (удобно при запуске из IDE/launch без передачи --pre-tension)
    if os.environ.get("PRE_TENSION") not in (None, ""):
        try:
            pre_tension = float(os.environ["PRE_TENSION"].strip())
        except ValueError:
            pass

    material_props = None
    if getattr(args, "material_library_file", None):
        material_props = _load_material_library_from_file(args.material_library_file)

    model = PlanarDiaphragmOpenCL(
        nx=24 * 4,
        ny=32 * 4,
        pre_tension_N_per_m=pre_tension,
        air_grid_step_mm=args.air_grid_step_mm,
        air_inject_mode=args.air_inject_mode,
        kernel_debug=debug_m_total,
        material_props=material_props,
    )
    test_topology = _build_test_topology_with_cotton_layer(model, gap_mm=1.0)
    model.set_custom_topology(
        element_position_xyz=test_topology["element_position_xyz"],
        element_size_xyz=test_topology["element_size_xyz"],
        neighbors=test_topology["neighbors"],
        material_index=test_topology["material_index"],
        boundary_mask_elements=test_topology["boundary_mask_elements"],
        visual_shape=test_topology["visual_shape"],
        preserve_velocity=False,
        rebuild_air=True,
        air_grid_step_mm=args.air_grid_step_mm,
    )
    dt = float(args.dt)
    duration = 0.05 if args.duration is None else float(args.duration)  # 50 мс
    force_shape = str(args.force_shape)
    if uniform_pressure:
        force_shape = "uniform"
    if force_shape == "uniform" and args.duration is None:
        duration = 0.0005
    if dt <= 0.0 or duration <= 0.0:
        raise ValueError("--dt и --duration должны быть > 0")
    if args.force_freq < 0.0 or args.force_freq_end < 0.0:
        raise ValueError("--force-freq и --force-freq-end должны быть >= 0")
    if args.air_grid_step_mm is not None and args.air_grid_step_mm <= 0.0:
        raise ValueError("--air-grid-step-mm должен быть > 0")
    n_steps = int(duration / dt)
    t = np.arange(n_steps, dtype=np.float64) * dt
    amp = float(args.force_amplitude)
    off = float(args.force_offset)
    f0 = float(args.force_freq)
    f1 = float(args.force_freq_end)
    phase = np.deg2rad(float(args.force_phase_deg))
    pressure = np.zeros(n_steps, dtype=np.float64)
    if force_shape == "uniform":
        pressure.fill(off + amp)
    elif force_shape == "sine":
        pressure = off + amp * np.sin(2.0 * np.pi * f0 * t + phase)
    elif force_shape == "square":
        pressure = off + amp * np.where(np.sin(2.0 * np.pi * f0 * t + phase) >= 0.0, 1.0, -1.0)
    elif force_shape == "chirp":
        if duration <= 0.0:
            raise ValueError("Для chirp duration должен быть > 0")
        k = (f1 - f0) / duration
        phase_t = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t) + phase
        pressure = off + amp * np.sin(phase_t)
    else:
        # impulse: оставляем историческое поведение, но с настраиваемой амплитудой и offset.
        pressure.fill(off)
        if n_steps > 0:
            pressure[0] = off + amp

    if do_validate:
        print("\n--- Валидация собственных частот ---")
        print(f"Численная модель и аналитика: pre_tension = {model.pre_tension} Н/м")
    else:
        print("\n--- Simulate (OpenCL) ---")
        print(f"Форма внешней силы: {force_shape}")
        print(
            f"Параметры силы: amp={amp:.6g} Па, offset={off:.6g} Па, "
            f"f0={f0:.6g} Гц, f1={f1:.6g} Гц, phase={float(args.force_phase_deg):.3f} deg"
        )
    print(f"Преднатяжение pre_tension = {pre_tension} Н/м")
    first_bad = np.array([-1], dtype=np.int32)
    if not do_validate:
        # Важно: history_disp_all нужна и в режиме --no-plot (для GUI/отладки),
        # поэтому record_history всегда включён, независимо от no_plot.
        hist = model.simulate(
            pressure,
            dt=dt,
            record_history=True,
            check_air_resistance=not debug_m_total,
            validate_steps=True,
        )
        if np.any(~np.isfinite(hist)):
            print("OpenCL: NaN/Inf в истории")
        else:
            print(f"OpenCL: max |u_center| = {np.max(np.abs(hist)) * 1e6:.4f} um")
    else:
        hist = np.array([])

    if debug_m_total:
        debug_elem = getattr(model, "_first_bad_elem", model.center_idx)
        if debug_elem < 0:
            debug_elem = 64 * 4 + 1
        if getattr(model, "_first_bad_step", -1) >= 0:
            print(f"\n--- Отладка M_total для элемента {debug_elem} (first_bad) ---")
        else:
            print(f"\n--- Отладка M_total для центрального элемента {debug_elem} (первые шаги) ---")
        model._set_rest_position()
        model.velocity.fill(0.0)
        debug_buf = np.zeros(model._DEBUG_BUF_DOUBLES, dtype=np.float64)
        n_debug_steps = min(35, n_steps)
        for step_idx in range(n_debug_steps):
            p = pressure[step_idx] if pressure.ndim == 1 else pressure[step_idx]
            model.step(dt, p, step_idx=step_idx, debug_elem=debug_elem, debug_buf_out=debug_buf)
            if step_idx >= 30 and not np.any(np.isfinite(debug_buf[1:34])):
                break

    if do_validate:
        validate_natural_frequencies(model, dt=2e-7, duration=0.02, impulse_velocity_z=0.01)

    if not no_plot and hist.size > 0 and np.all(np.isfinite(hist)):
        model.plot_time_and_spectrum(dt=dt)
        model.plot_displacement_map()
        ani = model.animate(dt=dt, skip=20, interval_ms=50)
        ani_air = model.animate_air_pressure_center_plane(dt=dt, skip=20, interval_ms=50)
        plt.show()

    return model, hist


if __name__ == "__main__":
    import sys
    cli_args = _parse_cli_args(sys.argv)
    run_cli_simulation(cli_args)
