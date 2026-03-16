# -*- coding: utf-8 -*-
"""
Генератор объёмной 3D топологии из мешей модели данных.
- Solid: воксельная сетка (PyVista voxelize_rectilinear).
- Membrane/Sensor: плоскостная генерация — меш считается плоскостью, ориентированной по XY/XZ/YZ.
  Толщина КЭ = толщина меша, шаг сетки = element_size_mm из настроек. КЭ имеют неодинаковые размеры.

Формат вывода (совместим с diaphragm_opencl.PlanarDiaphragmOpenCL.set_custom_topology):
- element_position_xyz: [n, 3] float64 — координаты центров КЭ в глобальной СК
- element_size_xyz: [n, 3] float64 — [0],[1] = размеры в плоскости (du, dv), [2] = толщина
  (area = size[0]*size[1], diaphragm использует это для давления и air coupling)
- neighbors: [n, 6] int32 — FACE_DIRS: +X,-X,+Y,-Y,+Z,-Z; -1 = нет соседа
- material_index: [n] uint8 — индекс строки в material_props (0=membrane, 4=sensor, ...)
- boundary_mask_elements: [n] int32 — 1 = граничный (периметр), 0 = внутренний

Дополнительные данные для diaphragm_opencl (НЕ из топологии):
- material_props: [n_materials, 7] float64 — density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_gain.
  Должен содержать строки для всех material_index из топологии. Вызов set_material_library().
- material_key_to_index: маппинг material_key меша -> индекс в material_props. Должен совпадать
  с порядком материалов в библиотеке (MaterialLibraryModel.materials).
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyvista as pv
except ImportError:
    pv = None

try:
    import trimesh
except ImportError:
    trimesh = None

from project_model import MeshEntity, MeshTransform

# Совместимость с diaphragm_opencl
FACE_DIRS = 6  # +X, -X, +Y, -Y, +Z, -Z
MAT_MEMBRANE = np.uint8(0)
MAT_SENSOR = np.uint8(4)
MAT_FOAM_VE3015 = np.uint8(1)


def _build_transform_matrix(
    translation: list[float],
    rotation_deg: list[float],
    scale: list[float],
) -> np.ndarray:
    """Строит 4x4 матрицу аффинного преобразования из translation, euler deg, scale."""
    tr = (list(translation) + [0.0, 0.0, 0.0])[:3]
    rot = (list(rotation_deg) + [0.0, 0.0, 0.0])[:3]
    scl = (list(scale) + [1.0, 1.0, 1.0])[:3]

    rx, ry, rz = math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2])
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    R = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ], dtype=np.float64)
    S = np.diag([scl[0], scl[1], scl[2]])
    M = np.eye(4)
    M[:3, :3] = R @ S
    M[:3, 3] = tr
    return M


def _apply_transform(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Применяет 4x4 матрицу к точкам [n, 3]. Возвращает [n, 3]."""
    n = points.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)
    pts = np.hstack([points, ones])
    return (M @ pts.T).T[:, :3]


def _polydata_to_vertices_faces(poly, transform: MeshTransform) -> tuple[np.ndarray, np.ndarray] | None:
    """Преобразует PyVista PolyData в (vertices, faces) с применённым transform."""
    if poly is None or not hasattr(poly, "points"):
        return None
    pts = np.asarray(poly.points, dtype=np.float64)
    M = _build_transform_matrix(
        list(transform.translation),
        list(transform.rotation_euler_deg),
        list(transform.scale),
    )
    pts = _apply_transform(pts, M)
    cells = None
    if hasattr(poly, "faces"):
        cells = poly.faces
    elif hasattr(poly, "cells"):
        cells = poly.cells
    if cells is None or (hasattr(cells, "size") and cells.size == 0):
        return None
    offset = 0
    faces_list = []
    while offset < cells.shape[0]:
        nv = int(cells[offset])
        offset += 1
        if offset + nv > cells.shape[0]:
            break
        idx = cells[offset : offset + nv]
        offset += nv
        if nv == 3:
            faces_list.append(idx)
        elif nv == 4:
            faces_list.append([idx[0], idx[1], idx[2]])
            faces_list.append([idx[0], idx[2], idx[3]])
    if not faces_list:
        return None
    faces = np.array(faces_list, dtype=np.int64)
    return pts, faces


def _build_polydata_cells(faces: np.ndarray) -> np.ndarray:
    """Строит массив cells для PyVista PolyData из треугольных faces [n, 3]."""
    n_tri = faces.shape[0]
    cells = np.hstack([np.full((n_tri, 1), 3, dtype=np.int64), faces]).ravel()
    return cells


# --- Плоскостная топология (membrane, sensor) ---

_PLANAR_TOL = 1e-6  # допуск для проверки плоскостности
_ANGLE_TOL_DEG = 2.0  # допуск угла (градусы) для ориентации по осям


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Нормали граней [n_faces, 3], не нормализованные."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    return n


def _analyse_planar_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    unit_scale: float = 1.0,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Анализирует меш как плоскость.
    Проверяет: 1) меш плоский; 2) плоскость ориентирована по XY, XZ или YZ (углы кратны 90°).
    Возвращает (info, error). info содержит: normal_axis, thickness, u_axis, v_axis, bbox, verts_2d, faces.
    """
    if vertices.shape[0] < 3 or faces.shape[0] < 1:
        return None, "Недостаточно вершин или граней"

    normals = _compute_face_normals(vertices, faces)
    areas = np.linalg.norm(normals, axis=1, keepdims=True)
    areas = np.where(areas > 1e-20, areas, 1.0)
    normals = normals / areas
    # Усреднённая норма (взвешенная по площади)
    avg_normal = np.mean(normals * areas, axis=0)
    nlen = np.linalg.norm(avg_normal)
    if nlen < 1e-12:
        return None, "Не удалось определить нормаль плоскости"
    avg_normal = avg_normal / nlen

    # Проверка плоскостности: все точки лежат в одной плоскости
    centroid = np.mean(vertices, axis=0)
    dists = np.abs(np.dot(vertices - centroid, avg_normal))
    max_dist = float(np.max(dists))
    extent_along_normal = max_dist * 2.0 if max_dist > 1e-20 else 0.0
    bbox = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    extent_max = float(np.max(bbox))
    planar_tol = max(_PLANAR_TOL * extent_max, 1e-9)
    if max_dist > planar_tol:
        return None, f"Меш не плоский: отклонение точек от плоскости {max_dist:.2e} > {planar_tol:.2e}"

    # Проверка ориентации: нормаль параллельна одной из осей (углы 0° или 90°)
    abs_n = np.abs(avg_normal)
    axis_idx = int(np.argmax(abs_n))
    if abs_n[axis_idx] < 1.0 - math.radians(_ANGLE_TOL_DEG):
        return None, (
            f"Плоскость не ориентирована по осям: нормаль {avg_normal}, "
            f"углы должны быть кратны 90° (допуск {_ANGLE_TOL_DEG}°)"
        )

    # Определение плоскости: normal_axis — ось нормали, u_axis/v_axis — оси в плоскости.
    # XY: normal=Z, u=X, v=Y. XZ: normal=Y, u=X, v=Z. YZ: normal=X, u=Y, v=Z.
    if axis_idx == 0:
        u_axis, v_axis = 1, 2  # плоскость YZ
    elif axis_idx == 1:
        u_axis, v_axis = 0, 2  # плоскость XZ
    else:
        u_axis, v_axis = 0, 1  # плоскость XY

    # Толщина = размах вершин вдоль нормали
    proj = np.dot(vertices, avg_normal)
    thickness = float(np.max(proj) - np.min(proj))
    if thickness < 1e-12:
        thickness = extent_max * 1e-6  # минимальная толщина для численной устойчивости

    # 2D проекция вершин в плоскости
    verts_2d = vertices[:, [u_axis, v_axis]].astype(np.float64)

    # Bbox в 2D
    u_min, u_max = float(np.min(verts_2d[:, 0])), float(np.max(verts_2d[:, 0]))
    v_min, v_max = float(np.min(verts_2d[:, 1])), float(np.max(verts_2d[:, 1]))

    return {
        "normal_axis": axis_idx,
        "normal": avg_normal,
        "thickness": thickness,
        "u_axis": u_axis,
        "v_axis": v_axis,
        "bbox_u": (u_min, u_max),
        "bbox_v": (v_min, v_max),
        "verts_2d": verts_2d,
        "faces": faces,
        "vertices": vertices,
        "centroid": centroid,
    }, None


def _point_in_triangle_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Проверка: точка p внутри треугольника abc в 2D (барицентрические координаты)."""
    v0 = c - a
    v1 = b - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-20:
        return False
    s = (d11 * d20 - d01 * d21) / denom
    t = (d00 * d21 - d01 * d20) / denom
    return s >= -1e-12 and t >= -1e-12 and (s + t) <= 1.0 + 1e-12


def _point_inside_mesh_2d(
    p: np.ndarray,
    verts_2d: np.ndarray,
    faces: np.ndarray,
) -> bool:
    """Проверка: точка p (2D) внутри меша (проекция треугольников)."""
    for i in range(faces.shape[0]):
        a = verts_2d[faces[i, 0]]
        b = verts_2d[faces[i, 1]]
        c = verts_2d[faces[i, 2]]
        if _point_in_triangle_2d(p, a, b, c):
            return True
    return False


def _generate_planar_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    material_index: int,
    element_size_mm: float,
    padding_mm: float,
    unit_scale: float,
    log_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Генерирует топологию для плоского меша (membrane/sensor).
    - Толщина КЭ = толщина меша.
    - Шаг сетки в плоскости = element_size_mm.
    - КЭ имеют неодинаковые размеры (граничные ячейки меньше при некратном делении).
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    empty = (
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 3), dtype=np.float64),
        np.full((0, FACE_DIRS), -1, dtype=np.int32),
        np.zeros(0, dtype=np.uint8),
        np.zeros(0, dtype=np.int32),
    )

    info, err = _analyse_planar_mesh(vertices, faces, unit_scale)
    if err is not None:
        _log(f"  Ошибка анализа плоскости: {err}")
        return empty

    step = float(element_size_mm) * unit_scale
    if step <= 0:
        step = 1e-3
    pad = float(padding_mm) * unit_scale

    u_min, u_max = info["bbox_u"]
    v_min, v_max = info["bbox_v"]
    u_min -= pad
    u_max += pad
    v_min -= pad
    v_max += pad

    Lu = u_max - u_min
    Lv = v_max - v_min
    if Lu <= 0 or Lv <= 0:
        _log("  Нулевой размер плоскости после padding")
        return empty

    # Сетка с некратным шагом — граничные КЭ меньше (неодинаковые размеры)
    u_edges = []
    x = u_min
    while x < u_max - 1e-12:
        u_edges.append(x)
        x += step
    u_edges.append(u_max)

    v_edges = []
    y = v_min
    while y < v_max - 1e-12:
        v_edges.append(y)
        y += step
    v_edges.append(v_max)

    u_edges = np.array(u_edges, dtype=np.float64)
    v_edges = np.array(v_edges, dtype=np.float64)
    nu = len(u_edges) - 1
    nv = len(v_edges) - 1

    verts_2d = info["verts_2d"]
    faces_arr = info["faces"]
    normal_axis = info["normal_axis"]
    thickness = info["thickness"]
    u_axis = info["u_axis"]
    v_axis = info["v_axis"]
    centroid = info["centroid"]

    # Собираем ячейки, центр которых внутри меша
    cells_data = []
    for i in range(nu):
        for j in range(nv):
            cu = (u_edges[i] + u_edges[i + 1]) * 0.5
            cv = (v_edges[j] + v_edges[j + 1]) * 0.5
            if _point_inside_mesh_2d(np.array([cu, cv]), verts_2d, faces_arr):
                du = u_edges[i + 1] - u_edges[i]
                dv = v_edges[j + 1] - v_edges[j]
                cells_data.append((i, j, du, dv, cu, cv))

    if not cells_data:
        _log("  Нет ячеек внутри контура меша")
        return empty

    n = len(cells_data)
    positions = np.zeros((n, 3), dtype=np.float64)
    sizes = np.zeros((n, 3), dtype=np.float64)
    neighbors = np.full((n, FACE_DIRS), -1, dtype=np.int32)
    mat_arr = np.full(n, material_index, dtype=np.uint8)
    boundary = np.zeros(n, dtype=np.int32)

    cell_map = {}
    for idx, (ii, jj, du, dv, cu, cv) in enumerate(cells_data):
        cell_map[(ii, jj)] = idx

    # FACE_DIRS: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
    # Связи только в плоскости мембраны. Оси плоскости (u_axis, v_axis) задают направления.
    # Для оси a: + = 2*a, - = 2*a+1. По нормали (normal_axis) соседей нет.
    dir_plus_u = 2 * u_axis
    dir_minus_u = 2 * u_axis + 1
    dir_plus_v = 2 * v_axis
    dir_minus_v = 2 * v_axis + 1
    dir_plus_n = 2 * normal_axis
    dir_minus_n = 2 * normal_axis + 1

    z_coord = float(centroid[normal_axis])

    # diaphragm_opencl: area = size[0]*size[1], thickness = size[2]. Всегда [0],[1] = in-plane, [2] = thickness.
    for idx, (ii, jj, du, dv, cu, cv) in enumerate(cells_data):
        pos_3d = np.zeros(3)
        pos_3d[u_axis] = cu
        pos_3d[v_axis] = cv
        pos_3d[normal_axis] = z_coord
        positions[idx] = pos_3d

        size_3d = np.array([du, dv, thickness], dtype=np.float64)
        sizes[idx] = size_3d

        # Соседи
        n_plus_u = cell_map.get((ii + 1, jj))
        n_minus_u = cell_map.get((ii - 1, jj))
        n_plus_v = cell_map.get((ii, jj + 1))
        n_minus_v = cell_map.get((ii, jj - 1))

        neighbors[idx, dir_plus_u] = n_plus_u if n_plus_u is not None else -1
        neighbors[idx, dir_minus_u] = n_minus_u if n_minus_u is not None else -1
        neighbors[idx, dir_plus_v] = n_plus_v if n_plus_v is not None else -1
        neighbors[idx, dir_minus_v] = n_minus_v if n_minus_v is not None else -1
        neighbors[idx, dir_plus_n] = -1
        neighbors[idx, dir_minus_n] = -1

        # Автоматические граничные условия: периметр = КЭ без соседей по хотя бы одной грани в плоскости
        if n_plus_u is None or n_minus_u is None or n_plus_v is None or n_minus_v is None:
            boundary[idx] = 1

    n_boundary = int(np.sum(boundary))
    _log(f"  Периметр: {n_boundary} КЭ (автоматические граничные условия)")

    return positions, sizes, neighbors, mat_arr, boundary


def _voxelize_single_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    element_size_mm: float,
    padding_mm: float,
    material_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Вокселизация одного меша через PyVista voxelize_rectilinear.
    Вызывается в worker-процессе.
    Возвращает (positions, sizes, neighbors, material_index, boundary).
    """
    empty_result = (
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 3), dtype=np.float64),
        np.full((0, FACE_DIRS), -1, dtype=np.int32),
        np.zeros(0, dtype=np.uint8),
        np.zeros(0, dtype=np.int32),
    )
    if pv is None:
        return empty_result

    try:
        cells = _build_polydata_cells(faces)
        poly = pv.PolyData(vertices, cells)
    except Exception:
        return empty_result

    bmin = np.array(vertices.min(axis=0), dtype=np.float64)
    bmax = np.array(vertices.max(axis=0), dtype=np.float64)
    extent = bmax - bmin

    # Единицы: CAD/STL обычно в мм. Если extent > 10 — считаем мм, иначе метры.
    extent_max = float(np.max(extent))
    if extent_max > 10.0:
        unit_scale = 1.0  # mesh в мм
    else:
        unit_scale = 1e-3  # mesh в метрах
    pad = float(padding_mm) * unit_scale
    bmin = bmin - pad
    bmax = bmax + pad
    extent = bmax - bmin

    dx = float(element_size_mm) * unit_scale
    if dx <= 0:
        dx = 1e-3

    nx = max(1, int(np.ceil(extent[0] / dx)))
    ny = max(1, int(np.ceil(extent[1] / dx)))
    nz = max(1, int(np.ceil(extent[2] / dx)))

    dx_act = extent[0] / nx
    dy_act = extent[1] / ny
    dz_act = extent[2] / nz
    elem_size = np.array([dx_act, dy_act, dz_act], dtype=np.float64)

    try:
        poly = poly.clean()
        poly.compute_normals(inplace=True, auto_orient_normals=True)
    except Exception:
        pass

    try:
        vox = poly.voxelize_rectilinear(spacing=(dx_act, dy_act, dz_act))
    except Exception:
        return empty_result

    # Маска может быть в point_data (старые версии) или cell_data (PyVista 0.47+)
    if "mask" in vox.cell_data:
        mask_arr = vox.cell_data["mask"]
    elif "mask" in vox.point_data:
        mask_arr = vox.point_data["mask"]
    else:
        mask_arr = None
    if mask_arr is None:
        return empty_result

    mask = np.asarray(mask_arr).ravel()
    dims = np.array(vox.dimensions, dtype=np.int32)
    dimx, dimy, dimz = dims[0], dims[1], dims[2]

    # dimensions = число точек; ячеек (nx, ny, nz) = (dimx-1, dimy-1, dimz-1)
    ncx = dimx - 1
    ncy = dimy - 1
    ncz = dimz - 1
    if ncx <= 0 or ncy <= 0 or ncz <= 0:
        return empty_result

    def _collect_voxels(foreground_val: int) -> list:
        out = []
        if mask.size == vox.n_cells:
            for i in range(ncx):
                for j in range(ncy):
                    for k in range(ncz):
                        cell_idx = i + j * ncx + k * ncx * ncy
                        if cell_idx < mask.size and mask[cell_idx] == foreground_val:
                            out.append((i, j, k))
        else:
            for i in range(ncx):
                for j in range(ncy):
                    for k in range(ncz):
                        pt_idx = (i + 1) + (j + 1) * dimx + (k + 1) * dimx * dimy
                        if pt_idx < mask.size and mask[pt_idx] == foreground_val:
                            out.append((i, j, k))
        return out

    voxel_list = _collect_voxels(1)
    if not voxel_list:
        voxel_list = _collect_voxels(0)
    if not voxel_list:
        return empty_result

    voxel_to_idx = {v: idx for idx, v in enumerate(voxel_list)}
    n = len(voxel_list)

    x_coords = np.asarray(vox.x)
    y_coords = np.asarray(vox.y)
    z_coords = np.asarray(vox.z)

    positions = np.zeros((n, 3), dtype=np.float64)
    sizes = np.tile(elem_size, (n, 1))
    neighbors = np.full((n, FACE_DIRS), -1, dtype=np.int32)
    mat_arr = np.full(n, material_index, dtype=np.uint8)
    boundary = np.zeros(n, dtype=np.int32)

    deltas = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    for idx, (ix, iy, iz) in enumerate(voxel_list):
        cx = (x_coords[ix] + x_coords[ix + 1]) * 0.5
        cy = (y_coords[iy] + y_coords[iy + 1]) * 0.5
        cz = (z_coords[iz] + z_coords[iz + 1]) * 0.5
        positions[idx] = [cx, cy, cz]

        for d, (di, dj, dk) in enumerate(deltas):
            ni, nj, nk = ix + di, iy + dj, iz + dk
            nkey = (ni, nj, nk)
            if nkey in voxel_to_idx:
                neighbors[idx, d] = voxel_to_idx[nkey]
            else:
                boundary[idx] = 1

    return positions, sizes, neighbors, mat_arr, boundary


def _get_mesh_vertices_faces_list(
    meshes: list[MeshEntity],
    polydata_by_id: dict[str, Any],
    load_mesh_fn,
    material_key_to_index: dict[str, int],
    log_fn=None,
) -> tuple[list[tuple[np.ndarray, np.ndarray, int, str]], list[tuple[np.ndarray, np.ndarray, int, str]]]:
    """
    Возвращает (solid_list, planar_list).
    solid_list: (vertices, faces, material_index, name) для вокселизации.
    planar_list: (vertices, faces, material_index, name) для плоскостной генерации (membrane, sensor).
    """
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    def _get_verts_faces(mesh: MeshEntity):
        verts, faces = None, None
        poly = polydata_by_id.get(mesh.mesh_id)
        if poly is not None:
            vf = _polydata_to_vertices_faces(poly, mesh.transform)
            if vf is not None:
                verts, faces = vf
        if verts is None and load_mesh_fn:
            raw = load_mesh_fn(mesh)
            if raw is not None and hasattr(raw, "vertices") and hasattr(raw, "faces"):
                pts = np.asarray(raw.vertices, dtype=np.float64)
                M = _build_transform_matrix(
                    list(mesh.transform.translation),
                    list(mesh.transform.rotation_euler_deg),
                    list(mesh.transform.scale),
                )
                pts = _apply_transform(pts, M)
                f = np.asarray(raw.faces, dtype=np.int64)
                if f.shape[1] == 4:
                    f3 = np.hstack([f[:, :3], f[:, [0, 2, 3]]]).reshape(-1, 3)
                else:
                    f3 = f
                verts, faces = pts, f3
        return verts, faces

    solid_list = []
    planar_list = []

    for mesh in meshes:
        role = (mesh.role or "solid").lower()
        verts, faces = _get_verts_faces(mesh)

        if verts is None:
            _log(f"  Меш '{mesh.name}' ({mesh.mesh_id}): не удалось получить геометрию — пропущен")
            continue

        name = mesh.name or mesh.mesh_id
        mat_key = (mesh.material_key or "").lower()
        if role == "membrane":
            mat_idx = material_key_to_index.get(mat_key or "membrane", int(MAT_MEMBRANE))
        elif role == "sensor":
            mat_idx = material_key_to_index.get(mat_key or "sensor", int(MAT_SENSOR))
        else:
            mat_idx = material_key_to_index.get(mat_key or "foam_ve3015", int(MAT_FOAM_VE3015))

        if role in ("membrane", "sensor"):
            _log(f"  Меш '{name}' ({mesh.mesh_id}): плоскостная генерация (роль {role})")
            planar_list.append((verts, faces, mat_idx, name))
        else:
            _log(f"  Меш '{name}' ({mesh.mesh_id}): вокселизация (роль {role})")
            solid_list.append((verts, faces, mat_idx, name))

    return solid_list, planar_list


def generate_topology_from_meshes(
    meshes: list[MeshEntity],
    polydata_by_id: dict[str, Any],
    load_mesh_fn,
    *,
    element_size_mm: float = 0.5,
    padding_mm: float = 0.0,
    material_key_to_index: dict[str, int] | None = None,
    max_workers: int | None = None,
    log_callback=None,
) -> dict[str, np.ndarray]:
    """
    Генерирует регулярную объёмную 3D топологию: воксельная сетка с элементами одинакового размера,
    заполняющая объём каждой детали отдельно. Меши не объединяются — связи только внутри одного меша.

    Использует PyVista (VTK) voxelize_rectilinear для вокселизации.

    element_size_mm: размер конечного элемента по всем осям (мм).
    padding_mm: отступ от bbox каждого меша (мм).
    max_workers: число процессов для параллельной вокселизации (по умолчанию cpu_count - 1).

    Возвращает словарь:
    - element_position_xyz: [n, 3]
    - element_size_xyz: [n, 3] — одинаковый размер внутри каждого меша
    - neighbors: [n, FACE_DIRS] — связи только внутри меша
    - material_index: [n]
    - boundary_mask_elements: [n]
    log_callback: вызывается с (msg: str) для вывода в лог.
    """
    def _log(msg: str) -> None:
        if log_callback:
            log_callback(msg)

    if material_key_to_index is None:
        material_key_to_index = {
            "membrane": int(MAT_MEMBRANE),
            "foam_ve3015": int(MAT_FOAM_VE3015),
            "sensor": int(MAT_SENSOR),
        }

    _log("=== Генерация топологии ===")
    _log(f"Мешей в проекте: {len(meshes)}")
    _log(f"PolyData в кэше: {list(polydata_by_id.keys())}")
    _log("Обработка мешей:")

    solid_data, planar_data = _get_mesh_vertices_faces_list(
        meshes, polydata_by_id, load_mesh_fn,
        material_key_to_index=material_key_to_index,
        log_fn=_log,
    )

    if not solid_data and not planar_data:
        _log("Нет мешей для генерации (все пропущены или геометрия недоступна).")
        return {
            "element_position_xyz": np.zeros((0, 3), dtype=np.float64),
            "element_size_xyz": np.zeros((0, 3), dtype=np.float64),
            "neighbors": np.full((0, FACE_DIRS), -1, dtype=np.int32),
            "material_index": np.zeros(0, dtype=np.uint8),
            "boundary_mask_elements": np.zeros(0, dtype=np.int32),
        }

    all_positions = []
    all_sizes = []
    all_neighbors = []
    all_material = []
    all_boundary = []
    offset = 0

    # 1. Плоскостная генерация (membrane, sensor)
    for verts, faces, mat_idx, name in planar_data:
        extent_max = float(np.max(np.max(verts, axis=0) - np.min(verts, axis=0)))
        unit_scale = 1.0 if extent_max > 10.0 else 1e-3
        pos, sizes, nbh, mat, bnd = _generate_planar_topology(
            verts, faces, mat_idx,
            element_size_mm=element_size_mm,
            padding_mm=padding_mm,
            unit_scale=unit_scale,
            log_fn=_log,
        )
        if len(pos) == 0:
            _log(f"  Меш '{name}': 0 элементов (плоскостная генерация)")
            continue
        _log(f"  Меш '{name}': {len(pos)} элементов (плоскость)")
        all_positions.append(pos)
        all_sizes.append(sizes)
        nbh_adj = np.where(nbh >= 0, nbh + offset, -1)
        all_neighbors.append(nbh_adj)
        all_material.append(mat)
        all_boundary.append(bnd)
        offset += len(pos)

    # 2. Вокселизация (solid)
    if solid_data:
        if pv is None:
            raise RuntimeError("PyVista требуется для вокселизации. Установите: pip install pyvista")
        n_workers = max_workers if max_workers is not None else max(1, (os.cpu_count() or 1) - 1)
        _log(f"Вокселизация {len(solid_data)} мешей, workers={n_workers}")
        tasks = [
            (verts.copy(), faces.copy(), element_size_mm, padding_mm, mat_idx)
            for verts, faces, mat_idx, _ in solid_data
        ]
        mesh_names = [name for _, _, _, name in solid_data]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_voxelize_single_mesh, *t): i for i, t in enumerate(tasks)}
            for future in as_completed(futures):
                task_idx = futures[future]
                pos, sizes, nbh, mat, bnd = future.result()
                mesh_name = mesh_names[task_idx] if task_idx < len(mesh_names) else f"#{task_idx}"
                if len(pos) == 0:
                    _log(f"  Меш '{mesh_name}': 0 элементов (вокселизация)")
                    continue
                _log(f"  Меш '{mesh_name}': {len(pos)} элементов")
                all_positions.append(pos)
                all_sizes.append(sizes)
                nbh_adj = np.where(nbh >= 0, nbh + offset, -1)
                all_neighbors.append(nbh_adj)
                all_material.append(mat)
                all_boundary.append(bnd)
                offset += len(pos)

    if not all_positions:
        _log("Итого: 0 элементов (все меши дали пустой результат).")
        return {
            "element_position_xyz": np.zeros((0, 3), dtype=np.float64),
            "element_size_xyz": np.zeros((0, 3), dtype=np.float64),
            "neighbors": np.full((0, FACE_DIRS), -1, dtype=np.int32),
            "material_index": np.zeros(0, dtype=np.uint8),
            "boundary_mask_elements": np.zeros(0, dtype=np.int32),
        }

    total = sum(len(p) for p in all_positions)
    _log(f"Итого: {total} элементов.")
    return {
        "element_position_xyz": np.vstack(all_positions),
        "element_size_xyz": np.vstack(all_sizes),
        "neighbors": np.vstack(all_neighbors),
        "material_index": np.concatenate(all_material),
        "boundary_mask_elements": np.concatenate(all_boundary),
    }


def generate_procedural_topology_membrane(*args, **kwargs) -> dict[str, np.ndarray]:
    """Устарело: используйте generate_topology_from_meshes с мешем role=membrane."""
    raise NotImplementedError(
        "Используйте generate_topology_from_meshes с мешем role=membrane. "
        "Импортируйте плоский меш и назначьте ему роль membrane."
    )


def generate_procedural_topology_sensor(*args, **kwargs) -> dict[str, np.ndarray]:
    """Устарело: используйте generate_topology_from_meshes с мешем role=sensor."""
    raise NotImplementedError(
        "Используйте generate_topology_from_meshes с мешем role=sensor. "
        "Импортируйте плоский меш и назначьте ему роль sensor."
    )
