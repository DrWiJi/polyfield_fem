# -*- coding: utf-8 -*-
"""
Генератор регулярной объёмной 3D топологии из мешей модели данных.
Объём деталей заполняется конечными элементами одинакового размера (воксельная сетка).
Формат совместим с diaphragm_opencl.PlanarDiaphragmOpenCL.set_custom_topology.

Использует PyVista (VTK) для вокселизации — voxelize_rectilinear.
Поддерживаемые типы мешей: solid, boundary (и др., кроме membrane, sensor).
MEMBRANE и SENSOR — процедурные меши, заготовка без реализации.
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
    log_fn=None,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """Возвращает список (vertices, faces, material_index) для мешей (кроме membrane, sensor)."""
    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)

    result = []
    for mesh in meshes:
        role = (mesh.role or "solid").lower()
        if role in ("membrane", "sensor"):
            _log(f"  Меш '{mesh.name}' ({mesh.mesh_id}): пропущен (роль {role})")
            continue

        verts, faces = None, None
        poly = polydata_by_id.get(mesh.mesh_id)
        if poly is not None:
            vf = _polydata_to_vertices_faces(poly, mesh.transform)
            if vf is not None:
                verts, faces = vf
                _log(f"  Меш '{mesh.name}' ({mesh.mesh_id}): PolyData, {verts.shape[0]} вершин, {faces.shape[0]} граней")
            else:
                _log(f"  Меш '{mesh.name}' ({mesh.mesh_id}): PolyData есть, но не удалось извлечь грани")

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
                _log(f"  Меш '{mesh.name}' ({mesh.mesh_id}): загружен из файла, {verts.shape[0]} вершин, {faces.shape[0]} граней")
            else:
                _log(f"  Меш '{mesh.name}' ({mesh.mesh_id}): загрузка из файла не удалась (load_mesh_fn вернул None)")

        if verts is None:
            _log(f"  Меш '{mesh.name}' ({mesh.mesh_id}): не удалось получить геометрию — пропущен")
            continue

        mat_idx = 1  # MAT_FOAM_VE3015
        result.append((verts, faces, mat_idx, mesh.name or mesh.mesh_id))

    return result


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

    mesh_data = _get_mesh_vertices_faces_list(meshes, polydata_by_id, load_mesh_fn, log_fn=_log)
    if not mesh_data:
        _log("Нет мешей для вокселизации (все пропущены или геометрия недоступна).")
        return {
            "element_position_xyz": np.zeros((0, 3), dtype=np.float64),
            "element_size_xyz": np.zeros((0, 3), dtype=np.float64),
            "neighbors": np.full((0, FACE_DIRS), -1, dtype=np.int32),
            "material_index": np.zeros(0, dtype=np.uint8),
            "boundary_mask_elements": np.zeros(0, dtype=np.int32),
        }

    if pv is None:
        raise RuntimeError("PyVista требуется для генерации топологии. Установите: pip install pyvista")

    n_workers = max_workers if max_workers is not None else max(1, (os.cpu_count() or 1) - 1)
    _log(f"Мешей для вокселизации: {len(mesh_data)}")
    _log(f"Параметры: element_size={element_size_mm} мм, padding={padding_mm} мм, workers={n_workers}")
    _log("Вокселизация...")

    tasks = [
        (verts.copy(), faces.copy(), element_size_mm, padding_mm, mat_idx)
        for verts, faces, mat_idx, _ in mesh_data
    ]
    mesh_names = [name for _, _, _, name in mesh_data]

    all_positions = []
    all_sizes = []
    all_neighbors = []
    all_material = []
    all_boundary = []
    offset = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_voxelize_single_mesh, *t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            task_idx = futures[future]
            pos, sizes, nbh, mat, bnd = future.result()
            mesh_name = mesh_names[task_idx] if task_idx < len(mesh_names) else f"#{task_idx}"
            if len(pos) == 0:
                _log(f"  Меш '{mesh_name}': 0 элементов (вокселизация не дала внутренних ячеек)")
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
    """Заготовка: процедурная топология для MEMBRANE. Не реализовано."""
    raise NotImplementedError("Процедурная топология MEMBRANE не реализована")


def generate_procedural_topology_sensor(*args, **kwargs) -> dict[str, np.ndarray]:
    """Заготовка: процедурная топология для SENSOR. Не реализовано."""
    raise NotImplementedError("Процедурная топология SENSOR не реализована")
