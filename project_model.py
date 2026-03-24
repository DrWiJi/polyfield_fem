# -*- coding: utf-8 -*-
"""
Data model for FE application project storage.

All user state is packed into Project:
- project_format_version (for future migrations)
- project metadata
- simulation source data (meshes with embedded geometry, mesh properties, simulation settings)
- unlimited list of simulation run records
"""

from __future__ import annotations

import base64
import gzip
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_FORMAT_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def mesh_encode(
    vertices: list,
    faces: list,
    normals: list | None = None,
) -> str:
    """
    Encode mesh geometry to compact base64 string.
    vertices: (N,3), faces: (M,3), normals: (N,3) optional vertex normals.
    """
    try:
        import numpy as np
        v = np.asarray(vertices, dtype=np.float32)
        f = np.asarray(faces, dtype=np.uint32)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
            raise ValueError("vertices must be (N,3), faces (M,3)")
        nv, nf = v.shape[0], f.shape[0]
        buf = struct.pack("II", nv, nf) + v.tobytes()
        if normals is not None:
            n = np.asarray(normals, dtype=np.float32)
            if n.shape == (nv, 3):
                buf += n.tobytes()
        buf += f.tobytes()
        return base64.b64encode(gzip.compress(buf)).decode("ascii")
    except ImportError:
        pass
    nv, nf = len(vertices), len(faces)
    parts = [struct.pack("II", nv, nf)]
    for v in vertices:
        parts.append(struct.pack("fff", float(v[0]), float(v[1]), float(v[2])))
    if normals is not None and len(normals) == nv:
        for n in normals:
            parts.append(struct.pack("fff", float(n[0]), float(n[1]), float(n[2])))
    for f in faces:
        parts.append(struct.pack("III", int(f[0]), int(f[1]), int(f[2])))
    buf = b"".join(parts)
    return base64.b64encode(gzip.compress(buf)).decode("ascii")


def mesh_decode(
    encoded: str,
) -> tuple[list[list[float]], list[list[int]], list[list[float]] | None] | None:
    """
    Decode mesh geometry from base64 string.
    Returns (vertices, faces, normals) or None on error.
    normals is None for legacy format (no normals stored).
    """
    if not encoded:
        return None
    try:
        buf = gzip.decompress(base64.b64decode(encoded))
        nv, nf = struct.unpack("II", buf[:8])
        offset = 8
        verts: list[list[float]] = []
        for _ in range(nv):
            x, y, z = struct.unpack_from("fff", buf, offset)
            verts.append([x, y, z])
            offset += 12
        normals: list[list[float]] | None = None
        old_format_size = 8 + nv * 12 + nf * 12
        if len(buf) >= old_format_size + nv * 12:
            normals = []
            for _ in range(nv):
                nx, ny, nz = struct.unpack_from("fff", buf, offset)
                normals.append([nx, ny, nz])
                offset += 12
        faces: list[list[int]] = []
        for _ in range(nf):
            a, b, c = struct.unpack_from("III", buf, offset)
            faces.append([a, b, c])
            offset += 12
        return (verts, faces, normals)
    except Exception:
        return None


def _vec3(raw: Any, default: list[float]) -> list[float]:
    vals = list(raw) if isinstance(raw, (list, tuple)) else list(default)
    vals = (vals + list(default))[:3]
    out: list[float] = []
    for i, fallback in enumerate(default):
        try:
            out.append(float(vals[i]))
        except Exception:  # noqa: BLE001
            out.append(float(fallback))
    return out


@dataclass
class MeshTransform:
    translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_euler_deg: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class MeshEntity:
    mesh_id: str
    name: str
    mesh_data: str = ""  # base64(gzip(vertices+faces)) - compact embedded geometry
    role: str = "solid"  # solid | membrane | sensor
    material_key: str = "membrane"
    visible: bool = True
    transform: MeshTransform = field(default_factory=MeshTransform)
    tags: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    boundary_groups: list[str] = field(default_factory=list)


@dataclass
class BoundaryCondition:
    """Boundary condition applied to specific meshes."""
    bc_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = "Boundary Condition"
    bc_type: str = "sphere"  # sphere, box, cylinder, tube (pyvista primitives)
    transform: MeshTransform = field(default_factory=MeshTransform)
    mesh_ids: list[str] = field(default_factory=list)  # IDs of meshes this BC applies to
    flags: dict[str, bool] = field(default_factory=lambda: {"fix_position": False})
    # Primitive-specific parameters (stored as generic dict for flexibility)
    parameters: dict[str, float] = field(default_factory=dict)


@dataclass
class SimulationSettings:
    dt: float = 1e-6
    duration: float = 0.05
    force_shape: str = "impulse"
    force_amplitude_pa: float = 10.0
    force_offset_pa: float = 0.0
    force_freq_hz: float = 1000.0
    force_freq_end_hz: float = 5000.0
    force_phase_deg: float = 0.0
    pre_tension_n_per_m: float = 10.0
    air_coupling_gain: float = 0.05
    air_grid_step_mm: float = 0.2
    air_pressure_history_every_steps: int = 10
    notes: str = ""


@dataclass
class SimulationSourceData:
    meshes: list[MeshEntity] = field(default_factory=list)
    simulation_settings: SimulationSettings = field(default_factory=SimulationSettings)
    material_library: list[list[float]] = field(default_factory=list)
    boundary_conditions: list[BoundaryCondition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationRunRecord:
    run_id: str
    created_at: str
    status: str = "created"  # created | running | completed | failed | cancelled
    started_at: str | None = None
    finished_at: str | None = None
    settings_snapshot: SimulationSettings | None = None
    source_snapshot: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # e.g. {"history_csv": "..."}
    log: str = ""
    error_message: str = ""


@dataclass
class Project:
    project_format_version: int
    name: str
    created_at: str
    updated_at: str
    source_data: SimulationSourceData = field(default_factory=SimulationSourceData)
    simulation_runs: list[SimulationRunRecord] = field(default_factory=list)

    @staticmethod
    def create(name: str) -> "Project":
        now = _now_iso()
        return Project(
            project_format_version=PROJECT_FORMAT_VERSION,
            name=name,
            created_at=now,
            updated_at=now,
        )

    def touch(self) -> None:
        self.updated_at = _now_iso()

    def add_mesh(self, name: str, role: str = "solid", material_key: str = "membrane") -> MeshEntity:
        mesh = MeshEntity(mesh_id=str(uuid4()), name=name, role=role, material_key=material_key)
        self.source_data.meshes.append(mesh)
        self.touch()
        return mesh

    def create_run_record(self, status: str = "created") -> SimulationRunRecord:
        rec = SimulationRunRecord(
            run_id=str(uuid4()),
            created_at=_now_iso(),
            status=status,
            settings_snapshot=SimulationSettings(**asdict(self.source_data.simulation_settings)),
            source_snapshot={
                "mesh_count": len(self.source_data.meshes),
                "materials_count": len(self.source_data.material_library),
            },
        )
        self.simulation_runs.append(rec)
        self.touch()
        return rec

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def save_json(self, path: str | Path, indent: int = 2) -> None:
        p = Path(path)
        p.write_text(self.to_json(indent=indent), encoding="utf-8")

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Project":
        version = int(data.get("project_format_version", data.get("model_version", PROJECT_FORMAT_VERSION)))
        if version != PROJECT_FORMAT_VERSION:
            # Placeholder for future migrations: raise or call migrate(data, version)
            raise ValueError(
                f"Unsupported project format version {version}. "
                f"Current version is {PROJECT_FORMAT_VERSION}."
            )
        src = data.get("source_data", {})

        meshes = []
        for raw in src.get("meshes", []):
            transform_raw = raw.get("transform", {})
            transform = MeshTransform(
                translation=_vec3(transform_raw.get("translation", [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]),
                rotation_euler_deg=_vec3(
                    transform_raw.get("rotation_euler_deg", [0.0, 0.0, 0.0]),
                    [0.0, 0.0, 0.0],
                ),
                scale=_vec3(transform_raw.get("scale", [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]),
            )
            mesh = MeshEntity(
                mesh_id=str(raw.get("mesh_id", uuid4())),
                name=str(raw.get("name", "UnnamedMesh")),
                mesh_data=str(raw.get("mesh_data", "")),
                role=str(raw.get("role", "solid")),
                material_key=str(raw.get("material_key", "membrane")),
                visible=bool(raw.get("visible", True)),
                transform=transform,
                tags=list(raw.get("tags", [])),
                properties=dict(raw.get("properties", {})),
                boundary_groups=list(raw.get("boundary_groups", [])),
            )
            meshes.append(mesh)

        sim_raw = src.get("simulation_settings", {})
        settings = SimulationSettings(
            dt=float(sim_raw.get("dt", 1e-6)),
            duration=float(sim_raw.get("duration", 0.05)),
            force_shape=str(sim_raw.get("force_shape", "impulse")),
            force_amplitude_pa=float(sim_raw.get("force_amplitude_pa", 10.0)),
            force_offset_pa=float(sim_raw.get("force_offset_pa", 0.0)),
            force_freq_hz=float(sim_raw.get("force_freq_hz", 1000.0)),
            force_freq_end_hz=float(sim_raw.get("force_freq_end_hz", 5000.0)),
            force_phase_deg=float(sim_raw.get("force_phase_deg", 0.0)),
            pre_tension_n_per_m=float(sim_raw.get("pre_tension_n_per_m", 10.0)),
            air_coupling_gain=float(sim_raw.get("air_coupling_gain", 0.05)),
            air_grid_step_mm=float(sim_raw.get("air_grid_step_mm", 0.2)),
            air_pressure_history_every_steps=int(sim_raw.get("air_pressure_history_every_steps", 10)),
            notes=str(sim_raw.get("notes", "")),
        )

        boundary_conditions = []
        for raw in src.get("boundary_conditions", []):
            tr_raw = raw.get("transform", {})
            bc_transform = MeshTransform(
                translation=_vec3(tr_raw.get("translation", [0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]),
                rotation_euler_deg=_vec3(
                    tr_raw.get("rotation_euler_deg", [0.0, 0.0, 0.0]),
                    [0.0, 0.0, 0.0],
                ),
                scale=_vec3(tr_raw.get("scale", [1.0, 1.0, 1.0]), [1.0, 1.0, 1.0]),
            )
            bc = BoundaryCondition(
                bc_id=str(raw.get("bc_id", uuid4())),
                name=str(raw.get("name", "Boundary Condition")),
                bc_type=str(raw.get("bc_type", "sphere")),
                transform=bc_transform,
                mesh_ids=list(raw.get("mesh_ids", [])),
                flags=dict(raw.get("flags", {"fix_position": False})),
                parameters=dict(raw.get("parameters", {})),
            )
            boundary_conditions.append(bc)

        source_data = SimulationSourceData(
            meshes=meshes,
            simulation_settings=settings,
            material_library=list(src.get("material_library", [])),
            boundary_conditions=boundary_conditions,
            metadata=dict(src.get("metadata", {})),
        )

        runs = []
        for raw in data.get("simulation_runs", []):
            snapshot_raw = raw.get("settings_snapshot")
            snapshot = None
            if isinstance(snapshot_raw, dict):
                snapshot = SimulationSettings(
                    dt=float(snapshot_raw.get("dt", settings.dt)),
                    duration=float(snapshot_raw.get("duration", settings.duration)),
                    force_shape=str(snapshot_raw.get("force_shape", settings.force_shape)),
                    force_amplitude_pa=float(snapshot_raw.get("force_amplitude_pa", settings.force_amplitude_pa)),
                    force_offset_pa=float(snapshot_raw.get("force_offset_pa", settings.force_offset_pa)),
                    force_freq_hz=float(snapshot_raw.get("force_freq_hz", settings.force_freq_hz)),
                    force_freq_end_hz=float(snapshot_raw.get("force_freq_end_hz", settings.force_freq_end_hz)),
                    force_phase_deg=float(snapshot_raw.get("force_phase_deg", settings.force_phase_deg)),
                    pre_tension_n_per_m=float(snapshot_raw.get("pre_tension_n_per_m", settings.pre_tension_n_per_m)),
                    air_coupling_gain=float(snapshot_raw.get("air_coupling_gain", settings.air_coupling_gain)),
                    air_grid_step_mm=float(snapshot_raw.get("air_grid_step_mm", settings.air_grid_step_mm)),
                    air_pressure_history_every_steps=int(
                        snapshot_raw.get(
                            "air_pressure_history_every_steps",
                            settings.air_pressure_history_every_steps,
                        )
                    ),
                    notes=str(snapshot_raw.get("notes", "")),
                )

            run = SimulationRunRecord(
                run_id=str(raw.get("run_id", uuid4())),
                created_at=str(raw.get("created_at", _now_iso())),
                status=str(raw.get("status", "created")),
                started_at=raw.get("started_at"),
                finished_at=raw.get("finished_at"),
                settings_snapshot=snapshot,
                source_snapshot=dict(raw.get("source_snapshot", {})),
                metrics=dict(raw.get("metrics", {})),
                artifacts=dict(raw.get("artifacts", {})),
                log=str(raw.get("log", "")),
                error_message=str(raw.get("error_message", "")),
            )
            runs.append(run)

        return Project(
            project_format_version=int(data.get("project_format_version", data.get("model_version", PROJECT_FORMAT_VERSION))),
            name=str(data.get("name", "Untitled Project")),
            created_at=str(data.get("created_at", _now_iso())),
            updated_at=str(data.get("updated_at", _now_iso())),
            source_data=source_data,
            simulation_runs=runs,
        )

    @staticmethod
    def from_json(text: str) -> "Project":
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("Project JSON root must be an object")
        return Project.from_dict(data)

    @staticmethod
    def load_json(path: str | Path) -> "Project":
        p = Path(path)
        return Project.from_json(p.read_text(encoding="utf-8"))


