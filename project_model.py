# -*- coding: utf-8 -*-
"""
Data model for FE application project storage.

All user state is packed into Project:
- model version (for future migrations)
- project metadata
- simulation source data (meshes, mesh properties, simulation settings)
- unlimited list of simulation run records
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_MODEL_VERSION = 1


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
    source_path: str = ""
    role: str = "solid"  # solid | membrane | boundary | sensor
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
    bc_type: str = "sphere"  # sphere, box, cylinder, plane, etc. (pyvista primitives)
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
    air_boundary_damping: float = 600.0
    air_bulk_damping: float = 120.0
    air_pressure_clip_pa: float = 2.0e4
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
    model_version: int
    name: str
    created_at: str
    updated_at: str
    source_data: SimulationSourceData = field(default_factory=SimulationSourceData)
    simulation_runs: list[SimulationRunRecord] = field(default_factory=list)

    @staticmethod
    def create(name: str) -> "Project":
        now = _now_iso()
        return Project(
            model_version=PROJECT_MODEL_VERSION,
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
        migrated = migrate_project_dict(data)
        src = migrated.get("source_data", {})

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
                source_path=str(raw.get("source_path", "")),
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
            air_boundary_damping=float(sim_raw.get("air_boundary_damping", 600.0)),
            air_bulk_damping=float(sim_raw.get("air_bulk_damping", 120.0)),
            air_pressure_clip_pa=float(sim_raw.get("air_pressure_clip_pa", 2.0e4)),
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
        for raw in migrated.get("simulation_runs", []):
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
                    air_boundary_damping=float(snapshot_raw.get("air_boundary_damping", settings.air_boundary_damping)),
                    air_bulk_damping=float(snapshot_raw.get("air_bulk_damping", settings.air_bulk_damping)),
                    air_pressure_clip_pa=float(snapshot_raw.get("air_pressure_clip_pa", settings.air_pressure_clip_pa)),
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
            model_version=int(migrated.get("model_version", PROJECT_MODEL_VERSION)),
            name=str(migrated.get("name", "Untitled Project")),
            created_at=str(migrated.get("created_at", _now_iso())),
            updated_at=str(migrated.get("updated_at", _now_iso())),
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


def migrate_project_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Migrate legacy project dicts to current model version."""
    if not isinstance(data, dict):
        raise ValueError("Project data must be a dict")

    out = dict(data)
    version = int(out.get("model_version", 0))
    while version < PROJECT_MODEL_VERSION:
        if version == 0:
            out = _migrate_v0_to_v1(out)
            version = 1
            continue
        raise ValueError(f"Unsupported migration path from version {version}")
    out["model_version"] = PROJECT_MODEL_VERSION
    return out


def _migrate_v0_to_v1(data: dict[str, Any]) -> dict[str, Any]:
    """
    Legacy bootstrap:
    - ensure required root fields
    - normalize source_data/simulation_runs containers
    """
    out = dict(data)
    now = _now_iso()
    out.setdefault("name", "Untitled Project")
    out.setdefault("created_at", now)
    out.setdefault("updated_at", now)
    if not isinstance(out.get("source_data"), dict):
        out["source_data"] = {}
    out["source_data"].setdefault("meshes", [])
    out["source_data"].setdefault("simulation_settings", {})
    out["source_data"].setdefault("material_library", [])
    out["source_data"].setdefault("boundary_conditions", [])
    out["source_data"].setdefault("metadata", {})
    if not isinstance(out.get("simulation_runs"), list):
        out["simulation_runs"] = []
    out["model_version"] = 1
    return out
