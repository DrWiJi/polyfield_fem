# -*- coding: utf-8 -*-
"""
Material library data model. Separate from simulation/project.
Based on diaphragm_opencl default materials.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MaterialEntry:
    """Single material: density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_gain."""

    name: str
    density: float  # kg/m³
    E_parallel: float  # Pa
    E_perp: float  # Pa
    poisson: float
    Cd: float
    eta_visc: float  # Pa·s
    coupling_gain: float  # 0..1

    def to_row(self) -> list[float]:
        """Row for np.ndarray / diaphragm_opencl format."""
        return [
            self.density,
            self.E_parallel,
            self.E_perp,
            self.poisson,
            self.Cd,
            self.eta_visc,
            self.coupling_gain,
        ]

    @staticmethod
    def from_row(name: str, row: list[float]) -> "MaterialEntry":
        row = (list(row) + [0.0] * 7)[:7]
        return MaterialEntry(
            name=name,
            density=float(row[0]),
            E_parallel=float(row[1]),
            E_perp=float(row[2]),
            poisson=float(row[3]),
            Cd=float(row[4]),
            eta_visc=float(row[5]),
            coupling_gain=float(row[6]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "MaterialEntry":
        return MaterialEntry(
            name=str(d.get("name", "Unnamed")),
            density=float(d.get("density", 1000.0)),
            E_parallel=float(d.get("E_parallel", 1e9)),
            E_perp=float(d.get("E_perp", 1e9)),
            poisson=float(d.get("poisson", 0.3)),
            Cd=float(d.get("Cd", 1.0)),
            eta_visc=float(d.get("eta_visc", 1.0)),
            coupling_gain=float(d.get("coupling_gain", 0.5)),
        )


def _stock_materials() -> list[MaterialEntry]:
    """Default library from diaphragm_opencl._build_default_material_library."""
    return [
        MaterialEntry(
            name="membrane",
            density=1380.0,
            E_parallel=5.0e9,
            E_perp=3.5e9,
            poisson=0.30,
            Cd=1.0,
            eta_visc=0.8,
            coupling_gain=0.90,
        ),
        MaterialEntry(
            name="foam_ve3015",
            density=55.0,
            E_parallel=0.08e6,
            E_perp=0.05e6,
            poisson=0.30,
            Cd=1.20,
            eta_visc=150.0,
            coupling_gain=0.25,
        ),
        MaterialEntry(
            name="sheepskin_leather",
            density=998.0,
            E_parallel=10.0e6,
            E_perp=7.0e6,
            poisson=0.40,
            Cd=1.05,
            eta_visc=12.0,
            coupling_gain=0.60,
        ),
        MaterialEntry(
            name="human_ear_avg",
            density=1080.0,
            E_parallel=1.80e6,
            E_perp=1.50e6,
            poisson=0.45,
            Cd=1.10,
            eta_visc=20.0,
            coupling_gain=0.50,
        ),
        MaterialEntry(
            name="sensor",
            density=1380.0,
            E_parallel=5.0e9,
            E_perp=3.5e9,
            poisson=0.30,
            Cd=1.0,
            eta_visc=0.8,
            coupling_gain=1.00,
        ),
        MaterialEntry(
            name="cotton_wool",
            density=250.0,
            E_parallel=0.03e6,
            E_perp=0.02e6,
            poisson=0.20,
            Cd=1.35,
            eta_visc=220.0,
            coupling_gain=0.30,
        ),
    ]


class MaterialLibraryModel:
    """Library of materials. Separate data model, not tied to project."""

    def __init__(self) -> None:
        self._materials: list[MaterialEntry] = list(_stock_materials())

    @property
    def materials(self) -> list[MaterialEntry]:
        return list(self._materials)

    def add(self, entry: MaterialEntry) -> int:
        self._materials.append(entry)
        return len(self._materials) - 1

    def update(self, index: int, entry: MaterialEntry) -> None:
        if 0 <= index < len(self._materials):
            self._materials[index] = entry

    def remove(self, index: int) -> None:
        if 0 <= index < len(self._materials):
            self._materials.pop(index)

    def clear_and_reset_to_stock(self) -> None:
        """Reset library to stock materials from diaphragm_opencl."""
        self._materials = list(_stock_materials())

    def merge(self, entries: list[MaterialEntry]) -> None:
        """Merge imported entries. Adds new materials, skips duplicates by name."""
        existing_names = {m.name.lower() for m in self._materials}
        for e in entries:
            if e.name.lower() not in existing_names:
                self._materials.append(e)
                existing_names.add(e.name.lower())

    def _unique_name(self, base: str) -> str:
        """Возвращает уникальное имя: base, base2, base3, ... при конфликтах."""
        existing = {m.name.lower() for m in self._materials}
        name = base
        n = 1
        while name.lower() in existing:
            n += 1
            name = f"{base}{n}"
        return name

    def ensure_material(self, name: str, density: float, E_parallel: float, E_perp: float,
                       poisson: float, Cd: float, eta_visc: float, coupling_gain: float) -> str:
        """
        Гарантирует наличие материала в библиотеке. Если name уже есть — возвращает name.
        Если нет — добавляет с указанными свойствами. При конфликте имён дописывает цифру.
        Возвращает фактическое имя материала (для обновления mesh.material_key).
        """
        for m in self._materials:
            if m.name.lower() == name.lower():
                return m.name
        unique = self._unique_name(name)
        entry = MaterialEntry(
            name=unique,
            density=density,
            E_parallel=E_parallel,
            E_perp=E_perp,
            poisson=poisson,
            Cd=Cd,
            eta_visc=eta_visc,
            coupling_gain=coupling_gain,
        )
        self._materials.append(entry)
        return unique

    def to_numpy_array(self) -> "np.ndarray":
        """Export as np.ndarray for diaphragm_opencl.set_material_library."""
        import numpy as np
        return np.array([m.to_row() for m in self._materials], dtype=np.float64)

    def save_json(self, path: str | Path) -> None:
        data = {"materials": [m.to_dict() for m in self._materials]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_json(self, path: str | Path) -> list[MaterialEntry]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        mats = data.get("materials", [])
        return [MaterialEntry.from_dict(m) for m in mats]

    def import_and_merge(self, path: str | Path) -> int:
        """Import from JSON and merge. Returns count of added materials."""
        imported = self.load_json(path)
        before = len(self._materials)
        self.merge(imported)
        return len(self._materials) - before

    def export_json(self, path: str | Path) -> None:
        self.save_json(path)
