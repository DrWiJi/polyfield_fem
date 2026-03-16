# -*- coding: utf-8 -*-
"""
Central application state model.
Holds project, material library, project path, dirty flag.
Signals notify when state changes; all windows work through this model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal

from project_model import Project

from .material_library_model import MaterialLibraryModel

try:
    import numpy as np
except ImportError:
    np = None


def _topology_to_jsonifiable(topo: dict[str, Any]) -> dict[str, Any]:
    """Convert topology dict (numpy arrays) to JSON-serializable form."""
    if np is None:
        return {}
    return {
        "element_position_xyz": topo["element_position_xyz"].tolist(),
        "element_size_xyz": topo["element_size_xyz"].tolist(),
        "neighbors": topo["neighbors"].tolist(),
        "material_index": topo["material_index"].tolist(),
        "boundary_mask_elements": topo["boundary_mask_elements"].tolist(),
    }


def _topology_from_jsonifiable(data: dict[str, Any]) -> dict[str, Any] | None:
    """Convert JSON-loaded topology back to numpy form."""
    if np is None or not data:
        return None
    try:
        return {
            "element_position_xyz": np.array(data["element_position_xyz"], dtype=np.float64),
            "element_size_xyz": np.array(data["element_size_xyz"], dtype=np.float64),
            "neighbors": np.array(data["neighbors"], dtype=np.int32),
            "material_index": np.array(data["material_index"], dtype=np.uint8),
            "boundary_mask_elements": np.array(data["boundary_mask_elements"], dtype=np.int32),
        }
    except (KeyError, TypeError, ValueError):
        return None


class AppModel(QObject):
    """
    Single source of truth for application state.
    - project: simulation data (meshes, settings)
    - material_library: materials
    - project_path, is_dirty
    """

    project_changed = Signal()
    material_library_changed = Signal()
    state_changed = Signal()  # dirty or path changed (for title bar etc.)
    selection_changed = Signal()  # index from selected_mesh_index
    transform_changed = Signal()  # selected mesh transform updated (from affine widget)
    viewport_closed = Signal()  # emitted when any viewport window closes (BC, etc.)
    topology_changed = Signal()  # generated topology updated

    def __init__(
        self,
        parent: QObject | None = None,
        material_library: MaterialLibraryModel | None = None,
    ) -> None:
        super().__init__(parent)
        self._project = Project.create("New Project")
        self._material_library = material_library if material_library is not None else MaterialLibraryModel()
        self._project_path: Path | None = None
        self._is_dirty = False
        self._selected_mesh_index: int | None = None
        self._generated_topology: dict[str, Any] | None = None

    @property
    def project(self) -> Project:
        return self._project

    @property
    def material_library(self) -> MaterialLibraryModel:
        return self._material_library

    @property
    def project_path(self) -> Path | None:
        return self._project_path

    @property
    def is_dirty(self) -> bool:
        return self._is_dirty

    @property
    def selected_mesh_index(self) -> int | None:
        return self._selected_mesh_index

    def get_generated_topology(self) -> dict[str, Any] | None:
        """Return stored generated topology (numpy arrays) or None."""
        return self._generated_topology

    def set_generated_topology(self, topology: dict[str, Any] | None) -> None:
        """Store generated topology. Emits topology_changed."""
        self._generated_topology = topology
        self.topology_changed.emit()

    def set_selection(self, idx: int | None, *, force: bool = False) -> None:
        """Set selected mesh index. Emits selection_changed when value changes or force=True."""
        if force or self._selected_mesh_index != idx:
            self._selected_mesh_index = idx
            self.selection_changed.emit()

    def set_project_path(self, path: Path | None) -> None:
        if self._project_path != path:
            self._project_path = path
            self.state_changed.emit()

    def touch(self) -> None:
        """Mark project as modified."""
        self._project.touch()
        if not self._is_dirty:
            self._is_dirty = True
            self.state_changed.emit()

    def clear_dirty(self) -> None:
        if self._is_dirty:
            self._is_dirty = False
            self.state_changed.emit()

    def new_project(self) -> None:
        self._project = Project.create("New Project")
        self._project_path = None
        self._is_dirty = False
        self._generated_topology = None
        self.project_changed.emit()
        self.state_changed.emit()

    def load_project(self, path: Path | str) -> None:
        p = Path(path)
        self._project = Project.load_json(p)
        self._project_path = p
        self._is_dirty = False
        self._restore_topology_from_project()
        self.project_changed.emit()
        self.state_changed.emit()

    def save_project(self, path: Path | str | None = None) -> bool:
        """Save project. If path is None, uses project_path. Raises on I/O error."""
        target = Path(path) if path else self._project_path
        if target is None:
            return False
        self._sync_topology_to_project()
        self._project.save_json(target)
        self._project_path = target
        self._is_dirty = False
        self.state_changed.emit()
        return True

    def notify_material_library_changed(self) -> None:
        """Call when material library is modified (add/edit/remove/import/reset)."""
        self.material_library_changed.emit()

    def notify_viewport_closed(self) -> None:
        """Call when any viewport window is closed. Other viewports should restore handlers (picking, affine widget)."""
        self.viewport_closed.emit()

    def _sync_topology_to_project(self) -> None:
        """Write generated topology to project metadata for persistence."""
        if self._generated_topology:
            self._project.source_data.metadata["generated_topology"] = _topology_to_jsonifiable(
                self._generated_topology
            )
        elif "generated_topology" in self._project.source_data.metadata:
            del self._project.source_data.metadata["generated_topology"]

    def _restore_topology_from_project(self) -> None:
        """Restore generated topology from project metadata after load."""
        raw = self._project.source_data.metadata.get("generated_topology")
        self._generated_topology = _topology_from_jsonifiable(raw) if raw else None
