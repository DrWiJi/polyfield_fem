# -*- coding: utf-8 -*-
"""
Central application state model.
Holds project, material library, project path, dirty flag.
Signals notify when state changes; all windows work through this model.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from project_model import Project

from .material_library_model import MaterialLibraryModel


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
        self.project_changed.emit()
        self.state_changed.emit()

    def load_project(self, path: Path | str) -> None:
        p = Path(path)
        self._project = Project.load_json(p)
        self._project_path = p
        self._is_dirty = False
        self.project_changed.emit()
        self.state_changed.emit()

    def save_project(self, path: Path | str | None = None) -> bool:
        """Save project. If path is None, uses project_path. Raises on I/O error."""
        target = Path(path) if path else self._project_path
        if target is None:
            return False
        self._project.save_json(target)
        self._project_path = target
        self._is_dirty = False
        self.state_changed.emit()
        return True

    def notify_material_library_changed(self) -> None:
        """Call when material library is modified (add/edit/remove/import/reset)."""
        self.material_library_changed.emit()
