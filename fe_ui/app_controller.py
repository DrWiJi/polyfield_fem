# -*- coding: utf-8 -*-
"""
Application controller: shared material library, multi-window management.
Each window has its own project; material library is shared across all windows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal

from .app_model import AppModel
from .material_library_model import MaterialLibraryModel

if TYPE_CHECKING:
    from .main_window import FeMainWindow


class AppController(QObject):
    """
    Application-level controller: shared material library and window creation.
    """

    material_library_changed = Signal()

    def __init__(
        self,
        parent: QObject | None = None,
        material_library_path: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self._shared_material_library = MaterialLibraryModel()
        if material_library_path:
            try:
                self._shared_material_library.load_from_file(material_library_path)
            except (OSError, json.JSONDecodeError):
                pass  # Keep stock materials on load error
        self._windows: list[FeMainWindow] = []

    @property
    def shared_material_library(self) -> MaterialLibraryModel:
        return self._shared_material_library

    def notify_material_library_changed(self) -> None:
        """Notify all windows that material library was modified."""
        self.material_library_changed.emit()

    def new_window(
        self,
        load_path: Path | str | None = None,
        auto_run_simulation: bool = False,
    ) -> FeMainWindow:
        """
        Create and show a new main window.
        If load_path is given, load that project; otherwise create new project.
        If auto_run_simulation is True and project has saved topology, simulation starts automatically.
        """
        from .main_window import FeMainWindow
        app_model = AppModel(material_library=self._shared_material_library)
        if load_path:
            app_model.load_project(load_path)
        should_auto_run = (
            auto_run_simulation
            and load_path is not None
            and app_model.get_generated_topology() is not None
        )
        window = FeMainWindow(app_model, self, auto_run_simulation=should_auto_run)
        self._windows.append(window)
        window.destroyed.connect(lambda: self._on_window_destroyed(window))
        window.show()
        return window

    def _on_window_destroyed(self, window: FeMainWindow) -> None:
        if window in self._windows:
            self._windows.remove(window)
