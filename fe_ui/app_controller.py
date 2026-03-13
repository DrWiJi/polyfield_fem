# -*- coding: utf-8 -*-
"""
Application controller: shared material library, multi-window management.
Each window has its own project; material library is shared across all windows.
"""

from __future__ import annotations

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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._shared_material_library = MaterialLibraryModel()
        self._windows: list[FeMainWindow] = []

    @property
    def shared_material_library(self) -> MaterialLibraryModel:
        return self._shared_material_library

    def notify_material_library_changed(self) -> None:
        """Notify all windows that material library was modified."""
        self.material_library_changed.emit()

    def new_window(self, load_path: Path | str | None = None) -> FeMainWindow:
        """
        Create and show a new main window.
        If load_path is given, load that project; otherwise create new project.
        """
        from .main_window import FeMainWindow
        app_model = AppModel(material_library=self._shared_material_library)
        if load_path:
            app_model.load_project(load_path)
        window = FeMainWindow(app_model, self)
        self._windows.append(window)
        window.destroyed.connect(lambda: self._on_window_destroyed(window))
        window.show()
        return window

    def _on_window_destroyed(self, window: FeMainWindow) -> None:
        if window in self._windows:
            self._windows.remove(window)
