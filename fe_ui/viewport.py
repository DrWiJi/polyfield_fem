# -*- coding: utf-8 -*-
"""
Viewport widget: placeholder or PyVista 3D renderer.
Depends: PySide6. Optional: pyvista, pyvistaqt.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:
    pv = None
    QtInteractor = None


class ViewportPlaceholder(QFrame):
    """Empty viewport with click callback for mocked selection."""

    def __init__(self, on_click_callback=None) -> None:
        super().__init__()
        self._on_click_callback = on_click_callback
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("viewportPlaceholder")
        self.setMinimumSize(520, 420)

        layout = QVBoxLayout(self)
        title = QLabel("Viewport (placeholder)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: 600;")
        hint = QLabel(
            "3D rendering is not connected.\n"
            "Select meshes from the Mesh List."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #888;")
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch(1)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._on_click_callback:
            self._on_click_callback()
        super().mousePressEvent(event)


def create_viewport(parent: QWidget, on_pick_callback=None) -> tuple[QWidget, object | None]:
    """
    Create viewport widget: PyVista if available, else placeholder.
    Returns (widget, plotter_or_none). Plotter is used for add_mesh, remove_actor, etc.
    """
    if QtInteractor is None or pv is None:
        return ViewportPlaceholder(on_pick_callback), None

    plotter = QtInteractor(parent)
    plotter.set_background("#1f1f1f")
    plotter.add_axes()
    plotter.show_grid(color="#555555")
    _setup_lighting(plotter)
    return plotter.interactor, plotter


def _setup_lighting(plotter) -> None:
    """Configure scene lighting for PyVista plotter."""
    if pv is None:
        return
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(
        position=(2.0, 2.5, 3.0),
        focal_point=(0.0, 0.0, 0.0),
        color="white",
        intensity=1.0,
        light_type="scene light",
    ))
    plotter.add_light(pv.Light(
        position=(-2.5, 1.0, 1.5),
        focal_point=(0.0, 0.0, 0.0),
        color="#cfd8ff",
        intensity=0.45,
        light_type="scene light",
    ))
    plotter.add_light(pv.Light(
        position=(0.0, -3.0, 2.0),
        focal_point=(0.0, 0.0, 0.0),
        color="#ffe7c9",
        intensity=0.30,
        light_type="scene light",
    ))


def has_pyvista() -> bool:
    """Check if PyVista 3D rendering is available."""
    return pv is not None and QtInteractor is not None
