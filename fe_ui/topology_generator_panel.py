# -*- coding: utf-8 -*-
"""
Окно генератора 3D топологии из мешей модели.
Образец оформления: BoundaryConditionsPanel.
Viewport показывает только сгенерированную топологию (меши не отображаются).
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from .viewport import TopologyViewport
from .widgets import ScientificDoubleSpinBox
from topology_generator import generate_topology_from_meshes


class TopologyGeneratorPanel(QDockWidget):
    """Окно генераторa 3D топологии. Отдельный ViewPort для просмотра топологии."""

    def __init__(self, parent=None) -> None:
        super().__init__("Topology Generator", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        widget = QWidget()
        main_layout = QHBoxLayout(widget)

        # Viewport — только топология
        self._topology_viewport = TopologyViewport(self)
        self._topology_viewport.setMinimumSize(320, 240)
        main_layout.addWidget(self._topology_viewport, 1)

        # Правая панель: параметры и кнопка
        params_group = QGroupBox("Параметры генерации")
        params_layout = QFormLayout(params_group)

        self.sp_element_size_mm = ScientificDoubleSpinBox()
        self.sp_element_size_mm.setRange(0.01, 100.0)
        self.sp_element_size_mm.setDecimals(3)
        self.sp_element_size_mm.setValue(0.5)
        self.sp_element_size_mm.setSuffix(" mm")
        params_layout.addRow("Размер КЭ (воксель):", self.sp_element_size_mm)

        self.sp_padding_mm = ScientificDoubleSpinBox()
        self.sp_padding_mm.setRange(0.0, 100.0)
        self.sp_padding_mm.setDecimals(3)
        self.sp_padding_mm.setValue(0.0)
        self.sp_padding_mm.setSuffix(" mm")
        params_layout.addRow("Отступ от bbox:", self.sp_padding_mm)

        self.btn_generate = QPushButton("Сгенерировать топологию")
        self.btn_generate.clicked.connect(self._on_generate)

        log_group = QGroupBox("Лог генерации")
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(120)
        self.log_text.setPlaceholderText("Нажмите «Сгенерировать топологию» для запуска…")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_text)

        right_layout = QVBoxLayout()
        right_layout.addWidget(params_group)
        right_layout.addWidget(self.btn_generate)
        right_layout.addWidget(log_group, 1)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        right_widget.setFixedWidth(280)
        main_layout.addWidget(right_widget, 0)

        self.setWidget(widget)
        self._main_window = parent
        self._refresh_from_model()

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self._topology_viewport, "close_viewport"):
            self._topology_viewport.close_viewport()
        super().closeEvent(event)
        if self._main_window and hasattr(self._main_window, "_on_topology_generator_closed"):
            self._main_window._on_topology_generator_closed()

    def _get_mesh_data(self):
        """Получить polydata и meshes из main window."""
        if self._main_window and hasattr(self._main_window, "_mesh_polydata_by_id") and hasattr(self._main_window, "_app"):
            return (
                self._main_window._mesh_polydata_by_id.copy(),
                self._main_window._app.project.source_data.meshes,
            )
        return {}, []

    def _get_load_mesh_fn(self):
        if self._main_window and hasattr(self._main_window, "_load_trimesh_for_entity"):
            return self._main_window._load_trimesh_for_entity
        return lambda m: None

    def _get_material_key_to_index(self) -> dict[str, int]:
        """Маппинг material_key -> индекс в material_props для diaphragm_opencl."""
        mat_map = {}
        if self._main_window and hasattr(self._main_window, "_app"):
            lib = getattr(self._main_window._app, "material_library", None)
            if lib and hasattr(lib, "materials"):
                for i, m in enumerate(lib.materials):
                    key = getattr(m, "name", str(m)).lower().strip()
                    if key:
                        mat_map[key] = i
        if not mat_map:
            from topology_generator import MAT_MEMBRANE, MAT_FOAM_VE3015, MAT_SENSOR
            mat_map = {"membrane": int(MAT_MEMBRANE), "foam_ve3015": int(MAT_FOAM_VE3015), "sensor": int(MAT_SENSOR)}
        return mat_map

    def _log(self, msg: str) -> None:
        """Добавить строку в лог с меткой времени."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")

    def _refresh_from_model(self) -> None:
        """Load and display topology from app model."""
        topo = None
        if self._main_window and hasattr(self._main_window, "_app"):
            topo = self._main_window._app.get_generated_topology()
        self._topology_viewport.set_topology(topo)

    def showEvent(self, event: QShowEvent) -> None:
        """Refresh topology display when window is shown."""
        super().showEvent(event)
        self._refresh_from_model()

    def _on_generate(self) -> None:
        polydata_by_id, meshes = self._get_mesh_data()
        self.log_text.clear()
        if not meshes:
            self._log("Нет мешей в проекте.")
            return

        element_size_mm = self.sp_element_size_mm.value()
        padding_mm = self.sp_padding_mm.value()

        try:
            topology = generate_topology_from_meshes(
                meshes,
                polydata_by_id,
                self._get_load_mesh_fn(),
                element_size_mm=element_size_mm,
                padding_mm=padding_mm,
                material_key_to_index=self._get_material_key_to_index(),
                log_callback=self._log,
            )
        except NotImplementedError as e:
            self._log(f"Ошибка: {e}")
            return
        except Exception as e:
            import traceback
            self._log(f"Ошибка: {e}")
            self._log(traceback.format_exc())
            return

        n = topology["element_position_xyz"].shape[0]
        if self._main_window and hasattr(self._main_window, "_app"):
            self._main_window._app.set_generated_topology(topology)
            self._main_window._app.touch()
        self._topology_viewport.set_topology(topology)
        self._log(f"Топология сохранена в проект ({n} элементов).")
