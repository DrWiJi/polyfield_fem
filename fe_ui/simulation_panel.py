# -*- coding: utf-8 -*-
"""
Simulation dock: solver params, excitation, run/stop, console.
Depends: PySide6 only. No project_model — uses dict for settings.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .constants import FORCE_SHAPES
from .widgets import ScientificDoubleSpinBox


class SimulationPanel(QDockWidget):
    """Solver parameters, excitation, run/stop buttons, console output."""

    run_clicked = Signal()
    stop_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Simulation", parent)
        self.setAllowedAreas(Qt.BottomDockWidgetArea)

        panel = QWidget()
        main_layout = QHBoxLayout(panel)

        # Solver
        box_solver = QGroupBox("Solver Parameters")
        solver_form = QFormLayout(box_solver)
        self.sp_dt = ScientificDoubleSpinBox()
        self.sp_dt.setDecimals(9)
        self.sp_dt.setRange(1e-9, 1.0)
        # Меньший dt для стабильности при сильных силах от границы
        self.sp_dt.setValue(1e-7)
        self.sp_dt.setSingleStep(1e-6)
        self.sp_dt.setSuffix(" s")
        self.sp_duration = ScientificDoubleSpinBox()
        self.sp_duration.setDecimals(4)
        self.sp_duration.setRange(1e-4, 30.0)
        self.sp_duration.setValue(0.05)
        self.sp_duration.setSuffix(" s")
        self.sp_air_coupling = ScientificDoubleSpinBox()
        self.sp_air_coupling.setRange(0.0, 1.0)
        self.sp_air_coupling.setSingleStep(0.01)
        self.sp_air_coupling.setValue(0.05)
        self.sp_air_grid_step_mm = ScientificDoubleSpinBox()
        self.sp_air_grid_step_mm.setRange(0.01, 50.0)
        self.sp_air_grid_step_mm.setDecimals(3)
        self.sp_air_grid_step_mm.setSingleStep(0.01)
        self.sp_air_grid_step_mm.setValue(0.2)
        self.sp_air_grid_step_mm.setSuffix(" mm")
        solver_form.addRow("dt", self.sp_dt)
        solver_form.addRow("Duration", self.sp_duration)
        solver_form.addRow("Air coupling gain", self.sp_air_coupling)
        solver_form.addRow("Air grid step", self.sp_air_grid_step_mm)

        # Excitation
        box_force = QGroupBox("Excitation")
        force_form = QFormLayout(box_force)
        self.cb_force_shape = QComboBox()
        self.cb_force_shape.addItems(list(FORCE_SHAPES))
        self.sp_force_amp = ScientificDoubleSpinBox()
        self.sp_force_amp.setRange(0.0, 1e6)
        self.sp_force_amp.setValue(10.0)
        self.sp_force_amp.setSuffix(" Pa")
        self.sp_force_freq = ScientificDoubleSpinBox()
        self.sp_force_freq.setRange(0.0, 100000.0)
        self.sp_force_freq.setValue(1000.0)
        self.sp_force_freq.setSuffix(" Hz")
        force_form.addRow("Shape", self.cb_force_shape)
        force_form.addRow("Amplitude", self.sp_force_amp)
        force_form.addRow("Freq", self.sp_force_freq)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_stop = QPushButton("Stop")
        self.btn_export = QPushButton("Export Case")
        self.btn_run.clicked.connect(self.run_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.btn_export.clicked.connect(self.export_clicked.emit)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_export)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Simulation console output...")

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(box_solver)
        left_layout.addWidget(box_force)
        left_layout.addLayout(btn_row)
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.console, 1)

        self.setWidget(panel)

    def get_settings(self) -> dict:
        return {
            "dt": float(self.sp_dt.value()),
            "duration": float(self.sp_duration.value()),
            "air_coupling_gain": float(self.sp_air_coupling.value()),
            "air_grid_step_mm": float(self.sp_air_grid_step_mm.value()),
            "force_shape": self.cb_force_shape.currentText(),
            "force_amplitude_pa": float(self.sp_force_amp.value()),
            "force_freq_hz": float(self.sp_force_freq.value()),
        }

    def set_settings(self, data: dict) -> None:
        self.sp_dt.setValue(float(data.get("dt", 1e-6)))
        self.sp_duration.setValue(float(data.get("duration", 0.05)))
        self.sp_air_coupling.setValue(float(data.get("air_coupling_gain", 0.05)))
        self.sp_air_grid_step_mm.setValue(float(data.get("air_grid_step_mm", 0.2)))
        self.cb_force_shape.setCurrentText(str(data.get("force_shape", "impulse")))
        self.sp_force_amp.setValue(float(data.get("force_amplitude_pa", 10.0)))
        self.sp_force_freq.setValue(float(data.get("force_freq_hz", 1000.0)))

    def set_running(self, is_running: bool) -> None:
        self.btn_run.setEnabled(not is_running)
        self.btn_stop.setEnabled(is_running)

    def append_console(self, text: str) -> None:
        if text:
            self.console.insertPlainText(text)

    def connect_dirty(self, slot) -> None:
        """Connect settings widgets to dirty slot."""
        self.sp_dt.valueChanged.connect(slot)
        self.sp_duration.valueChanged.connect(slot)
        self.sp_air_coupling.valueChanged.connect(slot)
        self.sp_air_grid_step_mm.valueChanged.connect(slot)
        self.cb_force_shape.currentIndexChanged.connect(slot)
        self.sp_force_amp.valueChanged.connect(slot)
        self.sp_force_freq.valueChanged.connect(slot)
