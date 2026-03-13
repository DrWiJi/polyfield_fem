# -*- coding: utf-8 -*-
"""
Draft UI for FE workflow:
- Empty viewport placeholder
- Mesh list panel
- Parameter editor panel
- Simulation panel with run button

No real model/data backend is used yet.
Selection sync is mocked:
- click mesh in list -> updates editor
- click empty viewport -> cycles selection in mesh list
"""

from __future__ import annotations

import sys
from pathlib import Path
import threading
import io
import contextlib
import traceback
import numpy as np

from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QCloseEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from project_model import MeshEntity, Project

try:
    import trimesh
except Exception:  # noqa: BLE001
    trimesh = None

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:  # noqa: BLE001
    pv = None
    QtInteractor = None


class ViewportPlaceholder(QFrame):
    """Empty viewport with click callback for mocked selection."""

    def __init__(self, on_click_callback) -> None:
        super().__init__()
        self._on_click_callback = on_click_callback
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("viewportPlaceholder")
        self.setMinimumSize(520, 420)

        layout = QVBoxLayout(self)
        title = QLabel("Viewport (draft placeholder)")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: 600;")
        hint = QLabel(
            "3D rendering is not connected yet.\n"
            "Click here to test selection sync with Mesh List."
        )
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #888;")
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addStretch(1)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton and self._on_click_callback is not None:
            self._on_click_callback()
        super().mousePressEvent(event)


class FeUiDraftMainWindow(QMainWindow):
    debug_test_run_finished = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.resize(1400, 900)
        self.project_path: Path | None = None
        self.project = Project.create("New Project")
        self._is_dirty = False
        self._is_loading_ui = False
        self._viewport_pick_index = -1
        self._mesh_actor_by_id: dict[str, object] = {}
        self._mesh_polydata_by_id: dict[str, object] = {}
        self._sim_process: QProcess | None = None
        self._debug_thread: threading.Thread | None = None
        self._debug_history_disp_all: list[np.ndarray] | None = None
        self._debug_log_text: str = ""
        self._debug_positions_xyz: np.ndarray | None = None
        self._debug_anim_timer: QTimer | None = None
        self._debug_anim_frame_idx: int = 0
        self._debug_glyph_mesh = None

        self._build_menu()
        self._build_central()
        self._build_mesh_list_dock()
        self._build_properties_dock()
        self._build_simulation_dock()
        self.debug_test_run_finished.connect(self._finish_debug_test_run)
        self._wire_dirty_signals()
        self._load_project_to_ui()
        self._refresh_mesh_list()

    def _build_menu(self) -> None:
        menu_file = self.menuBar().addMenu("File")
        act_new = QAction("New Project", self)
        act_import_mesh = QAction("Import Mesh...", self)
        act_load = QAction("Load...", self)
        act_save = QAction("Save", self)
        act_save_as = QAction("Save As...", self)
        act_exit = QAction("Exit", self)
        act_new.triggered.connect(self._action_new_project)
        act_import_mesh.triggered.connect(self._action_import_mesh)
        act_load.triggered.connect(self._action_load_project)
        act_save.triggered.connect(self._action_save_project)
        act_save_as.triggered.connect(self._action_save_project_as)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_new)
        menu_file.addAction(act_import_mesh)
        menu_file.addSeparator()
        menu_file.addAction(act_load)
        menu_file.addAction(act_save)
        menu_file.addAction(act_save_as)
        menu_file.addSeparator()
        menu_file.addAction(act_exit)

        menu_debug = self.menuBar().addMenu("Debug")
        act_test_run = QAction("Test Run", self)
        act_test_run.triggered.connect(self._action_debug_test_run)
        menu_debug.addAction(act_test_run)
        act_test_vis = QAction("Test Visualization", self)
        act_test_vis.triggered.connect(self._action_debug_test_visualization)
        menu_debug.addAction(act_test_vis)

    def _build_central(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Horizontal)
        self.viewport = self._create_viewport_widget()
        splitter.addWidget(self.viewport)
        splitter.setStretchFactor(0, 1)

        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _create_viewport_widget(self) -> QWidget:
        if QtInteractor is None or pv is None:
            return ViewportPlaceholder(self._mock_pick_from_viewport)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1f1f1f")
        self.plotter.add_axes()
        self.plotter.show_grid(color="#555555")
        self._setup_viewport_lighting()
        return self.plotter.interactor

    def _setup_viewport_lighting(self) -> None:
        if not hasattr(self, "plotter"):
            return
        self.plotter.remove_all_lights()
        key = pv.Light(
            position=(2.0, 2.5, 3.0),
            focal_point=(0.0, 0.0, 0.0),
            color="white",
            intensity=1.0,
            light_type="scene light",
        )
        fill = pv.Light(
            position=(-2.5, 1.0, 1.5),
            focal_point=(0.0, 0.0, 0.0),
            color="#cfd8ff",
            intensity=0.45,
            light_type="scene light",
        )
        back = pv.Light(
            position=(0.0, -3.0, 2.0),
            focal_point=(0.0, 0.0, 0.0),
            color="#ffe7c9",
            intensity=0.30,
            light_type="scene light",
        )
        self.plotter.add_light(key)
        self.plotter.add_light(fill)
        self.plotter.add_light(back)

    def _build_mesh_list_dock(self) -> None:
        dock = QDockWidget("Mesh List", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.mesh_search = QLineEdit()
        self.mesh_search.setPlaceholderText("Search mesh...")
        self.mesh_search.textChanged.connect(self._refresh_mesh_list)

        self.mesh_list = QListWidget()
        self.mesh_list.currentRowChanged.connect(self._on_mesh_selected)

        button_row = QHBoxLayout()
        self.btn_add_mesh = QPushButton("Add")
        self.btn_remove_mesh = QPushButton("Remove")
        self.btn_isolate_mesh = QPushButton("Isolate")
        self.btn_add_mesh.clicked.connect(self._action_add_mesh)
        self.btn_remove_mesh.clicked.connect(self._action_remove_selected_mesh)
        button_row.addWidget(self.btn_add_mesh)
        button_row.addWidget(self.btn_remove_mesh)
        button_row.addWidget(self.btn_isolate_mesh)

        layout.addWidget(self.mesh_search)
        layout.addWidget(self.mesh_list, 1)
        layout.addLayout(button_row)

        dock.setWidget(panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_properties_dock(self) -> None:
        dock = QDockWidget("Mesh Parameter Editor", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        tabs = QTabWidget()
        tabs.addTab(self._build_identity_tab(), "Identity")
        tabs.addTab(self._build_material_tab(), "Material")
        tabs.addTab(self._build_membrane_tab(), "Membrane")
        tabs.addTab(self._build_transform_tab(), "Transform")
        tabs.addTab(self._build_boundary_tab(), "Boundary")
        self.mesh_editor_tabs = tabs

        self.editor_info = QLabel("No mesh selected")
        self.editor_info.setStyleSheet("color: #888;")

        btn_row = QHBoxLayout()
        self.btn_apply_mesh = QPushButton("Apply Mesh Params")
        self.btn_reset_mesh = QPushButton("Reset")
        self.btn_apply_mesh.clicked.connect(self._apply_selected_mesh_editor_to_model)
        self.btn_reset_mesh.clicked.connect(self._reload_selected_mesh_from_model)
        btn_row.addWidget(self.btn_apply_mesh)
        btn_row.addWidget(self.btn_reset_mesh)

        layout.addWidget(self.editor_info)
        layout.addWidget(tabs, 1)
        layout.addLayout(btn_row)

        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_identity_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.ed_name = QLineEdit()
        self.cb_role = QComboBox()
        self.cb_role.addItems(["solid", "membrane", "boundary", "sensor"])
        self.cb_role.currentTextChanged.connect(self._update_membrane_tab_visibility)
        self.cb_visible = QCheckBox()
        form.addRow("Name", self.ed_name)
        form.addRow("Role", self.cb_role)
        form.addRow("Visible", self.cb_visible)
        return w

    def _build_material_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.cb_material = QComboBox()
        self.cb_material.addItems(
            ["membrane", "foam_ve3015", "sheepskin_leather", "human_ear_avg"]
        )
        self.sp_density = QDoubleSpinBox()
        self.sp_density.setRange(1.0, 10000.0)
        self.sp_density.setValue(1380.0)
        self.sp_density.setSuffix(" kg/m^3")
        self.sp_E = QDoubleSpinBox()
        self.sp_E.setRange(1e3, 2e11)
        self.sp_E.setDecimals(0)
        self.sp_E.setValue(5.0e9)
        self.sp_E.setSuffix(" Pa")
        self.sp_poisson = QDoubleSpinBox()
        self.sp_poisson.setRange(0.0, 0.499)
        self.sp_poisson.setSingleStep(0.01)
        self.sp_poisson.setValue(0.30)
        form.addRow("Material Preset", self.cb_material)
        form.addRow("Density", self.sp_density)
        form.addRow("Young Modulus", self.sp_E)
        form.addRow("Poisson", self.sp_poisson)
        return w

    def _build_membrane_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.sp_thickness_mm = QDoubleSpinBox()
        self.sp_thickness_mm.setRange(0.001, 50.0)
        self.sp_thickness_mm.setDecimals(4)
        self.sp_thickness_mm.setValue(0.012)
        self.sp_thickness_mm.setSuffix(" mm")
        self.sp_pretension = QDoubleSpinBox()
        self.sp_pretension.setRange(0.0, 10000.0)
        self.sp_pretension.setValue(10.0)
        self.sp_pretension.setSuffix(" N/m")
        self.cb_fixed_edge = QComboBox()
        self.cb_fixed_edge.addItems(["none", "FIXED_EDGE", "FIXED_ALL"])
        form.addRow("Thickness", self.sp_thickness_mm)
        form.addRow("Pre-tension", self.sp_pretension)
        form.addRow("Boundary Group", self.cb_fixed_edge)
        return w

    def _build_transform_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        self.sp_tx = QDoubleSpinBox()
        self.sp_ty = QDoubleSpinBox()
        self.sp_tz = QDoubleSpinBox()
        for sp in (self.sp_tx, self.sp_ty, self.sp_tz):
            sp.setRange(-1_000_000.0, 1_000_000.0)
            sp.setDecimals(6)
            sp.setSingleStep(0.1)
            sp.setValue(0.0)

        self.sp_rx = QDoubleSpinBox()
        self.sp_ry = QDoubleSpinBox()
        self.sp_rz = QDoubleSpinBox()
        for sp in (self.sp_rx, self.sp_ry, self.sp_rz):
            sp.setRange(-360.0, 360.0)
            sp.setDecimals(3)
            sp.setSingleStep(1.0)
            sp.setValue(0.0)
            sp.setSuffix(" deg")

        self.sp_sx = QDoubleSpinBox()
        self.sp_sy = QDoubleSpinBox()
        self.sp_sz = QDoubleSpinBox()
        for sp in (self.sp_sx, self.sp_sy, self.sp_sz):
            sp.setRange(0.001, 1000.0)
            sp.setDecimals(6)
            sp.setSingleStep(0.01)
            sp.setValue(1.0)

        form.addRow("Translate X", self.sp_tx)
        form.addRow("Translate Y", self.sp_ty)
        form.addRow("Translate Z", self.sp_tz)
        form.addRow("Rotate X", self.sp_rx)
        form.addRow("Rotate Y", self.sp_ry)
        form.addRow("Rotate Z", self.sp_rz)
        form.addRow("Scale X", self.sp_sx)
        form.addRow("Scale Y", self.sp_sy)
        form.addRow("Scale Z", self.sp_sz)
        return w

    def _build_boundary_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        groups_box = QGroupBox("Boundary / Region Tags")
        g_layout = QGridLayout(groups_box)
        self.ed_bc_fixed = QLineEdit("FIXED_EDGE")
        self.ed_bc_load = QLineEdit("PRESSURE_ZONE")
        self.ed_bc_contact = QLineEdit("CONTACT_ZONE")
        g_layout.addWidget(QLabel("Fixed"), 0, 0)
        g_layout.addWidget(self.ed_bc_fixed, 0, 1)
        g_layout.addWidget(QLabel("Load"), 1, 0)
        g_layout.addWidget(self.ed_bc_load, 1, 1)
        g_layout.addWidget(QLabel("Contact"), 2, 0)
        g_layout.addWidget(self.ed_bc_contact, 2, 1)
        layout.addWidget(groups_box)

        self.ed_notes = QTextEdit()
        self.ed_notes.setPlaceholderText("Per-mesh notes, processing hints, exporter flags...")
        layout.addWidget(self.ed_notes, 1)
        return w

    def _build_simulation_dock(self) -> None:
        dock = QDockWidget("Simulation", self)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea)

        panel = QWidget()
        layout = QGridLayout(panel)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        box_solver = QGroupBox("Solver Parameters")
        solver_form = QFormLayout(box_solver)
        self.sp_dt = QDoubleSpinBox()
        self.sp_dt.setDecimals(9)
        self.sp_dt.setRange(1e-9, 1.0)
        self.sp_dt.setValue(1e-6)
        self.sp_dt.setSingleStep(1e-6)
        self.sp_dt.setSuffix(" s")
        self.sp_duration = QDoubleSpinBox()
        self.sp_duration.setDecimals(4)
        self.sp_duration.setRange(1e-4, 30.0)
        self.sp_duration.setValue(0.05)
        self.sp_duration.setSuffix(" s")
        self.sp_air_coupling = QDoubleSpinBox()
        self.sp_air_coupling.setRange(0.0, 1.0)
        self.sp_air_coupling.setSingleStep(0.01)
        self.sp_air_coupling.setValue(0.05)
        self.sp_air_grid_step_mm = QDoubleSpinBox()
        self.sp_air_grid_step_mm.setRange(0.01, 50.0)
        self.sp_air_grid_step_mm.setDecimals(3)
        self.sp_air_grid_step_mm.setSingleStep(0.01)
        self.sp_air_grid_step_mm.setValue(0.2)
        self.sp_air_grid_step_mm.setSuffix(" mm")
        solver_form.addRow("dt", self.sp_dt)
        solver_form.addRow("Duration", self.sp_duration)
        solver_form.addRow("Air coupling gain", self.sp_air_coupling)
        solver_form.addRow("Air grid step", self.sp_air_grid_step_mm)

        box_force = QGroupBox("Excitation")
        force_form = QFormLayout(box_force)
        self.cb_force_shape = QComboBox()
        self.cb_force_shape.addItems(["impulse", "uniform", "sine", "square", "chirp"])
        self.sp_force_amp = QDoubleSpinBox()
        self.sp_force_amp.setRange(0.0, 1e6)
        self.sp_force_amp.setValue(10.0)
        self.sp_force_amp.setSuffix(" Pa")
        self.sp_force_freq = QDoubleSpinBox()
        self.sp_force_freq.setRange(0.0, 100000.0)
        self.sp_force_freq.setValue(1000.0)
        self.sp_force_freq.setSuffix(" Hz")
        force_form.addRow("Shape", self.cb_force_shape)
        force_form.addRow("Amplitude", self.sp_force_amp)
        force_form.addRow("Freq", self.sp_force_freq)

        btn_row = QHBoxLayout()
        main_column = QHBoxLayout()
        console_column = QHBoxLayout()
        self.btn_run = QPushButton("Run Simulation")
        self.btn_stop = QPushButton("Stop")
        self.btn_export_case = QPushButton("Export Case")
        self.btn_run.clicked.connect(self._action_run_simulation)
        self.btn_stop.clicked.connect(self._action_stop_simulation)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_export_case)

        self.sim_console = QTextEdit()
        self.sim_console.setReadOnly(True)
        self.sim_console.setPlaceholderText("Simulation console output will appear here...")
        console_column.addWidget(self.sim_console)

        main_column.addWidget(box_solver)
        main_column.addWidget(box_force)

        # set column layouts for buttons and console
        layout.addLayout(main_column, 0, 0)
        layout.addLayout(console_column, 0, 1)
        layout.addLayout(btn_row, 1, 0)

        dock.setWidget(panel)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _append_sim_console(self, text: str) -> None:
        if not text:
            return
        self.sim_console.insertPlainText(text)

    def _set_sim_running_state(self, is_running: bool) -> None:
        self.btn_run.setEnabled(not is_running)
        self.btn_stop.setEnabled(is_running)

    def _append_debug_log(self, text: str) -> None:
        if not text:
            return
        self._append_sim_console("\n[Debug Test Run]\n")
        self._append_sim_console(text)

    def _finish_debug_test_run(self) -> None:
        log_text = self._debug_log_text or ""
        if log_text:
            self._append_debug_log(log_text)
        self._append_sim_console(
            f"\n[UI] Debug test run finished. Frames in history_disp_all: {len(self._debug_history_disp_all or [])}\n"
        )

    def _action_debug_test_run(self) -> None:
        if self._debug_thread is not None and self._debug_thread.is_alive():
            QMessageBox.information(self, "Debug Test Run", "Debug simulation is already running.")
            return

        self._append_sim_console("[UI] Starting debug test run in background thread...\n")

        def worker() -> None:
            try:
                import diaphragm_opencl as cl_model

                log_buf = io.StringIO()
                history_disp_all: list[np.ndarray] | None = None

                try:
                    with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
                        # Строим аргументы так же, как при CLI-запуске.
                        argv = [
                            "diaphragm_opencl.py",
                            "--no-plot",
                            "--dt",
                            "1e-7",
                            "--duration",
                            "0.001",
                            "--force-shape",
                            "impulse",
                            "--force-amplitude",
                            "0.001",
                            "--force-freq",
                            "200",
                            "--force-freq-end",
                            "5000",
                            "--air-inject-mode",
                            "reduce",
                            "--debug",
                        ]
                        args = cl_model._parse_cli_args(argv)

                        # Выполняем симуляцию стандартным CLI-путём.
                        model, hist_center = cl_model.run_cli_simulation(args)
                        _ = hist_center

                        history_disp_all = list(model.history_disp_all) if getattr(
                            model, "history_disp_all", None
                        ) is not None else []

                except Exception:
                    log_buf.write("\n[Debug Test Run] Exception during simulation:\n")
                    log_buf.write(traceback.format_exc())

                log_text = log_buf.getvalue()
                self._debug_history_disp_all = history_disp_all or []
                self._debug_positions_xyz = None
                self._debug_log_text = log_text
                # emit signal; Qt обеспечит queued connection между потоками
                self.debug_test_run_finished.emit()

            finally:
                self._debug_thread = None

        self._debug_thread = threading.Thread(target=worker, name="DebugTestRunThread", daemon=True)
        self._debug_thread.start()

    def _action_debug_test_visualization(self) -> None:
        if QtInteractor is None or pv is None or not hasattr(self, "plotter"):
            QMessageBox.information(
                self,
                "Test Visualization",
                "3D viewport (pyvista) is not available in this build.\n"
                "Install pyvista and pyvistaqt to enable in-viewport animation.",
            )
            return
        frames = self._debug_history_disp_all or []
        if not frames:
            QMessageBox.warning(
                self,
                "Test Visualization",
                "No debug history available.\nRun Debug → Test Run first to collect history_disp_all.",
            )
            return

        # Debug: проверяем, отличаются ли кадры истории между собой (по максимальному отклонению).
        try:
            first = np.asarray(frames[0], dtype=np.float64)
            max_diff = 0.0
            for f in frames[1:]:
                arr = np.asarray(f, dtype=np.float64)
                if arr.shape != first.shape:
                    max_diff = float("inf")
                    break
                diff = float(np.max(np.abs(arr - first)))
                if diff > max_diff:
                    max_diff = diff
            if not np.isfinite(max_diff):
                self._append_sim_console(
                    "[Debug Visualization] history_disp_all frames have different shapes; animation will change.\n"
                )
            elif max_diff == 0.0:
                self._append_sim_console(
                    "[Debug Visualization] All frames in history_disp_all are exactly identical; "
                    "animation will show no visible change.\n"
                )
            else:
                vmin = float(np.min(first))
                vmax = float(np.max(first))
                self._append_sim_console(
                    f"[Debug Visualization] history_disp_all varies across frames. "
                    f"First frame range: [{vmin:.3e}, {vmax:.3e}], max diff between frames: {max_diff:.3e}\n"
                )
        except Exception:
            self._append_sim_console(
                "[Debug Visualization] Failed to inspect history_disp_all for debugging.\n"
            )
            first = np.asarray(frames[0], dtype=np.float64)

        first = np.asarray(frames[0], dtype=np.float64)
        if first.ndim != 2:
            QMessageBox.warning(
                self,
                "Test Visualization",
                "history_disp_all does not contain 2D frames.\n"
                "Visualization in viewport expects 2D displacement maps.",
            )
            return
        # В модели: frame shape (ny, nx), frame[j,i] = значение в узле (x_index=i, y_index=j).
        # PyVista StructuredGrid: размерности (dimx, dimy)=(nx, ny), точка (i,j) имеет flat index = i + j*nx.
        # Поэтому flat должен быть: flat[i + j*nx] = frame[j,i]. Массив z[i,j]=frame[j,i] shape (nx,ny)
        # даёт нужный порядок при z.ravel(order='F') (первый индекс меняется быстрее).
        ny, nx = first.shape
        xs, ys = np.meshgrid(
            np.linspace(-0.5, 0.5, nx),
            np.linspace(-0.5, 0.5, ny),
            indexing="ij",
        )
        scale_z = 1e10
        z0 = (first.T * scale_z).astype(np.float64)  # (nx, ny), z0[i,j] = first[j,i]
        grid = pv.StructuredGrid(xs, ys, z0)
        grid["uz"] = first.T.ravel(order="F")
        self._debug_surface = grid
        if self._debug_anim_timer is not None:
            self._debug_anim_timer.stop()
            self._debug_anim_timer.deleteLater()
            self._debug_anim_timer = None
        try:
            actor_old = self._mesh_actor_by_id.get("__debug_surface__")
            if actor_old is not None:
                self.plotter.remove_actor(actor_old, reset_camera=False)
        except Exception:
            pass
        actor = self.plotter.add_mesh(
            self._debug_surface,
            name="debug_surface",
            scalars="uz",
            cmap="RdBu",
            show_edges=False,
        )
        self._mesh_actor_by_id["__debug_surface__"] = actor
        self.plotter.reset_camera()
        self.plotter.render()
        self._debug_anim_frame_idx = 0
        def update_frame() -> None:
            if not frames:
                return
            idx = self._debug_anim_frame_idx % len(frames)
            frame = np.asarray(frames[idx], dtype=np.float64)
            if frame.shape != (ny, nx):
                return
            # Тот же порядок точек: flat = i + j*nx → ravel(order='F')
            z = (frame.T * scale_z).astype(np.float64)
            pts = self._debug_surface.points
            pts[:, 2] = z.ravel(order="F")
            self._debug_surface.points = pts
            self._debug_surface["uz"] = frame.T.ravel(order="F")
            self._debug_anim_frame_idx = (self._debug_anim_frame_idx + 1) % len(frames)
            self.plotter.render()
        self._debug_anim_timer = QTimer(self)
        self._debug_anim_timer.timeout.connect(update_frame)
        self._debug_anim_timer.start(1)

    def _action_run_simulation(self) -> None:
        if self._sim_process is not None and self._sim_process.state() != QProcess.NotRunning:
            self._append_sim_console("[UI] Simulation process is already running.\n")
            return
        self._apply_simulation_settings_to_model()
        script_path = Path(__file__).with_name("diaphragm_opencl.py")
        if not script_path.exists():
            self._append_sim_console(f"[UI] Script not found: {script_path}\n")
            return

        args = [
            str(script_path),
            "--no-plot",
            "--dt",
            str(float(self.sp_dt.value())),
            "--duration",
            str(float(self.sp_duration.value())),
            "--force-shape",
            str(self.cb_force_shape.currentText()),
            "--force-amplitude",
            str(float(self.sp_force_amp.value())),
            "--force-freq",
            str(float(self.sp_force_freq.value())),
        ]

        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(args)
        proc.setWorkingDirectory(str(script_path.parent))
        proc.setProcessChannelMode(QProcess.SeparateChannels)
        proc.readyReadStandardOutput.connect(self._on_sim_stdout)
        proc.readyReadStandardError.connect(self._on_sim_stderr)
        proc.finished.connect(self._on_sim_finished)
        proc.errorOccurred.connect(self._on_sim_error)

        self._sim_process = proc
        self._set_sim_running_state(True)
        self._append_sim_console(f"[UI] Starting: {sys.executable} {' '.join(args)}\n")
        proc.start()

    def _action_stop_simulation(self) -> None:
        proc = self._sim_process
        if proc is None or proc.state() == QProcess.NotRunning:
            self._append_sim_console("[UI] No running simulation process.\n")
            self._set_sim_running_state(False)
            return
        self._append_sim_console("[UI] Stop requested. Killing process...\n")
        proc.kill()

    def _on_sim_stdout(self) -> None:
        proc = self._sim_process
        if proc is None:
            return
        data = bytes(proc.readAllStandardOutput())
        if data:
            self._append_sim_console(data.decode("utf-8", errors="replace"))

    def _on_sim_stderr(self) -> None:
        proc = self._sim_process
        if proc is None:
            return
        data = bytes(proc.readAllStandardError())
        if data:
            self._append_sim_console(data.decode("utf-8", errors="replace"))

    def _on_sim_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        status = "normal" if exit_status == QProcess.NormalExit else "crash"
        self._append_sim_console(f"[UI] Simulation finished. exit_code={exit_code}, status={status}\n")
        self._set_sim_running_state(False)
        self._sim_process = None

    def _on_sim_error(self, error: QProcess.ProcessError) -> None:
        self._append_sim_console(f"[UI] Process error: {int(error)}\n")

    def _refresh_mesh_list(self) -> None:
        query = self.mesh_search.text().strip().lower() if hasattr(self, "mesh_search") else ""
        self.mesh_list.blockSignals(True)
        self.mesh_list.clear()
        for i, mesh in enumerate(self.project.source_data.meshes):
            text = f"{mesh.name}  [{mesh.role}]  <{mesh.material_key}>"
            if query and query not in text.lower():
                continue
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, i)
            self.mesh_list.addItem(item)
        self.mesh_list.blockSignals(False)
        if self.mesh_list.count() > 0 and self.mesh_list.currentRow() < 0:
            self.mesh_list.setCurrentRow(0)

    def _selected_mesh_index(self) -> int | None:
        item = self.mesh_list.currentItem()
        if item is None:
            return None
        idx = item.data(Qt.UserRole)
        return int(idx) if idx is not None else None

    def _on_mesh_selected(self, _: int) -> None:
        idx = self._selected_mesh_index()
        if idx is None:
            self.editor_info.setText("No mesh selected")
            self._set_mesh_editor_enabled(False)
            self._update_membrane_tab_visibility("solid")
            self._update_viewport_selection()
            return
        self._set_mesh_editor_enabled(True)
        mesh = self.project.source_data.meshes[idx]
        was_loading = self._is_loading_ui
        self._is_loading_ui = True
        try:
            self.editor_info.setText(f"Selected: {mesh.name}")
            self.ed_name.setText(mesh.name)
            self.cb_role.setCurrentText(mesh.role)
            self.cb_material.setCurrentText(mesh.material_key)
            self.cb_visible.setChecked(bool(mesh.visible))
            self.sp_density.setValue(float(mesh.properties.get("density", 1380.0)))
            self.sp_E.setValue(float(mesh.properties.get("young_modulus", 5.0e9)))
            self.sp_poisson.setValue(float(mesh.properties.get("poisson", 0.30)))
            self.sp_thickness_mm.setValue(float(mesh.properties.get("thickness_mm", 0.012)))
            self.sp_pretension.setValue(float(mesh.properties.get("pre_tension_n_per_m", 10.0)))
            translation = list(mesh.transform.translation) if mesh.transform is not None else [0.0, 0.0, 0.0]
            rotation = (
                list(mesh.transform.rotation_euler_deg) if mesh.transform is not None else [0.0, 0.0, 0.0]
            )
            scale = list(mesh.transform.scale) if mesh.transform is not None else [1.0, 1.0, 1.0]
            translation = (translation + [0.0, 0.0, 0.0])[:3]
            rotation = (rotation + [0.0, 0.0, 0.0])[:3]
            scale = (scale + [1.0, 1.0, 1.0])[:3]
            self.sp_tx.setValue(float(translation[0]))
            self.sp_ty.setValue(float(translation[1]))
            self.sp_tz.setValue(float(translation[2]))
            self.sp_rx.setValue(float(rotation[0]))
            self.sp_ry.setValue(float(rotation[1]))
            self.sp_rz.setValue(float(rotation[2]))
            self.sp_sx.setValue(float(scale[0]))
            self.sp_sy.setValue(float(scale[1]))
            self.sp_sz.setValue(float(scale[2]))
            groups = mesh.boundary_groups
            self.cb_fixed_edge.setCurrentText(groups[0] if groups else "none")
            self.ed_notes.setPlainText(str(mesh.properties.get("notes", "")))
        finally:
            self._is_loading_ui = was_loading
        self._update_membrane_tab_visibility(mesh.role)
        self._update_viewport_selection()

    def _mock_pick_from_viewport(self) -> None:
        if hasattr(self, "plotter"):
            return
        if self.mesh_list.count() == 0:
            return
        self._viewport_pick_index = (self._viewport_pick_index + 1) % self.mesh_list.count()
        self.mesh_list.setCurrentRow(self._viewport_pick_index)

    def _load_project_to_ui(self) -> None:
        self._is_loading_ui = True
        try:
            sim = self.project.source_data.simulation_settings
            self.sp_dt.setValue(sim.dt)
            self.sp_duration.setValue(sim.duration)
            self.sp_air_coupling.setValue(sim.air_coupling_gain)
            self.sp_air_grid_step_mm.setValue(sim.air_grid_step_mm)
            self.cb_force_shape.setCurrentText(sim.force_shape)
            self.sp_force_amp.setValue(sim.force_amplitude_pa)
            self.sp_force_freq.setValue(sim.force_freq_hz)
            self._load_boundary_defaults_to_ui()
            self._update_window_title()
            if self.project.source_data.meshes:
                self.mesh_list.setCurrentRow(0)
                self._on_mesh_selected(0)
            else:
                self._clear_mesh_editor()
            self._rebuild_viewport_from_project()
        finally:
            self._is_loading_ui = False

    def _apply_simulation_settings_to_model(self) -> None:
        sim = self.project.source_data.simulation_settings
        sim.dt = float(self.sp_dt.value())
        sim.duration = float(self.sp_duration.value())
        sim.air_coupling_gain = float(self.sp_air_coupling.value())
        sim.air_grid_step_mm = float(self.sp_air_grid_step_mm.value())
        sim.force_shape = str(self.cb_force_shape.currentText())
        sim.force_amplitude_pa = float(self.sp_force_amp.value())
        sim.force_freq_hz = float(self.sp_force_freq.value())
        self._apply_boundary_defaults_to_project()
        self.project.touch()
        self._is_dirty = True
        self._update_window_title()

    def _apply_selected_mesh_editor_to_model(self) -> None:
        idx = self._selected_mesh_index()
        if idx is None:
            return
        mesh = self.project.source_data.meshes[idx]
        mesh.name = self.ed_name.text().strip() or mesh.name
        mesh.role = self.cb_role.currentText()
        mesh.material_key = self.cb_material.currentText()
        mesh.visible = bool(self.cb_visible.isChecked())
        mesh.properties["density"] = float(self.sp_density.value())
        mesh.properties["young_modulus"] = float(self.sp_E.value())
        mesh.properties["poisson"] = float(self.sp_poisson.value())
        mesh.properties["thickness_mm"] = float(self.sp_thickness_mm.value())
        mesh.properties["pre_tension_n_per_m"] = float(self.sp_pretension.value())
        mesh.transform.translation = [float(self.sp_tx.value()), float(self.sp_ty.value()), float(self.sp_tz.value())]
        mesh.transform.rotation_euler_deg = [
            float(self.sp_rx.value()),
            float(self.sp_ry.value()),
            float(self.sp_rz.value()),
        ]
        mesh.transform.scale = [float(self.sp_sx.value()), float(self.sp_sy.value()), float(self.sp_sz.value())]
        notes = self.ed_notes.toPlainText().strip()
        if notes:
            mesh.properties["notes"] = notes
        elif "notes" in mesh.properties:
            mesh.properties.pop("notes")
        group = self.cb_fixed_edge.currentText()
        mesh.boundary_groups = [] if group == "none" else [group]
        self.project.touch()
        self._is_dirty = True
        self._refresh_mesh_list()
        self._update_viewport_selection()
        self._update_window_title()

    def _reload_selected_mesh_from_model(self) -> None:
        if self.mesh_list.count() == 0:
            self._clear_mesh_editor()
            return
        self._on_mesh_selected(self.mesh_list.currentRow())

    def _action_add_mesh(self) -> None:
        n = len(self.project.source_data.meshes) + 1
        mesh = self.project.add_mesh(name=f"Mesh_{n}", role="solid", material_key="membrane")
        mesh.properties["density"] = 1380.0
        mesh.properties["young_modulus"] = 5.0e9
        mesh.properties["poisson"] = 0.30
        self._is_dirty = True
        self._refresh_mesh_list()
        self.mesh_list.setCurrentRow(self.mesh_list.count() - 1)
        self._update_window_title()

    def _action_remove_selected_mesh(self) -> None:
        idx = self._selected_mesh_index()
        if idx is None:
            return
        removed = self.project.source_data.meshes.pop(idx)
        self._remove_mesh_actor(removed.mesh_id)
        self.project.touch()
        self._is_dirty = True
        self._refresh_mesh_list()
        if self.mesh_list.count() == 0:
            self._clear_mesh_editor()
        self._update_window_title()

    def _remove_mesh_actor(self, mesh_id: str) -> None:
        actor = self._mesh_actor_by_id.pop(mesh_id, None)
        self._mesh_polydata_by_id.pop(mesh_id, None)
        if actor is not None and hasattr(self, "plotter"):
            self.plotter.remove_actor(actor, reset_camera=False)
            self.plotter.render()

    def _action_import_mesh(self) -> None:
        if trimesh is None:
            QMessageBox.critical(
                self,
                "Import Error",
                "trimesh is not available. Install it with:\n\npip install trimesh",
            )
            return
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Mesh",
            "",
            (
                "Mesh files (*.stl *.obj *.ply *.off *.glb *.gltf *.dae *.3mf *.xaml);;"
                "All Files (*.*)"
            ),
        )
        if not file_paths:
            return

        imported_indices: list[int] = []
        for file_path in file_paths:
            src = Path(file_path)
            try:
                loaded = trimesh.load(file_path, force="scene")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(self, "Import Warning", f"Failed to import {src.name}:\n{exc}")
                continue

            geometries = []
            if hasattr(loaded, "geometry") and isinstance(loaded.geometry, dict):
                geometries = list(loaded.geometry.items())
            elif hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
                geometries = [(src.stem, loaded)]

            for geom_name, geom in geometries:
                if not hasattr(geom, "vertices") or not hasattr(geom, "faces"):
                    continue
                if len(geom.vertices) == 0 or len(geom.faces) == 0:
                    continue
                if len(geometries) == 1:
                    mesh_name = src.stem
                else:
                    mesh_name = f"{src.stem}:{geom_name}"
                new_mesh = self.project.add_mesh(name=mesh_name, role="solid", material_key="membrane")
                new_mesh.source_path = str(src)
                new_mesh.properties["trimesh_geom_name"] = str(geom_name)
                new_mesh.properties["vertex_count"] = int(len(geom.vertices))
                new_mesh.properties["face_count"] = int(len(geom.faces))
                new_mesh.properties["is_watertight"] = bool(getattr(geom, "is_watertight", False))
                self._add_mesh_to_viewport(new_mesh, geom)
                imported_indices.append(len(self.project.source_data.meshes) - 1)

        if not imported_indices:
            QMessageBox.information(self, "Import Mesh", "No polygonal meshes were imported.")
            return

        self._is_dirty = True
        self._refresh_mesh_list()
        self.mesh_list.setCurrentRow(imported_indices[-1])
        self._update_window_title()

    def _trimesh_to_polydata(self, tri_mesh) -> object | None:
        if pv is None:
            return None
        faces = np.asarray(tri_mesh.faces, dtype=np.int64)
        vertices = np.asarray(tri_mesh.vertices, dtype=np.float64)
        if faces.size == 0 or vertices.size == 0:
            return None
        face_cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        return pv.PolyData(vertices, face_cells)

    def _add_mesh_to_viewport(self, mesh: MeshEntity, tri_mesh) -> None:
        if not hasattr(self, "plotter"):
            return
        poly = self._trimesh_to_polydata(tri_mesh)
        if poly is None:
            return
        actor = self.plotter.add_mesh(
            poly,
            color="#9A9A9A",
            smooth_shading=True,
            pickable=True,
            name=f"mesh_{mesh.mesh_id}",
            show_edges=False,
            reset_camera=False,
        )
        self._mesh_polydata_by_id[mesh.mesh_id] = poly
        self._mesh_actor_by_id[mesh.mesh_id] = actor
        if hasattr(actor, "SetVisibility"):
            actor.SetVisibility(1 if mesh.visible else 0)
        self._apply_actor_transform(mesh, actor)
        self.plotter.reset_camera()
        self._update_viewport_selection()
        self.plotter.render()

    def _update_viewport_selection(self) -> None:
        if not hasattr(self, "plotter"):
            return
        selected_idx = self._selected_mesh_index()
        selected_id = None
        if selected_idx is not None and 0 <= selected_idx < len(self.project.source_data.meshes):
            selected_id = self.project.source_data.meshes[selected_idx].mesh_id
        mesh_by_id = {mesh.mesh_id: mesh for mesh in self.project.source_data.meshes}
        for mesh_id, actor in self._mesh_actor_by_id.items():
            mesh = mesh_by_id.get(mesh_id)
            if mesh is not None:
                self._apply_actor_transform(mesh, actor)
            is_visible = True if mesh is None else bool(mesh.visible)
            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(1 if is_visible else 0)
            if not is_visible:
                continue
            color = "#F0D070" if mesh_id == selected_id else "#9A9A9A"
            prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
            if prop is not None:
                prop.SetColor(*pv.Color(color).float_rgb)
        self.plotter.render()

    def _load_trimesh_for_entity(self, mesh: MeshEntity):
        if trimesh is None or not mesh.source_path:
            return None
        src = Path(mesh.source_path)
        if not src.exists():
            return None
        try:
            loaded = trimesh.load(str(src), force="scene")
        except Exception:  # noqa: BLE001
            return None
        if hasattr(loaded, "geometry") and isinstance(loaded.geometry, dict):
            target = str(mesh.properties.get("trimesh_geom_name", ""))
            if target and target in loaded.geometry:
                return loaded.geometry[target]
            if loaded.geometry:
                return next(iter(loaded.geometry.values()))
            return None
        if hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
            return loaded
        return None

    def _rebuild_viewport_from_project(self) -> None:
        if not hasattr(self, "plotter"):
            return
        self.plotter.clear()
        self.plotter.add_axes()
        self.plotter.show_grid(color="#555555")
        self._setup_viewport_lighting()
        self._mesh_actor_by_id.clear()
        self._mesh_polydata_by_id.clear()
        for mesh in self.project.source_data.meshes:
            tri = self._load_trimesh_for_entity(mesh)
            if tri is not None:
                self._add_mesh_to_viewport(mesh, tri)
        self._update_viewport_selection()

    def _update_window_title(self) -> None:
        path_part = str(self.project_path) if self.project_path is not None else "unsaved"
        dirty_mark = "*" if self._is_dirty else ""
        self.setWindowTitle(f"FE Import/Markup Draft UI{dirty_mark} - {self.project.name} ({path_part})")

    def _mark_dirty(self) -> None:
        if self._is_loading_ui:
            return
        if not self._is_dirty:
            self._is_dirty = True
            self._update_window_title()

    def _confirm_save_if_dirty(self) -> bool:
        if not self._is_dirty:
            return True
        reply = QMessageBox.question(
            self,
            "Unsaved Changes",
            "Project has unsaved changes. Save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Cancel:
            return False
        if reply == QMessageBox.Discard:
            return True
        if reply == QMessageBox.Save:
            ok = self._save_project_internal(force_save_as_when_needed=True)
            return ok
        return False

    def _save_project_internal(self, force_save_as_when_needed: bool) -> bool:
        self._apply_simulation_settings_to_model()
        self._apply_selected_mesh_editor_to_model()
        if self.project_path is None:
            if not force_save_as_when_needed:
                return False
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Project As",
                "project.json",
                "Project JSON (*.json);;All Files (*.*)",
            )
            if not file_path:
                return False
            if not file_path.lower().endswith(".json"):
                file_path += ".json"
            self.project_path = Path(file_path)
        try:
            self.project.save_json(self.project_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Save Error", f"Failed to save project:\n{exc}")
            return False
        self._is_dirty = False
        self._update_window_title()
        return True

    def _action_new_project(self) -> None:
        if not self._confirm_save_if_dirty():
            return
        self.project = Project.create("New Project")
        self.project_path = None
        self._is_dirty = False
        self._load_project_to_ui()
        self._refresh_mesh_list()

    def _action_load_project(self) -> None:
        if not self._confirm_save_if_dirty():
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            "",
            "Project JSON (*.json);;All Files (*.*)",
        )
        if not file_path:
            return
        try:
            self.project = Project.load_json(file_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load Error", f"Failed to load project:\n{exc}")
            return
        self.project_path = Path(file_path)
        self._is_dirty = False
        self._load_project_to_ui()
        self._refresh_mesh_list()

    def _clear_mesh_editor(self) -> None:
        self._set_mesh_editor_enabled(False)
        self.editor_info.setText("No mesh selected")
        self.ed_name.setText("")
        self.cb_role.setCurrentText("solid")
        self._update_membrane_tab_visibility("solid")
        self.cb_material.setCurrentText("membrane")
        self.cb_visible.setChecked(True)
        self.sp_density.setValue(1380.0)
        self.sp_E.setValue(5.0e9)
        self.sp_poisson.setValue(0.30)
        self.sp_thickness_mm.setValue(0.012)
        self.sp_pretension.setValue(10.0)
        self.sp_tx.setValue(0.0)
        self.sp_ty.setValue(0.0)
        self.sp_tz.setValue(0.0)
        self.sp_rx.setValue(0.0)
        self.sp_ry.setValue(0.0)
        self.sp_rz.setValue(0.0)
        self.sp_sx.setValue(1.0)
        self.sp_sy.setValue(1.0)
        self.sp_sz.setValue(1.0)
        self.cb_fixed_edge.setCurrentText("none")
        self.ed_notes.setPlainText("")

    def _set_mesh_editor_enabled(self, enabled: bool) -> None:
        self.mesh_editor_tabs.setEnabled(enabled)
        self.btn_apply_mesh.setEnabled(enabled)
        self.btn_reset_mesh.setEnabled(enabled)

    def _update_membrane_tab_visibility(self, role: str) -> None:
        tab_idx = self.mesh_editor_tabs.indexOf(self.sp_thickness_mm.parentWidget())
        if tab_idx < 0:
            return
        self.mesh_editor_tabs.setTabVisible(tab_idx, role == "membrane")

    def _apply_actor_transform(self, mesh: MeshEntity, actor: object) -> None:
        tr = list(mesh.transform.translation) if mesh.transform is not None else [0.0, 0.0, 0.0]
        rot = list(mesh.transform.rotation_euler_deg) if mesh.transform is not None else [0.0, 0.0, 0.0]
        scl = list(mesh.transform.scale) if mesh.transform is not None else [1.0, 1.0, 1.0]
        tr = (tr + [0.0, 0.0, 0.0])[:3]
        rot = (rot + [0.0, 0.0, 0.0])[:3]
        scl = (scl + [1.0, 1.0, 1.0])[:3]
        if hasattr(actor, "SetPosition"):
            actor.SetPosition(float(tr[0]), float(tr[1]), float(tr[2]))
        if hasattr(actor, "SetOrientation"):
            actor.SetOrientation(float(rot[0]), float(rot[1]), float(rot[2]))
        if hasattr(actor, "SetScale"):
            actor.SetScale(float(scl[0]), float(scl[1]), float(scl[2]))

    def _apply_transform_live(self, _value: float) -> None:
        if self._is_loading_ui:
            return
        idx = self._selected_mesh_index()
        if idx is None:
            return
        mesh = self.project.source_data.meshes[idx]
        mesh.transform.translation = [float(self.sp_tx.value()), float(self.sp_ty.value()), float(self.sp_tz.value())]
        mesh.transform.rotation_euler_deg = [
            float(self.sp_rx.value()),
            float(self.sp_ry.value()),
            float(self.sp_rz.value()),
        ]
        mesh.transform.scale = [float(self.sp_sx.value()), float(self.sp_sy.value()), float(self.sp_sz.value())]
        actor = self._mesh_actor_by_id.get(mesh.mesh_id)
        if actor is not None:
            self._apply_actor_transform(mesh, actor)
        if hasattr(self, "plotter"):
            self.plotter.render()
        self.project.touch()
        self._is_dirty = True
        self._update_window_title()

    def _load_boundary_defaults_to_ui(self) -> None:
        md = self.project.source_data.metadata
        bc = md.get("boundary_defaults", {})
        fixed = str(bc.get("fixed", "FIXED_EDGE"))
        load = str(bc.get("load", "PRESSURE_ZONE"))
        contact = str(bc.get("contact", "CONTACT_ZONE"))
        self.ed_bc_fixed.setText(fixed)
        self.ed_bc_load.setText(load)
        self.ed_bc_contact.setText(contact)
        self._refresh_fixed_edge_options()

    def _apply_boundary_defaults_to_project(self) -> None:
        fixed = self.ed_bc_fixed.text().strip() or "FIXED_EDGE"
        load = self.ed_bc_load.text().strip() or "PRESSURE_ZONE"
        contact = self.ed_bc_contact.text().strip() or "CONTACT_ZONE"
        self.project.source_data.metadata["boundary_defaults"] = {
            "fixed": fixed,
            "load": load,
            "contact": contact,
        }
        self._refresh_fixed_edge_options()

    def _refresh_fixed_edge_options(self) -> None:
        current = self.cb_fixed_edge.currentText()
        fixed_group = self.ed_bc_fixed.text().strip() or "FIXED_EDGE"
        options = ["none", fixed_group, "FIXED_ALL"]
        self.cb_fixed_edge.blockSignals(True)
        self.cb_fixed_edge.clear()
        self.cb_fixed_edge.addItems(options)
        if current in options:
            self.cb_fixed_edge.setCurrentText(current)
        else:
            self.cb_fixed_edge.setCurrentText("none")
        self.cb_fixed_edge.blockSignals(False)

    def _action_save_project(self) -> None:
        self._save_project_internal(force_save_as_when_needed=True)

    def _action_save_project_as(self) -> None:
        prev_path = self.project_path
        self.project_path = None
        ok = self._save_project_internal(force_save_as_when_needed=True)
        if not ok:
            self.project_path = prev_path
            self._update_window_title()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._sim_process is not None and self._sim_process.state() != QProcess.NotRunning:
            self._sim_process.kill()
        if self._confirm_save_if_dirty():
            event.accept()
        else:
            event.ignore()

    def _wire_dirty_signals(self) -> None:
        # Identity
        self.ed_name.textEdited.connect(lambda _text: self._mark_dirty())
        self.cb_role.currentIndexChanged.connect(lambda _idx: self._mark_dirty())
        self.cb_visible.stateChanged.connect(lambda _state: self._mark_dirty())

        # Material tab
        self.cb_material.currentIndexChanged.connect(lambda _idx: self._mark_dirty())
        self.sp_density.valueChanged.connect(lambda _v: self._mark_dirty())
        self.sp_E.valueChanged.connect(lambda _v: self._mark_dirty())
        self.sp_poisson.valueChanged.connect(lambda _v: self._mark_dirty())

        # Membrane tab
        self.sp_thickness_mm.valueChanged.connect(lambda _v: self._mark_dirty())
        self.sp_pretension.valueChanged.connect(lambda _v: self._mark_dirty())
        self.cb_fixed_edge.currentIndexChanged.connect(lambda _idx: self._mark_dirty())

        # Transform tab
        self.sp_tx.valueChanged.connect(self._apply_transform_live)
        self.sp_ty.valueChanged.connect(self._apply_transform_live)
        self.sp_tz.valueChanged.connect(self._apply_transform_live)
        self.sp_rx.valueChanged.connect(self._apply_transform_live)
        self.sp_ry.valueChanged.connect(self._apply_transform_live)
        self.sp_rz.valueChanged.connect(self._apply_transform_live)
        self.sp_sx.valueChanged.connect(self._apply_transform_live)
        self.sp_sy.valueChanged.connect(self._apply_transform_live)
        self.sp_sz.valueChanged.connect(self._apply_transform_live)

        # Boundary tab
        self.ed_bc_fixed.textEdited.connect(lambda _text: self._mark_dirty())
        self.ed_bc_load.textEdited.connect(lambda _text: self._mark_dirty())
        self.ed_bc_contact.textEdited.connect(lambda _text: self._mark_dirty())
        self.ed_notes.textChanged.connect(self._mark_dirty)

        # Simulation tab
        self.sp_dt.valueChanged.connect(lambda _v: self._mark_dirty())
        self.sp_duration.valueChanged.connect(lambda _v: self._mark_dirty())
        self.sp_air_coupling.valueChanged.connect(lambda _v: self._mark_dirty())
        self.sp_air_grid_step_mm.valueChanged.connect(lambda _v: self._mark_dirty())
        self.cb_force_shape.currentIndexChanged.connect(lambda _idx: self._mark_dirty())
        self.sp_force_amp.valueChanged.connect(lambda _v: self._mark_dirty())
        self.sp_force_freq.valueChanged.connect(lambda _v: self._mark_dirty())


def main() -> int:
    app = QApplication(sys.argv)
    window = FeUiDraftMainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
