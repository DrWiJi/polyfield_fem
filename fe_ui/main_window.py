# -*- coding: utf-8 -*-
"""
Main window: orchestrates panels, project model, viewport, simulation.
Depends: fe_ui panels, project_model, optional trimesh/pyvista.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QWidget,
    QVBoxLayout,
)

from project_model import MeshEntity, Project

from .mesh_editor_panel import MeshEditorPanel
from .mesh_list_panel import MeshListPanel
from .simulation_panel import SimulationPanel
from .viewport import create_viewport, has_pyvista

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import pyvista as pv
except Exception:
    pv = None


class FeMainWindow(QMainWindow):
    """Main window orchestrating all panels and project."""

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
        self._plotter = None

        # Debug (for Test Run / Test Visualization)
        self._debug_thread = None
        self._debug_history_disp_all = None
        self._debug_log_text = ""
        self._debug_anim_timer = None
        self._debug_anim_frame_idx = 0

        self._build_ui()
        self._connect_signals()
        self._load_project_to_ui()
        self._refresh_mesh_list()

    def _build_ui(self) -> None:
        self._build_menu()
        self._build_central()
        self._build_docks()

    def _build_menu(self) -> None:
        menu_file = self.menuBar().addMenu("File")
        for label, slot in [
            ("New Project", self._action_new_project),
            ("Import Mesh...", self._action_import_mesh),
        ]:
            act = QAction(label, self)
            act.triggered.connect(slot)
            menu_file.addAction(act)
        menu_file.addSeparator()
        for label, slot in [
            ("Load...", self._action_load_project),
            ("Save", self._action_save_project),
            ("Save As...", self._action_save_project_as),
        ]:
            act = QAction(label, self)
            act.triggered.connect(slot)
            menu_file.addAction(act)
        menu_file.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
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
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        viewport_widget, self._plotter = create_viewport(self, self._mock_pick_from_viewport)
        self.viewport_widget = viewport_widget
        splitter.addWidget(viewport_widget)
        splitter.setStretchFactor(0, 1)
        layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _build_docks(self) -> None:
        self.mesh_list = MeshListPanel(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.mesh_list)

        self.mesh_editor = MeshEditorPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.mesh_editor)

        self.simulation = SimulationPanel(self)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.simulation)

    def _connect_signals(self) -> None:
        self.debug_test_run_finished.connect(self._finish_debug_test_run)

        self.mesh_list.selection_changed.connect(self._on_mesh_selected)
        self.mesh_list.search_changed.connect(lambda _: self._refresh_mesh_list())
        self.mesh_list.add_clicked.connect(self._action_add_mesh)
        self.mesh_list.remove_clicked.connect(self._action_remove_selected_mesh)

        self.mesh_editor.apply_clicked.connect(self._apply_mesh_editor_to_model)
        self.mesh_editor.reset_clicked.connect(self._reload_mesh_from_model)
        self.mesh_editor.cb_role.currentTextChanged.connect(self._update_membrane_visibility)
        self.mesh_editor.connect_dirty(lambda: self._mark_dirty())
        self.mesh_editor.connect_transform_live(self._apply_transform_live)

        self.simulation.run_clicked.connect(self._action_run_simulation)
        self.simulation.stop_clicked.connect(self._action_stop_simulation)
        self.simulation.connect_dirty(lambda: self._mark_dirty())

    def _refresh_mesh_list(self) -> None:
        query = self.mesh_list.get_search_filter()
        items = []
        for i, mesh in enumerate(self.project.source_data.meshes):
            text = f"{mesh.name}  [{mesh.role}]  <{mesh.material_key}>"
            if query and query not in text.lower():
                continue
            items.append((text, i))
        self.mesh_list.set_meshes(items)
        if self.mesh_list.count() > 0 and self.mesh_list.get_selected_index() is None:
            self.mesh_list.set_selection_by_row(0)

    def _selected_mesh_index(self) -> int | None:
        return self.mesh_list.get_selected_index()

    def _on_mesh_selected(self, idx: int | None) -> None:
        if idx is None:
            self.mesh_editor.set_info("No mesh selected")
            self.mesh_editor.set_enabled(False)
            self.mesh_editor.set_membrane_tab_visible(False)
            self._update_viewport_selection()
            return
        mesh = self.project.source_data.meshes[idx]
        self._is_loading_ui = True
        try:
            self.mesh_editor.set_info(f"Selected: {mesh.name}")
            self.mesh_editor.set_enabled(True)
            self.mesh_editor.set_data(self._mesh_to_editor_dict(mesh))
            self.mesh_editor.set_membrane_tab_visible(mesh.role == "membrane")
        finally:
            self._is_loading_ui = False
        self._update_viewport_selection()

    def _mesh_to_editor_dict(self, mesh: MeshEntity) -> dict:
        tr = list(mesh.transform.translation) if mesh.transform else [0, 0, 0]
        rot = list(mesh.transform.rotation_euler_deg) if mesh.transform else [0, 0, 0]
        scl = list(mesh.transform.scale) if mesh.transform else [1, 1, 1]
        return {
            "name": mesh.name,
            "role": mesh.role,
            "material_key": mesh.material_key,
            "visible": mesh.visible,
            "density": mesh.properties.get("density", 1380.0),
            "young_modulus": mesh.properties.get("young_modulus", 5.0e9),
            "poisson": mesh.properties.get("poisson", 0.30),
            "thickness_mm": mesh.properties.get("thickness_mm", 0.012),
            "pre_tension_n_per_m": mesh.properties.get("pre_tension_n_per_m", 10.0),
            "translation": (tr + [0, 0, 0])[:3],
            "rotation_euler_deg": (rot + [0, 0, 0])[:3],
            "scale": (scl + [1, 1, 1])[:3],
            "boundary_groups": mesh.boundary_groups or [],
            "notes": mesh.properties.get("notes", ""),
        }

    def _apply_mesh_editor_to_model(self) -> None:
        idx = self._selected_mesh_index()
        if idx is None:
            return
        mesh = self.project.source_data.meshes[idx]
        data = self.mesh_editor.get_data()
        mesh.name = data["name"] or mesh.name
        mesh.role = data["role"]
        mesh.material_key = data["material_key"]
        mesh.visible = data["visible"]
        mesh.properties["density"] = data["density"]
        mesh.properties["young_modulus"] = data["young_modulus"]
        mesh.properties["poisson"] = data["poisson"]
        mesh.properties["thickness_mm"] = data["thickness_mm"]
        mesh.properties["pre_tension_n_per_m"] = data["pre_tension_n_per_m"]
        mesh.transform.translation = data["translation"]
        mesh.transform.rotation_euler_deg = data["rotation_euler_deg"]
        mesh.transform.scale = data["scale"]
        if data.get("notes"):
            mesh.properties["notes"] = data["notes"]
        elif "notes" in mesh.properties:
            mesh.properties.pop("notes")
        mesh.boundary_groups = data["boundary_groups"] or []
        self.project.touch()
        self._is_dirty = True
        self._refresh_mesh_list()
        self._update_viewport_selection()
        self._update_window_title()

    def _reload_mesh_from_model(self) -> None:
        idx = self._selected_mesh_index()
        if idx is not None:
            self._on_mesh_selected(idx)

    def _update_membrane_visibility(self, role: str) -> None:
        self.mesh_editor.set_membrane_tab_visible(role == "membrane")

    def _apply_transform_live(self, _=None) -> None:
        if self._is_loading_ui:
            return
        idx = self._selected_mesh_index()
        if idx is None:
            return
        mesh = self.project.source_data.meshes[idx]
        mesh.transform.translation = [
            float(self.mesh_editor.sp_tx.value()),
            float(self.mesh_editor.sp_ty.value()),
            float(self.mesh_editor.sp_tz.value()),
        ]
        mesh.transform.rotation_euler_deg = [
            float(self.mesh_editor.sp_rx.value()),
            float(self.mesh_editor.sp_ry.value()),
            float(self.mesh_editor.sp_rz.value()),
        ]
        mesh.transform.scale = [
            float(self.mesh_editor.sp_sx.value()),
            float(self.mesh_editor.sp_sy.value()),
            float(self.mesh_editor.sp_sz.value()),
        ]
        actor = self._mesh_actor_by_id.get(mesh.mesh_id)
        if actor and hasattr(actor, "SetPosition"):
            tr, rot, scl = mesh.transform.translation, mesh.transform.rotation_euler_deg, mesh.transform.scale
            actor.SetPosition(*tr)
            actor.SetOrientation(*rot)
            actor.SetScale(*scl)
        if self._plotter:
            self._plotter.render()
        self.project.touch()
        self._is_dirty = True
        self._update_window_title()

    def _mock_pick_from_viewport(self) -> None:
        if self._plotter is not None:
            return
        if self.mesh_list.count() == 0:
            return
        self._viewport_pick_index = (self._viewport_pick_index + 1) % self.mesh_list.count()
        self.mesh_list.set_selection_by_row(self._viewport_pick_index)

    def _update_viewport_selection(self) -> None:
        if not self._plotter or not pv:
            return
        idx = self._selected_mesh_index()
        selected_id = None
        if idx is not None and 0 <= idx < len(self.project.source_data.meshes):
            selected_id = self.project.source_data.meshes[idx].mesh_id
        mesh_by_id = {m.mesh_id: m for m in self.project.source_data.meshes}
        for mesh_id, actor in self._mesh_actor_by_id.items():
            mesh = mesh_by_id.get(mesh_id)
            if mesh:
                self._apply_actor_transform(mesh, actor)
            visible = mesh.visible if mesh else True
            if hasattr(actor, "SetVisibility"):
                actor.SetVisibility(1 if visible else 0)
            if not visible:
                continue
            color = "#F0D070" if mesh_id == selected_id else "#9A9A9A"
            prop = actor.GetProperty() if hasattr(actor, "GetProperty") else None
            if prop:
                prop.SetColor(*pv.Color(color).float_rgb)
        self._plotter.render()

    def _apply_actor_transform(self, mesh: MeshEntity, actor: object) -> None:
        tr = list(mesh.transform.translation) if mesh.transform else [0, 0, 0]
        rot = list(mesh.transform.rotation_euler_deg) if mesh.transform else [0, 0, 0]
        scl = list(mesh.transform.scale) if mesh.transform else [1, 1, 1]
        tr, rot, scl = (tr + [0, 0, 0])[:3], (rot + [0, 0, 0])[:3], (scl + [1, 1, 1])[:3]
        if hasattr(actor, "SetPosition"):
            actor.SetPosition(*tr)
        if hasattr(actor, "SetOrientation"):
            actor.SetOrientation(*rot)
        if hasattr(actor, "SetScale"):
            actor.SetScale(*scl)

    def _action_add_mesh(self) -> None:
        n = len(self.project.source_data.meshes) + 1
        self.project.add_mesh(name=f"Mesh_{n}", role="solid", material_key="membrane")
        mesh = self.project.source_data.meshes[-1]
        mesh.properties.update(density=1380.0, young_modulus=5.0e9, poisson=0.30)
        self._is_dirty = True
        self._refresh_mesh_list()
        self.mesh_list.set_selection_by_row(self.mesh_list.count() - 1)
        self._update_window_title()

    def _action_remove_selected_mesh(self) -> None:
        idx = self._selected_mesh_index()
        if idx is None:
            return
        mesh = self.project.source_data.meshes.pop(idx)
        self._remove_mesh_actor(mesh.mesh_id)
        self.project.touch()
        self._is_dirty = True
        self._refresh_mesh_list()
        if self.mesh_list.count() == 0:
            self.mesh_editor.set_info("No mesh selected")
            self.mesh_editor.set_enabled(False)
        self._update_window_title()

    def _remove_mesh_actor(self, mesh_id: str) -> None:
        actor = self._mesh_actor_by_id.pop(mesh_id, None)
        self._mesh_polydata_by_id.pop(mesh_id, None)
        if actor and self._plotter:
            self._plotter.remove_actor(actor, reset_camera=False)
            self._plotter.render()

    def _action_import_mesh(self) -> None:
        if trimesh is None:
            QMessageBox.critical(self, "Import Error", "trimesh is not available. pip install trimesh")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Mesh", "",
            "Mesh files (*.stl *.obj *.ply *.off *.glb *.gltf);;All Files (*.*)",
        )
        if not paths:
            return
        for fp in paths:
            self._import_mesh_file(Path(fp))
        self._is_dirty = True
        self._refresh_mesh_list()
        self._update_window_title()

    def _import_mesh_file(self, src: Path) -> None:
        try:
            loaded = trimesh.load(str(src), force="scene")
        except Exception as e:
            QMessageBox.warning(self, "Import Warning", f"Failed to import {src.name}:\n{e}")
            return
        geoms = []
        if hasattr(loaded, "geometry") and isinstance(loaded.geometry, dict):
            geoms = list(loaded.geometry.items())
        elif hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
            geoms = [(src.stem, loaded)]
        for gname, geom in geoms:
            if not hasattr(geom, "vertices") or not hasattr(geom, "faces") or len(geom.vertices) == 0:
                continue
            name = src.stem if len(geoms) == 1 else f"{src.stem}:{gname}"
            mesh = self.project.add_mesh(name=name, role="solid", material_key="membrane")
            mesh.source_path = str(src)
            mesh.properties["trimesh_geom_name"] = str(gname)
            mesh.properties["vertex_count"] = len(geom.vertices)
            mesh.properties["face_count"] = len(geom.faces)
            self._add_mesh_to_viewport(mesh, geom)

    def _trimesh_to_polydata(self, tri_mesh):
        if not pv:
            return None
        import numpy as np
        verts = np.asarray(tri_mesh.vertices, dtype=np.float64)
        faces = np.asarray(tri_mesh.faces, dtype=np.int64)
        cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
        return pv.PolyData(verts, cells)

    def _add_mesh_to_viewport(self, mesh: MeshEntity, tri_mesh) -> None:
        if not self._plotter or not pv:
            return
        poly = self._trimesh_to_polydata(tri_mesh)
        if poly is None:
            return
        actor = self._plotter.add_mesh(
            poly, color="#9A9A9A", smooth_shading=True, pickable=True,
            name=f"mesh_{mesh.mesh_id}", show_edges=False, reset_camera=False,
        )
        self._mesh_polydata_by_id[mesh.mesh_id] = poly
        self._mesh_actor_by_id[mesh.mesh_id] = actor
        if hasattr(actor, "SetVisibility"):
            actor.SetVisibility(1 if mesh.visible else 0)
        self._apply_actor_transform(mesh, actor)
        self._plotter.reset_camera()
        self._update_viewport_selection()
        self._plotter.render()

    def _load_trimesh_for_entity(self, mesh: MeshEntity):
        if not trimesh or not mesh.source_path:
            return None
        src = Path(mesh.source_path)
        if not src.exists():
            return None
        try:
            loaded = trimesh.load(str(src), force="scene")
        except Exception:
            return None
        if hasattr(loaded, "geometry") and isinstance(loaded.geometry, dict):
            target = str(mesh.properties.get("trimesh_geom_name", ""))
            if target and target in loaded.geometry:
                return loaded.geometry[target]
            if loaded.geometry:
                return next(iter(loaded.geometry.values()))
        if hasattr(loaded, "vertices") and hasattr(loaded, "faces"):
            return loaded
        return None

    def _rebuild_viewport_from_project(self) -> None:
        if not self._plotter or not pv:
            return
        self._plotter.clear()
        self._plotter.add_axes()
        self._plotter.show_grid(color="#555555")
        from .viewport import _setup_lighting
        try:
            import pyvista as _pv
            self._plotter.remove_all_lights()
            self._plotter.add_light(_pv.Light(position=(2, 2.5, 3), focal_point=(0, 0, 0), color="white", intensity=1))
            self._plotter.add_light(_pv.Light(position=(-2.5, 1, 1.5), focal_point=(0, 0, 0), color="#cfd8ff", intensity=0.45))
            self._plotter.add_light(_pv.Light(position=(0, -3, 2), focal_point=(0, 0, 0), color="#ffe7c9", intensity=0.3))
        except Exception:
            pass
        self._mesh_actor_by_id.clear()
        self._mesh_polydata_by_id.clear()
        for mesh in self.project.source_data.meshes:
            tri = self._load_trimesh_for_entity(mesh)
            if tri:
                self._add_mesh_to_viewport(mesh, tri)
        self._update_viewport_selection()

    def _load_project_to_ui(self) -> None:
        self._is_loading_ui = True
        try:
            sim = self.project.source_data.simulation_settings
            self.simulation.set_settings({
                "dt": sim.dt,
                "duration": sim.duration,
                "air_coupling_gain": sim.air_coupling_gain,
                "air_grid_step_mm": sim.air_grid_step_mm,
                "force_shape": sim.force_shape,
                "force_amplitude_pa": sim.force_amplitude_pa,
                "force_freq_hz": sim.force_freq_hz,
            })
            md = self.project.source_data.metadata
            bc = md.get("boundary_defaults", {})
            self.mesh_editor.ed_bc_fixed.setText(str(bc.get("fixed", "FIXED_EDGE")))
            self.mesh_editor.ed_bc_load.setText(str(bc.get("load", "PRESSURE_ZONE")))
            self.mesh_editor.ed_bc_contact.setText(str(bc.get("contact", "CONTACT_ZONE")))
            fixed = self.mesh_editor.ed_bc_fixed.text().strip() or "FIXED_EDGE"
            self.mesh_editor.set_fixed_edge_options(["none", fixed, "FIXED_ALL"])
            self._update_window_title()
            self._refresh_mesh_list()
            if self.project.source_data.meshes:
                self.mesh_list.set_selection_by_row(0)
                self._on_mesh_selected(0)
            self._rebuild_viewport_from_project()
        finally:
            self._is_loading_ui = False

    def _apply_simulation_to_model(self) -> None:
        data = self.simulation.get_settings()
        sim = self.project.source_data.simulation_settings
        sim.dt = data["dt"]
        sim.duration = data["duration"]
        sim.air_coupling_gain = data["air_coupling_gain"]
        sim.air_grid_step_mm = data["air_grid_step_mm"]
        sim.force_shape = data["force_shape"]
        sim.force_amplitude_pa = data["force_amplitude_pa"]
        sim.force_freq_hz = data["force_freq_hz"]
        self.project.source_data.metadata["boundary_defaults"] = {
            "fixed": self.mesh_editor.ed_bc_fixed.text().strip() or "FIXED_EDGE",
            "load": self.mesh_editor.ed_bc_load.text().strip() or "PRESSURE_ZONE",
            "contact": self.mesh_editor.ed_bc_contact.text().strip() or "CONTACT_ZONE",
        }
        fixed = self.mesh_editor.ed_bc_fixed.text().strip() or "FIXED_EDGE"
        self.mesh_editor.set_fixed_edge_options(["none", fixed, "FIXED_ALL"])
        self.project.touch()
        self._is_dirty = True

    def _action_run_simulation(self) -> None:
        if self._sim_process and self._sim_process.state() != QProcess.NotRunning:
            self.simulation.append_console("[UI] Simulation already running.\n")
            return
        self._apply_simulation_to_model()
        script = Path(__file__).resolve().parent.parent / "diaphragm_opencl.py"
        if not script.exists():
            self.simulation.append_console(f"[UI] Script not found: {script}\n")
            return
        args = [
            str(script), "--no-plot",
            "--dt", str(self.simulation.sp_dt.value()),
            "--duration", str(self.simulation.sp_duration.value()),
            "--force-shape", self.simulation.cb_force_shape.currentText(),
            "--force-amplitude", str(self.simulation.sp_force_amp.value()),
            "--force-freq", str(self.simulation.sp_force_freq.value()),
        ]
        proc = QProcess(self)
        proc.setProgram(sys.executable)
        proc.setArguments(args)
        proc.setWorkingDirectory(str(script.parent))
        proc.setProcessChannelMode(QProcess.SeparateChannels)
        proc.readyReadStandardOutput.connect(lambda: self._on_sim_stdout(proc))
        proc.readyReadStandardError.connect(lambda: self._on_sim_stderr(proc))
        proc.finished.connect(self._on_sim_finished)
        proc.errorOccurred.connect(lambda e: self.simulation.append_console(f"[UI] Process error: {int(e)}\n"))
        self._sim_process = proc
        self.simulation.set_running(True)
        self.simulation.append_console(f"[UI] Starting: {sys.executable} {' '.join(args)}\n")
        proc.start()

    def _on_sim_stdout(self, proc: QProcess) -> None:
        if proc:
            data = bytes(proc.readAllStandardOutput())
            if data:
                self.simulation.append_console(data.decode("utf-8", errors="replace"))

    def _on_sim_stderr(self, proc: QProcess) -> None:
        if proc:
            data = bytes(proc.readAllStandardError())
            if data:
                self.simulation.append_console(data.decode("utf-8", errors="replace"))

    def _on_sim_finished(self, code: int, status: QProcess.ExitStatus) -> None:
        s = "normal" if status == QProcess.NormalExit else "crash"
        self.simulation.append_console(f"[UI] Simulation finished. exit_code={code}, status={s}\n")
        self.simulation.set_running(False)
        self._sim_process = None

    def _action_stop_simulation(self) -> None:
        if self._sim_process and self._sim_process.state() != QProcess.NotRunning:
            self._sim_process.kill()
        else:
            self.simulation.set_running(False)

    def _action_debug_test_run(self) -> None:
        import threading
        import io
        import contextlib
        import traceback
        import numpy as np

        if self._debug_thread and self._debug_thread.is_alive():
            QMessageBox.information(self, "Debug Test Run", "Debug simulation is already running.")
            return
        self.simulation.append_console("[UI] Starting debug test run in background...\n")

        def worker():
            try:
                import diaphragm_opencl as cl_model
                log_buf = io.StringIO()
                history = None
                try:
                    with contextlib.redirect_stdout(log_buf), contextlib.redirect_stderr(log_buf):
                        argv = ["diaphragm_opencl.py", "--no-plot", "--dt", "1e-7", "--duration", "0.001",
                                "--force-shape", "impulse", "--force-amplitude", "0.001", "--force-freq", "200",
                                "--force-freq-end", "5000", "--air-inject-mode", "reduce", "--debug"]
                        args = cl_model._parse_cli_args(argv)
                        model, _ = cl_model.run_cli_simulation(args)
                        history = list(model.history_disp_all) if getattr(model, "history_disp_all", None) else []
                except Exception:
                    log_buf.write("\n[Debug] Exception:\n" + traceback.format_exc())
                self._debug_history_disp_all = history or []
                self._debug_log_text = log_buf.getvalue()
                self.debug_test_run_finished.emit()
            finally:
                self._debug_thread = None

        self._debug_thread = threading.Thread(target=worker, daemon=True)
        self._debug_thread.start()

    def _finish_debug_test_run(self) -> None:
        if self._debug_log_text:
            self.simulation.append_console("\n[Debug Test Run]\n" + self._debug_log_text)
        n = len(self._debug_history_disp_all or [])
        self.simulation.append_console(f"\n[UI] Debug test run finished. Frames: {n}\n")

    def _action_debug_test_visualization(self) -> None:
        if not has_pyvista() or not self._plotter:
            QMessageBox.information(self, "Test Visualization", "PyVista viewport not available.")
            return
        frames = self._debug_history_disp_all or []
        if not frames:
            QMessageBox.warning(self, "Test Visualization", "No debug history. Run Debug → Test Run first.")
            return
        import numpy as np
        first = np.asarray(frames[0], dtype=np.float64)
        if first.ndim != 2:
            QMessageBox.warning(self, "Test Visualization", "history_disp_all is not 2D.")
            return
        ny, nx = first.shape
        xs, ys = np.meshgrid(np.linspace(-0.5, 0.5, nx), np.linspace(-0.5, 0.5, ny), indexing="ij")
        scale_z = 1e10
        z0 = (first.T * scale_z).astype(np.float64)
        grid = pv.StructuredGrid(xs, ys, z0)
        grid["uz"] = first.T.ravel(order="F")
        self._debug_surface = grid
        if self._debug_anim_timer:
            self._debug_anim_timer.stop()
            self._debug_anim_timer.deleteLater()
        try:
            old = self._mesh_actor_by_id.pop("__debug_surface__", None)
            if old:
                self._plotter.remove_actor(old, reset_camera=False)
        except Exception:
            pass
        actor = self._plotter.add_mesh(grid, name="debug_surface", scalars="uz", cmap="RdBu", show_edges=False)
        self._mesh_actor_by_id["__debug_surface__"] = actor
        self._plotter.reset_camera()
        self._plotter.render()
        self._debug_anim_frame_idx = 0

        def update_frame():
            if not frames:
                return
            idx = self._debug_anim_frame_idx % len(frames)
            frame = np.asarray(frames[idx], dtype=np.float64)
            if frame.shape != (ny, nx):
                return
            z = (frame.T * scale_z).astype(np.float64)
            pts = self._debug_surface.points
            pts[:, 2] = z.ravel(order="F")
            self._debug_surface.points = pts
            self._debug_surface["uz"] = frame.T.ravel(order="F")
            self._debug_anim_frame_idx = (self._debug_anim_frame_idx + 1) % len(frames)
            self._plotter.render()

        self._debug_anim_timer = QTimer(self)
        self._debug_anim_timer.timeout.connect(update_frame)
        self._debug_anim_timer.start(50)

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
        fp, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project JSON (*.json);;All (*.*)")
        if not fp:
            return
        try:
            self.project = Project.load_json(fp)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return
        self.project_path = Path(fp)
        self._is_dirty = False
        self._load_project_to_ui()
        self._refresh_mesh_list()

    def _action_save_project(self) -> None:
        self._save_internal(force_save_as=True)

    def _action_save_project_as(self) -> None:
        prev = self.project_path
        self.project_path = None
        if not self._save_internal(force_save_as=True):
            self.project_path = prev
        self._update_window_title()

    def _save_internal(self, force_save_as: bool) -> bool:
        self._apply_simulation_to_model()
        if self._selected_mesh_index() is not None:
            self._apply_mesh_editor_to_model()
        if self.project_path is None:
            if not force_save_as:
                return False
            fp, _ = QFileDialog.getSaveFileName(self, "Save As", "project.json", "Project JSON (*.json);;All (*.*)")
            if not fp:
                return False
            if not fp.lower().endswith(".json"):
                fp += ".json"
            self.project_path = Path(fp)
        try:
            self.project.save_json(self.project_path)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return False
        self._is_dirty = False
        self._update_window_title()
        return True

    def _confirm_save_if_dirty(self) -> bool:
        if not self._is_dirty:
            return True
        r = QMessageBox.question(
            self, "Unsaved Changes",
            "Project has unsaved changes. Save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if r == QMessageBox.Cancel:
            return False
        if r == QMessageBox.Discard:
            return True
        return self._save_internal(force_save_as=True)

    def _mark_dirty(self) -> None:
        if self._is_loading_ui:
            return
        if not self._is_dirty:
            self._is_dirty = True
            self._update_window_title()

    def _update_window_title(self) -> None:
        path = str(self.project_path) if self.project_path else "unsaved"
        mark = "*" if self._is_dirty else ""
        self.setWindowTitle(f"FE UI{mark} - {self.project.name} ({path})")

    def closeEvent(self, event) -> None:
        if self._sim_process and self._sim_process.state() != QProcess.NotRunning:
            self._sim_process.kill()
        if self._confirm_save_if_dirty():
            event.accept()
        else:
            event.ignore()
