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
from pyvistaqt import QtInteractor

from project_model import MeshEntity

from .app_controller import AppController
from .app_model import AppModel
from .material_library_window import MaterialLibraryWindow
from .mesh_editor_panel import MeshEditorPanel
from .mesh_list_panel import MeshListPanel
from .simulation_panel import SimulationPanel
from .boundary_conditions_panel import BoundaryConditionsPanel
from .topology_generator_panel import TopologyGeneratorPanel
from .viewport import MainViewport, create_viewport, has_pyvista

try:
    from pyvista.plotting import _vtk as _pv_vtk
except Exception:
    _pv_vtk = None

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import pyvista as pv
except Exception:
    pv = None


class FeMainWindow(QMainWindow):
    """Main window orchestrating all panels and project."""

    debug_test_run_finished = Signal()
    mesh_viewport_changed = Signal()  # Emitted when mesh geometry/list changes (add/remove/rebuild)
    bc_changed = Signal()  # Emitted when boundary conditions change

    def __init__(self, app_model: AppModel, app_controller: AppController | None = None) -> None:
        super().__init__()
        self._app = app_model
        self._app_controller = app_controller
        self.resize(1400, 900)
        self._is_loading_ui = False
        self._mesh_actor_by_id: dict[str, object] = {}
        self._mesh_polydata_by_id: dict[str, object] = {}
        self._sim_process: QProcess | None = None
        self._plotter: QtInteractor | None = None

        # Debug (for Test Run / Test Visualization)
        self._debug_thread = None
        self._debug_history_disp_all = None
        self._debug_log_text = ""
        self._debug_anim_timer = None
        self._debug_anim_frame_idx = 0

        self._material_library_window = None
        self._boundary_conditions_window = None
        self._topology_generator_window = None
        self._affine_widget = None
        self._affine_widget_mesh_id: str | None = None
        self._mesh_pick_deselect_observer = None

        self._build_ui()
        self._connect_signals()
        self._app.project_changed.connect(self._on_project_changed)
        self._app.viewport_closed.connect(self._on_viewport_closed)
        lib_signal = self._app_controller.material_library_changed if self._app_controller else self._app.material_library_changed
        lib_signal.connect(self._refresh_material_options)
        self._app.state_changed.connect(self._update_window_title)
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
        if self._app_controller:
            menu_file.addSeparator()
            act_new_win = QAction("New Window", self)
            act_new_win.triggered.connect(self._action_new_window)
            menu_file.addAction(act_new_win)
            act_open_win = QAction("Open in New Window...", self)
            act_open_win.triggered.connect(self._action_open_in_new_window)
            menu_file.addAction(act_open_win)
        menu_file.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        menu_window = self.menuBar().addMenu("Window")
        act_material_lib = QAction("Material Library", self)
        act_material_lib.triggered.connect(self._action_open_material_library)
        menu_window.addAction(act_material_lib)
        menu_window.addSeparator()
        self.act_mesh_list = QAction("Mesh List", self)
        self.act_mesh_list.setCheckable(True)
        self.act_mesh_list.setChecked(True)
        self.act_mesh_list.triggered.connect(self._window_toggle_mesh_list)
        menu_window.addAction(self.act_mesh_list)

        self.act_mesh_editor = QAction("Mesh Parameter Editor", self)
        self.act_mesh_editor.setCheckable(True)
        self.act_mesh_editor.setChecked(True)
        self.act_mesh_editor.triggered.connect(self._window_toggle_mesh_editor)
        menu_window.addAction(self.act_mesh_editor)

        self.act_simulation = QAction("Simulation", self)
        self.act_simulation.setCheckable(True)
        self.act_simulation.setChecked(True)
        self.act_simulation.triggered.connect(self._window_toggle_simulation)
        menu_window.addAction(self.act_simulation)

        self.act_bc_panel = QAction("Boundary Conditions", self)
        self.act_bc_panel.setCheckable(True)
        self.act_bc_panel.setChecked(False)
        self.act_bc_panel.triggered.connect(self._window_toggle_boundary_conditions)
        menu_window.addAction(self.act_bc_panel)

        self.act_topology_generator = QAction("Topology Generator", self)
        self.act_topology_generator.setCheckable(True)
        self.act_topology_generator.setChecked(False)
        self.act_topology_generator.triggered.connect(self._window_toggle_topology_generator)
        menu_window.addAction(self.act_topology_generator)

        menu_window.addSeparator()
        act_float_mesh_list = QAction("Mesh List in Separate Window", self)
        act_float_mesh_list.triggered.connect(lambda: self._window_open_floating(self.mesh_list))
        menu_window.addAction(act_float_mesh_list)

        act_float_mesh_editor = QAction("Mesh Parameter Editor in Separate Window", self)
        act_float_mesh_editor.triggered.connect(lambda: self._window_open_floating(self.mesh_editor))
        menu_window.addAction(act_float_mesh_editor)

        act_float_simulation = QAction("Simulation in Separate Window", self)
        act_float_simulation.triggered.connect(lambda: self._window_open_floating(self.simulation))
        menu_window.addAction(act_float_simulation)

        act_float_bc = QAction("Boundary Conditions in Separate Window", self)
        act_float_bc.triggered.connect(self._action_open_boundary_conditions)
        menu_window.addAction(act_float_bc)

        act_float_topology = QAction("Topology Generator in Separate Window", self)
        act_float_topology.triggered.connect(self._action_open_topology_generator)
        menu_window.addAction(act_float_topology)

        menu_window.addSeparator()
        act_reset_layout = QAction("Reset Layout", self)
        act_reset_layout.triggered.connect(self._window_reset_layout)
        menu_window.addAction(act_reset_layout)

        menu_debug = self.menuBar().addMenu("Debug")
        act_test_run = QAction("Test Run", self)
        act_test_run.triggered.connect(self._action_debug_test_run)
        menu_debug.addAction(act_test_run)
        act_test_vis = QAction("Test Visualization", self)
        act_test_vis.triggered.connect(self._action_debug_test_visualization)
        menu_debug.addAction(act_test_vis)

    def _action_open_material_library(self) -> None:
        if self._material_library_window is None:
            self._material_library_window = MaterialLibraryWindow(self, self._app, self._app_controller)
            self._material_library_window.setWindowFlags(
                self._material_library_window.windowFlags() | Qt.Window
            )
        self._material_library_window.show()
        self._material_library_window.raise_()
        self._material_library_window.activateWindow()

    def _window_open_floating(self, dock: QWidget) -> None:
        """Show dock and open it in a separate floating window."""
        dock.setVisible(True)
        dock.setFloating(True)
        if dock == self.mesh_list:
            self.act_mesh_list.setChecked(True)
        elif dock == self.mesh_editor:
            self.act_mesh_editor.setChecked(True)
        elif dock == self.simulation:
            self.act_simulation.setChecked(True)

    def _action_open_boundary_conditions(self) -> None:
        """Open Boundary Conditions panel in a separate floating window."""
        if self._boundary_conditions_window is None:
            self._boundary_conditions_window = BoundaryConditionsPanel(self)
            self._boundary_conditions_window.setWindowFlags(
                self._boundary_conditions_window.windowFlags() | Qt.Window
            )
            # Connect signals
            self._boundary_conditions_window.bc_created.connect(self._on_bc_created)
            self._boundary_conditions_window.bc_deleted.connect(self._on_bc_deleted)
            self._boundary_conditions_window.bc_updated.connect(self._on_bc_updated)
            self._boundary_conditions_window.bc_selected.connect(self._on_bc_selected)
        self._boundary_conditions_window.show()
        self._boundary_conditions_window.raise_()
        self._boundary_conditions_window.activateWindow()
        self.act_bc_panel.setChecked(True)

    def _window_toggle_boundary_conditions(self) -> None:
        """Toggle Boundary Conditions panel visibility."""
        if self.act_bc_panel.isChecked():
            self._action_open_boundary_conditions()
        else:
            if self._boundary_conditions_window:
                self._boundary_conditions_window.close()
                self._boundary_conditions_window = None

    def _action_open_topology_generator(self) -> None:
        """Open Topology Generator panel in a separate floating window."""
        if self._topology_generator_window is None:
            self._topology_generator_window = TopologyGeneratorPanel(self)
            self._topology_generator_window.setWindowFlags(
                self._topology_generator_window.windowFlags() | Qt.Window
            )
        self._topology_generator_window.show()
        self._topology_generator_window.raise_()
        self._topology_generator_window.activateWindow()
        self.act_topology_generator.setChecked(True)

    def _window_toggle_topology_generator(self) -> None:
        """Toggle Topology Generator panel visibility."""
        if self.act_topology_generator.isChecked():
            self._action_open_topology_generator()
        else:
            if self._topology_generator_window:
                self._topology_generator_window.close()
                self._topology_generator_window = None

    def _on_topology_generator_closed(self) -> None:
        """Called when Topology Generator window is closed."""
        self._topology_generator_window = None
        self.act_topology_generator.setChecked(False)

    def _window_toggle_mesh_list(self) -> None:
        visible = self.act_mesh_list.isChecked()
        self.mesh_list.setVisible(visible)

    def _window_toggle_mesh_editor(self) -> None:
        visible = self.act_mesh_editor.isChecked()
        self.mesh_editor.setVisible(visible)

    def _window_toggle_simulation(self) -> None:
        visible = self.act_simulation.isChecked()
        self.simulation.setVisible(visible)

    def _on_project_changed(self) -> None:
        """Called when project is replaced (new/load)."""
        self._load_project_to_ui()
        self._refresh_mesh_list()

    def _on_mesh_actors_updated(self, mesh_actor_by_id: dict) -> None:
        """Called when UnifiedMeshViewport refreshes mesh display."""
        self._mesh_actor_by_id.clear()
        self._mesh_actor_by_id.update(mesh_actor_by_id)
        self._update_viewport_selection()

    def _refresh_material_options(self) -> None:
        """Refresh mesh editor material combo from MaterialLibraryModel."""
        names = [m.name for m in self._app.material_library.materials]
        self.mesh_editor.set_material_options(names)

    def _window_reset_layout(self) -> None:
        self.act_mesh_list.setChecked(True)
        self.act_mesh_editor.setChecked(True)
        self.act_simulation.setChecked(True)
        self.act_bc_panel.setChecked(False)
        self.act_topology_generator.setChecked(False)
        self.mesh_list.setVisible(True)
        self.mesh_editor.setVisible(True)
        self.simulation.setVisible(True)
        self.mesh_list.setFloating(False)
        self.mesh_editor.setFloating(False)
        self.simulation.setFloating(False)
        if self._boundary_conditions_window:
            self._boundary_conditions_window.close()
            self._boundary_conditions_window = None
        if self._topology_generator_window:
            self._topology_generator_window.close()
            self._topology_generator_window = None
        self.addDockWidget(Qt.LeftDockWidgetArea, self.mesh_list)
        self.addDockWidget(Qt.RightDockWidgetArea, self.mesh_editor)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.simulation)

    def _build_central(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        splitter = QSplitter(Qt.Horizontal)
        self._mesh_viewport = MainViewport(
            self,
            get_mesh_data=lambda: (self._mesh_polydata_by_id.copy(), self._app.project.source_data.meshes),
            refresh_signals=[
                self._app.project_changed,
                self.mesh_viewport_changed,
            ],
            pickable=True,
            mesh_color="#9A9A9A",
        )
        self._plotter = self._mesh_viewport.plotter
        self.viewport_widget = self._mesh_viewport
        self._mesh_viewport.mesh_actors_updated.connect(self._on_mesh_actors_updated)
        splitter.addWidget(self._mesh_viewport)
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

        self.mesh_list.selection_changed.connect(self._app.set_selection)
        self._app.selection_changed.connect(self._on_mesh_selected)
        self._app.transform_changed.connect(self._on_transform_changed)
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
        for i, mesh in enumerate(self._app.project.source_data.meshes):
            text = f"{mesh.name}  [{mesh.role}]  <{mesh.material_key}>"
            if query and query not in text.lower():
                continue
            items.append((text, i))
        self.mesh_list.set_meshes(items)
        if self.mesh_list.count() > 0 and self.mesh_list.get_selected_index() is None:
            self.mesh_list.set_selection_by_row(0)

    def _selected_mesh_index(self) -> int | None:
        return self._app.selected_mesh_index

    def _on_mesh_selected(self) -> None:
        """React to selection change — read index from app model."""
        idx = self._app.selected_mesh_index
        if idx is None:
            self.mesh_editor.set_info("No mesh selected")
            self.mesh_editor.set_enabled(False)
            self.mesh_editor.set_membrane_tab_visible(False)
            self._update_viewport_selection()
            return
        mesh = self._app.project.source_data.meshes[idx]
        self._is_loading_ui = True
        try:
            self.mesh_editor.set_info(f"Selected: {mesh.name}")
            self.mesh_editor.set_enabled(True)
            self.mesh_editor.set_data(self._mesh_to_editor_dict(mesh))
            self.mesh_editor.set_membrane_tab_visible(mesh.role == "membrane")
        finally:
            self._is_loading_ui = False
        self._update_viewport_selection()

    def _on_transform_changed(self) -> None:
        """Refresh mesh editor from model when transform changes (affine widget drag)."""
        idx = self._app.selected_mesh_index
        if idx is None:
            return
        if idx >= len(self._app.project.source_data.meshes):
            return
        mesh = self._app.project.source_data.meshes[idx]
        self._is_loading_ui = True
        try:
            self.mesh_editor.set_data(self._mesh_to_editor_dict(mesh))
        finally:
            self._is_loading_ui = False

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
        mesh = self._app.project.source_data.meshes[idx]
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
        self._app.touch()
        self._refresh_mesh_list()

    def _on_bc_created(self, bc_data: dict) -> None:
        """Handle new boundary condition creation."""
        from project_model import BoundaryCondition, MeshTransform
        bc = BoundaryCondition(
            bc_id=bc_data["bc_id"],
            name=bc_data["name"],
            bc_type=bc_data["bc_type"],
            transform=MeshTransform(
                translation=bc_data["translation"],
                rotation_euler_deg=bc_data["rotation_euler_deg"],
                scale=bc_data["scale"],
            ),
            mesh_ids=bc_data["mesh_ids"],
            flags=bc_data["flags"],
            parameters=bc_data["parameters"],
        )
        self._app.project.source_data.boundary_conditions.append(bc)
        self._app.touch()
        self._refresh_bc_list(select_bc_id=bc.bc_id)
        self.bc_changed.emit()

    def _on_bc_deleted(self, bc_id: str) -> None:
        """Handle boundary condition deletion."""
        bc_list = self._app.project.source_data.boundary_conditions
        self._app.project.source_data.boundary_conditions = [
            bc for bc in bc_list if bc.bc_id != bc_id
        ]
        self._app.touch()
        self._refresh_bc_list()
        self.bc_changed.emit()

    def _on_bc_updated(self, bc_id: str, bc_data: dict) -> None:
        """Handle boundary condition update."""
        bc_list = self._app.project.source_data.boundary_conditions
        for bc in bc_list:
            if bc.bc_id == bc_id:
                bc.name = bc_data["name"]
                bc.bc_type = bc_data["bc_type"]
                bc.transform.translation = bc_data["translation"]
                bc.transform.rotation_euler_deg = bc_data["rotation_euler_deg"]
                bc.transform.scale = bc_data["scale"]
                bc.mesh_ids = bc_data["mesh_ids"]
                bc.flags = bc_data["flags"]
                bc.parameters = bc_data["parameters"]
                break
        self._app.touch()
        self._refresh_bc_list(select_bc_id=bc_id)
        self.bc_changed.emit()

    def _on_bc_selected(self, bc_id: str) -> None:
        """Handle boundary condition selection."""
        pass

    def _close_boundary_conditions_window(self) -> None:
        """Close boundary conditions window."""
        if self._boundary_conditions_window:
            self._boundary_conditions_window.close()
            self._boundary_conditions_window = None
            self.act_bc_panel.setChecked(False)

    def _on_boundary_conditions_window_closed(self) -> None:
        """Called when BC window is closed (e.g. via X button). Clear reference and notify other viewports."""
        if self._boundary_conditions_window:
            self._boundary_conditions_window = None
            self.act_bc_panel.setChecked(False)
        self._app.notify_viewport_closed()

    def _on_viewport_closed(self) -> None:
        """React to any viewport window closing. Restore picking and affine widget."""
        QTimer.singleShot(0, self._restore_viewport_handlers)

    def _restore_viewport_handlers(self) -> None:
        """Re-setup picking and affine widget after another viewport closed (OpenGL context restored)."""
        if not self._plotter or not pv:
            return
        try:
            if hasattr(self._plotter, "disable_picking"):
                self._plotter.disable_picking()
            setattr(self._plotter, "_picker_in_use", False)
            if self._mesh_pick_deselect_observer is not None and hasattr(self._plotter, "iren"):
                try:
                    self._plotter.iren.remove_observer(self._mesh_pick_deselect_observer)
                except Exception:
                    pass
                self._mesh_pick_deselect_observer = None
            self._remove_affine_widget()
            self._affine_widget_mesh_id = None
            self._plotter.render()
            self._update_viewport_selection()
        except Exception:
            pass

    def _refresh_bc_list(self, select_bc_id: str | None = None) -> None:
        """Refresh boundary conditions list in all panels."""
        bc_list = self._app.project.source_data.boundary_conditions
        # Update BC window if it exists
        if self._boundary_conditions_window:
            self._boundary_conditions_window.set_boundary_conditions(bc_list, select_bc_id=select_bc_id)
        self._update_viewport_selection()
        self._update_window_title()

    def _reload_mesh_from_model(self) -> None:
        idx = self._selected_mesh_index()
        if idx is not None:
            self._app.set_selection(idx, force=True)

    def _update_membrane_visibility(self, role: str) -> None:
        self.mesh_editor.set_membrane_tab_visible(role == "membrane")

    def _apply_transform_live(self, _=None) -> None:
        if self._is_loading_ui:
            return
        idx = self._selected_mesh_index()
        if idx is None:
            return
        mesh = self._app.project.source_data.meshes[idx]
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
        self._app.touch()
        self._update_window_title()
        self.mesh_viewport_changed.emit()

    def _setup_mesh_picking(self) -> None:
        """Enable PyVista mesh picking on left click when no mesh is selected.
        
        This method sets up mesh picking functionality, which allows users to select meshes by clicking on them in the viewport.
        It is only enabled when no mesh is currently selected to avoid conflicts with the AffineWidget3D, which also uses left-click interactions.
        When a mesh is selected, picking is disabled to prevent interference with the widget's transform handles.
        """
        if not self._plotter or not hasattr(self._plotter, "enable_mesh_picking"):
            return

        def _on_viewport_mesh_picked(picked_actor):
            # Identify the mesh ID from the picked actor or its polydata
            mesh_id = None
            for mid, actor in self._mesh_actor_by_id.items():
                if mid == "__debug_surface__":
                    continue
                if actor is picked_actor:
                    mesh_id = mid
                    break
            if mesh_id is None and picked_actor is not None and hasattr(picked_actor, "GetMapper"):
                mapper = picked_actor.GetMapper()
                if mapper and hasattr(mapper, "GetInput") and mapper.GetInput():
                    polydata = mapper.GetInput()
                    for mid, poly in self._mesh_polydata_by_id.items():
                        if mid != "__debug_surface__" and poly is polydata:
                            mesh_id = mid
                            break
            if mesh_id is None:
                return
            for i, m in enumerate(self._app.project.source_data.meshes):
                if m.mesh_id == mesh_id:
                    def _select(idx=i):
                        self._app.set_selection(idx)
                        self.mesh_list.set_selection_by_model_index(idx)
                    QTimer.singleShot(0, _select)
                    break

        def _on_left_press(_interactor, _event):
            # Handle deselection when clicking on empty space
            picker = getattr(self._plotter.iren, "picker", None)
            if picker is None or not hasattr(picker, "GetActor"):
                return
            if picker.GetActor() is None:
                def _deselect():
                    self._app.set_selection(None)
                    self.mesh_list.set_selection_by_row(-1)
                    self._setup_mesh_picking()  # Re-enable picking after deselection
                QTimer.singleShot(0, _deselect)

        try:
            if self._mesh_pick_deselect_observer is not None:
                self._plotter.iren.remove_observer(self._mesh_pick_deselect_observer)
                self._mesh_pick_deselect_observer = None
            self._plotter.enable_mesh_picking(
                callback=_on_viewport_mesh_picked,
                use_actor=True,
                show=False,
                left_clicking=True,
            )
            self._mesh_pick_deselect_observer = self._plotter.iren.add_observer(
                "LeftButtonPressEvent", _on_left_press
            )
        except Exception:
            pass

    def _update_viewport_selection(self) -> None:
        if not self._plotter or not pv:
            return
        idx = self._selected_mesh_index()
        selected_id = None
        if idx is not None and 0 <= idx < len(self._app.project.source_data.meshes):
            selected_id = self._app.project.source_data.meshes[idx].mesh_id
        # Enable mesh picking (LMB) only when no mesh selected; AffineWidget uses LMB when selected
        # This is a key collision point: picking and widget both use left-click, so they are mutually exclusive
        if selected_id is None and not getattr(self._plotter, "_picker_in_use", False):
            self._setup_mesh_picking()
        mesh_by_id = {m.mesh_id: m for m in self._app.project.source_data.meshes}
        for mesh_id, actor in self._mesh_actor_by_id.items():
            if mesh_id == "__debug_surface__":
                continue
            mesh = mesh_by_id.get(mesh_id)
            if mesh:
                if mesh_id == self._affine_widget_mesh_id:
                    # Widget active: do NOT overwrite actor transform - widget manages it during drag.
                    # Only apply model transform when widget is first added (in _sync_affine_widget).
                    pass
                else:
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
        self._sync_affine_widget(selected_id)
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
        if hasattr(actor, "SetUserMatrix"):
            actor.SetUserMatrix(None)

    def _sync_affine_widget(self, selected_id: str | None) -> None:
        """Show AffineWidget3D on selected mesh, hide when none selected.

        Uses hide (disable) / show (enable) and move (update origin) instead of
        remove+create when possible. Remove+create only when selection changes.

        Complexity: PyVista API quirks (optional add_affine_transform_widget, different
        widget removal methods), VTK callbacks run in different thread (QTimer.singleShot
        for Qt main thread), closure capture of selected_id, defensive checks for missing
        plotter/pv/actor.
        
        Key collision: AffineWidget uses left-click for interaction, conflicting with mesh picking.
        When widget is active, picking is disabled. Widget expects actor at origin, so transform
        is baked into user_matrix to avoid double application of transforms.
        """
        if selected_id == self._affine_widget_mesh_id:
            self._update_affine_widget_origin()
            return
        self._remove_affine_widget()
        self._affine_widget_mesh_id = None
        if not selected_id or not self._plotter or not pv:
            return
        actor = self._mesh_actor_by_id.get(selected_id)
        if not actor:
            return
        add_fn = getattr(self._plotter, "add_affine_transform_widget", None)
        if not add_fn:
            return
        try:
            if getattr(self._plotter, "_picker_in_use", False) and hasattr(self._plotter, "disable_picking"):
                self._plotter.disable_picking()
            mesh = next((m for m in self._app.project.source_data.meshes if m.mesh_id == selected_id), None)
            if not mesh:
                return

            # AffineWidget expects actor at origin; otherwise translation doubles with position.
            # Bake Position/Orientation/Scale into user_matrix and reset them to identity.
            # This is confusing: widget manipulates user_matrix, but model updates separately.
            tr = list(mesh.transform.translation) if mesh.transform else [0, 0, 0]
            rot = list(mesh.transform.rotation_euler_deg) if mesh.transform else [0, 0, 0]
            scl = list(mesh.transform.scale) if mesh.transform else [1, 1, 1]
            tr, rot, scl = (tr + [0, 0, 0])[:3], (rot + [0, 0, 0])[:3], (scl + [1, 1, 1])[:3]
            M = self._build_transform_matrix(tr, rot, scl)
            if M is not None:
                actor.SetPosition(0, 0, 0)
                actor.SetOrientation(0, 0, 0)
                actor.SetScale(1, 1, 1)
                actor.user_matrix = M

            def on_release(_user_matrix):
                def _apply():
                    self._apply_affine_matrix_to_mesh(selected_id, _user_matrix)
                QTimer.singleShot(0, _apply)

            self._affine_widget = add_fn(
                actor,
                release_callback=on_release,
            )
            self._affine_widget_mesh_id = selected_id
        except Exception:
            self._affine_widget = None
            self._affine_widget_mesh_id = None

    def _update_affine_widget_origin(self) -> None:
        """Move widget to current actor position (same selection, transform changed)."""
        if self._affine_widget is None or self._affine_widget_mesh_id is None:
            return
        actor = self._mesh_actor_by_id.get(self._affine_widget_mesh_id)
        if not actor or not hasattr(self._affine_widget, "origin"):
            return
        try:
            center = getattr(actor, "center", None)
            if center is not None:
                self._affine_widget.origin = tuple(center)
                if self._plotter:
                    self._plotter.render()
        except Exception:
            pass

    def _remove_affine_widget(self) -> None:
        if self._affine_widget is None:
            return
        # Restore actor to Position/Orientation/Scale mode (we use user_matrix while widget is active)
        mesh_id = self._affine_widget_mesh_id
        if mesh_id:
            mesh = next((m for m in self._app.project.source_data.meshes if m.mesh_id == mesh_id), None)
            actor = self._mesh_actor_by_id.get(mesh_id)
            if mesh and actor:
                self._apply_actor_transform(mesh, actor)
        try:
            if hasattr(self._affine_widget, "remove"):
                self._affine_widget.remove()
            elif hasattr(self._affine_widget, "Off"):
                self._affine_widget.Off()
            elif hasattr(self._affine_widget, "disable"):
                self._affine_widget.disable()
        except Exception:
            pass
        self._affine_widget = None
        self._affine_widget_mesh_id = None

    def _build_transform_matrix(self, tr: list[float], rot: list[float], scl: list[float]):
        """Build 4x4 matrix from translation, rotation (euler deg), scale using VTK."""
        import numpy as np

        if _pv_vtk is None or np is None:
            return None
        try:
            t = _pv_vtk.vtkTransform()
            t.SetPosition(tr[0], tr[1], tr[2])
            t.SetOrientation(rot[0], rot[1], rot[2])
            t.SetScale(scl[0], scl[1], scl[2])
            m = t.GetMatrix()
            arr = np.eye(4)
            for i in range(4):
                for j in range(4):
                    arr[i, j] = m.GetElement(i, j)
            return arr
        except Exception:
            return None

    def _decompose_matrix_to_transform(self, mat) -> tuple[list[float], list[float], list[float]] | None:
        """Decompose 4x4 matrix into translation, rotation (deg), scale. Returns (tr, rot, scl) or None."""
        import numpy as np

        if _pv_vtk is None or np is None:
            return None
        try:
            m = _pv_vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    m.SetElement(i, j, float(mat[i, j]))
            t = _pv_vtk.vtkTransform()
            t.SetMatrix(m)
            tr = [t.GetPosition()[k] for k in range(3)]
            rot = [t.GetOrientation()[k] for k in range(3)]
            scl = [t.GetScale()[k] for k in range(3)]
            return (tr, rot, scl)
        except Exception:
            return None

    def _update_mesh_from_affine_matrix(self, mesh_id: str, user_matrix, *, use_full_matrix: bool = False) -> bool:
        """Update MeshEntity from matrix. If use_full_matrix, user_matrix is the full transform
        (we bake Position/Orientation/Scale into user_matrix before adding the widget).
        
        This is a confusing part: when use_full_matrix=True (for widget release), user_matrix contains
        the entire transform. When False (for live updates), it's incremental. The logic differs
        because the widget's interaction mode changes how transforms are applied.
        """
        import numpy as np

        mesh = next((m for m in self._app.project.source_data.meshes if m.mesh_id == mesh_id), None)
        if not mesh:
            return False
        M_user = np.array(user_matrix) if user_matrix is not None else np.eye(4)
        if M_user.shape != (4, 4):
            M_user = np.eye(4)
        if use_full_matrix:
            M_total = M_user
        else:
            tr = list(mesh.transform.translation) if mesh.transform else [0, 0, 0]
            rot = list(mesh.transform.rotation_euler_deg) if mesh.transform else [0, 0, 0]
            scl = list(mesh.transform.scale) if mesh.transform else [1, 1, 1]
            tr, rot, scl = (tr + [0, 0, 0])[:3], (rot + [0, 0, 0])[:3], (scl + [1, 1, 1])[:3]
            M_base = self._build_transform_matrix(tr, rot, scl)
            M_total = (M_base @ M_user) if M_base is not None else M_user

        decomposed = self._decompose_matrix_to_transform(M_total)
        if decomposed is None:
            return False
        tr, rot, scl = decomposed
        mesh.transform.translation = tr
        mesh.transform.rotation_euler_deg = rot
        mesh.transform.scale = scl
        return True

    def _apply_affine_matrix_to_mesh(self, mesh_id: str, user_matrix) -> None:
        """On release: user_matrix is the full transform (we baked Position/Orientation/Scale into it
        before adding the widget). Update model only — actor already has correct user_matrix from widget.
        
        This method handles the complex interaction between the AffineWidget's user_matrix and the mesh model.
        The widget manipulates the actor's user_matrix directly, but the model needs to be updated separately.
        Since the initial transform was baked into user_matrix, the widget's output is the full new transform.
        Potential confusion: actor's transform is in user_matrix mode, while model is updated here.
        """
        if not self._update_mesh_from_affine_matrix(mesh_id, user_matrix, use_full_matrix=True):
            return
        self._app.touch()
        self._app.transform_changed.emit()
        self._update_window_title()
        self._update_affine_widget_origin()
        if self._plotter:
            self._plotter.render()

    def _action_add_mesh(self) -> None:
        n = len(self._app.project.source_data.meshes) + 1
        self._app.project.add_mesh(name=f"Mesh_{n}", role="solid", material_key="membrane")
        mesh = self._app.project.source_data.meshes[-1]
        mesh.properties.update(density=1380.0, young_modulus=5.0e9, poisson=0.30)
        self._app.touch()
        self._refresh_mesh_list()
        self.mesh_list.set_selection_by_row(self.mesh_list.count() - 1)
        self._update_window_title()

    def _action_remove_selected_mesh(self) -> None:
        idx = self._selected_mesh_index()
        if idx is None:
            return
        mesh = self._app.project.source_data.meshes.pop(idx)
        self._remove_mesh_actor(mesh.mesh_id)
        self._app.touch()
        self._refresh_mesh_list()
        if self.mesh_list.count() == 0:
            self.mesh_editor.set_info("No mesh selected")
            self.mesh_editor.set_enabled(False)
        self._update_window_title()

    def _remove_mesh_actor(self, mesh_id: str) -> None:
        if mesh_id == self._affine_widget_mesh_id:
            self._remove_affine_widget()
            self._affine_widget_mesh_id = None
        self._mesh_actor_by_id.pop(mesh_id, None)
        self._mesh_polydata_by_id.pop(mesh_id, None)
        self.mesh_viewport_changed.emit()

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
        self._app.touch()
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
            mesh = self._app.project.add_mesh(name=name, role="solid", material_key="membrane")
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
        if not pv:
            return
        poly = self._trimesh_to_polydata(tri_mesh)
        if poly is None:
            return
        self._mesh_polydata_by_id[mesh.mesh_id] = poly
        self.mesh_viewport_changed.emit()

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
        if not pv:
            return
        self._remove_affine_widget()
        self._affine_widget_mesh_id = None
        self._mesh_actor_by_id.clear()
        self._mesh_polydata_by_id.clear()
        for mesh in self._app.project.source_data.meshes:
            tri = self._load_trimesh_for_entity(mesh)
            if tri:
                poly = self._trimesh_to_polydata(tri)
                if poly:
                    self._mesh_polydata_by_id[mesh.mesh_id] = poly
        self.mesh_viewport_changed.emit()

    def _load_project_to_ui(self) -> None:
        self._is_loading_ui = True
        try:
            sim = self._app.project.source_data.simulation_settings
            self.simulation.set_settings({
                "dt": sim.dt,
                "duration": sim.duration,
                "air_coupling_gain": sim.air_coupling_gain,
                "air_grid_step_mm": sim.air_grid_step_mm,
                "force_shape": sim.force_shape,
                "force_amplitude_pa": sim.force_amplitude_pa,
                "force_freq_hz": sim.force_freq_hz,
            })
            md = self._app.project.source_data.metadata
            bc = md.get("boundary_defaults", {})
            fixed = str(bc.get("fixed", "FIXED_EDGE"))
            self.mesh_editor.set_fixed_edge_options(["none", fixed, "FIXED_ALL"])
            self._refresh_material_options()
            self._update_window_title()
            self._refresh_mesh_list()
            if self._app.project.source_data.meshes:
                self.mesh_list.set_selection_by_row(0)
            else:
                self._app.set_selection(None)
            self._rebuild_viewport_from_project()
        finally:
            self._is_loading_ui = False

    def _apply_simulation_to_model(self) -> None:
        data = self.simulation.get_settings()
        sim = self._app.project.source_data.simulation_settings
        sim.dt = data["dt"]
        sim.duration = data["duration"]
        sim.air_coupling_gain = data["air_coupling_gain"]
        sim.air_grid_step_mm = data["air_grid_step_mm"]
        sim.force_shape = data["force_shape"]
        sim.force_amplitude_pa = data["force_amplitude_pa"]
        sim.force_freq_hz = data["force_freq_hz"]
        fixed = self.mesh_editor.cb_fixed_edge.currentText()
        if fixed == "none":
            fixed = "FIXED_EDGE"
        self._app.project.source_data.metadata["boundary_defaults"] = {
            "fixed": fixed,
        }
        self.mesh_editor.set_fixed_edge_options(["none", fixed, "FIXED_ALL"])
        self._app.touch()

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
        if self._mesh_viewport and self._plotter:
            self._mesh_viewport.remove_extra_actor("__debug_surface__")
            actor = self._plotter.add_mesh(grid, name="debug_surface", scalars="uz", cmap="RdBu", show_edges=False)
            self._mesh_viewport.add_extra_actor("__debug_surface__", actor, already_in_scene=True)
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
        self._debug_anim_timer.start(1)

    def _action_new_project(self) -> None:
        if not self._confirm_save_if_dirty():
            return
        self._app.new_project()

    def _action_load_project(self) -> None:
        if not self._confirm_save_if_dirty():
            return
        fp, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Project JSON (*.json);;All (*.*)")
        if not fp:
            return
        try:
            self._app.load_project(fp)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))
            return

    def _action_save_project(self) -> None:
        self._save_internal(force_save_as=True)  # True: show Save As dialog when no path

    def _action_save_project_as(self) -> None:
        prev = self._app.project_path
        self._app.set_project_path(None)
        if not self._save_internal(force_save_as=True):
            self._app.set_project_path(prev)
        self._update_window_title()

    def _action_new_window(self) -> None:
        """Create new window with empty project."""
        if self._app_controller:
            self._app_controller.new_window()

    def _action_open_in_new_window(self) -> None:
        """Open project in new window."""
        if not self._app_controller:
            return
        fp, _ = QFileDialog.getOpenFileName(self, "Open in New Window", "", "Project JSON (*.json);;All (*.*)")
        if not fp:
            return
        try:
            self._app_controller.new_window(load_path=fp)
        except Exception as e:
            QMessageBox.critical(self, "Open Error", str(e))

    def _save_internal(self, force_save_as: bool) -> bool:
        self._apply_simulation_to_model()
        if self._selected_mesh_index() is not None:
            self._apply_mesh_editor_to_model()
        if self._app.project_path is None:
            if not force_save_as:
                return False
            fp, _ = QFileDialog.getSaveFileName(self, "Save As", "project.json", "Project JSON (*.json);;All (*.*)")
            if not fp:
                return False
            if not fp.lower().endswith(".json"):
                fp += ".json"
            self._app.set_project_path(Path(fp))
        try:
            if not self._app.save_project():
                return False
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            return False
        self._update_window_title()
        return True

    def _confirm_save_if_dirty(self) -> bool:
        if not self._app.is_dirty:
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
        self._app.touch()

    def _update_window_title(self) -> None:
        path = str(self._app.project_path) if self._app.project_path else "unsaved"
        mark = "*" if self._app.is_dirty else ""
        self.setWindowTitle(f"FE UI{mark} - {self._app.project.name} ({path})")

    def closeEvent(self, event) -> None:
        if self._sim_process and self._sim_process.state() != QProcess.NotRunning:
            self._sim_process.kill()
        if self._confirm_save_if_dirty():
            event.accept()
        else:
            event.ignore()
