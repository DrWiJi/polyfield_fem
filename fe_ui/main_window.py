# -*- coding: utf-8 -*-
"""
Main window: orchestrates panels, project model, viewport, simulation.
Depends: fe_ui panels, project_model, optional trimesh/pyvista.
"""

from __future__ import annotations 

import copy 
import json 
import logging 
import sys 
import threading 
from pathlib import Path 

logger =logging .getLogger (__name__ )

from PySide6 .QtCore import QObject ,QProcess ,Qt ,QTimer ,Signal 
from PySide6 .QtGui import QAction 
from PySide6 .QtWidgets import (
QApplication ,
QFileDialog ,
QMainWindow ,
QMessageBox ,
QSplitter ,
QWidget ,
QVBoxLayout ,
)
from pyvistaqt import QtInteractor 

from project_model import MeshEntity ,mesh_decode ,mesh_encode 

from .app_controller import AppController 
from .app_model import AppModel 
from .material_library_window import MaterialLibraryWindow 
from .mesh_editor_panel import MeshEditorPanel 
from .mesh_list_panel import MeshListPanel 
from .simulation_panel import DEFAULT_SERVER_PORT ,SimulationPanel 
from .simulation_client import SimulationClientBridge 
from .results_panel import ResultsPanel ,SimulationResultsData 
from .boundary_conditions_panel import BoundaryConditionsPanel 
from .constants import PROJECT_EXT 
from .topology_generator_panel import TopologyGeneratorPanel 
from .viewport import MainViewport ,create_viewport ,has_pyvista 

try :
    from pyvista .plotting import _vtk as _pv_vtk 
except Exception :
    _pv_vtk =None 

try :
    import trimesh 
except Exception :
    trimesh =None 

try :
    import numpy as np 
except Exception :
    np =None 

try :
    import pyvista as pv 
except Exception :
    pv =None 


class _SimulationBridge (QObject ):
    """Bridge to emit simulation completion from worker thread to main thread."""
    finished =Signal ()


def _get_vertex_normals (geom ,verts :"np.ndarray",faces :"np.ndarray")->"np.ndarray | None":
    """Get vertex normals from trimesh geometry, or compute area-weighted from faces. Returns (N,3) float32."""
    if np is None :
        return None 
    try :
        if hasattr (geom ,"vertex_normals"):
            vn =np .asarray (geom .vertex_normals ,dtype =np .float32 )
            if vn .shape ==verts .shape :
                nlen =np .linalg .norm (vn ,axis =1 ,keepdims =True )
                nlen =np .where (nlen >1e-20 ,nlen ,1.0 )
                return (vn /nlen ).astype (np .float32 )
    except Exception :
        pass 
        # Compute from face normals (area-weighted average per vertex)
    v0 =verts [faces [:,0 ]]
    v1 =verts [faces [:,1 ]]
    v2 =verts [faces [:,2 ]]
    fn =np .cross (v1 -v0 ,v2 -v0 )
    areas =np .linalg .norm (fn ,axis =1 ,keepdims =True )
    areas =np .where (areas >1e-20 ,areas ,1.0 )
    fn =fn /areas 
    areas =areas .ravel ()
    vn =np .zeros_like (verts ,dtype =np .float64 )
    np .add .at (vn ,faces [:,0 ],fn *areas [:,None ])
    np .add .at (vn ,faces [:,1 ],fn *areas [:,None ])
    np .add .at (vn ,faces [:,2 ],fn *areas [:,None ])
    nlen =np .linalg .norm (vn ,axis =1 ,keepdims =True )
    nlen =np .where (nlen >1e-20 ,nlen ,1.0 )
    return (vn /nlen ).astype (np .float32 )


class FeMainWindow (QMainWindow ):
    """Main window orchestrating all panels and project."""

    debug_test_run_finished =Signal ()
    simulation_finished =Signal (object )# Emits SimulationResultsData when run completes
    mesh_viewport_changed =Signal ()# Emitted when mesh geometry/list changes (add/remove/rebuild)
    bc_changed =Signal ()# Emitted when boundary conditions change

    def __init__ (
    self ,
    app_model :AppModel ,
    app_controller :AppController |None =None ,
    *,
    auto_run_simulation :bool =False ,
    )->None :
        super ().__init__ ()
        self ._app =app_model 
        self ._app_controller =app_controller 
        self ._auto_run_simulation =auto_run_simulation 
        self .resize (1400 ,900 )
        self ._is_loading_ui =False 
        self ._mesh_actor_by_id :dict [str ,object ]={}
        self ._mesh_polydata_by_id :dict [str ,object ]={}
        self ._sim_process :QProcess |None =None 
        self ._sim_thread =None 
        self ._sim_bridge =_SimulationBridge (self )
        self ._sim_bridge .finished .connect (self ._finish_simulation_run )
        self ._sim_client =SimulationClientBridge (self )
        self ._sim_server_thread :threading .Thread |None =None 
        self ._plotter :QtInteractor |None =None 

        # Debug (for Test Run / Test Visualization)
        self ._debug_thread =None 
        self ._debug_history_disp_all =None 
        self ._debug_log_text =""
        self ._debug_anim_timer =None 
        self ._debug_anim_frame_idx =0 

        self ._material_library_window =None 
        self ._boundary_conditions_window =None 
        self ._topology_generator_window =None 
        self ._affine_widget =None 
        self ._affine_widget_mesh_id :str |None =None 
        self ._mesh_pick_deselect_observer =None 
        self ._mesh_pick_press_observer =None 
        self ._mesh_picked_this_click =False # Flag: mesh was picked this click (avoids picker race with many actors)
        self ._mesh_pick_press_pos :tuple [int ,int ]|None =None # (x, y) on LeftButtonPress for drag detection

        # Last run / results for Simulation panel export (network-equivalent files)
        self ._last_sim_results_dict :dict |None =None 
        self ._last_run_params_snapshot :dict |None =None 
        self ._last_run_material_library :list |None =None 

        self ._build_ui ()
        self ._connect_signals ()
        self ._app .project_changed .connect (self ._on_project_changed )
        self ._app .viewport_closed .connect (self ._on_viewport_closed )
        lib_signal =self ._app_controller .material_library_changed if self ._app_controller else self ._app .material_library_changed 
        lib_signal .connect (self ._refresh_material_options )
        self ._app .state_changed .connect (self ._update_window_title )
        self ._load_project_to_ui ()
        self ._refresh_mesh_list ()
        if self ._auto_run_simulation :
            QTimer .singleShot (100 ,self ._action_run_simulation )

    def _build_ui (self )->None :
        self ._build_menu ()
        self ._build_central ()
        self ._build_docks ()

    def _build_menu (self )->None :
        menu_file =self .menuBar ().addMenu ("File")
        for label ,slot in [
        ("New Project",self ._action_new_project ),
        ("Import Mesh...",self ._action_import_mesh ),
        ]:
            act =QAction (label ,self )
            act .triggered .connect (slot )
            menu_file .addAction (act )
        menu_file .addSeparator ()
        for label ,slot in [
        ("Load...",self ._action_load_project ),
        ("Save",self ._action_save_project ),
        ("Save As...",self ._action_save_project_as ),
        ]:
            act =QAction (label ,self )
            act .triggered .connect (slot )
            menu_file .addAction (act )
        if self ._app_controller :
            menu_file .addSeparator ()
            act_new_win =QAction ("New Window",self )
            act_new_win .triggered .connect (self ._action_new_window )
            menu_file .addAction (act_new_win )
            act_open_win =QAction ("Open in New Window...",self )
            act_open_win .triggered .connect (self ._action_open_in_new_window )
            menu_file .addAction (act_open_win )
        menu_file .addSeparator ()
        act_exit =QAction ("Exit",self )
        act_exit .triggered .connect (self .close )
        menu_file .addAction (act_exit )

        menu_window =self .menuBar ().addMenu ("Window")
        act_material_lib =QAction ("Material Library",self )
        act_material_lib .triggered .connect (self ._action_open_material_library )
        menu_window .addAction (act_material_lib )
        menu_window .addSeparator ()
        self .act_mesh_list =QAction ("Mesh List",self )
        self .act_mesh_list .setCheckable (True )
        self .act_mesh_list .setChecked (True )
        self .act_mesh_list .triggered .connect (self ._window_toggle_mesh_list )
        menu_window .addAction (self .act_mesh_list )

        self .act_mesh_editor =QAction ("Mesh Parameter Editor",self )
        self .act_mesh_editor .setCheckable (True )
        self .act_mesh_editor .setChecked (True )
        self .act_mesh_editor .triggered .connect (self ._window_toggle_mesh_editor )
        menu_window .addAction (self .act_mesh_editor )

        self .act_simulation =QAction ("Simulation",self )
        self .act_simulation .setCheckable (True )
        self .act_simulation .setChecked (True )
        self .act_simulation .triggered .connect (self ._window_toggle_simulation )
        menu_window .addAction (self .act_simulation )

        self .act_bc_panel =QAction ("Boundary Conditions",self )
        self .act_bc_panel .setCheckable (True )
        self .act_bc_panel .setChecked (False )
        self .act_bc_panel .triggered .connect (self ._window_toggle_boundary_conditions )
        menu_window .addAction (self .act_bc_panel )

        self .act_topology_generator =QAction ("Topology Generator",self )
        self .act_topology_generator .setCheckable (True )
        self .act_topology_generator .setChecked (False )
        self .act_topology_generator .triggered .connect (self ._window_toggle_topology_generator )
        menu_window .addAction (self .act_topology_generator )

        menu_window .addSeparator ()
        act_float_mesh_list =QAction ("Mesh List in Separate Window",self )
        act_float_mesh_list .triggered .connect (lambda :self ._window_open_floating (self .mesh_list ))
        menu_window .addAction (act_float_mesh_list )

        act_float_mesh_editor =QAction ("Mesh Parameter Editor in Separate Window",self )
        act_float_mesh_editor .triggered .connect (lambda :self ._window_open_floating (self .mesh_editor ))
        menu_window .addAction (act_float_mesh_editor )

        act_float_simulation =QAction ("Simulation in Separate Window",self )
        act_float_simulation .triggered .connect (lambda :self ._window_open_floating (self .simulation ))
        menu_window .addAction (act_float_simulation )

        self .act_results =QAction ("Results",self )
        self .act_results .triggered .connect (self ._window_show_results )
        menu_window .addAction (self .act_results )

        act_float_bc =QAction ("Boundary Conditions in Separate Window",self )
        act_float_bc .triggered .connect (self ._action_open_boundary_conditions )
        menu_window .addAction (act_float_bc )

        act_float_topology =QAction ("Topology Generator in Separate Window",self )
        act_float_topology .triggered .connect (self ._action_open_topology_generator )
        menu_window .addAction (act_float_topology )

        menu_window .addSeparator ()
        act_reset_layout =QAction ("Reset Layout",self )
        act_reset_layout .triggered .connect (self ._window_reset_layout )
        menu_window .addAction (act_reset_layout )

        menu_debug =self .menuBar ().addMenu ("Debug")
        act_test_run =QAction ("Test Run",self )
        act_test_run .triggered .connect (self ._action_debug_test_run )
        menu_debug .addAction (act_test_run )
        act_test_vis =QAction ("Test Visualization",self )
        act_test_vis .triggered .connect (self ._action_debug_test_visualization )
        menu_debug .addAction (act_test_vis )

    def _action_open_material_library (self )->None :
        if self ._material_library_window is None :
            self ._material_library_window =MaterialLibraryWindow (self ,self ._app ,self ._app_controller )
            self ._material_library_window .setWindowFlags (
            self ._material_library_window .windowFlags ()|Qt .Window 
            )
        self ._material_library_window .show ()
        self ._material_library_window .raise_ ()
        self ._material_library_window .activateWindow ()

    def _window_open_floating (self ,dock :QWidget )->None :
        """Show dock and open it in a separate floating window."""
        dock .setVisible (True )
        dock .setFloating (True )
        if dock ==self .mesh_list :
            self .act_mesh_list .setChecked (True )
        elif dock ==self .mesh_editor :
            self .act_mesh_editor .setChecked (True )
        elif dock ==self .simulation :
            self .act_simulation .setChecked (True )

    def _action_open_boundary_conditions (self )->None :
        """Open Boundary Conditions panel in a separate floating window."""
        if self ._boundary_conditions_window is None :
            self ._boundary_conditions_window =BoundaryConditionsPanel (self )
            self ._boundary_conditions_window .setWindowFlags (
            self ._boundary_conditions_window .windowFlags ()|Qt .Window 
            )
            # Connect signals
            self ._boundary_conditions_window .bc_created .connect (self ._on_bc_created )
            self ._boundary_conditions_window .bc_deleted .connect (self ._on_bc_deleted )
            self ._boundary_conditions_window .bc_updated .connect (self ._on_bc_updated )
            self ._boundary_conditions_window .bc_selected .connect (self ._on_bc_selected )
        self ._boundary_conditions_window .show ()
        self ._boundary_conditions_window .raise_ ()
        self ._boundary_conditions_window .activateWindow ()
        self ._refresh_bc_list ()
        self .act_bc_panel .setChecked (True )

    def _window_toggle_boundary_conditions (self )->None :
        """Toggle Boundary Conditions panel visibility."""
        if self .act_bc_panel .isChecked ():
            self ._action_open_boundary_conditions ()
        else :
            if self ._boundary_conditions_window :
                self ._boundary_conditions_window .close ()
                self ._boundary_conditions_window =None 

    def _action_open_topology_generator (self )->None :
        """Open Topology Generator panel in a separate floating window."""
        if self ._topology_generator_window is None :
            self ._topology_generator_window =TopologyGeneratorPanel (self )
            self ._topology_generator_window .setWindowFlags (
            self ._topology_generator_window .windowFlags ()|Qt .Window 
            )
        self ._topology_generator_window .show ()
        self ._topology_generator_window .raise_ ()
        self ._topology_generator_window .activateWindow ()
        self .act_topology_generator .setChecked (True )

    def _window_toggle_topology_generator (self )->None :
        """Toggle Topology Generator panel visibility."""
        if self .act_topology_generator .isChecked ():
            self ._action_open_topology_generator ()
        else :
            if self ._topology_generator_window :
                self ._topology_generator_window .close ()
                self ._topology_generator_window =None 

    def _on_topology_generator_closed (self )->None :
        """Called when Topology Generator window is closed."""
        self ._topology_generator_window =None 
        self .act_topology_generator .setChecked (False )

    def _window_toggle_mesh_list (self )->None :
        visible =self .act_mesh_list .isChecked ()
        self .mesh_list .setVisible (visible )

    def _window_toggle_mesh_editor (self )->None :
        visible =self .act_mesh_editor .isChecked ()
        self .mesh_editor .setVisible (visible )

    def _window_toggle_simulation (self )->None :
        visible =self .act_simulation .isChecked ()
        self .simulation .setVisible (visible )

    def _window_show_results (self )->None :
        self .results .setVisible (True )
        self .results .raise_ ()
        self .results .activateWindow ()

    def _on_project_changed (self )->None :
        """Called when project is replaced (new/load)."""
        self ._load_project_to_ui ()
        self ._refresh_mesh_list ()

    def _on_mesh_actors_updated (self ,mesh_actor_by_id :dict )->None :
        """Called when UnifiedMeshViewport refreshes mesh display.
        Actors are replaced — remove widget first (it was attached to old actors)."""
        self ._remove_affine_widget ()
        self ._affine_widget_mesh_id =None 
        self ._mesh_actor_by_id .clear ()
        self ._mesh_actor_by_id .update (mesh_actor_by_id )
        self ._update_viewport_selection ()

    def _refresh_material_options (self )->None :
        """Refresh mesh editor material combo and params from MaterialLibraryModel."""
        names =[m .name for m in self ._app .material_library .materials ]
        self .mesh_editor .set_material_options (names )
        self .mesh_editor .refresh_material_params_from_library ()

    def _window_reset_layout (self )->None :
        self .act_mesh_list .setChecked (True )
        self .act_mesh_editor .setChecked (True )
        self .act_simulation .setChecked (True )
        self .act_bc_panel .setChecked (False )
        self .act_topology_generator .setChecked (False )
        self .mesh_list .setVisible (True )
        self .mesh_editor .setVisible (True )
        self .simulation .setVisible (True )
        self .mesh_list .setFloating (False )
        self .mesh_editor .setFloating (False )
        self .simulation .setFloating (False )
        if self ._boundary_conditions_window :
            self ._boundary_conditions_window .close ()
            self ._boundary_conditions_window =None 
        if self ._topology_generator_window :
            self ._topology_generator_window .close ()
            self ._topology_generator_window =None 
        self .addDockWidget (Qt .LeftDockWidgetArea ,self .mesh_list )
        self .addDockWidget (Qt .RightDockWidgetArea ,self .mesh_editor )
        self .addDockWidget (Qt .BottomDockWidgetArea ,self .simulation )

    def _build_central (self )->None :
        root =QWidget ()
        layout =QVBoxLayout (root )
        layout .setContentsMargins (6 ,6 ,6 ,6 )
        splitter =QSplitter (Qt .Horizontal )
        self ._mesh_viewport =MainViewport (
        self ,
        get_mesh_data =lambda :(self ._mesh_polydata_by_id .copy (),self ._app .project .source_data .meshes ),
        refresh_signals =[
        self ._app .project_changed ,
        self .mesh_viewport_changed ,
        ],
        pickable =True ,
        mesh_color ="#9A9A9A",
        )
        self ._plotter =self ._mesh_viewport .plotter 
        self .viewport_widget =self ._mesh_viewport 
        self ._mesh_viewport .mesh_actors_updated .connect (self ._on_mesh_actors_updated )
        splitter .addWidget (self ._mesh_viewport )
        splitter .setStretchFactor (0 ,1 )
        layout .addWidget (splitter )
        self .setCentralWidget (root )

    def _build_docks (self )->None :
        self .mesh_list =MeshListPanel (self )
        self .addDockWidget (Qt .LeftDockWidgetArea ,self .mesh_list )

        self .mesh_editor =MeshEditorPanel (self )
        self .addDockWidget (Qt .RightDockWidgetArea ,self .mesh_editor )

        self .simulation =SimulationPanel (self )
        self .addDockWidget (Qt .BottomDockWidgetArea ,self .simulation )

        self .results =ResultsPanel (self )
        self .addDockWidget (Qt .RightDockWidgetArea ,self .results )
        self .results .setFloating (True )
        self .results .setVisible (False )

    def _connect_signals (self )->None :
        self .debug_test_run_finished .connect (self ._finish_debug_test_run )

        self .mesh_list .selection_changed .connect (self ._app .set_selection )
        self ._app .selection_changed .connect (self ._on_mesh_selected )
        self ._app .transform_changed .connect (self ._on_transform_changed )
        self .mesh_list .search_changed .connect (lambda _ :self ._refresh_mesh_list ())
        self .mesh_list .remove_clicked .connect (self ._action_remove_selected_mesh )

        self .mesh_editor .set_material_provider (lambda :self ._app .material_library )
        self .mesh_editor .apply_clicked .connect (self ._apply_mesh_editor_to_model )
        self .mesh_editor .connect_apply_on_change ()
        self .mesh_editor .reset_clicked .connect (self ._reload_mesh_from_model )
        self .mesh_editor .cb_role .currentTextChanged .connect (self ._update_membrane_visibility )
        self .mesh_editor .connect_dirty (lambda :self ._mark_dirty ())
        self .mesh_editor .connect_transform_live (self ._apply_transform_live )

        self .simulation .run_clicked .connect (self ._action_run_simulation )
        self .simulation .stop_clicked .connect (self ._action_stop_simulation )
        self .simulation .export_results_clicked .connect (self ._action_export_simulation_results )
        self .simulation .export_run_case_clicked .connect (self ._action_export_run_case )
        self .simulation .connect_requested .connect (self ._on_simulation_connect_requested )
        self .simulation .disconnect_requested .connect (self ._on_simulation_disconnect_requested )
        self .simulation .connect_dirty (lambda :self ._mark_dirty ())
        self .simulation_finished .connect (self ._on_simulation_finished )

        self ._sim_client .log_received .connect (self ._on_sim_client_log )
        self ._sim_client .status_received .connect (self ._on_sim_client_status )
        self ._sim_client .results_received .connect (self ._on_sim_client_results )
        self ._sim_client .connected_changed .connect (self ._on_sim_client_connected_changed )

    def _refresh_mesh_list (self )->None :
        query =self .mesh_list .get_search_filter ()
        items =[]
        for i ,mesh in enumerate (self ._app .project .source_data .meshes ):
            text =f"{mesh .name }  [{mesh .role }]  <{mesh .material_key }>"
            if query and query not in text .lower ():
                continue 
            items .append ((text ,i ))
        preserve_idx =self ._app .selected_mesh_index 
        self .mesh_list .set_meshes (items ,preserve_model_index =preserve_idx )

    def _selected_mesh_index (self )->int |None :
        return self ._app .selected_mesh_index 

    def _on_mesh_selected (self )->None :
        """React to selection change — read index from app model."""
        idx =self ._app .selected_mesh_index 
        if idx is None :
            self .mesh_editor .set_info ("No mesh selected")
            self .mesh_editor .set_enabled (False )
            self .mesh_editor .set_membrane_tab_visible (False )
            self ._update_viewport_selection ()
            return 
        mesh =self ._app .project .source_data .meshes [idx ]
        self ._is_loading_ui =True 
        try :
            self .mesh_editor .set_info (f"Selected: {mesh .name }")
            self .mesh_editor .set_enabled (True )
            self .mesh_editor .set_data (self ._mesh_to_editor_dict (mesh ))
            self .mesh_editor .set_membrane_tab_visible (mesh .role =="membrane")
        finally :
            self ._is_loading_ui =False 
        self ._update_viewport_selection ()

    def _on_transform_changed (self )->None :
        """Refresh mesh editor from model when transform changes (affine widget drag)."""
        idx =self ._app .selected_mesh_index 
        if idx is None :
            return 
        if idx >=len (self ._app .project .source_data .meshes ):
            return 
        mesh =self ._app .project .source_data .meshes [idx ]
        self ._is_loading_ui =True 
        try :
            self .mesh_editor .set_data (self ._mesh_to_editor_dict (mesh ))
        finally :
            self ._is_loading_ui =False 

    def _mesh_to_editor_dict (self ,mesh :MeshEntity )->dict :
        tr =list (mesh .transform .translation )if mesh .transform else [0 ,0 ,0 ]
        rot =list (mesh .transform .rotation_euler_deg )if mesh .transform else [0 ,0 ,0 ]
        scl =list (mesh .transform .scale )if mesh .transform else [1 ,1 ,1 ]
        return {
        "name":mesh .name ,
        "role":mesh .role ,
        "material_key":mesh .material_key ,
        "visible":mesh .visible ,
        "density":mesh .properties .get ("density",1380.0 ),
        "E_parallel":mesh .properties .get ("E_parallel",mesh .properties .get ("young_modulus",5.0e9 )),
        "E_perp":mesh .properties .get ("E_perp",3.5e9 ),
        "poisson":mesh .properties .get ("poisson",0.30 ),
        "Cd":mesh .properties .get ("Cd",1.0 ),
        "eta_visc":mesh .properties .get ("eta_visc",0.8 ),
        "acoustic_impedance":mesh .properties .get ("acoustic_impedance",mesh .properties .get ("coupling_gain",1e6 )),
        "thickness_mm":mesh .properties .get ("thickness_mm",0.012 ),
        "pre_tension_n_per_m":mesh .properties .get ("pre_tension_n_per_m",10.0 ),
        "translation":(tr +[0 ,0 ,0 ])[:3 ],
        "rotation_euler_deg":(rot +[0 ,0 ,0 ])[:3 ],
        "scale":(scl +[1 ,1 ,1 ])[:3 ],
        "boundary_groups":mesh .boundary_groups or [],
        "notes":mesh .properties .get ("notes",""),
        }

    def _apply_mesh_editor_to_model (self )->None :
        if self ._is_loading_ui :
            return 
        idx =self ._selected_mesh_index ()
        if idx is None :
            return 
        mesh =self ._app .project .source_data .meshes [idx ]
        data =self .mesh_editor .get_data ()
        mesh .name =data ["name"]or mesh .name 
        mesh .role =data ["role"]
        mesh .material_key =data ["material_key"]
        mesh .visible =data ["visible"]
        # Material parameters (density, E_parallel, ...) only in the library, not saved in mesh
        mesh .properties ["thickness_mm"]=data ["thickness_mm"]
        mesh .properties ["pre_tension_n_per_m"]=data ["pre_tension_n_per_m"]
        mesh .transform .translation =data ["translation"]
        mesh .transform .rotation_euler_deg =data ["rotation_euler_deg"]
        mesh .transform .scale =data ["scale"]
        if data .get ("notes"):
            mesh .properties ["notes"]=data ["notes"]
        elif "notes"in mesh .properties :
            mesh .properties .pop ("notes")
        mesh .boundary_groups =data ["boundary_groups"]or []
        self ._app .touch ()
        self ._refresh_mesh_list ()

    def _on_bc_created (self ,bc_data :dict )->None :
        """Handle new boundary condition creation."""
        from project_model import BoundaryCondition ,MeshTransform 
        bc =BoundaryCondition (
        bc_id =bc_data ["bc_id"],
        name =bc_data ["name"],
        bc_type =bc_data ["bc_type"],
        transform =MeshTransform (
        translation =bc_data ["translation"],
        rotation_euler_deg =bc_data ["rotation_euler_deg"],
        scale =bc_data ["scale"],
        ),
        mesh_ids =bc_data ["mesh_ids"],
        flags =bc_data ["flags"],
        parameters =bc_data ["parameters"],
        )
        self ._app .project .source_data .boundary_conditions .append (bc )
        self ._app .touch ()
        self ._refresh_bc_list (select_bc_id =bc .bc_id )
        self .bc_changed .emit ()

    def _on_bc_deleted (self ,bc_id :str )->None :
        """Handle boundary condition deletion."""
        bc_list =self ._app .project .source_data .boundary_conditions 
        self ._app .project .source_data .boundary_conditions =[
        bc for bc in bc_list if bc .bc_id !=bc_id 
        ]
        self ._app .touch ()
        self ._refresh_bc_list ()
        self .bc_changed .emit ()

    def _on_bc_updated (self ,bc_id :str ,bc_data :dict )->None :
        """Handle boundary condition update."""
        bc_list =self ._app .project .source_data .boundary_conditions 
        for bc in bc_list :
            if bc .bc_id ==bc_id :
                bc .name =bc_data ["name"]
                bc .bc_type =bc_data ["bc_type"]
                bc .transform .translation =bc_data ["translation"]
                bc .transform .rotation_euler_deg =bc_data ["rotation_euler_deg"]
                bc .transform .scale =bc_data ["scale"]
                bc .mesh_ids =bc_data ["mesh_ids"]
                bc .flags =bc_data ["flags"]
                bc .parameters =bc_data ["parameters"]
                break 
        self ._app .touch ()
        self ._refresh_bc_list (select_bc_id =bc_id )
        self .bc_changed .emit ()

    def _on_bc_selected (self ,bc_id :str )->None :
        """Handle boundary condition selection."""
        pass 

    def _close_boundary_conditions_window (self )->None :
        """Close boundary conditions window."""
        if self ._boundary_conditions_window :
            self ._boundary_conditions_window .close ()
            self ._boundary_conditions_window =None 
            self .act_bc_panel .setChecked (False )

    def _on_boundary_conditions_window_closed (self )->None :
        """Called when BC window is closed (e.g. via X button). Clear reference and notify other viewports."""
        if self ._boundary_conditions_window :
            self ._boundary_conditions_window =None 
            self .act_bc_panel .setChecked (False )
        self ._app .notify_viewport_closed ()

    def _on_viewport_closed (self )->None :
        """React to any viewport window closing. Restore picking and affine widget."""
        QTimer .singleShot (0 ,self ._restore_viewport_handlers )

    def _restore_viewport_handlers (self )->None :
        """Re-setup picking and affine widget after another viewport closed (OpenGL context restored)."""
        if not self ._plotter or not pv :
            return 
        try :
            if hasattr (self ._plotter ,"disable_picking"):
                self ._plotter .disable_picking ()
            setattr (self ._plotter ,"_picker_in_use",False )
            if hasattr (self ._plotter ,"iren"):
                for obs in (self ._mesh_pick_deselect_observer ,self ._mesh_pick_press_observer ):
                    if obs is not None :
                        try :
                            self ._plotter .iren .remove_observer (obs )
                        except Exception :
                            pass 
            self ._mesh_pick_deselect_observer =None 
            self ._mesh_pick_press_observer =None 
            self ._remove_affine_widget ()
            self ._affine_widget_mesh_id =None 
            self ._plotter .render ()
            self ._update_viewport_selection ()
        except Exception :
            pass 

    def _refresh_bc_list (self ,select_bc_id :str |None =None )->None :
        """Refresh boundary conditions list in all panels."""
        bc_list =self ._app .project .source_data .boundary_conditions 
        # Update BC window if it exists
        if self ._boundary_conditions_window :
            self ._boundary_conditions_window .set_boundary_conditions (bc_list ,select_bc_id =select_bc_id )
        self ._update_viewport_selection ()
        self ._update_window_title ()

    def _reload_mesh_from_model (self )->None :
        idx =self ._selected_mesh_index ()
        if idx is not None :
            self ._app .set_selection (idx ,force =True )

    def _update_membrane_visibility (self ,role :str )->None :
        self .mesh_editor .set_membrane_tab_visible (role =="membrane")

    def _apply_transform_live (self ,_ =None )->None :
        if self ._is_loading_ui :
            return 
        idx =self ._selected_mesh_index ()
        if idx is None :
            return 
        mesh =self ._app .project .source_data .meshes [idx ]
        mesh .transform .translation =[
        float (self .mesh_editor .sp_tx .value ()),
        float (self .mesh_editor .sp_ty .value ()),
        float (self .mesh_editor .sp_tz .value ()),
        ]
        mesh .transform .rotation_euler_deg =[
        float (self .mesh_editor .sp_rx .value ()),
        float (self .mesh_editor .sp_ry .value ()),
        float (self .mesh_editor .sp_rz .value ()),
        ]
        mesh .transform .scale =[
        float (self .mesh_editor .sp_sx .value ()),
        float (self .mesh_editor .sp_sy .value ()),
        float (self .mesh_editor .sp_sz .value ()),
        ]
        actor =self ._mesh_actor_by_id .get (mesh .mesh_id )
        if actor and hasattr (actor ,"SetPosition"):
            tr ,rot ,scl =mesh .transform .translation ,mesh .transform .rotation_euler_deg ,mesh .transform .scale 
            actor .SetPosition (*tr )
            actor .SetOrientation (*rot )
            actor .SetScale (*scl )
        if self ._plotter :
            self ._plotter .render ()
        self ._app .touch ()
        self ._update_window_title ()
        # Emit transform_changed for BC viewport; main viewport actor already updated above
        self ._app .transform_changed .emit ()

    def _setup_mesh_picking (self )->None :
        """Enable PyVista mesh picking on left click when no mesh is selected.
        
        This method sets up mesh picking functionality, which allows users to select meshes by clicking on them in the viewport.
        It is only enabled when no mesh is currently selected to avoid conflicts with the AffineWidget3D, which also uses left-click interactions.
        When a mesh is selected, picking is disabled to prevent interference with the widget's transform handles.
        """
        if not self ._plotter or not hasattr (self ._plotter ,"enable_mesh_picking"):
            return 

            # Always disable first to ensure clean state — PyVista raises if picking is already enabled.
            # Without this, on 2nd/3rd re-enable the picker state can get corrupted and the title disappears.
        if hasattr (self ._plotter ,"disable_picking"):
            try :
                self ._plotter .disable_picking ()
            except Exception :
                pass 

        def _on_viewport_mesh_picked (picked_actor ):
        # Identify the mesh ID from the picked actor or its polydata
            mesh_id =None 
            for mid ,actor in self ._mesh_actor_by_id .items ():
                if mid =="__debug_surface__":
                    continue 
                if actor is picked_actor :
                    mesh_id =mid 
                    break 
            if mesh_id is None and picked_actor is not None and hasattr (picked_actor ,"GetMapper"):
                mapper =picked_actor .GetMapper ()
                if mapper and hasattr (mapper ,"GetInput")and mapper .GetInput ():
                    polydata =mapper .GetInput ()
                    for mid ,poly in self ._mesh_polydata_by_id .items ():
                        if mid !="__debug_surface__"and poly is polydata :
                            mesh_id =mid 
                            break 
            if mesh_id is None :
                return 
                # Mark that a mesh was picked (used by _on_left_press to avoid deselecting)
            self ._mesh_picked_this_click =True 
            for i ,m in enumerate (self ._app .project .source_data .meshes ):
                if m .mesh_id ==mesh_id :
                    def _select (idx =i ):
                        self ._app .set_selection (idx )
                        self .mesh_list .set_selection_by_model_index (idx )
                    QTimer .singleShot (0 ,_select )
                    break 

        _DRAG_THRESHOLD_PX =5 # Treat as click if movement less than this

        def _on_left_press (_obj ,_event ):
        # Record press position to distinguish click from drag (camera rotate).
            x ,y =self ._plotter .iren .get_event_position ()
            self ._mesh_pick_press_pos =(x ,y )

        def _on_left_release (_obj ,_event ):
        # Deselect only on click (not drag). Dragging = camera rotate — don't deselect.
            if self ._mesh_picked_this_click :
                self ._mesh_picked_this_click =False 
                self ._mesh_pick_press_pos =None 
                return 
            press =self ._mesh_pick_press_pos 
            self ._mesh_pick_press_pos =None 
            if press is None :
                return 
                # Check if it was a drag (rotate) - don't deselect
            rx ,ry =self ._plotter .iren .get_event_position ()
            dx ,dy =abs (rx -press [0 ]),abs (ry -press [1 ])
            if dx >_DRAG_THRESHOLD_PX or dy >_DRAG_THRESHOLD_PX :
                return # Was a drag, not a click
            def _deselect ():
                self ._app .set_selection (None )
                self .mesh_list .set_selection_by_row (-1 )
                self ._setup_mesh_picking ()# Re-enable picking after deselection
            QTimer .singleShot (0 ,_deselect )

        try :
            self ._mesh_picked_this_click =False 
            self ._mesh_pick_press_pos =None 
            if self ._mesh_pick_deselect_observer is not None :
                self ._plotter .iren .remove_observer (self ._mesh_pick_deselect_observer )
                self ._mesh_pick_deselect_observer =None 
            if self ._mesh_pick_press_observer is not None :
                self ._plotter .iren .remove_observer (self ._mesh_pick_press_observer )
                self ._mesh_pick_press_observer =None 
            self ._plotter .enable_mesh_picking (
            callback =_on_viewport_mesh_picked ,
            use_actor =True ,
            show =False ,
            left_clicking =True ,
            show_message =True ,
            )
            self ._mesh_pick_press_observer =self ._plotter .iren .add_observer (
            "LeftButtonPressEvent",_on_left_press 
            )
            self ._mesh_pick_deselect_observer =self ._plotter .iren .add_observer (
            "LeftButtonReleaseEvent",_on_left_release 
            )
        except Exception as e :
            logger .warning ("Failed to setup mesh picking: %s",e )

    def _update_viewport_selection (self )->None :
        if not self ._plotter or not pv :
            return 
        idx =self ._selected_mesh_index ()
        selected_id =None 
        if idx is not None and 0 <=idx <len (self ._app .project .source_data .meshes ):
            selected_id =self ._app .project .source_data .meshes [idx ].mesh_id 
            # Enable mesh picking (LMB) only when no mesh selected; AffineWidget uses LMB when selected
            # This is a key collision point: picking and widget both use left-click, so they are mutually exclusive
        if selected_id is None and not getattr (self ._plotter ,"_picker_in_use",False ):
            self ._setup_mesh_picking ()
        mesh_by_id ={m .mesh_id :m for m in self ._app .project .source_data .meshes }
        for mesh_id ,actor in self ._mesh_actor_by_id .items ():
            if mesh_id =="__debug_surface__":
                continue 
            mesh =mesh_by_id .get (mesh_id )
            if mesh :
                if mesh_id ==self ._affine_widget_mesh_id :
                # Widget active: do NOT overwrite actor transform - widget manages it during drag.
                # Only apply model transform when widget is first added (in _sync_affine_widget).
                    pass 
                else :
                    self ._apply_actor_transform (mesh ,actor )
            visible =mesh .visible if mesh else True 
            if hasattr (actor ,"SetVisibility"):
                actor .SetVisibility (1 if visible else 0 )
            if not visible :
                continue 
            color ="#F0D070"if mesh_id ==selected_id else "#9A9A9A"
            prop =actor .GetProperty ()if hasattr (actor ,"GetProperty")else None 
            if prop :
                prop .SetColor (*pv .Color (color ).float_rgb )
        self ._sync_affine_widget (selected_id )
        self ._plotter .render ()

    def _apply_actor_transform (self ,mesh :MeshEntity ,actor :object )->None :
        tr =list (mesh .transform .translation )if mesh .transform else [0 ,0 ,0 ]
        rot =list (mesh .transform .rotation_euler_deg )if mesh .transform else [0 ,0 ,0 ]
        scl =list (mesh .transform .scale )if mesh .transform else [1 ,1 ,1 ]
        tr ,rot ,scl =(tr +[0 ,0 ,0 ])[:3 ],(rot +[0 ,0 ,0 ])[:3 ],(scl +[1 ,1 ,1 ])[:3 ]
        if hasattr (actor ,"SetPosition"):
            actor .SetPosition (*tr )
        if hasattr (actor ,"SetOrientation"):
            actor .SetOrientation (*rot )
        if hasattr (actor ,"SetScale"):
            actor .SetScale (*scl )
        if hasattr (actor ,"SetUserMatrix"):
            actor .SetUserMatrix (None )

    def _sync_affine_widget (self ,selected_id :str |None )->None :
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
        if selected_id ==self ._affine_widget_mesh_id :
            self ._update_affine_widget_origin ()
            return 
        self ._remove_affine_widget ()
        self ._affine_widget_mesh_id =None 
        if not selected_id or not self ._plotter or not pv :
            return 
        actor =self ._mesh_actor_by_id .get (selected_id )
        if not actor :
            return 
        add_fn =getattr (self ._plotter ,"add_affine_transform_widget",None )
        if not add_fn :
            logger .warning ("add_affine_transform_widget not available (PyVista 0.47+ required)")
            return 
        try :
            if getattr (self ._plotter ,"_picker_in_use",False )and hasattr (self ._plotter ,"disable_picking"):
                self ._plotter .disable_picking ()
            mesh =next ((m for m in self ._app .project .source_data .meshes if m .mesh_id ==selected_id ),None )
            if not mesh :
                return 

                # AffineWidget expects actor at origin; otherwise translation doubles with position.
                # Bake Position/Orientation/Scale into user_matrix and reset them to identity.
                # This is confusing: widget manipulates user_matrix, but model updates separately.
            tr =list (mesh .transform .translation )if mesh .transform else [0 ,0 ,0 ]
            rot =list (mesh .transform .rotation_euler_deg )if mesh .transform else [0 ,0 ,0 ]
            scl =list (mesh .transform .scale )if mesh .transform else [1 ,1 ,1 ]
            tr ,rot ,scl =(tr +[0 ,0 ,0 ])[:3 ],(rot +[0 ,0 ,0 ])[:3 ],(scl +[1 ,1 ,1 ])[:3 ]
            M =self ._build_transform_matrix (tr ,rot ,scl )
            if M is not None :
                actor .SetPosition (0 ,0 ,0 )
                actor .SetOrientation (0 ,0 ,0 )
                actor .SetScale (1 ,1 ,1 )
                actor .user_matrix =M 

            def on_release (_user_matrix ):
                def _apply ():
                    self ._apply_affine_matrix_to_mesh (selected_id ,_user_matrix )
                QTimer .singleShot (0 ,_apply )

                # Constant widget size regardless of geometry: scale inversely with actor length.
                # PyVista uses widget_size = scale * actor_length, so scale = const / length.
            actor_length =max (float (actor .GetLength ()),0.01 )if hasattr (actor ,"GetLength")else 1.0 
            const_widget_scale =max (0.15 ,6.0 /actor_length )# ~6 world units or 15% of mesh min (3x)

            self ._affine_widget =add_fn (
            actor ,
            release_callback =on_release ,
            scale =const_widget_scale ,
            line_radius =0.05 ,# Thicker lines for visibility
            always_visible =True ,
            )
            self ._affine_widget_mesh_id =selected_id 
            if self ._plotter :
                self ._plotter .render ()
        except Exception as e :
            logger .warning ("Failed to add AffineWidget3D: %s",e )
            self ._affine_widget =None 
            self ._affine_widget_mesh_id =None 

    def _update_affine_widget_origin (self )->None :
        """Move widget to current actor position (same selection, transform changed)."""
        if self ._affine_widget is None or self ._affine_widget_mesh_id is None :
            return 
        actor =self ._mesh_actor_by_id .get (self ._affine_widget_mesh_id )
        if not actor or not hasattr (self ._affine_widget ,"origin"):
            return 
        try :
            center =getattr (actor ,"center",None )
            if center is not None :
                self ._affine_widget .origin =tuple (center )
                if self ._plotter :
                    self ._plotter .render ()
        except Exception :
            pass 

    def _remove_affine_widget (self )->None :
        if self ._affine_widget is None :
            return 
            # Restore actor to Position/Orientation/Scale mode (we use user_matrix while widget is active)
        mesh_id =self ._affine_widget_mesh_id 
        if mesh_id :
            mesh =next ((m for m in self ._app .project .source_data .meshes if m .mesh_id ==mesh_id ),None )
            actor =self ._mesh_actor_by_id .get (mesh_id )
            if mesh and actor :
                self ._apply_actor_transform (mesh ,actor )
        try :
            if hasattr (self ._affine_widget ,"remove"):
                self ._affine_widget .remove ()
            elif hasattr (self ._affine_widget ,"Off"):
                self ._affine_widget .Off ()
            elif hasattr (self ._affine_widget ,"disable"):
                self ._affine_widget .disable ()
        except Exception :
            pass 
        self ._affine_widget =None 
        self ._affine_widget_mesh_id =None 

    def _build_transform_matrix (self ,tr :list [float ],rot :list [float ],scl :list [float ]):
        """Build 4x4 matrix from translation, rotation (euler deg), scale using VTK."""
        import numpy as np 

        if _pv_vtk is None or np is None :
            return None 
        try :
            t =_pv_vtk .vtkTransform ()
            t .SetPosition (tr [0 ],tr [1 ],tr [2 ])
            t .SetOrientation (rot [0 ],rot [1 ],rot [2 ])
            t .SetScale (scl [0 ],scl [1 ],scl [2 ])
            m =t .GetMatrix ()
            arr =np .eye (4 )
            for i in range (4 ):
                for j in range (4 ):
                    arr [i ,j ]=m .GetElement (i ,j )
            return arr 
        except Exception :
            return None 

    def _decompose_matrix_to_transform (self ,mat )->tuple [list [float ],list [float ],list [float ]]|None :
        """Decompose 4x4 matrix into translation, rotation (deg), scale. Returns (tr, rot, scl) or None."""
        import numpy as np 

        if _pv_vtk is None or np is None :
            return None 
        try :
            m =_pv_vtk .vtkMatrix4x4 ()
            for i in range (4 ):
                for j in range (4 ):
                    m .SetElement (i ,j ,float (mat [i ,j ]))
            t =_pv_vtk .vtkTransform ()
            t .SetMatrix (m )
            tr =[t .GetPosition ()[k ]for k in range (3 )]
            rot =[t .GetOrientation ()[k ]for k in range (3 )]
            scl =[t .GetScale ()[k ]for k in range (3 )]
            return (tr ,rot ,scl )
        except Exception :
            return None 

    def _update_mesh_from_affine_matrix (self ,mesh_id :str ,user_matrix ,*,use_full_matrix :bool =False )->bool :
        """Update MeshEntity from matrix. If use_full_matrix, user_matrix is the full transform
        (we bake Position/Orientation/Scale into user_matrix before adding the widget).
        
        This is a confusing part: when use_full_matrix=True (for widget release), user_matrix contains
        the entire transform. When False (for live updates), it's incremental. The logic differs
        because the widget's interaction mode changes how transforms are applied.
        """
        import numpy as np 

        mesh =next ((m for m in self ._app .project .source_data .meshes if m .mesh_id ==mesh_id ),None )
        if not mesh :
            return False 
        M_user =np .array (user_matrix )if user_matrix is not None else np .eye (4 )
        if M_user .shape !=(4 ,4 ):
            M_user =np .eye (4 )
        if use_full_matrix :
            M_total =M_user 
        else :
            tr =list (mesh .transform .translation )if mesh .transform else [0 ,0 ,0 ]
            rot =list (mesh .transform .rotation_euler_deg )if mesh .transform else [0 ,0 ,0 ]
            scl =list (mesh .transform .scale )if mesh .transform else [1 ,1 ,1 ]
            tr ,rot ,scl =(tr +[0 ,0 ,0 ])[:3 ],(rot +[0 ,0 ,0 ])[:3 ],(scl +[1 ,1 ,1 ])[:3 ]
            M_base =self ._build_transform_matrix (tr ,rot ,scl )
            M_total =(M_base @M_user )if M_base is not None else M_user 

        decomposed =self ._decompose_matrix_to_transform (M_total )
        if decomposed is None :
            return False 
        tr ,rot ,scl =decomposed 
        mesh .transform .translation =tr 
        mesh .transform .rotation_euler_deg =rot 
        mesh .transform .scale =scl 
        return True 

    def _apply_affine_matrix_to_mesh (self ,mesh_id :str ,user_matrix )->None :
        """On release: user_matrix is the full transform (we baked Position/Orientation/Scale into it
        before adding the widget). Update model only — actor already has correct user_matrix from widget.
        
        This method handles the complex interaction between the AffineWidget's user_matrix and the mesh model.
        The widget manipulates the actor's user_matrix directly, but the model needs to be updated separately.
        Since the initial transform was baked into user_matrix, the widget's output is the full new transform.
        Potential confusion: actor's transform is in user_matrix mode, while model is updated here.
        """
        if not self ._update_mesh_from_affine_matrix (mesh_id ,user_matrix ,use_full_matrix =True ):
            return 
        self ._app .touch ()
        self ._app .transform_changed .emit ()
        self ._update_window_title ()
        self ._update_affine_widget_origin ()
        if self ._plotter :
            self ._plotter .render ()

    def _action_add_mesh (self )->None :
        n =len (self ._app .project .source_data .meshes )+1 
        self ._app .project .add_mesh (name =f"Mesh_{n }",role ="solid",material_key ="membrane")
        mesh =self ._app .project .source_data .meshes [-1 ]
        mesh .properties .update (
        density =1380.0 ,E_parallel =5.0e9 ,E_perp =3.5e9 ,poisson =0.30 ,
        Cd =1.0 ,eta_visc =0.8 ,acoustic_impedance =1e6 ,
        )
        self ._app .touch ()
        self ._refresh_mesh_list ()
        self .mesh_list .set_selection_by_row (self .mesh_list .count ()-1 )
        self ._update_window_title ()

    def _action_remove_selected_mesh (self )->None :
        idx =self ._selected_mesh_index ()
        if idx is None :
            return 
        mesh =self ._app .project .source_data .meshes .pop (idx )
        self ._remove_mesh_actor (mesh .mesh_id )
        self ._app .touch ()
        self ._refresh_mesh_list ()
        if self .mesh_list .count ()==0 :
            self .mesh_editor .set_info ("No mesh selected")
            self .mesh_editor .set_enabled (False )
        self ._update_window_title ()

    def _remove_mesh_actor (self ,mesh_id :str )->None :
        if mesh_id ==self ._affine_widget_mesh_id :
            self ._remove_affine_widget ()
            self ._affine_widget_mesh_id =None 
        self ._mesh_actor_by_id .pop (mesh_id ,None )
        self ._mesh_polydata_by_id .pop (mesh_id ,None )
        self .mesh_viewport_changed .emit ()

    def _action_import_mesh (self )->None :
        if trimesh is None :
            QMessageBox .critical (self ,"Import Error","trimesh is not available. pip install trimesh")
            return 
        paths ,_ =QFileDialog .getOpenFileNames (
        self ,"Import Mesh","",
        "Mesh files (*.stl *.obj *.ply *.off *.glb *.gltf);;All Files (*.*)",
        )
        if not paths :
            return 
        all_errors :list [str ]=[]
        all_warnings :list [str ]=[]
        for fp in paths :
            errs ,warns =self ._import_mesh_file (Path (fp ))
            all_errors .extend (errs )
            all_warnings .extend (warns )
        self ._app .touch ()
        self ._refresh_mesh_list ()
        self ._update_window_title ()
        self ._show_import_summary (all_errors ,all_warnings )

    def _trimesh_load_kwargs (self ,path :Path )->dict :
        """Format-specific kwargs for trimesh.load to split objects/groups where supported."""
        ext =path .suffix .lower ()
        if ext ==".obj":
            return {"force":"scene","split_objects":True ,"split_groups":True }
        return {"force":"scene"}

    def _show_import_summary (self ,errors :list [str ],warnings :list [str ])->None :
        """Show a dialog with all import errors and warnings, if any."""
        if not errors and not warnings :
            return 
        lines :list [str ]=[]
        if errors :
            lines .append ("Errors:")
            lines .extend (f"  • {e }"for e in errors )
        if warnings :
            if lines :
                lines .append ("")
            lines .append ("Warnings:")
            lines .extend (f"  • {w }"for w in warnings )
        msg ="\n".join (lines )
        title ="Import Issues"if (errors and warnings )else ("Import Error"if errors else "Import Warnings")
        icon =QMessageBox .Critical if errors else QMessageBox .Warning 
        mb =QMessageBox (self )
        mb .setWindowTitle (title )
        mb .setIcon (icon )
        mb .setText (msg )
        mb .setStandardButtons (QMessageBox .Ok )
        mb .exec ()

    def _import_mesh_file (self ,src :Path )->tuple [list [str ],list [str ]]:
        """Import mesh file. Returns (errors, warnings)."""
        errors :list [str ]=[]
        warnings :list [str ]=[]

        try :
            load_kwargs =self ._trimesh_load_kwargs (src )
            logger .info ("Loading mesh: %s with kwargs=%s",src ,load_kwargs )
            loaded =trimesh .load (str (src ),**load_kwargs )
        except Exception as e :
            err_msg =f"{src .name }: {e }"
            errors .append (err_msg )
            logger .exception ("Failed to import mesh %s",src )
            return (errors ,warnings )

        if loaded is None :
            err_msg =f"{src .name }: trimesh.load returned None"
            errors .append (err_msg )
            logger .error ("trimesh.load returned None for %s",src )
            return (errors ,warnings )

        geoms :list [tuple [str ,object ]]=[]
        if hasattr (loaded ,"geometry")and isinstance (loaded .geometry ,dict ):
            geoms =list (loaded .geometry .items ())
            logger .info ("Loaded scene from %s: %d geometry objects",src .name ,len (geoms ))
        elif hasattr (loaded ,"vertices")and hasattr (loaded ,"faces"):
            geoms =[(src .stem ,loaded )]
            logger .info ("Loaded single mesh from %s",src .name )
        else :
            warn_msg =f"{src .name }: Unknown format (no geometry dict, no vertices/faces)"
            warnings .append (warn_msg )
            logger .warning ("Unknown loaded format for %s: %s",src ,type (loaded ))
            return (errors ,warnings )

        imported_count =0 
        for gname ,geom in geoms :
        # Validate geometry
            if not hasattr (geom ,"vertices")or not hasattr (geom ,"faces"):
                warn_msg =f"{src .name } object '{gname }': missing vertices or faces (skipped)"
                warnings .append (warn_msg )
                logger .warning ("Skipping object '%s' in %s: no vertices/faces",gname ,src .name )
                continue 

            try :
                nv =len (geom .vertices )
                nf =len (geom .faces )
            except (TypeError ,AttributeError )as e :
                warn_msg =f"{src .name } object '{gname }': invalid geometry ({e }) (skipped)"
                warnings .append (warn_msg )
                logger .warning ("Skipping object '%s' in %s: %s",gname ,src .name ,e )
                continue 

            if nv ==0 :
                warn_msg =f"{src .name } object '{gname }': empty (0 vertices) (skipped)"
                warnings .append (warn_msg )
                logger .warning ("Skipping empty object '%s' in %s: no vertices",gname ,src .name )
                continue 

            if nf ==0 :
                warn_msg =f"{src .name } object '{gname }': no polygons (0 faces) (skipped)"
                warnings .append (warn_msg )
                logger .warning ("Skipping object '%s' in %s: no faces (invisible)",gname ,src .name )
                continue 

            name =src .stem if len (geoms )==1 else f"{src .stem }:{gname }"
            try :
                mesh =self ._app .project .add_mesh (name =name ,role ="solid",material_key ="membrane")
                verts =np .asarray (geom .vertices ,dtype =np .float32 )
                faces_arr =np .asarray (geom .faces ,dtype =np .int64 )
                if faces_arr .shape [1 ]==4 :
                    faces_arr =np .hstack ([faces_arr [:,:3 ],faces_arr [:,[0 ,2 ,3 ]]]).reshape (-1 ,3 )
                normals =_get_vertex_normals (geom ,verts ,faces_arr )
                mesh .mesh_data =mesh_encode (verts ,faces_arr ,normals )
                mesh .properties .update (
                density =1380.0 ,E_parallel =5.0e9 ,E_perp =3.5e9 ,poisson =0.30 ,
                Cd =1.0 ,eta_visc =0.8 ,acoustic_impedance =1e6 ,
                )
                mesh .properties ["trimesh_geom_name"]=str (gname )
                mesh .properties ["vertex_count"]=nv 
                mesh .properties ["face_count"]=nf 
                add_err =self ._add_mesh_to_viewport (mesh ,geom )
                if add_err :
                    errors .append (add_err )
                else :
                    imported_count +=1 
                    logger .info ("Imported '%s' from %s: %d verts, %d faces",name ,src .name ,nv ,nf )
            except Exception as e :
                err_msg =f"{src .name } object '{gname }': {e }"
                errors .append (err_msg )
                logger .exception ("Failed to add mesh '%s' from %s",gname ,src .name )

        if imported_count ==0 and geoms and not errors :
            warn_msg =f"{src .name }: no valid geometry imported (all objects empty or invalid)"
            warnings .append (warn_msg )
            logger .warning ("No valid geometry imported from %s",src .name )

        return (errors ,warnings )

    def _trimesh_to_polydata (self ,tri_mesh ,normals =None ):
        """Convert trimesh to PyVista PolyData. Optionally set vertex normals."""
        if not pv or np is None :
            return None 
        try :
            verts =np .asarray (tri_mesh .vertices ,dtype =np .float64 )
            faces =np .asarray (tri_mesh .faces ,dtype =np .int64 )
            if len (verts )==0 or len (faces )==0 :
                logger .warning ("trimesh_to_polydata: empty mesh (verts=%d, faces=%d)",len (verts ),len (faces ))
                return None 
            cells =np .hstack ([np .full ((faces .shape [0 ],1 ),3 ,dtype =np .int64 ),faces ]).ravel ()
            poly =pv .PolyData (verts ,cells )
            if normals is not None :
                n_arr =np .asarray (normals ,dtype =np .float64 )
                if n_arr .shape ==verts .shape :
                    poly ["Normals"]=n_arr 
            elif hasattr (tri_mesh ,"vertex_normals"):
                try :
                    vn =np .asarray (tri_mesh .vertex_normals ,dtype =np .float64 )
                    if vn .shape ==verts .shape :
                        poly ["Normals"]=vn 
                except Exception :
                    poly .compute_normals (inplace =True )
            else :
                poly .compute_normals (inplace =True )
            return poly 
        except Exception as e :
            logger .exception ("trimesh_to_polydata failed: %s",e )
            return None 

    def _add_mesh_to_viewport (self ,mesh :MeshEntity ,tri_mesh )->str |None :
        """Add mesh to viewport. Returns error message or None on success."""
        if not pv :
            return "pyvista is not available"
        poly =self ._trimesh_to_polydata (tri_mesh )
        if poly is None :
            return f"Failed to convert mesh '{mesh .name }' to PolyData"
        self ._mesh_polydata_by_id [mesh .mesh_id ]=poly 
        self .mesh_viewport_changed .emit ()
        return None 

    def _load_trimesh_for_entity (self ,mesh :MeshEntity ):
        """Load trimesh geometry for a MeshEntity from embedded mesh_data. Returns (tri_mesh, normals) or (tri_mesh, None)."""
        if not trimesh or not mesh .mesh_data :
            logger .debug ("_load_trimesh_for_entity: trimesh or mesh_data missing for %s",mesh .name )
            return None ,None 
        decoded =mesh_decode (mesh .mesh_data )
        if decoded is None :
            logger .warning ("_load_trimesh_for_entity: failed to decode mesh_data for %s",mesh .name )
            return None ,None 
        verts ,faces ,normals =decoded 
        if not verts or not faces :
            logger .warning ("_load_trimesh_for_entity: empty geometry for %s",mesh .name )
            return None ,None 
        try :
            tri =trimesh .Trimesh (vertices =verts ,faces =faces )
            if normals is not None :
                tri ._cache ["vertex_normals"]=np .asarray (normals ,dtype =np .float64 )
            return tri ,np .asarray (normals )if normals else None 
        except Exception as e :
            logger .exception ("_load_trimesh_for_entity: failed to create Trimesh for %s: %s",mesh .name ,e )
            return None ,None 

    def _rebuild_viewport_from_project (self )->None :
        if not pv :
            return 
        self ._remove_affine_widget ()
        self ._affine_widget_mesh_id =None 
        self ._mesh_actor_by_id .clear ()
        self ._mesh_polydata_by_id .clear ()
        failed :list [str ]=[]
        for mesh in self ._app .project .source_data .meshes :
            tri ,normals =self ._load_trimesh_for_entity (mesh )
            if tri is not None :
                poly =self ._trimesh_to_polydata (tri ,normals )
                if poly :
                    self ._mesh_polydata_by_id [mesh .mesh_id ]=poly 
                else :
                    failed .append (f"{mesh .name }: PolyData conversion failed")
            else :
                failed .append (f"{mesh .name }: failed to load geometry (no embedded mesh_data)")
        if failed :
            logger .warning ("_rebuild_viewport_from_project: %d mesh(es) failed to load: %s",len (failed ),failed )
        self .mesh_viewport_changed .emit ()

    def _ensure_materials_in_library (self )->None :
        'Checks all meshes: materials must be in the library. Missing ones are added.'
        lib =self ._app .material_library 
        for mesh in self ._app .project .source_data .meshes :
            key =(mesh .material_key or "membrane").strip ()
            if not key :
                key ="membrane"
            found =any (m .name .lower ()==key .lower ()for m in lib .materials )
            if not found :
                d =mesh .properties 
                actual =lib .ensure_material (
                name =key ,
                density =float (d .get ("density",1380.0 )),
                E_parallel =float (d .get ("E_parallel",d .get ("young_modulus",5.0e9 ))),
                E_perp =float (d .get ("E_perp",3.5e9 )),
                poisson =float (d .get ("poisson",0.30 )),
                Cd =float (d .get ("Cd",1.0 )),
                eta_visc =float (d .get ("eta_visc",0.8 )),
                acoustic_impedance =float (d .get ("acoustic_impedance",d .get ("coupling_gain",1e6 ))),
                acoustic_inject =float (
                d .get ("acoustic_inject",1.0 if key .lower ()=="membrane"else 1.0 )
                ),
                )
                if actual !=key :
                    mesh .material_key =actual 
        if self ._app_controller :
            self ._app_controller .notify_material_library_changed ()
        else :
            self ._app .notify_material_library_changed ()
        self ._refresh_material_options ()

    def _load_project_to_ui (self )->None :
        self ._is_loading_ui =True 
        try :
            self ._ensure_materials_in_library ()
            sim =self ._app .project .source_data .simulation_settings 
            self .simulation .set_settings ({
            "dt":sim .dt ,
            "duration":sim .duration ,
            "air_coupling_gain":sim .air_coupling_gain ,
            "air_grid_step_mm":sim .air_grid_step_mm ,
            "air_pressure_history_every_steps":sim .air_pressure_history_every_steps ,
            "force_shape":sim .force_shape ,
            "excitation_mode":sim .excitation_mode ,
            "force_amplitude_pa":sim .force_amplitude_pa ,
            "force_freq_hz":sim .force_freq_hz ,
            })
            md =self ._app .project .source_data .metadata 
            bc =md .get ("boundary_defaults",{})
            fixed =str (bc .get ("fixed","FIXED_EDGE"))
            self .mesh_editor .set_fixed_edge_options (["none",fixed ,"FIXED_ALL"])
            self ._refresh_material_options ()
            self ._update_window_title ()
            self ._refresh_mesh_list ()
            self ._refresh_bc_list ()
            if self ._app .project .source_data .meshes :
                self .mesh_list .set_selection_by_row (0 )
            else :
                self ._app .set_selection (None )
            self ._rebuild_viewport_from_project ()
        finally :
            self ._is_loading_ui =False 

    def _apply_simulation_to_model (self )->None :
        data =self .simulation .get_settings ()
        sim =self ._app .project .source_data .simulation_settings 
        sim .dt =data ["dt"]
        sim .duration =data ["duration"]
        sim .air_coupling_gain =data ["air_coupling_gain"]
        sim .air_grid_step_mm =data ["air_grid_step_mm"]
        sim .air_pressure_history_every_steps =int (data .get ("air_pressure_history_every_steps",10 ))
        sim .force_shape =data ["force_shape"]
        sim .excitation_mode =str (data .get ("excitation_mode","external"))
        sim .force_amplitude_pa =data ["force_amplitude_pa"]
        sim .force_freq_hz =data ["force_freq_hz"]
        fixed =self .mesh_editor .cb_fixed_edge .currentText ()
        if fixed =="none":
            fixed ="FIXED_EDGE"
        self ._app .project .source_data .metadata ["boundary_defaults"]={
        "fixed":fixed ,
        }
        self .mesh_editor .set_fixed_edge_options (["none",fixed ,"FIXED_ALL"])
        self ._app .touch ()

    def _on_simulation_connect_requested (self ,host :str ,port :int ,use_local_server :bool )->None :
        """Handle Connect: start local server if needed, then connect client."""
        if use_local_server :
            self ._start_local_simulation_server (port )
        ok =self ._sim_client .connect_to_server (host ,port )
        if not ok :
            self .simulation .append_console (f"[UI] Failed to connect to {host }:{port }\n")
            self .simulation .set_connection_status (False ,"Connection failed")

    def _on_simulation_disconnect_requested (self )->None :
        """Handle Disconnect: disconnect client, stop local server."""
        self ._sim_client .disconnect_from_server ()
        self ._stop_local_simulation_server ()
        self .simulation .set_connection_status (False ,"Disconnected")

    def _start_local_simulation_server (self ,port :int )->None :
        """Start simulation server in background thread (for Local mode)."""
        if self ._sim_server_thread and self ._sim_server_thread .is_alive ():
            return 
        try :
            from simulation_server import run_server 
        except ImportError :
            self .simulation .append_console ("[UI] simulation_server module not found. Run server manually.\n")
            return 
        self ._sim_server_thread =threading .Thread (
        target =run_server ,
        kwargs ={"host":"127.0.0.1","port":port },
        daemon =True ,
        )
        self ._sim_server_thread .start ()
        import time 
        time .sleep (0.3 )# Give server time to bind

    def _stop_local_simulation_server (self )->None :
        """Stop local server thread (server runs until process exit; we just clear ref)."""
        self ._sim_server_thread =None 

    def _on_sim_client_log (self ,text :str )->None :
        self .simulation .append_console (text )

    def _on_sim_client_status (self ,state :str ,message :str )->None :
        if state =="running":
            self .simulation .set_running (True )
            self .simulation .set_connection_status (True ,"Connected (running)")
        elif state in ("finished","error","stopping","stopped"):
            self .simulation .set_running (False )
            status ="Connected"if self ._sim_client .is_connected ()else "Disconnected"
            self .simulation .set_connection_status (self ._sim_client .is_connected (),status )

    def _on_sim_client_results (self ,data :dict )->None :
        sim_data =SimulationResultsData .from_packed_dict (data )
        if sim_data .has_time_data ():
            self .results .set_results (sim_data )
            self .results .setVisible (True )
            self .results .raise_ ()
            self .results .activateWindow ()
        else :
            self .results .set_results (None )
        if sim_data .has_time_data ():
            try :
                self ._last_sim_results_dict =dict (data )
            except Exception :
                self ._last_sim_results_dict =None 
        else :
            self ._last_sim_results_dict =None 

    def _on_sim_client_connected_changed (self ,connected :bool )->None :
        if connected :
            self .simulation .set_connection_status (True ,"Connected")
        else :
            self .simulation .set_connection_status (False ,"Disconnected")

    def _action_run_simulation (self )->None :
        if not self ._sim_client .is_connected ():
            self .simulation .append_console ("[UI] Connect to simulation server first (Local or Remote).\n")
            return 
        if self .simulation .btn_run .isEnabled ()is False :
            self .simulation .append_console ("[UI] Simulation already running.\n")
            return 
        self ._apply_simulation_to_model ()
        params =self .simulation .get_settings ()
        if (
            str (params .get ("excitation_mode",""))=="external_velocity_override"
            and float (params .get ("force_amplitude_pa",0.0 ))<=0.0
        ):
            self .simulation .append_console (
                "[UI] external_velocity_override requires Velocity > 0 m/s.\n"
            )
            return
        material_library =None 
        if self ._app and hasattr (self ._app ,"material_library"):
            try :
                lib =self ._app .material_library .to_numpy_array ()
                if lib is not None and lib .size >0 :
                    material_library =lib .tolist ()
            except Exception :
                pass 
        topology =None 
        if self ._app and hasattr (self ._app ,"get_generated_topology"):
            topology =self ._app .get_generated_topology ()
        if topology is not None :
            params =dict (params )
            params ["topology"]=topology 
        self ._last_run_params_snapshot =copy .deepcopy (params )
        self ._last_run_material_library =copy .deepcopy (material_library )if material_library else []
        ok =self ._sim_client .run_simulation (params ,material_library )
        if not ok :
            self .simulation .append_console ("[UI] Failed to send run command.\n")
        else :
            self .simulation .append_console ("[UI] Run command sent to server.\n")

    def _finish_simulation_run (self )->None :
        """Called when in-process simulation completes (from worker via bridge signal)."""
        self .simulation .set_running (False )
        self ._sim_thread =None 
        log =getattr (self ,"_sim_log_text","")
        if log :
            self .simulation .append_console ("\n[Simulation output]\n"+log )
        sim_data =getattr (self ,"_sim_results_data",None )
        self .simulation .append_console (f"[UI] Simulation finished. Results: {'OK'if sim_data and sim_data .has_time_data ()else 'no data'}\n")
        if sim_data and sim_data .has_time_data ():
            self .results .set_results (sim_data )
            self .results .setVisible (True )
            self .results .raise_ ()
            self .results .activateWindow ()
            self ._last_sim_results_dict =sim_data .to_results_dict ()
        else :
            self .results .set_results (None )
            self ._last_sim_results_dict =None 

    def _on_simulation_finished (self ,data :SimulationResultsData )->None :
        """Handle simulation_finished signal (legacy / alternative path)."""
        if data :
            self .results .set_results (data )
            self .results .setVisible (True )
            self .results .raise_ ()
            self .results .activateWindow ()
            if data .has_time_data ():
                self ._last_sim_results_dict =data .to_results_dict ()

    def _action_export_simulation_results (self )->None :
        if not self ._last_sim_results_dict :
            QMessageBox .information (
            self ,
            "Export results",
            "No results to export. Run a simulation and wait until results arrive.",
            )
            return 
        path ,selected_filter =QFileDialog .getSaveFileName (
        self ,
        "Export simulation results",
        "",
        "Network JSON (*.json);;Pickle (*.pkl)",
        )
        if not path :
            return 
        try :
            from simulation_io import save_results_pickle ,save_results_wire_json 
        except ImportError as e :
            QMessageBox .warning (self ,"Export results",f"simulation_io not available: {e }")
            return 
        p =Path (path )
        use_json =p .suffix .lower ()==".json"or ("json"in selected_filter .lower ()and p .suffix .lower ()!=".pkl")
        try :
            if use_json :
                if p .suffix .lower ()!=".json":
                    p =p .with_suffix (".json")
                save_results_wire_json (p ,self ._last_sim_results_dict )
            else :
                if p .suffix .lower ()!=".pkl":
                    p =p .with_suffix (".pkl")
                save_results_pickle (p ,self ._last_sim_results_dict )
        except Exception as ex :
            QMessageBox .critical (self ,"Export results",str (ex ))
            return 
        self .simulation .append_console (f"[UI] Exported results to {p }\n")

    def _action_export_run_case (self )->None :
        if not self ._last_run_params_snapshot :
            QMessageBox .information (
            self ,
            "Export run case",
            "No run case yet. Press Run Simulation once (command must be sent to the server).",
            )
            return 
        path ,_ =QFileDialog .getSaveFileName (
        self ,
        "Export run case (CLI replay)",
        "",
        "Run case pickle (*.pkl)",
        )
        if not path :
            return 
        try :
            from simulation_io import save_run_case_pickle 
        except ImportError as e :
            QMessageBox .warning (self ,"Export run case",f"simulation_io not available: {e }")
            return 
        p =Path (path )
        if p .suffix .lower ()!=".pkl":
            p =p .with_suffix (".pkl")
        params =copy .deepcopy (self ._last_run_params_snapshot )
        topology =params .pop ("topology",None )
        mat_lib =self ._last_run_material_library or []
        try :
            save_run_case_pickle (p ,params ,mat_lib ,topology )
        except Exception as ex :
            QMessageBox .critical (self ,"Export run case",str (ex ))
            return 
        self .simulation .append_console (f"[UI] Exported run case to {p } (python diaphragm_opencl.py --sim-file ...)\n")

    def _action_stop_simulation (self )->None :
        if self ._sim_client .is_connected ():
            self ._sim_client .stop_simulation ()
            self .simulation .append_console ("[UI] Stop requested (current run will complete).\n")
        self .simulation .set_running (False )

    def _action_debug_test_run (self )->None :
        import threading 
        import io 
        import contextlib 
        import traceback 
        import numpy as np 

        if self ._debug_thread and self ._debug_thread .is_alive ():
            QMessageBox .information (self ,"Debug Test Run","Debug simulation is already running.")
            return 
        self .simulation .append_console ("[UI] Starting debug test run in background...\n")

        def worker ():
            try :
                import diaphragm_opencl as cl_model 
                log_buf =io .StringIO ()
                history =None 
                try :
                    argv =["diaphragm_opencl.py","--no-plot","--dt","1e-7","--duration","0.001",
                    "--force-shape","impulse","--force-amplitude","0.001","--force-freq","200",
                    "--force-freq-end","5000","--debug"]
                    material_rows =None 
                    if self ._app and hasattr (self ._app ,"material_library"):
                        try :
                            lib =self ._app .material_library .to_numpy_array ()
                            if lib is not None and lib .size >0 :
                                material_rows =lib .tolist ()
                        except Exception :
                            material_rows =None 
                    with contextlib .redirect_stdout (log_buf ),contextlib .redirect_stderr (log_buf ):
                        args =cl_model ._parse_cli_args (argv )
                        model ,_ =cl_model .run_cli_simulation (args ,material_library_rows =material_rows )
                        history =list (model .history_disp_all )if getattr (model ,"history_disp_all",None )else []
                except Exception :
                    log_buf .write ("\n[Debug] Exception:\n"+traceback .format_exc ())
                self ._debug_history_disp_all =history or []
                self ._debug_log_text =log_buf .getvalue ()
                self .debug_test_run_finished .emit ()
            finally :
                self ._debug_thread =None 

        self ._debug_thread =threading .Thread (target =worker ,daemon =True )
        self ._debug_thread .start ()

    def _finish_debug_test_run (self )->None :
        if self ._debug_log_text :
            self .simulation .append_console ("\n[Debug Test Run]\n"+self ._debug_log_text )
        n =len (self ._debug_history_disp_all or [])
        self .simulation .append_console (f"\n[UI] Debug test run finished. Frames: {n }\n")

    def _action_debug_test_visualization (self )->None :
        if not has_pyvista ()or not self ._plotter :
            QMessageBox .information (self ,"Test Visualization","PyVista viewport not available.")
            return 
        frames =self ._debug_history_disp_all or []
        if not frames :
            QMessageBox .warning (self ,"Test Visualization","No debug history. Run Debug → Test Run first.")
            return 
        import numpy as np 
        first =np .asarray (frames [0 ],dtype =np .float64 )
        if first .ndim !=2 :
            QMessageBox .warning (self ,"Test Visualization","history_disp_all is not 2D.")
            return 
        ny ,nx =first .shape 
        xs ,ys =np .meshgrid (np .linspace (-0.5 ,0.5 ,nx ),np .linspace (-0.5 ,0.5 ,ny ),indexing ="ij")
        scale_z =1e10 
        z0 =(first .T *scale_z ).astype (np .float64 )
        grid =pv .StructuredGrid (xs ,ys ,z0 )
        grid ["uz"]=first .T .ravel (order ="F")
        self ._debug_surface =grid 
        if self ._debug_anim_timer :
            self ._debug_anim_timer .stop ()
            self ._debug_anim_timer .deleteLater ()
        if self ._mesh_viewport and self ._plotter :
            self ._mesh_viewport .remove_extra_actor ("__debug_surface__")
            actor =self ._plotter .add_mesh (grid ,name ="debug_surface",scalars ="uz",cmap ="RdBu",show_edges =False )
            self ._mesh_viewport .add_extra_actor ("__debug_surface__",actor ,already_in_scene =True )
        self ._plotter .reset_camera ()
        self ._plotter .render ()
        self ._debug_anim_frame_idx =0 

        def update_frame ():
            if not frames :
                return 
            idx =self ._debug_anim_frame_idx %len (frames )
            frame =np .asarray (frames [idx ],dtype =np .float64 )
            if frame .shape !=(ny ,nx ):
                return 
            z =(frame .T *scale_z ).astype (np .float64 )
            pts =self ._debug_surface .points 
            pts [:,2 ]=z .ravel (order ="F")
            self ._debug_surface .points =pts 
            self ._debug_surface ["uz"]=frame .T .ravel (order ="F")
            self ._debug_anim_frame_idx =(self ._debug_anim_frame_idx +1 )%len (frames )
            self ._plotter .render ()

        self ._debug_anim_timer =QTimer (self )
        self ._debug_anim_timer .timeout .connect (update_frame )
        self ._debug_anim_timer .start (1 )

    def _action_new_project (self )->None :
        if not self ._confirm_save_if_dirty ():
            return 
        self ._app .new_project ()

    def _action_load_project (self )->None :
        if not self ._confirm_save_if_dirty ():
            return 
        fp ,_ =QFileDialog .getOpenFileName (self ,"Load Project","",f"Project (*{PROJECT_EXT });;Legacy (*.json);;All (*.*)")
        if not fp :
            return 
        try :
            self ._app .load_project (fp )
        except Exception as e :
            QMessageBox .critical (self ,"Load Error",str (e ))
            return 

    def _action_save_project (self )->None :
        self ._save_internal (force_save_as =True )# True: show Save As dialog when no path

    def _action_save_project_as (self )->None :
        prev =self ._app .project_path 
        self ._app .set_project_path (None )
        if not self ._save_internal (force_save_as =True ):
            self ._app .set_project_path (prev )
        self ._update_window_title ()

    def _action_new_window (self )->None :
        """Create new window with empty project."""
        if self ._app_controller :
            self ._app_controller .new_window ()

    def _action_open_in_new_window (self )->None :
        """Open project in new window."""
        if not self ._app_controller :
            return 
        fp ,_ =QFileDialog .getOpenFileName (self ,"Open in New Window","",f"Project (*{PROJECT_EXT });;Legacy (*.json);;All (*.*)")
        if not fp :
            return 
        try :
            self ._app_controller .new_window (load_path =fp )
        except Exception as e :
            QMessageBox .critical (self ,"Open Error",str (e ))

    def _save_internal (self ,force_save_as :bool )->bool :
        self ._apply_simulation_to_model ()
        if self ._selected_mesh_index ()is not None :
            self ._apply_mesh_editor_to_model ()
        if self ._app .project_path is None :
            if not force_save_as :
                return False 
            fp ,_ =QFileDialog .getSaveFileName (self ,"Save As",f"project{PROJECT_EXT }",f"Project (*{PROJECT_EXT });;All (*.*)")
            if not fp :
                return False 
            if not fp .lower ().endswith (PROJECT_EXT ):
                fp +=PROJECT_EXT 
            self ._app .set_project_path (Path (fp ))
        try :
            if not self ._app .save_project ():
                return False 
        except Exception as e :
            QMessageBox .critical (self ,"Save Error",str (e ))
            return False 
        self ._update_window_title ()
        return True 

    def _confirm_save_if_dirty (self )->bool :
        if not self ._app .is_dirty :
            return True 
        r =QMessageBox .question (
        self ,"Unsaved Changes",
        "Project has unsaved changes. Save before continuing?",
        QMessageBox .Save |QMessageBox .Discard |QMessageBox .Cancel ,
        QMessageBox .Save ,
        )
        if r ==QMessageBox .Cancel :
            return False 
        if r ==QMessageBox .Discard :
            return True 
        return self ._save_internal (force_save_as =True )

    def _mark_dirty (self )->None :
        if self ._is_loading_ui :
            return 
        self ._app .touch ()

    def _update_window_title (self )->None :
        path =str (self ._app .project_path )if self ._app .project_path else "unsaved"
        mark ="*"if self ._app .is_dirty else ""
        self .setWindowTitle (f"FE UI{mark } - {self ._app .project .name } ({path })")

    def closeEvent (self ,event )->None :
        if self ._sim_process and self ._sim_process .state ()!=QProcess .NotRunning :
            self ._sim_process .kill ()
        if self ._confirm_save_if_dirty ():
            event .accept ()
        else :
            event .ignore ()
