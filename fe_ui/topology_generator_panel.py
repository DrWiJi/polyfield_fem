# -*- coding: utf-8 -*-
'Window for a 3D topology generator from model meshes.\nSample design: BoundaryConditionsPanel.\nViewport shows only the generated topology (meshes are not displayed).'

from __future__ import annotations 

from datetime import datetime 

from PySide6 .QtCore import Qt 
from PySide6 .QtGui import QCloseEvent ,QShowEvent 
from PySide6 .QtWidgets import (
QApplication ,
QCheckBox ,
QComboBox ,
QDockWidget ,
QFormLayout ,
QGroupBox ,
QHBoxLayout ,
QLabel ,
QPlainTextEdit ,
QPushButton ,
QSpinBox ,
QSlider ,
QVBoxLayout ,
QWidget ,
)
from .viewport import TopologyViewport 
from .widgets import ScientificDoubleSpinBox 
from topology_generator import generate_topology_from_meshes 


class TopologyGeneratorPanel (QDockWidget ):
    '3D topology generator window. Separate ViewPort for viewing the topology.'

    def __init__ (self ,parent =None )->None :
        super ().__init__ ("Topology Generator",parent )
        self .setAllowedAreas (Qt .LeftDockWidgetArea |Qt .RightDockWidgetArea )

        widget =QWidget ()
        main_layout =QHBoxLayout (widget )

        # Viewport - topology only
        viewport_container =QWidget ()
        viewport_layout =QVBoxLayout (viewport_container )
        viewport_layout .setContentsMargins (0 ,0 ,0 ,0 )
        self ._topology_viewport =TopologyViewport (self )
        self ._topology_viewport .setMinimumSize (320 ,240 )
        viewport_layout .addWidget (self ._topology_viewport ,1 )

        # Layer Slider: Hides FEs with a Z center above the value
        self ._layer_slider =QSlider ()
        self ._layer_slider .setOrientation (Qt .Horizontal )
        self ._layer_slider .setRange (0 ,100 )
        self ._layer_slider .setValue (100 )
        self ._layer_slider .setEnabled (False )
        self ._layer_z_min =0.0 
        self ._layer_z_max =1.0 
        self ._layer_slider_label =QLabel ('Layers: all')
        viewport_layout .addWidget (self ._layer_slider_label )
        viewport_layout .addWidget (self ._layer_slider )

        # Level of detail: 1:N — render every Nth CE
        lod_layout =QHBoxLayout ()
        lod_layout .addWidget (QLabel ('Detail:'))
        self ._lod_combo =QComboBox ()
        self ._lod_combo .addItems (['1:1 (all)',"1:2","1:3","1:4","1:5","1:10"])
        self ._lod_combo .setCurrentIndex (0 )# 1:1 default (avoid misleading apparent cell size)
        self ._lod_combo .currentIndexChanged .connect (self ._on_lod_changed )
        lod_layout .addWidget (self ._lod_combo )
        viewport_layout .addLayout (lod_layout )

        self ._topology_viewport .layer_range_changed .connect (self ._on_layer_range_changed )
        self ._layer_slider .valueChanged .connect (self ._on_layer_slider_changed )

        main_layout .addWidget (viewport_container ,1 )

        # Right panel: options and button
        params_group =QGroupBox ('Generation parameters')
        params_layout =QFormLayout (params_group )

        self .sp_element_size_mm =ScientificDoubleSpinBox ()
        self .sp_element_size_mm .setRange (0.01 ,100.0 )
        self .sp_element_size_mm .setDecimals (3 )
        self .sp_element_size_mm .setValue (0.5 )
        self .sp_element_size_mm .setSuffix (" mm")
        params_layout .addRow ('FE size (voxel):',self .sp_element_size_mm )

        self .sp_padding_mm =ScientificDoubleSpinBox ()
        self .sp_padding_mm .setRange (0.0 ,100.0 )
        self .sp_padding_mm .setDecimals (3 )
        self .sp_padding_mm .setValue (0.0 )
        self .sp_padding_mm .setSuffix (" mm")
        params_layout .addRow ('Indent from bbox:',self .sp_padding_mm )

        self .chk_generate_air =QCheckBox ('Generate air FE grid')
        self .chk_generate_air .setChecked (True )
        params_layout .addRow ('Air grid:',self .chk_generate_air )

        self .sp_max_air_cells =QSpinBox ()
        self .sp_max_air_cells .setRange (10_000 ,2_000_000_000 )
        self .sp_max_air_cells .setSingleStep (100_000 )
        self .sp_max_air_cells .setValue (1_200_000 )
        self .sp_max_air_cells .setToolTip ('Safety cap: if exceeded, air step is increased automatically')
        params_layout .addRow ('Max air cells:',self .sp_max_air_cells )

        self .btn_generate =QPushButton ('Generate topology')
        self .btn_generate .clicked .connect (self ._on_generate )
        self .btn_draw =QPushButton ('Draw topology')
        self .btn_draw .clicked .connect (self ._on_draw_topology )
        self .btn_draw .setEnabled (False )

        log_group =QGroupBox ('Generation log')
        self .log_text =QPlainTextEdit ()
        self .log_text .setReadOnly (True )
        self .log_text .setMinimumHeight (120 )
        self .log_text .setPlaceholderText ('Click "Generate Topology" to run...')
        log_layout =QVBoxLayout (log_group )
        log_layout .addWidget (self .log_text )

        right_layout =QVBoxLayout ()
        right_layout .addWidget (params_group )
        btn_row =QHBoxLayout ()
        btn_row .addWidget (self .btn_generate )
        btn_row .addWidget (self .btn_draw )
        right_layout .addLayout (btn_row )
        right_layout .addWidget (log_group ,1 )

        right_widget =QWidget ()
        right_widget .setLayout (right_layout )
        right_widget .setFixedWidth (280 )
        main_layout .addWidget (right_widget ,0 )

        self .setWidget (widget )
        self ._main_window =parent 
        self ._refresh_from_model ()

    def closeEvent (self ,event :QCloseEvent )->None :
        if hasattr (self ._topology_viewport ,"close_viewport"):
            self ._topology_viewport .close_viewport ()
        super ().closeEvent (event )
        if self ._main_window and hasattr (self ._main_window ,"_on_topology_generator_closed"):
            self ._main_window ._on_topology_generator_closed ()

    def _get_mesh_data (self ):
        'Get polydata and meshes from the main window.'
        if self ._main_window and hasattr (self ._main_window ,"_mesh_polydata_by_id")and hasattr (self ._main_window ,"_app"):
            return (
            self ._main_window ._mesh_polydata_by_id .copy (),
            self ._main_window ._app .project .source_data .meshes ,
            )
        return {},[]

    def _get_load_mesh_fn (self ):
        if self ._main_window and hasattr (self ._main_window ,"_load_trimesh_for_entity"):
            def _load (m ):
                tri ,_ =self ._main_window ._load_trimesh_for_entity (m )
                return tri 
            return _load 
        return lambda m :None 

    def _get_material_key_to_index (self )->dict [str ,int ]:
        'Mapping material_key -> index in material_props for diaphragm_opencl.'
        mat_map ={}
        if self ._main_window and hasattr (self ._main_window ,"_app"):
            lib =getattr (self ._main_window ._app ,"material_library",None )
            if lib and hasattr (lib ,"ensure_material"):
                try :
                    lib .ensure_material (
                    "air",
                    density =1.225 ,
                    E_parallel =1.42e5 ,
                    E_perp =1.42e5 ,
                    poisson =0.0 ,
                    Cd =0.0 ,
                    eta_visc =1.8e-5 ,
                    coupling_gain =1.00 ,
                    acoustic_inject =0.0 ,
                    )
                except Exception :
                    pass
            if lib and hasattr (lib ,"materials"):
                for i ,m in enumerate (lib .materials ):
                    key =getattr (m ,"name",str (m )).lower ().strip ()
                    if key :
                        mat_map [key ]=i 
        if not mat_map :
            from topology_generator import MAT_AIR ,MAT_MEMBRANE ,MAT_FOAM_VE3015 ,MAT_SENSOR 
            mat_map ={
            "membrane":int (MAT_MEMBRANE ),
            "foam_ve3015":int (MAT_FOAM_VE3015 ),
            "sensor":int (MAT_SENSOR ),
            "air":int (MAT_AIR ),
            }
        elif "air"not in mat_map and mat_map :
            mat_map ["air"]=max (mat_map .values ()) +1
        return mat_map 

    def _log (self ,msg :str )->None :
        'Add a line to the log with a timestamp. Immediately updates the UI.'
        ts =datetime .now ().strftime ("%H:%M:%S")
        self .log_text .appendPlainText (f"[{ts }] {msg }")
        self .log_text .verticalScrollBar ().setValue (self .log_text .verticalScrollBar ().maximum ())
        QApplication .processEvents ()

    def _on_layer_range_changed (self ,z_min :float ,z_max :float )->None :
        'Update slider range when changing topology.'
        self ._layer_z_min =z_min 
        self ._layer_z_max =z_max 
        self ._layer_slider .setEnabled (True )
        self ._layer_slider .setValue (100 )
        self ._topology_viewport .set_layer_cutoff (None )
        self ._layer_slider_label .setText (f"Layers: Z ≤ {z_max :.3f} (all)")

    def _lod_value (self )->int :
        'Current LOD value from the combo box (1, 2, 3, 4, 5, 10).'
        idx =self ._lod_combo .currentIndex ()
        return [1 ,2 ,3 ,4 ,5 ,10 ][idx ]

    def _on_lod_changed (self )->None :
        'Update display detail.'
        self ._topology_viewport .set_lod (self ._lod_value ())

    def _on_layer_slider_changed (self ,value :int )->None :
        'Hide FEs with a Z center higher than the value.'
        if value >=100 :
            self ._topology_viewport .set_layer_cutoff (None )
            self ._layer_slider_label .setText (f"Layers: Z ≤ {self ._layer_z_max :.3f} (all)")
        else :
            z_cutoff =self ._layer_z_min +(self ._layer_z_max -self ._layer_z_min )*(value /100.0 )
            self ._topology_viewport .set_layer_cutoff (z_cutoff )
            self ._layer_slider_label .setText (f"Layers: Z ≤ {z_cutoff :.3f}")

    def _refresh_from_model (self )->None :
        """Load topology metadata from app model, but do not auto-render."""
        topo =None 
        if self ._main_window and hasattr (self ._main_window ,"_app"):
            topo =self ._main_window ._app .get_generated_topology ()
        self ._topology_viewport .set_lod (self ._lod_value ())
        self ._topology_viewport .set_topology (None )
        self .btn_draw .setEnabled (topo is not None )
        pos =topo .get ("element_position_xyz")if topo else None 
        if topo is None or pos is None or getattr (pos ,"size",0 )==0 :
            self ._layer_slider .setEnabled (False )
            self ._layer_slider_label .setText ('Layers: -')
        else :
            self ._layer_slider .setEnabled (False )
            self ._layer_slider_label .setText ('Layers: press "Draw topology"')

    def showEvent (self ,event :QShowEvent )->None :
        """Refresh topology display when window is shown."""
        super ().showEvent (event )
        self ._refresh_from_model ()

    def _on_generate (self )->None :
        polydata_by_id ,meshes =self ._get_mesh_data ()
        self .log_text .clear ()
        if not meshes :
            self ._log ('There are no meshes in the project.')
            return 

        element_size_mm =self .sp_element_size_mm .value ()
        padding_mm =self .sp_padding_mm .value ()
        generate_air_grid =bool (self .chk_generate_air .isChecked ())
        max_air_cells =int (self .sp_max_air_cells .value ())

        boundary_conditions =[]
        if self ._main_window and hasattr (self ._main_window ,"_app"):
            boundary_conditions =getattr (
            self ._main_window ._app .project .source_data ,
            "boundary_conditions",
            [],
            )or []

        try :
            topology =generate_topology_from_meshes (
            meshes ,
            polydata_by_id ,
            self ._get_load_mesh_fn (),
            element_size_mm =element_size_mm ,
            padding_mm =padding_mm ,
            generate_air_grid =generate_air_grid ,
            max_air_cells =max_air_cells ,
            material_key_to_index =self ._get_material_key_to_index (),
            boundary_conditions =boundary_conditions ,
            log_callback =self ._log ,
            )
        except NotImplementedError as e :
            self ._log (f"Error: {e }")
            return 
        except Exception as e :
            import traceback 
            self ._log (f"Error: {e }")
            self ._log (traceback .format_exc ())
            return 

        n =topology ["element_position_xyz"].shape [0 ]
        if self ._main_window and hasattr (self ._main_window ,"_app"):
            self ._main_window ._app .set_generated_topology (topology )
            self ._main_window ._app .touch ()
        self ._topology_viewport .set_topology (None )
        self .btn_draw .setEnabled (True )
        n_air =int (topology .get ("air_element_position_xyz",[]).shape [0 ])
        self ._layer_slider .setEnabled (False )
        self ._layer_slider_label .setText ('Layers: press "Draw topology"')
        self ._log (f"Topology saved to project ({n } solid elements, {n_air } air elements). Click \"Draw topology\" to render.")

    def _on_draw_topology (self )->None :
        topo =None
        if self ._main_window and hasattr (self ._main_window ,"_app"):
            topo =self ._main_window ._app .get_generated_topology ()
        if topo is None :
            self ._log ('There is no generated topology to draw.')
            return
        self ._topology_viewport .set_lod (self ._lod_value ())
        self ._topology_viewport .set_topology (topo )
