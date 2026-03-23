# -*- coding: utf-8 -*-
"""
Mesh parameter editor dock with Identity, Material, Membrane, Transform tabs.
Depends: PySide6 only. Uses dict for mesh data — no project_model.
"""

from __future__ import annotations 

from PySide6 .QtCore import Signal 
from PySide6 .QtWidgets import (
QCheckBox ,
QComboBox ,
QDockWidget ,
QFormLayout ,
QGridLayout ,
QGroupBox ,
QHBoxLayout ,
QLabel ,
QLineEdit ,
QPushButton ,
QTabWidget ,
QTextEdit ,
QVBoxLayout ,
QWidget ,
)
from .constants import ROLES 
from .widgets import ScientificDoubleSpinBox 


class MeshEditorPanel (QDockWidget ):
    """Parameter editor with tabs. Emits apply/reset, holds form state."""

    apply_clicked =Signal ()
    reset_clicked =Signal ()
    data_changed =Signal ()# any field change (for dirty tracking)

    def __init__ (self ,parent =None )->None :
        super ().__init__ ("Mesh Parameter Editor",parent )
        from PySide6 .QtCore import Qt 
        self .setAllowedAreas (Qt .LeftDockWidgetArea |Qt .RightDockWidgetArea )

        panel =QWidget ()
        layout =QVBoxLayout (panel )

        self .info_label =QLabel ("No mesh selected")
        self .info_label .setStyleSheet ("color: #888;")

        self .tabs =QTabWidget ()
        self .tabs .addTab (self ._build_identity_tab (),"Identity")
        self .tabs .addTab (self ._build_material_tab (),"Material")
        self ._membrane_tab =self ._build_membrane_tab ()
        self .tabs .addTab (self ._membrane_tab ,"Membrane")
        self .tabs .addTab (self ._build_transform_tab (),"Transform")

        btn_row =QHBoxLayout ()
        self .btn_reset =QPushButton ("Reset")
        self .btn_reset .clicked .connect (self .reset_clicked .emit )
        btn_row .addWidget (self .btn_reset )

        layout .addWidget (self .info_label )
        layout .addWidget (self .tabs ,1 )
        layout .addLayout (btn_row )

        self .setWidget (panel )
        self .set_enabled (False )# disabled until mesh is selected
        self ._material_provider =None # callable returning MaterialLibraryModel
        self ._loading_data =False 

    def _build_identity_tab (self )->QWidget :
        w =QWidget ()
        form =QFormLayout (w )
        self .ed_name =QLineEdit ()
        self .cb_role =QComboBox ()
        self .cb_role .addItems (list (ROLES ))
        self .cb_role .currentTextChanged .connect (self ._on_role_changed )
        self .cb_visible =QCheckBox ()
        self .ed_notes =QTextEdit ()
        self .ed_notes .setPlaceholderText ("Notes")
        self .ed_notes .setMaximumHeight (60 )
        form .addRow ("Name",self .ed_name )
        form .addRow ("Role",self .cb_role )
        form .addRow ("Visible",self .cb_visible )
        form .addRow ("Notes",self .ed_notes )
        return w 

    def _build_material_tab (self )->QWidget :
        w =QWidget ()
        form =QFormLayout (w )
        self .cb_material =QComboBox ()
        self .cb_material .currentTextChanged .connect (self ._on_material_preset_changed )
        self .sp_density =ScientificDoubleSpinBox ()
        self .sp_density .setRange (1.0 ,50000.0 )
        self .sp_density .setValue (1380.0 )
        self .sp_density .setSuffix (" kg/m³")
        self .sp_E_parallel =ScientificDoubleSpinBox ()
        self .sp_E_parallel .setRange (1e3 ,2e12 )
        self .sp_E_parallel .setDecimals (0 )
        self .sp_E_parallel .setValue (5.0e9 )
        self .sp_E_parallel .setSuffix (" Pa")
        self .sp_E_perp =ScientificDoubleSpinBox ()
        self .sp_E_perp .setRange (1e3 ,2e12 )
        self .sp_E_perp .setDecimals (0 )
        self .sp_E_perp .setValue (3.5e9 )
        self .sp_E_perp .setSuffix (" Pa")
        self .sp_poisson =ScientificDoubleSpinBox ()
        self .sp_poisson .setRange (0.0 ,0.499 )
        self .sp_poisson .setSingleStep (0.01 )
        self .sp_poisson .setValue (0.30 )
        self .sp_Cd =ScientificDoubleSpinBox ()
        self .sp_Cd .setRange (0.5 ,2.0 )
        self .sp_Cd .setSingleStep (0.05 )
        self .sp_Cd .setValue (1.0 )
        self .sp_eta_visc =ScientificDoubleSpinBox ()
        self .sp_eta_visc .setRange (0.0 ,1000.0 )
        self .sp_eta_visc .setDecimals (2 )
        self .sp_eta_visc .setValue (0.8 )
        self .sp_eta_visc .setSuffix (" Pa·s")
        self .sp_coupling_gain =ScientificDoubleSpinBox ()
        self .sp_coupling_gain .setRange (0.0 ,1.0 )
        self .sp_coupling_gain .setSingleStep (0.05 )
        self .sp_coupling_gain .setValue (0.9 )
        form .addRow ("Material Preset",self .cb_material )
        mat_params =[
        self .sp_density ,self .sp_E_parallel ,self .sp_E_perp ,self .sp_poisson ,
        self .sp_Cd ,self .sp_eta_visc ,self .sp_coupling_gain ,
        ]
        for sp in mat_params :
            sp .setReadOnly (True )
        form .addRow ("Density",self .sp_density )
        form .addRow ("E_parallel",self .sp_E_parallel )
        form .addRow ("E_perp",self .sp_E_perp )
        form .addRow ("Poisson",self .sp_poisson )
        form .addRow ("Cd",self .sp_Cd )
        form .addRow ("η_visc",self .sp_eta_visc )
        form .addRow ("Coupling gain",self .sp_coupling_gain )
        return w 

    def _build_membrane_tab (self )->QWidget :
        w =QWidget ()
        form =QFormLayout (w )
        self .sp_thickness_mm =ScientificDoubleSpinBox ()
        self .sp_thickness_mm .setRange (0.001 ,50.0 )
        self .sp_thickness_mm .setDecimals (4 )
        self .sp_thickness_mm .setValue (0.012 )
        self .sp_thickness_mm .setSuffix (" mm")
        self .sp_pretension =ScientificDoubleSpinBox ()
        self .sp_pretension .setRange (0.0 ,10000.0 )
        self .sp_pretension .setValue (10.0 )
        self .sp_pretension .setSuffix (" N/m")
        self .cb_fixed_edge =QComboBox ()
        self .cb_fixed_edge .addItems (["none","FIXED_EDGE","FIXED_ALL"])
        form .addRow ("Thickness",self .sp_thickness_mm )
        form .addRow ("Pre-tension",self .sp_pretension )
        form .addRow ("Boundary Group",self .cb_fixed_edge )
        return w 

    def _build_transform_tab (self )->QWidget :
        w =QWidget ()
        form =QFormLayout (w )
        self .sp_tx =ScientificDoubleSpinBox ()
        self .sp_ty =ScientificDoubleSpinBox ()
        self .sp_tz =ScientificDoubleSpinBox ()
        for sp in (self .sp_tx ,self .sp_ty ,self .sp_tz ):
            sp .setRange (-1_000_000.0 ,1_000_000.0 )
            sp .setDecimals (6 )
            sp .setSingleStep (0.1 )
            sp .setValue (0.0 )

        self .sp_rx =ScientificDoubleSpinBox ()
        self .sp_ry =ScientificDoubleSpinBox ()
        self .sp_rz =ScientificDoubleSpinBox ()
        for sp in (self .sp_rx ,self .sp_ry ,self .sp_rz ):
            sp .setRange (-360.0 ,360.0 )
            sp .setDecimals (3 )
            sp .setSingleStep (1.0 )
            sp .setValue (0.0 )
            sp .setSuffix (" deg")

        self .sp_sx =ScientificDoubleSpinBox ()
        self .sp_sy =ScientificDoubleSpinBox ()
        self .sp_sz =ScientificDoubleSpinBox ()
        for sp in (self .sp_sx ,self .sp_sy ,self .sp_sz ):
            sp .setRange (0.001 ,1000.0 )
            sp .setDecimals (6 )
            sp .setSingleStep (0.01 )
            sp .setValue (1.0 )

        form .addRow ("Translate X",self .sp_tx )
        form .addRow ("Translate Y",self .sp_ty )
        form .addRow ("Translate Z",self .sp_tz )
        form .addRow ("Rotate X",self .sp_rx )
        form .addRow ("Rotate Y",self .sp_ry )
        form .addRow ("Rotate Z",self .sp_rz )
        form .addRow ("Scale X",self .sp_sx )
        form .addRow ("Scale Y",self .sp_sy )
        form .addRow ("Scale Z",self .sp_sz )
        return w 

    def _on_role_changed (self ,role :str )->None :
        self .set_membrane_tab_visible (role =="membrane")

    def set_material_provider (self ,provider )->None :
        """Set callable returning MaterialLibraryModel for preset lookup."""
        self ._material_provider =provider 

    def _refresh_material_params_from_library (self ,fallback :dict |None =None )->None :
        """Fill material params from library for current preset. Use fallback dict if material not found."""
        name =self .cb_material .currentText ().strip ()if self .cb_material else ""
        if not name and not fallback :
            return 
        mat =None 
        if name and self ._material_provider :
            lib =self ._material_provider ()
            if lib :
                mat =next ((m for m in lib .materials if m .name ==name ),None )
        spinboxes =[
        self .sp_density ,self .sp_E_parallel ,self .sp_E_perp ,self .sp_poisson ,
        self .sp_Cd ,self .sp_eta_visc ,self .sp_coupling_gain ,
        ]
        for sp in spinboxes :
            sp .blockSignals (True )
        try :
            if mat is not None :
                self .sp_density .setValue (mat .density )
                self .sp_E_parallel .setValue (mat .E_parallel )
                self .sp_E_perp .setValue (mat .E_perp )
                self .sp_poisson .setValue (mat .poisson )
                self .sp_Cd .setValue (mat .Cd )
                self .sp_eta_visc .setValue (mat .eta_visc )
                self .sp_coupling_gain .setValue (mat .coupling_gain )
            elif fallback :
                self .sp_density .setValue (float (fallback .get ("density",1380.0 )))
                self .sp_E_parallel .setValue (float (fallback .get ("E_parallel",fallback .get ("young_modulus",5.0e9 ))))
                self .sp_E_perp .setValue (float (fallback .get ("E_perp",3.5e9 )))
                self .sp_poisson .setValue (float (fallback .get ("poisson",0.30 )))
                self .sp_Cd .setValue (float (fallback .get ("Cd",1.0 )))
                self .sp_eta_visc .setValue (float (fallback .get ("eta_visc",0.8 )))
                self .sp_coupling_gain .setValue (float (fallback .get ("coupling_gain",0.9 )))
        finally :
            for sp in spinboxes :
                sp .blockSignals (False )

    def refresh_material_params_from_library (self )->None :
        """Public: refresh material params from library (e.g. when library changes)."""
        self ._refresh_material_params_from_library ()

    def _on_material_preset_changed (self ,name :str )->None :
        """When material preset selected, fill all material params from library."""
        if self ._loading_data or not name or not self ._material_provider :
            return 
        self ._refresh_material_params_from_library ()
        self .data_changed .emit ()

    def set_membrane_tab_visible (self ,visible :bool )->None :
        idx =self .tabs .indexOf (self ._membrane_tab )
        if idx >=0 :
            self .tabs .setTabVisible (idx ,visible )

    def set_enabled (self ,enabled :bool )->None :
        """Disable entire panel when no mesh selected — no tab switching, no editing."""
        w =self .widget ()
        if w :
            w .setEnabled (enabled )
        self .tabs .setEnabled (enabled )
        for i in range (self .tabs .count ()):
            self .tabs .setTabEnabled (i ,enabled )

    def set_info (self ,text :str )->None :
        self .info_label .setText (text )

    def get_data (self )->dict :
        """Return current form values as dict (for main_window to apply to model)."""
        group =self .cb_fixed_edge .currentText ()
        boundary_groups =[]if group =="none"else [group ]
        notes =self .ed_notes .toPlainText ().strip ()
        return {
        "name":self .ed_name .text ().strip (),
        "role":self .cb_role .currentText (),
        "material_key":self .cb_material .currentText (),
        "visible":bool (self .cb_visible .isChecked ()),
        "density":float (self .sp_density .value ()),
        "E_parallel":float (self .sp_E_parallel .value ()),
        "E_perp":float (self .sp_E_perp .value ()),
        "poisson":float (self .sp_poisson .value ()),
        "Cd":float (self .sp_Cd .value ()),
        "eta_visc":float (self .sp_eta_visc .value ()),
        "coupling_gain":float (self .sp_coupling_gain .value ()),
        "thickness_mm":float (self .sp_thickness_mm .value ()),
        "pre_tension_n_per_m":float (self .sp_pretension .value ()),
        "translation":[float (self .sp_tx .value ()),float (self .sp_ty .value ()),float (self .sp_tz .value ())],
        "rotation_euler_deg":[float (self .sp_rx .value ()),float (self .sp_ry .value ()),float (self .sp_rz .value ())],
        "scale":[float (self .sp_sx .value ()),float (self .sp_sy .value ()),float (self .sp_sz .value ())],
        "boundary_groups":boundary_groups ,
        "notes":notes if notes else None ,
        }

    def set_data (self ,data :dict )->None :
        """Load form from dict (from main_window, from MeshEntity).
        Material params are always taken from library; fallback to data if material not found."""
        self ._loading_data =True 
        try :
            self .ed_name .setText (data .get ("name",""))
            self .cb_role .setCurrentText (str (data .get ("role","solid")))
            self .cb_material .blockSignals (True )
            self .cb_material .setCurrentText (str (data .get ("material_key","membrane")))
            self .cb_material .blockSignals (False )
            self .cb_visible .setChecked (bool (data .get ("visible",True )))
            self ._refresh_material_params_from_library (fallback =data )
            self .sp_thickness_mm .setValue (float (data .get ("thickness_mm",0.012 )))
            self .sp_pretension .setValue (float (data .get ("pre_tension_n_per_m",10.0 )))
            tr =(data .get ("translation")or [0 ,0 ,0 ])+[0 ,0 ,0 ]
            rot =(data .get ("rotation_euler_deg")or [0 ,0 ,0 ])+[0 ,0 ,0 ]
            scl =(data .get ("scale")or [1 ,1 ,1 ])+[1 ,1 ,1 ]
            for sp in (self .sp_tx ,self .sp_ty ,self .sp_tz ,self .sp_rx ,self .sp_ry ,self .sp_rz ,
            self .sp_sx ,self .sp_sy ,self .sp_sz ):
                sp .blockSignals (True )
            self .sp_tx .setValue (float (tr [0 ]))
            self .sp_ty .setValue (float (tr [1 ]))
            self .sp_tz .setValue (float (tr [2 ]))
            self .sp_rx .setValue (float (rot [0 ]))
            self .sp_ry .setValue (float (rot [1 ]))
            self .sp_rz .setValue (float (rot [2 ]))
            self .sp_sx .setValue (float (scl [0 ]))
            self .sp_sy .setValue (float (scl [1 ]))
            self .sp_sz .setValue (float (scl [2 ]))
            for sp in (self .sp_tx ,self .sp_ty ,self .sp_tz ,self .sp_rx ,self .sp_ry ,self .sp_rz ,
            self .sp_sx ,self .sp_sy ,self .sp_sz ):
                sp .blockSignals (False )
            groups =data .get ("boundary_groups")or []
            self .cb_fixed_edge .setCurrentText (groups [0 ]if groups else "none")
            self .ed_notes .setPlainText (str (data .get ("notes","")))
        finally :
            self ._loading_data =False 

    def set_fixed_edge_options (self ,options :list [str ])->None :
        """Refresh fixed edge combo from boundary defaults."""
        current =self .cb_fixed_edge .currentText ()
        self .cb_fixed_edge .blockSignals (True )
        self .cb_fixed_edge .clear ()
        self .cb_fixed_edge .addItems (options )
        if current in options :
            self .cb_fixed_edge .setCurrentText (current )
        else :
            self .cb_fixed_edge .setCurrentText ("none")
        self .cb_fixed_edge .blockSignals (False )

    def set_material_options (self ,names :list [str ])->None :
        """Refresh material combo from MaterialLibraryModel."""
        current =self .cb_material .currentText ()
        self .cb_material .blockSignals (True )
        self .cb_material .clear ()
        if names :
            self .cb_material .addItems (names )
            if current in names :
                self .cb_material .setCurrentText (current )
            else :
                self .cb_material .setCurrentText (names [0 ])
        self .cb_material .blockSignals (False )

    def _emit_apply_if_ready (self )->None :
        """Emit apply_clicked only when not loading data (avoids apply during set_data)."""
        if not self ._loading_data :
            self .apply_clicked .emit ()

    def connect_apply_on_change (self )->None :
        """Connect all value-changing widgets to instant apply (no Apply button)."""
        self .ed_name .textEdited .connect (self ._emit_apply_if_ready )
        self .cb_role .currentTextChanged .connect (self ._emit_apply_if_ready )
        self .cb_visible .stateChanged .connect (self ._emit_apply_if_ready )
        self .cb_material .currentTextChanged .connect (self ._emit_apply_if_ready )
        self .sp_thickness_mm .valueChanged .connect (self ._emit_apply_if_ready )
        self .sp_pretension .valueChanged .connect (self ._emit_apply_if_ready )
        self .cb_fixed_edge .currentIndexChanged .connect (self ._emit_apply_if_ready )
        self .ed_notes .textChanged .connect (self ._emit_apply_if_ready )
        for sp in (self .sp_tx ,self .sp_ty ,self .sp_tz ,self .sp_rx ,self .sp_ry ,self .sp_rz ,
        self .sp_sx ,self .sp_sy ,self .sp_sz ):
            sp .valueChanged .connect (self ._emit_apply_if_ready )

    def connect_dirty (self ,slot )->None :
        """Connect all value-changing widgets to slot (e.g. mark_dirty)."""
        self .ed_name .textEdited .connect (slot )
        self .cb_role .currentIndexChanged .connect (slot )
        self .cb_visible .stateChanged .connect (slot )
        self .cb_material .currentIndexChanged .connect (slot )
        # Material parameters (density, E_parallel, ...) read-only, do not connect
        self .sp_thickness_mm .valueChanged .connect (slot )
        self .sp_pretension .valueChanged .connect (slot )
        self .cb_fixed_edge .currentIndexChanged .connect (slot )

    def connect_transform_live (self ,slot )->None :
        """Connect transform spinboxes to slot (for live viewport update)."""
        for sp in (self .sp_tx ,self .sp_ty ,self .sp_tz ,self .sp_rx ,self .sp_ry ,self .sp_rz ,
        self .sp_sx ,self .sp_sy ,self .sp_sz ):
            sp .valueChanged .connect (slot )
