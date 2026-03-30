# -*- coding: utf-8 -*-
"""
Simulation dock: solver params, excitation, run/stop, console, server connection.
Depends: PySide6 only. No project_model — uses dict for settings.
"""

from __future__ import annotations 

from PySide6 .QtCore import Qt ,Signal 
from PySide6 .QtWidgets import (
QComboBox ,
QDockWidget ,
QFormLayout ,
QGroupBox ,
QHBoxLayout ,
QLabel ,
QLineEdit ,
QPushButton ,
QSpinBox ,
QTextEdit ,
QVBoxLayout ,
QWidget ,
)

from .constants import FORCE_SHAPES 
from .widgets import ScientificDoubleSpinBox 

DEFAULT_SERVER_PORT =8765 


class SimulationPanel (QDockWidget ):
    """Solver parameters, excitation, run/stop buttons, console output, server connection."""

    run_clicked =Signal ()
    stop_clicked =Signal ()
    export_results_clicked =Signal ()
    export_run_case_clicked =Signal ()
    connect_requested =Signal (str ,int ,bool )# host, port, use_local_server
    disconnect_requested =Signal ()

    def __init__ (self ,parent =None )->None :
        super ().__init__ ("Simulation",parent )
        self .setAllowedAreas (Qt .BottomDockWidgetArea )

        panel =QWidget ()
        main_layout =QHBoxLayout (panel )

        # Connection
        box_conn =QGroupBox ("Server Connection")
        conn_form =QFormLayout (box_conn )
        self .cb_connection_mode =QComboBox ()
        self .cb_connection_mode .addItems (["Local (built-in server)","Remote"])
        self .cb_connection_mode .currentIndexChanged .connect (self ._on_connection_mode_changed )
        self .le_host =QLineEdit ()
        self .le_host .setPlaceholderText ("127.0.0.1 or remote IP")
        self .le_host .setText ("127.0.0.1")
        self .sp_port =QSpinBox ()
        self .sp_port .setRange (1 ,65535 )
        self .sp_port .setValue (DEFAULT_SERVER_PORT )
        conn_form .addRow ("Mode",self .cb_connection_mode )
        conn_form .addRow ("Host",self .le_host )
        conn_form .addRow ("Port",self .sp_port )
        conn_btn_row =QHBoxLayout ()
        self .btn_connect =QPushButton ("Connect")
        self .btn_disconnect =QPushButton ("Disconnect")
        self .btn_connect .clicked .connect (self ._on_connect_clicked )
        self .btn_disconnect .clicked .connect (self .disconnect_requested .emit )
        self .btn_disconnect .setEnabled (False )
        conn_btn_row .addWidget (self .btn_connect )
        conn_btn_row .addWidget (self .btn_disconnect )
        conn_form .addRow ("",conn_btn_row )
        self .label_conn_status =QLabel ("Disconnected")
        self .label_conn_status .setStyleSheet ("color: gray;")
        conn_form .addRow ("Status",self .label_conn_status )
        self ._on_connection_mode_changed ()

        # Solver
        box_solver =QGroupBox ("Solver Parameters")
        solver_form =QFormLayout (box_solver )
        self .sp_dt =ScientificDoubleSpinBox ()
        self .sp_dt .setDecimals (9 )
        self .sp_dt .setRange (1e-9 ,1.0 )
        # Lower dt for stability under strong boundary forces
        self .sp_dt .setValue (1e-7 )
        self .sp_dt .setSingleStep (1e-6 )
        self .sp_dt .setSuffix (" s")
        self .sp_duration =ScientificDoubleSpinBox ()
        self .sp_duration .setDecimals (4 )
        self .sp_duration .setRange (1e-4 ,30.0 )
        self .sp_duration .setValue (0.05 )
        self .sp_duration .setSuffix (" s")
        self .sp_air_coupling =ScientificDoubleSpinBox ()
        self .sp_air_coupling .setRange (0.0 ,1.0 )
        self .sp_air_coupling .setSingleStep (0.01 )
        self .sp_air_coupling .setValue (0.05 )
        # Air grid step is defined by generated topology parameters (Topology Generator panel).
        # Keep a hidden compatibility value for old project/run-case payloads.
        self ._air_grid_step_mm =0.2
        self .sp_air_pressure_hist_every =QSpinBox ()
        self .sp_air_pressure_hist_every .setRange (1 ,1_000_000 )
        self .sp_air_pressure_hist_every .setValue (10 )
        solver_form .addRow ("dt",self .sp_dt )
        solver_form .addRow ("Duration",self .sp_duration )
        solver_form .addRow ("Air coupling gain",self .sp_air_coupling )
        solver_form .addRow ("Air pressure history every N steps",self .sp_air_pressure_hist_every )

        # Excitation
        box_force =QGroupBox ("Excitation")
        force_form =QFormLayout (box_force )
        self .cb_force_shape =QComboBox ()
        self .cb_force_shape .addItems (list (FORCE_SHAPES ))
        self .sp_force_amp =ScientificDoubleSpinBox ()
        self .sp_force_amp .setRange (0.0 ,1e6 )
        self .sp_force_amp .setValue (10.0 )
        self .sp_force_amp .setSuffix (" Pa")
        self .sp_force_freq =ScientificDoubleSpinBox ()
        self .sp_force_freq .setRange (0.0 ,100000.0 )
        self .sp_force_freq .setValue (1000.0 )
        self .sp_force_freq .setSuffix (" Hz")
        force_form .addRow ("Shape",self .cb_force_shape )
        force_form .addRow ("Amplitude",self .sp_force_amp )
        force_form .addRow ("Freq",self .sp_force_freq )

        btn_row =QHBoxLayout ()
        self .btn_run =QPushButton ("Run Simulation")
        self .btn_stop =QPushButton ("Stop")
        self .btn_export_results =QPushButton ("Export results…")
        self .btn_export_run_case =QPushButton ("Export run case…")
        self .btn_run .clicked .connect (self .run_clicked .emit )
        self .btn_stop .clicked .connect (self .stop_clicked .emit )
        self .btn_export_results .clicked .connect (self .export_results_clicked .emit )
        self .btn_export_run_case .clicked .connect (self .export_run_case_clicked .emit )
        self .btn_stop .setEnabled (False )
        btn_row .addWidget (self .btn_run )
        btn_row .addWidget (self .btn_stop )
        btn_row .addWidget (self .btn_export_results )
        btn_row .addWidget (self .btn_export_run_case )

        self .console =QTextEdit ()
        self .console .setReadOnly (True )
        self .console .setPlaceholderText ("Simulation console output...")

        self ._connected =False 
        self ._running =False 

        left_panel =QWidget ()
        left_layout =QVBoxLayout (left_panel )
        left_layout .addWidget (box_conn )
        left_layout .addWidget (box_solver )
        left_layout .addWidget (box_force )
        left_layout .addLayout (btn_row )
        main_layout .addWidget (left_panel )
        main_layout .addWidget (self .console ,1 )

        self .setWidget (panel )
        self .set_connection_status (False ,"Disconnected")

    def get_settings (self )->dict :
        return {
        "dt":float (self .sp_dt .value ()),
        "duration":float (self .sp_duration .value ()),
        "air_coupling_gain":float (self .sp_air_coupling .value ()),
        "air_grid_step_mm":float (self ._air_grid_step_mm ),
        "air_pressure_history_every_steps":int (self .sp_air_pressure_hist_every .value ()),
        "force_shape":self .cb_force_shape .currentText (),
        "force_amplitude_pa":float (self .sp_force_amp .value ()),
        "force_freq_hz":float (self .sp_force_freq .value ()),
        }

    def set_settings (self ,data :dict )->None :
        self .sp_dt .setValue (float (data .get ("dt",1e-6 )))
        self .sp_duration .setValue (float (data .get ("duration",0.05 )))
        self .sp_air_coupling .setValue (float (data .get ("air_coupling_gain",0.05 )))
        self ._air_grid_step_mm =float (data .get ("air_grid_step_mm",0.2 ))
        self .sp_air_pressure_hist_every .setValue (int (data .get ("air_pressure_history_every_steps",10 )))
        self .cb_force_shape .setCurrentText (str (data .get ("force_shape","impulse")))
        self .sp_force_amp .setValue (float (data .get ("force_amplitude_pa",10.0 )))
        self .sp_force_freq .setValue (float (data .get ("force_freq_hz",1000.0 )))

    def set_running (self ,is_running :bool )->None :
        self ._running =is_running 
        self .btn_run .setEnabled (self ._connected and not is_running )
        self .btn_stop .setEnabled (is_running )

    def append_console (self ,text :str )->None :
        if text :
            self .console .insertPlainText (text )

    def connect_dirty (self ,slot )->None :
        """Connect settings widgets to dirty slot."""
        self .sp_dt .valueChanged .connect (slot )
        self .sp_duration .valueChanged .connect (slot )
        self .sp_air_coupling .valueChanged .connect (slot )
        self .sp_air_pressure_hist_every .valueChanged .connect (slot )
        self .cb_force_shape .currentIndexChanged .connect (slot )
        self .sp_force_amp .valueChanged .connect (slot )
        self .sp_force_freq .valueChanged .connect (slot )

    def _on_connection_mode_changed (self )->None :
        is_remote =self .cb_connection_mode .currentIndex ()==1 
        self .le_host .setEnabled (is_remote )
        if not is_remote :
            self .le_host .setText ("127.0.0.1")

    def _on_connect_clicked (self )->None :
        host =self .le_host .text ().strip ()or "127.0.0.1"
        port =self .sp_port .value ()
        use_local =self .cb_connection_mode .currentIndex ()==0 
        self .connect_requested .emit (host ,port ,use_local )

    def set_connection_status (self ,connected :bool ,status_text :str |None =None )->None :
        """Update connection UI state."""
        self ._connected =connected 
        self .btn_connect .setEnabled (not connected )
        self .btn_disconnect .setEnabled (connected )
        self .cb_connection_mode .setEnabled (not connected )
        self .le_host .setEnabled (not connected and self .cb_connection_mode .currentIndex ()==1 )
        self .sp_port .setEnabled (not connected )
        self .btn_run .setEnabled (connected and not self ._running )
        if status_text is not None :
            self .label_conn_status .setText (status_text )
        if connected :
            self .label_conn_status .setStyleSheet ("color: green;")
        else :
            self .label_conn_status .setStyleSheet ("color: gray;")
