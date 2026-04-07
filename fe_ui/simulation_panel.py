# -*- coding: utf-8 -*-
"""
Simulation dock: solver params, excitation, run/stop, console, server connection.
Depends: PySide6 only. No project_model — uses dict for settings.
"""

from __future__ import annotations 

from PySide6 .QtCore import Qt ,Signal 
from PySide6 .QtGui import QTextCursor
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

from .constants import EXCITATION_MODES ,FORCE_SHAPES 
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
        self .cb_excitation_mode =QComboBox ()
        self .cb_excitation_mode .addItems (list (EXCITATION_MODES ))
        self .cb_excitation_mode .currentTextChanged .connect (self ._update_excitation_controls )
        self .lbl_force_amplitude_pa =QLabel ("Amplitude (Pa)")
        self .sp_force_amplitude_pa =ScientificDoubleSpinBox ()
        self .sp_force_amplitude_pa .setRange (0.0 ,1e6 )
        self .sp_force_amplitude_pa .setValue (10.0 )
        self .sp_force_amplitude_pa .setSuffix (" Pa")
        self .lbl_force_velocity_mps =QLabel ("Velocity (m/s)")
        self .sp_force_velocity_mps =ScientificDoubleSpinBox ()
        # QDoubleSpinBox stores/returns values quantized to `decimals()`.
        # Default is 2, which would round 1e-4 to 0.00 -> value()==0.
        self .sp_force_velocity_mps .setDecimals (12)
        self .sp_force_velocity_mps .setRange (1e-20, 10)
        self .sp_force_velocity_mps .setValue (0.001 )
        self .sp_force_velocity_mps .setSingleStep (1e-20)
        self .sp_force_velocity_mps .setSuffix (" m/s")
        self .lbl_force_freq_start =QLabel ("Frequency (Hz)")
        self .sp_force_freq_start =ScientificDoubleSpinBox ()
        self .sp_force_freq_start .setRange (0.0 ,100000.0 )
        self .sp_force_freq_start .setValue (1000.0 )
        self .sp_force_freq_start .setSuffix (" Hz")
        self .lbl_force_freq_end =QLabel ("End frequency (Hz)")
        self .sp_force_freq_end =ScientificDoubleSpinBox ()
        self .sp_force_freq_end .setRange (0.0 ,100000.0 )
        self .sp_force_freq_end .setValue (5000.0 )
        self .sp_force_freq_end .setSuffix (" Hz")
        force_form .addRow ("Shape",self .cb_force_shape )
        self .cb_force_shape .currentTextChanged .connect (self ._update_frequency_controls )
        force_form .addRow ("Mode",self .cb_excitation_mode )
        force_form .addRow (self .lbl_force_amplitude_pa ,self .sp_force_amplitude_pa )
        force_form .addRow (self .lbl_force_velocity_mps ,self .sp_force_velocity_mps )
        force_form .addRow (self .lbl_force_freq_start ,self .sp_force_freq_start )
        force_form .addRow (self .lbl_force_freq_end ,self .sp_force_freq_end )
        self ._update_excitation_controls (self .cb_excitation_mode .currentText ())
        self ._update_frequency_controls (self .cb_force_shape .currentText ())

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
        "excitation_mode":self .cb_excitation_mode .currentText (),
        "force_amplitude_pa":float (self .sp_force_amplitude_pa .value ()),
        "force_velocity_mps":float (self .sp_force_velocity_mps .value ()),
        "force_freq_hz":float (self .sp_force_freq_start .value ()),
        "force_freq_end_hz":float (self .sp_force_freq_end .value ()),
        }

    def set_settings (self ,data :dict )->None :
        self .sp_dt .setValue (float (data .get ("dt",1e-6 )))
        self .sp_duration .setValue (float (data .get ("duration",0.05 )))
        self .sp_air_coupling .setValue (float (data .get ("air_coupling_gain",0.05 )))
        self ._air_grid_step_mm =float (data .get ("air_grid_step_mm",0.2 ))
        self .sp_air_pressure_hist_every .setValue (int (data .get ("air_pressure_history_every_steps",10 )))
        self .sp_force_amplitude_pa .setValue (float (data .get ("force_amplitude_pa",10.0 )))
        self .sp_force_velocity_mps .setValue (float (data .get ("force_velocity_mps",data .get ("force_amplitude_pa",10.0 ))))
        self .cb_force_shape .setCurrentText (str (data .get ("force_shape","impulse")))
        self .cb_excitation_mode .setCurrentText (str (data .get ("excitation_mode","external")))
        self .sp_force_freq_start .setValue (float (data .get ("force_freq_hz",1000.0 )))
        self .sp_force_freq_end .setValue (float (data .get ("force_freq_end_hz",5000.0 )))
        self ._update_excitation_controls (self .cb_excitation_mode .currentText ())
        self ._update_frequency_controls (self .cb_force_shape .currentText ())

    def set_running (self ,is_running :bool )->None :
        self ._running =is_running 
        self .btn_run .setEnabled (self ._connected and not is_running )
        self .btn_stop .setEnabled (is_running )

    def append_console (self ,text :str )->None :
        if text :
            # Always append to end and keep viewport at latest logs.
            self .console .moveCursor (QTextCursor .End )
            self .console .insertPlainText (text )
            self .console .moveCursor (QTextCursor .End )
            self .console .ensureCursorVisible ()

    def connect_dirty (self ,slot )->None :
        """Connect settings widgets to dirty slot."""
        self .sp_dt .valueChanged .connect (slot )
        self .sp_duration .valueChanged .connect (slot )
        self .sp_air_coupling .valueChanged .connect (slot )
        self .sp_air_pressure_hist_every .valueChanged .connect (slot )
        self .cb_force_shape .currentIndexChanged .connect (slot )
        self .cb_excitation_mode .currentIndexChanged .connect (slot )
        self .sp_force_amplitude_pa .valueChanged .connect (slot )
        self .sp_force_velocity_mps .valueChanged .connect (slot )
        self .sp_force_freq_start .valueChanged .connect (slot )
        self .sp_force_freq_end .valueChanged .connect (slot )

    def _update_excitation_controls (self ,mode :str )->None :
        is_velocity =str (mode )=="external_velocity_override"
        self .lbl_force_amplitude_pa .setVisible (not is_velocity )
        self .sp_force_amplitude_pa .setVisible (not is_velocity )
        self .lbl_force_velocity_mps .setVisible (is_velocity )
        self .sp_force_velocity_mps .setVisible (is_velocity )
        self ._update_frequency_controls (self .cb_force_shape .currentText ())

    def _update_frequency_controls (self ,shape :str )->None :
        s =str (shape ).strip ().lower ()
        need_start =s in ("sine","square","chirp","sweep_tone")
        need_end =s in ("chirp","sweep_tone")
        self .lbl_force_freq_start .setText (
        "Start frequency (Hz)"if need_end else "Frequency (Hz)"
        )
        self .lbl_force_freq_start .setVisible (need_start )
        self .sp_force_freq_start .setVisible (need_start )
        self .lbl_force_freq_end .setVisible (need_end )
        self .sp_force_freq_end .setVisible (need_end )

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
