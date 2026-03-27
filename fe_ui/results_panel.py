# -*- coding: utf-8 -*-
"""
Results visualization panel: displays all simulation data from the kernel.
Charts: time domain (displacement center), spectrum (FFT), displacement map, air pressure.
"""

from __future__ import annotations 

from PySide6 .QtCore import Qt ,QTimer ,Signal 
from PySide6 .QtGui import QShowEvent 
from PySide6 .QtWidgets import (
QDockWidget ,
QFileDialog ,
QHBoxLayout ,
QLabel ,
QPushButton ,
QComboBox ,
QScrollArea ,
QSlider ,
QTabWidget ,
QVBoxLayout ,
QWidget ,
)

try :
    import numpy as np 
except ImportError :
    np =None 

try :
    from matplotlib .backends .backend_qtagg import FigureCanvasQTAgg 
    from matplotlib .colors import TwoSlopeNorm 
    from matplotlib .figure import Figure 
    HAS_MATPLOTLIB =True 
except ImportError :
    HAS_MATPLOTLIB =False 
    TwoSlopeNorm =None # type: ignore[misc,assignment]


class SimulationResultsData :
    """Container for all simulation results from the kernel."""

    def __init__ (
    self ,
    history_disp_center :list |None =None ,
    history_disp_all :list |None =None ,
    history_air_pressure_xy_center_z :list |None =None ,
    history_air_pressure_step :int =1 ,
    history_air_center_xz :list |None =None ,
    dt :float =1e-6 ,
    width_mm :float =0.0 ,
    height_mm :float =0.0 ,
    air_extent :tuple |None =None ,
    )->None :
        self .history_disp_center =[]if history_disp_center is None else (history_disp_center .tolist ()if hasattr (history_disp_center ,"tolist")else list (history_disp_center ))
        self .history_disp_all =[]if history_disp_all is None else list (history_disp_all )
        self .history_air_pressure_xy_center_z =[]if history_air_pressure_xy_center_z is None else list (history_air_pressure_xy_center_z )
        self .history_air_pressure_step =max (1 ,int (history_air_pressure_step ))
        self .history_air_center_xz =[]if history_air_center_xz is None else list (history_air_center_xz )
        self .dt =dt 
        self .width_mm =width_mm 
        self .height_mm =height_mm 
        self .air_extent =air_extent # (x0_mm, x1_mm, z0_mm, z1_mm)

    def has_time_data (self )->bool :
        return bool (self .history_disp_center and np is not None )

    def has_displacement_map (self )->bool :
        if not self .history_disp_all or np is None :
            return False 
        return self ._get_disp_shape ()is not None 

    def _get_disp_shape (self )->tuple |None :
        if not self .history_disp_all :
            return None 
        arr =np .asarray (self .history_disp_all [0 ],dtype =float )
        if arr .ndim !=2 :
            return None 
        return arr .shape 

    def has_air (self )->bool :
        return bool (self .history_air_center_xz and np is not None )

    def has_air_pressure_history (self )->bool :
        return bool (self .history_air_pressure_xy_center_z and np is not None )

    @classmethod 
    def from_packed_dict (cls ,data :dict )->"SimulationResultsData":
        """Build from network / file payload (same keys as simulation_io.pack_simulation_results)."""

        def _to_list (val ):
            if val is None :
                return []
            if hasattr (val ,"tolist"):
                return val .ravel ().tolist ()
            return list (val )if isinstance (val ,(list ,tuple ))else []

        air_ext =data .get ("air_extent")
        if air_ext is not None and isinstance (air_ext ,list ):
            air_ext =tuple (air_ext )
        hc =data .get ("history_disp_center")
        hda =data .get ("history_disp_all")
        hap =data .get ("history_air_pressure_xy_center_z")
        hap_step =int (data .get ("history_air_pressure_step",1 ))
        hac =data .get ("history_air_center_xz")
        return cls (
        history_disp_center =_to_list (hc )if hc is not None else [],
        history_disp_all =hda if (hda is not None and isinstance (hda ,(list ,tuple )))else [],
        history_air_pressure_xy_center_z =hap if (hap is not None and isinstance (hap ,(list ,tuple )))else [],
        history_air_pressure_step =hap_step ,
        history_air_center_xz =hac if (hac is not None and isinstance (hac ,(list ,tuple )))else [],
        dt =float (data .get ("dt",1e-6 )),
        width_mm =float (data .get ("width_mm",0 )),
        height_mm =float (data .get ("height_mm",0 )),
        air_extent =air_ext ,
        )

    def to_results_dict (self )->dict :
        'Dictionary for export (same set of keys as the server / simulation_io.pack_simulation_results).'
        ae =self .air_extent 
        if isinstance (ae ,tuple ):
            ae =list (ae )
        return {
        "history_disp_center":self .history_disp_center ,
        "history_disp_all":self .history_disp_all ,
        "history_air_pressure_xy_center_z":self .history_air_pressure_xy_center_z ,
        "history_air_pressure_step":self .history_air_pressure_step ,
        "history_air_center_xz":self .history_air_center_xz ,
        "dt":self .dt ,
        "width_mm":self .width_mm ,
        "height_mm":self .height_mm ,
        "air_extent":ae ,
        }


class ResultsPanel (QDockWidget ):
    """Dock for simulation results: time, spectrum, displacement map."""

    def __init__ (self ,parent =None )->None :
        super ().__init__ ("Results",parent )
        self .setAllowedAreas (Qt .BottomDockWidgetArea |Qt .RightDockWidgetArea )

        self ._data :SimulationResultsData |None =None 
        self ._disp_frame_idx =0 
        self ._air_pressure_frame_idx =0
        self ._air_pressure_vmin :float |None =None
        self ._air_pressure_vmax :float |None =None
        self ._air_pressure_playing =False
        self ._air_pressure_play_timer =QTimer (self )
        self ._air_pressure_play_timer .timeout .connect (self ._on_air_pressure_play_tick )

        scroll =QScrollArea ()
        scroll .setWidgetResizable (True )
        scroll .setHorizontalScrollBarPolicy (Qt .ScrollBarAsNeeded )
        scroll .setVerticalScrollBarPolicy (Qt .ScrollBarAsNeeded )

        content =QWidget ()
        layout =QVBoxLayout (content )

        btn_bar =QHBoxLayout ()
        self ._btn_load =QPushButton ("Load from file...")
        self ._btn_load .clicked .connect (self ._on_load_from_file )
        btn_bar .addWidget (self ._btn_load )
        btn_bar .addStretch ()
        layout .addLayout (btn_bar )

        if not HAS_MATPLOTLIB or np is None :
            layout .addWidget (QLabel ("Results: matplotlib/numpy required for charts."))
        else :
            self ._tabs =QTabWidget ()
            self ._tab_time =QWidget ()
            self ._tab_spectrum =QWidget ()
            self ._tab_disp =QWidget ()
            self ._tab_air_pressure =QWidget ()

            layout_t =QVBoxLayout (self ._tab_time )
            self ._canvas_time =FigureCanvasQTAgg (Figure (figsize =(6 ,3 )))
            layout_t .addWidget (self ._canvas_time )

            layout_s =QVBoxLayout (self ._tab_spectrum )
            self ._canvas_spectrum =FigureCanvasQTAgg (Figure (figsize =(6 ,3 )))
            layout_s .addWidget (self ._canvas_spectrum )

            layout_d =QVBoxLayout (self ._tab_disp )
            self ._canvas_disp =FigureCanvasQTAgg (Figure (figsize =(5 ,4 )))
            self ._slider_disp =QSlider (Qt .Horizontal )
            self ._slider_disp .setMinimum (0 )
            self ._slider_disp .setMaximum (0 )
            self ._slider_disp .valueChanged .connect (self ._on_disp_slider )
            self ._label_disp =QLabel ("Frame 0")
            layout_d .addWidget (self ._canvas_disp )
            layout_d .addWidget (self ._label_disp )
            layout_d .addWidget (self ._slider_disp )

            layout_ap =QVBoxLayout (self ._tab_air_pressure )
            self ._canvas_air_pressure =FigureCanvasQTAgg (Figure (figsize =(6 ,3 )))
            self ._label_air_pressure =QLabel ("Frame 0")
            self ._slider_air_pressure =QSlider (Qt .Horizontal )
            self ._slider_air_pressure .setMinimum (0 )
            self ._slider_air_pressure .setMaximum (0 )
            self ._slider_air_pressure .valueChanged .connect (self ._on_air_pressure_slider )
            ap_controls =QHBoxLayout ()
            self ._btn_air_pressure_play =QPushButton ("Play")
            self ._btn_air_pressure_play .setCheckable (True )
            self ._btn_air_pressure_play .setEnabled (False )
            self ._btn_air_pressure_play .toggled .connect (self ._toggle_air_pressure_playback )
            ap_controls .addWidget (self ._btn_air_pressure_play )
            ap_controls .addWidget (QLabel ("Speed:"))
            self ._cmb_air_pressure_speed =QComboBox ()
            self ._cmb_air_pressure_speed .addItems (["1x","2x","4x","8x","16x"])
            self ._cmb_air_pressure_speed .setCurrentIndex (0 )
            self ._cmb_air_pressure_speed .setToolTip ("Playback speed multiplier (frame skipping)")
            ap_controls .addWidget (self ._cmb_air_pressure_speed )
            ap_controls .addStretch ()
            layout_ap .addWidget (self ._canvas_air_pressure )
            layout_ap .addWidget (self ._label_air_pressure )
            layout_ap .addWidget (self ._slider_air_pressure )
            layout_ap .addLayout (ap_controls )

            self ._tabs .addTab (self ._tab_time ,"Time")
            self ._tabs .addTab (self ._tab_spectrum ,"Spectrum")
            self ._tabs .addTab (self ._tab_disp ,"Displacement Map")
            self ._tabs .addTab (self ._tab_air_pressure ,"Air Pressure")

            layout .addWidget (self ._tabs )

        self ._empty_label =QLabel ("No simulation results. Run simulation to see charts.")
        layout .addWidget (self ._empty_label )

        scroll .setWidget (content )
        self .setWidget (scroll )

    def showEvent (self ,event :QShowEvent )->None :
        super ().showEvent (event )
        if self ._data is not None :
            self ._refresh_all ()

    def set_results (self ,data :SimulationResultsData |None )->None :
        """Update charts with new simulation data."""
        self ._stop_air_pressure_playback ()
        self ._data =data 
        self ._air_pressure_vmin =None
        self ._air_pressure_vmax =None
        if self .isVisible ():
            self ._refresh_all ()

    def _on_load_from_file (self )->None :
        """Load results same as network: .pkl (server save) or .json (wire export with data_b64)."""
        path ,_ =QFileDialog .getOpenFileName (
        self ,
        "Load simulation results",
        "",
        "Simulation results (*.pkl *.json);;Pickle (*.pkl);;JSON wire (*.json);;All (*)",
        )
        if not path :
            return 
        try :
            from simulation_io import load_simulation_results_file 

            packed =load_simulation_results_file (path )
            sim_data =SimulationResultsData .from_packed_dict (packed )
            if sim_data .has_time_data ():
                self .set_results (sim_data )
                self .setVisible (True )
            else :
                from PySide6 .QtWidgets import QMessageBox 

                QMessageBox .warning (self ,"Load Results","File has no valid time data.")
        except Exception as e :
            from PySide6 .QtWidgets import QMessageBox 

            QMessageBox .critical (self ,"Load Results",f"Failed to load: {e }")

    def _refresh_all (self )->None :
        if not HAS_MATPLOTLIB or np is None :
            return 
        if self ._data is None or not self ._data .has_time_data ():
            self ._empty_label .setVisible (True )
            if hasattr (self ,"_tabs"):
                self ._tabs .setVisible (False )
            return 
        self ._empty_label .setVisible (False )
        if hasattr (self ,"_tabs"):
            self ._tabs .setVisible (True )
        self ._plot_time ()
        self ._plot_spectrum ()
        self ._plot_displacement_map ()
        self ._plot_air_pressure_history ()

    def _plot_time (self )->None :
        if not self ._data or not self ._data .has_time_data ():
            return 
        hist =np .asarray (self ._data .history_disp_center ,dtype =np .float64 )
        t_ms =np .arange (len (hist ))*self ._data .dt *1e3 
        ax =self ._canvas_time .figure .clear ()
        ax =self ._canvas_time .figure .add_subplot (111 )
        ax .plot (t_ms ,hist *1e6 )
        ax .set_xlabel ("Time, ms")
        ax .set_ylabel ("Center displacement, µm")
        ax .set_title ("Diaphragm center displacement")
        ax .grid (True ,alpha =0.3 )
        self ._canvas_time .figure .tight_layout ()
        self ._canvas_time .draw ()

    def _plot_spectrum (self )->None :
        if not self ._data or not self ._data .has_time_data ():
            return 
        hist =np .asarray (self ._data .history_disp_center ,dtype =np .float64 )
        if len (hist )<4 :
            return 
        freq =np .fft .rfftfreq (len (hist ),self ._data .dt )
        spec =np .abs (np .fft .rfft (hist ))
        mask =(freq >0 )&(freq <=20_000 )

        ax =self ._canvas_spectrum .figure .clear ()
        ax =self ._canvas_spectrum .figure .add_subplot (111 )
        ax .loglog (freq [mask ],np .maximum (spec [mask ],1e-20 ))
        ax .set_xlim (1.0 ,20_000 )
        ax .set_xlabel ("Frequency, Hz")
        ax .set_ylabel ("Amplitude")
        ax .set_title ("Spectrum")
        ax .grid (True ,alpha =0.3 ,which ="both")
        self ._canvas_spectrum .figure .tight_layout ()
        self ._canvas_spectrum .draw ()

    def _plot_displacement_map (self )->None :
        if not self ._data or not self ._data .has_displacement_map ():
            return 
        frames =self ._data .history_disp_all 
        if not frames :
            return 
        self ._slider_disp .setMaximum (max (0 ,len (frames )-1 ))
        self ._plot_disp_frame (self ._disp_frame_idx )

    def _plot_disp_frame (self ,idx :int )->None :
        if not self ._data or not self ._data .history_disp_all :
            return 
        frames =self ._data .history_disp_all 
        idx =max (0 ,min (idx ,len (frames )-1 ))
        self ._disp_frame_idx =idx 
        frame =np .asarray (frames [idx ],dtype =np .float64 )
        if frame .ndim !=2 :
            return 
        extent =[0.0 ,self ._data .width_mm ,0.0 ,self ._data .height_mm ]if self ._data .width_mm >0 else None 
        ax =self ._canvas_disp .figure .clear ()
        ax =self ._canvas_disp .figure .add_subplot (111 )
        im =ax .imshow (
        frame *1e6 ,
        cmap ="RdBu",
        origin ="lower",
        extent =extent or [0 ,1 ,0 ,1 ],
        aspect ="auto",
        )
        ax .set_xlabel ("X, mm")
        ax .set_ylabel ("Y, mm")
        ax .set_title (f"Displacement uz (µm), frame {idx }")
        self ._canvas_disp .figure .colorbar (im ,ax =ax ,label ="uz, µm")
        self ._label_disp .setText (f"Frame {idx } / {len (frames )-1 }")
        self ._canvas_disp .figure .tight_layout ()
        self ._canvas_disp .draw ()

    def _on_disp_slider (self ,value :int )->None :
        self ._disp_frame_idx =value 
        self ._plot_disp_frame (value )

    def _plot_air_pressure_history (self )->None :
        if not self ._data :
            return
        frames =self ._data .history_air_pressure_xy_center_z
        if hasattr (self ,"_btn_air_pressure_play"):
            self ._btn_air_pressure_play .setEnabled (len (frames )>1 )
            if len (frames )<=1 :
                self ._stop_air_pressure_playback ()
        self ._air_pressure_vmin =None
        self ._air_pressure_vmax =None
        if frames :
            gmin =None
            gmax =None
            for fr in frames :
                arr =np .asarray (fr ,dtype =np .float64 )
                if arr .ndim !=2 :
                    continue
                if not np .isfinite (arr ).any ():
                    continue
                fmin =float (np .nanmin (arr ))
                fmax =float (np .nanmax (arr ))
                gmin =fmin if gmin is None else min (gmin ,fmin )
                gmax =fmax if gmax is None else max (gmax ,fmax )
            # Symmetric limits around 0 so RdBu uses the full range for both signs; avoids
            # TwoSlopeNorm + [gmin,gmax] squashing almost all variation into one hue when |gmin|>>gmax.
            if gmin is not None and gmax is not None :
                vabs =max (abs (gmin ),abs (gmax ),1e-12 )
                self ._air_pressure_vmin =-vabs 
                self ._air_pressure_vmax =vabs 
            else :
                self ._air_pressure_vmin =gmin 
                self ._air_pressure_vmax =gmax 
        self ._slider_air_pressure .setMaximum (max (0 ,len (frames )-1 ))
        self ._plot_air_pressure_frame (self ._air_pressure_frame_idx )

    def _plot_air_pressure_frame (self ,idx :int )->None :
        ax =self ._canvas_air_pressure .figure .clear ()
        ax =self ._canvas_air_pressure .figure .add_subplot (111 )
        if not self ._data or not self ._data .has_air_pressure_history ():
            ax .set_title ("Air pressure XY slice at center Z")
            ax .text (0.5 ,0.5 ,"No air pressure history",ha ="center",va ="center",transform =ax .transAxes )
            ax .set_axis_off ()
            self ._canvas_air_pressure .figure .tight_layout ()
            self ._canvas_air_pressure .draw ()
            return
        frames =self ._data .history_air_pressure_xy_center_z
        idx =max (0 ,min (idx ,len (frames )-1 ))
        self ._air_pressure_frame_idx =idx
        frame =np .asarray (frames [idx ],dtype =np .float64 )
        if frame .ndim !=2 :
            ax .set_title ("Air pressure XY slice at center Z")
            ax .text (0.5 ,0.5 ,"Invalid frame shape",ha ="center",va ="center",transform =ax .transAxes )
            ax .set_axis_off ()
            self ._canvas_air_pressure .figure .tight_layout ()
            self ._canvas_air_pressure .draw ()
            return
        # 2D pressure map: symmetric scale about p=0 (see _plot_air_pressure_history) + RdBu.
        vmin =self ._air_pressure_vmin
        vmax =self ._air_pressure_vmax
        if vmin is not None and vmax is not None and vmax <=vmin :
            vmax =vmin +1e-12
        frame_plot =np .ma .masked_invalid (np .asarray (frame ,dtype =np .float64 ))
        im_kw :dict ={
        "cmap":"RdBu",
        "origin":"lower",
        "aspect":"auto",
        "interpolation":"nearest",
        "extent":[0.0 ,float (frame .shape [1 ]),0.0 ,float (frame .shape [0 ])],
        }
        if (
        vmin is not None 
        and vmax is not None 
        and TwoSlopeNorm is not None 
        and vmin <0.0 <vmax 
        ):
            im_kw ["norm"]=TwoSlopeNorm (vmin =vmin ,vcenter =0.0 ,vmax =vmax )
        elif vmin is not None and vmax is not None :
            im_kw ["vmin"]=vmin 
            im_kw ["vmax"]=vmax 
        im =ax .imshow (frame_plot ,**im_kw )
        t_ms =idx *self ._data .dt *float (self ._data .history_air_pressure_step )*1e3
        ax .set_xlabel ("Air column index (x)")
        ax .set_ylabel ("Air column index (z)")
        if frame .shape [0 ]<=1 or frame .shape [1 ]<=1 :
            ax .set_title (
                f"2D air pressure map (degenerate slice {frame .shape [0 ]}x{frame .shape [1 ]}), "
                f"t = {t_ms :.2f} ms"
            )
        else :
            ax .set_title (f"Air pressure XZ slice (center Y voxel row), t = {t_ms :.2f} ms")
        self ._canvas_air_pressure .figure .colorbar (im ,ax =ax ,label ="p, Pa")
        self ._label_air_pressure .setText (f"Frame {idx } / {len (frames )-1 }")
        self ._canvas_air_pressure .figure .tight_layout ()
        self ._canvas_air_pressure .draw ()

    def _on_air_pressure_slider (self ,value :int )->None :
        self ._air_pressure_frame_idx =value
        self ._plot_air_pressure_frame (value )

    def _toggle_air_pressure_playback (self ,playing :bool )->None :
        if playing :
            frames =self ._data .history_air_pressure_xy_center_z if self ._data else []
            if len (frames )<=1 :
                self ._stop_air_pressure_playback ()
                return
            self ._air_pressure_playing =True
            if hasattr (self ,"_btn_air_pressure_play"):
                self ._btn_air_pressure_play .setText ("Pause")
            self ._air_pressure_play_timer .start (1)
        else :
            self ._stop_air_pressure_playback ()

    def _stop_air_pressure_playback (self )->None :
        self ._air_pressure_playing =False
        if self ._air_pressure_play_timer .isActive ():
            self ._air_pressure_play_timer .stop ()
        if hasattr (self ,"_btn_air_pressure_play"):
            if self ._btn_air_pressure_play .isChecked ():
                self ._btn_air_pressure_play .blockSignals (True )
                self ._btn_air_pressure_play .setChecked (False )
                self ._btn_air_pressure_play .blockSignals (False )
            self ._btn_air_pressure_play .setText ("Play")

    def _on_air_pressure_play_tick (self )->None :
        if not self ._air_pressure_playing or not self ._data :
            self ._stop_air_pressure_playback ()
            return
        frames =self ._data .history_air_pressure_xy_center_z
        n =len (frames )
        if n <=1 :
            self ._stop_air_pressure_playback ()
            return
        speed =1
        if hasattr (self ,"_cmb_air_pressure_speed"):
            txt =self ._cmb_air_pressure_speed .currentText ().strip ().lower ()
            if txt .endswith ("x"):
                txt =txt [:-1 ]
            try :
                speed =max (1 ,int (txt ))
            except Exception :
                speed =1
        nxt =(self ._air_pressure_frame_idx +speed )%n
        self ._slider_air_pressure .setValue (nxt )

    def _plot_air_pressure (self )->None :
        if not self ._data or not self ._data .has_air ():
            return 
        frames =self ._data .history_air_center_xz 
        if not frames :
            return 
        self ._slider_air .setMaximum (max (0 ,len (frames )-1 ))
        self ._plot_air_frame (self ._air_frame_idx )

    def _plot_air_frame (self ,idx :int )->None :
        if not self ._data or not self ._data .history_air_center_xz :
            return 
        frames =self ._data .history_air_center_xz 
        idx =max (0 ,min (idx ,len (frames )-1 ))
        self ._air_frame_idx =idx 
        frame =np .asarray (frames [idx ],dtype =np .float64 )
        extent =self ._data .air_extent or [0 ,1 ,0 ,1 ]
        ax =self ._canvas_air .figure .clear ()
        ax =self ._canvas_air .figure .add_subplot (111 )
        im =ax .imshow (
        frame ,
        cmap ="RdBu",
        origin ="lower",
        extent =extent ,
        aspect ="auto",
        )
        t_ms =idx *self ._data .dt *1e3 
        ax .set_xlabel ("X, mm")
        ax .set_ylabel ("Z, mm")
        ax .set_title (f"Air pressure center slice (X-Z), t = {t_ms :.2f} ms")
        self ._canvas_air .figure .colorbar (im ,ax =ax ,label ="p, Pa")
        self ._label_air .setText (f"Frame {idx } / {len (frames )-1 }")
        self ._canvas_air .figure .tight_layout ()
        self ._canvas_air .draw ()

    def _on_air_slider (self ,value :int )->None :
        self ._air_frame_idx =value 
        self ._plot_air_frame (value )
