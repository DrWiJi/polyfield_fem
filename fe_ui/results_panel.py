# -*- coding: utf-8 -*-
"""
Results visualization panel: displays all simulation data from the kernel.
Charts: time domain (displacement center), spectrum (FFT), displacement map, air pressure.
"""

from __future__ import annotations 

from PySide6 .QtCore import Qt ,Signal 
from PySide6 .QtGui import QShowEvent 
from PySide6 .QtWidgets import (
QDockWidget ,
QFileDialog ,
QHBoxLayout ,
QLabel ,
QPushButton ,
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
    from matplotlib .figure import Figure 
    HAS_MATPLOTLIB =True 
except ImportError :
    HAS_MATPLOTLIB =False 


class SimulationResultsData :
    """Container for all simulation results from the kernel."""

    def __init__ (
    self ,
    history_disp_center :list |None =None ,
    history_disp_all :list |None =None ,
    history_air_center_xz :list |None =None ,
    dt :float =1e-6 ,
    width_mm :float =0.0 ,
    height_mm :float =0.0 ,
    air_extent :tuple |None =None ,
    )->None :
        self .history_disp_center =[]if history_disp_center is None else (history_disp_center .tolist ()if hasattr (history_disp_center ,"tolist")else list (history_disp_center ))
        self .history_disp_all =[]if history_disp_all is None else list (history_disp_all )
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
        hac =data .get ("history_air_center_xz")
        return cls (
        history_disp_center =_to_list (hc )if hc is not None else [],
        history_disp_all =hda if (hda is not None and isinstance (hda ,(list ,tuple )))else [],
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

            self ._tabs .addTab (self ._tab_time ,"Time")
            self ._tabs .addTab (self ._tab_spectrum ,"Spectrum")
            self ._tabs .addTab (self ._tab_disp ,"Displacement Map")

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
        self ._data =data 
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
