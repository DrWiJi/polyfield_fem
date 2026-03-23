# -*- coding: utf-8 -*-
"""
Viewport widget: placeholder or PyVista 3D renderer.
Unified mesh viewport: subscribes to model signals, renders meshes, supports extra actors.
Depends: PySide6. Optional: pyvista, pyvistaqt.
"""

from __future__ import annotations 

from collections .abc import Callable 
from typing import Any 

from PySide6 .QtCore import Qt ,Signal 
from PySide6 .QtGui import QMouseEvent 
from PySide6 .QtWidgets import QFrame ,QLabel ,QVBoxLayout ,QWidget 

try :
    import pyvista as pv 
    from pyvistaqt import QtInteractor 
except Exception :
    pv =None 
    QtInteractor =None 


class ViewportPlaceholder (QFrame ):
    """Empty viewport with click callback for mocked selection."""

    def __init__ (self ,on_click_callback =None )->None :
        super ().__init__ ()
        self ._on_click_callback =on_click_callback 
        self .setFrameShape (QFrame .StyledPanel )
        self .setObjectName ("viewportPlaceholder")
        self .setMinimumSize (520 ,420 )

        layout =QVBoxLayout (self )
        title =QLabel ("Viewport (placeholder)")
        title .setAlignment (Qt .AlignCenter )
        title .setStyleSheet ("font-weight: 600;")
        hint =QLabel (
        "3D rendering is not connected.\n"
        "Select meshes from the Mesh List."
        )
        hint .setAlignment (Qt .AlignCenter )
        hint .setStyleSheet ("color: #888;")
        layout .addStretch (1 )
        layout .addWidget (title )
        layout .addWidget (hint )
        layout .addStretch (1 )

    def mousePressEvent (self ,event :QMouseEvent )->None :
        if event .button ()==Qt .LeftButton and self ._on_click_callback :
            self ._on_click_callback ()
        super ().mousePressEvent (event )


def create_viewport (parent :QWidget ,on_pick_callback =None )->tuple [QWidget ,object |None ]:
    """
    Create viewport widget: PyVista if available, else placeholder.
    Returns (widget, plotter_or_none). Plotter is used for add_mesh, remove_actor, etc.
    """
    if QtInteractor is None or pv is None :
        return ViewportPlaceholder (on_pick_callback ),None 

    plotter =QtInteractor (parent )
    plotter .set_background ("#1f1f1f")
    plotter .add_axes ()
    plotter .show_grid (color ="#555555")
    _setup_lighting (plotter )
    return plotter .interactor ,plotter 


def _setup_lighting (plotter )->None :
    """Configure scene lighting for PyVista plotter."""
    if pv is None :
        return 
    plotter .remove_all_lights ()
    plotter .add_light (pv .Light (
    position =(2.0 ,2.5 ,3.0 ),
    focal_point =(0.0 ,0.0 ,0.0 ),
    color ="white",
    intensity =1.0 ,
    light_type ="scene light",
    ))
    plotter .add_light (pv .Light (
    position =(-2.5 ,1.0 ,1.5 ),
    focal_point =(0.0 ,0.0 ,0.0 ),
    color ="#cfd8ff",
    intensity =0.45 ,
    light_type ="scene light",
    ))
    plotter .add_light (pv .Light (
    position =(0.0 ,-3.0 ,2.0 ),
    focal_point =(0.0 ,0.0 ,0.0 ),
    color ="#ffe7c9",
    intensity =0.30 ,
    light_type ="scene light",
    ))


def has_pyvista ()->bool :
    """Check if PyVista 3D rendering is available."""
    return pv is not None and QtInteractor is not None 


class UnifiedMeshViewport (QWidget ):
    """
    Unified viewport that renders meshes from model data and reacts to change signals.
    Supports adding extra visualization objects via add_extra_actor/remove_extra_actor.
    """

    mesh_actors_updated =Signal (object )# dict[str, actor] when mesh display is refreshed

    def __init__ (
    self ,
    parent :QWidget |None ,
    get_mesh_data :Callable [[],tuple [dict [str ,Any ],list ]],
    refresh_signals :list |None =None ,
    *,
    pickable :bool =False ,
    mesh_color :str ="#9A9A9A",
    )->None :
        """
        Args:
            parent: Parent widget.
            get_mesh_data: Callable returning (polydata_by_id, meshes).
            refresh_signals: List of Qt signals to connect to refresh_meshes.
            pickable: Whether mesh actors are pickable.
            mesh_color: Default mesh color.
        """
        super ().__init__ (parent )
        self ._get_mesh_data =get_mesh_data 
        self ._pickable =pickable 
        self ._mesh_color =mesh_color 
        self ._mesh_actor_by_id :dict [str ,Any ]={}
        self ._extra_actors :dict [str ,Any ]={}
        self ._refresh_connections :list =[]
        self ._refresh_signals :list =list (refresh_signals )if refresh_signals else []
        self ._viewport_closed =False 

        if QtInteractor is None or pv is None :
            self ._plotter =None 
            self ._interactor =ViewportPlaceholder ()
            layout =QVBoxLayout (self )
            layout .addWidget (self ._interactor )
            return 

        self ._plotter =QtInteractor (self )
        self ._plotter .set_background ("#1f1f1f")
        self ._interactor =self ._plotter .interactor 
        layout =QVBoxLayout (self )
        layout .setContentsMargins (0 ,0 ,0 ,0 )
        layout .addWidget (self ._interactor )

        _setup_lighting (self ._plotter )
        self ._add_scene_basics ()

        if refresh_signals :
            for sig in refresh_signals :
                if hasattr (sig ,"connect"):
                    conn =sig .connect (self .refresh_meshes )
                    self ._refresh_connections .append (conn )

        self .refresh_meshes ()

    def _add_scene_basics (self )->None :
        """Add axes and grid (call after clear)."""
        if self ._plotter :
            self ._plotter .add_axes ()
            self ._plotter .show_grid (color ="#555555")

    def disconnect_refresh_signals (self )->None :
        """Disconnect from refresh signals. Call when window is closed to prevent render errors."""
        for conn in self ._refresh_connections :
            try :
                if hasattr (conn ,"disconnect"):
                    conn .disconnect ()
            except Exception :
                pass 
        self ._refresh_connections .clear ()
        # Fallback: disconnect by signal (PySide6 Connection may not support disconnect)
        for sig in self ._refresh_signals :
            try :
                if hasattr (sig ,"disconnect"):
                    sig .disconnect (self .refresh_meshes )
            except (TypeError ,RuntimeError ):
                pass 
        self ._refresh_signals .clear ()

    def close_viewport (self )->None :
        """
        Prepare viewport for window close. Disconnects signals and releases VTK render window.
        Must call plotter.close() BEFORE Qt destroys the widget to avoid wglMakeCurrent on invalid handle.
        """
        self ._viewport_closed =True 
        self .disconnect_refresh_signals ()
        if self ._plotter :
            try :
                self ._plotter .close ()
            except Exception :
                pass 
            self ._plotter =None 

    def _meshes_list_changed (self ,polydata_by_id :dict ,mesh_by_id :dict )->bool :
        """Return True if the set of visible mesh IDs differs from current actors."""
        current_visible =frozenset (
        mid for mid ,m in mesh_by_id .items ()
        if m .visible and mid in polydata_by_id and mid !="__debug_surface__"
        )
        our_actors =frozenset (self ._mesh_actor_by_id .keys ())
        return current_visible !=our_actors 

    def _update_mesh_transforms (self ,mesh_by_id :dict )->None :
        """Update actor positions only (no geometry recreation)."""
        for mesh_id ,actor in self ._mesh_actor_by_id .items ():
            mesh =mesh_by_id .get (mesh_id )
            if not mesh or not mesh .visible :
                continue 
            tr =list (mesh .transform .translation )if mesh .transform else [0 ,0 ,0 ]
            rot =list (mesh .transform .rotation_euler_deg )if mesh .transform else [0 ,0 ,0 ]
            scl =list (mesh .transform .scale )if mesh .transform else [1 ,1 ,1 ]
            tr =(tr +[0 ,0 ,0 ])[:3 ]
            rot =(rot +[0 ,0 ,0 ])[:3 ]
            scl =(scl +[1 ,1 ,1 ])[:3 ]
            if hasattr (actor ,"SetPosition"):
                actor .SetPosition (*tr )
            if hasattr (actor ,"SetOrientation"):
                actor .SetOrientation (*rot )
            if hasattr (actor ,"SetScale"):
                actor .SetScale (*scl )

    def refresh_meshes (self )->None :
        """Refresh mesh display from model data. Keeps extra actors.
        Recreates geometry only when mesh list changes; otherwise only updates actor positions."""
        if getattr (self ,"_viewport_closed",False ):
            return 
        if not self ._plotter or not pv :
            return 
        try :
            polydata_by_id ,meshes =self ._get_mesh_data ()
            mesh_by_id ={m .mesh_id :m for m in meshes }

            if not self ._meshes_list_changed (polydata_by_id ,mesh_by_id ):
            # Same mesh list: only update transforms, no geometry recreation
                self ._update_mesh_transforms (mesh_by_id )
                if self ._plotter :
                    self ._plotter .render ()
                return # Don't emit mesh_actors_updated - actors unchanged, avoids affine widget reset

                # Mesh list changed: full rebuild
            for mesh_id ,actor in list (self ._mesh_actor_by_id .items ()):
                try :
                    self ._plotter .remove_actor (actor ,reset_camera =False )
                except Exception :
                    pass 
            self ._mesh_actor_by_id .clear ()

            for mesh_id ,poly in polydata_by_id .items ():
                if mesh_id =="__debug_surface__":
                    continue 
                mesh =mesh_by_id .get (mesh_id )
                if not mesh or not mesh .visible :
                    continue 
                actor =self ._plotter .add_mesh (
                poly ,
                color =self ._mesh_color ,
                smooth_shading =False ,
                pickable =self ._pickable ,
                name =f"mesh_{mesh_id }",
                show_edges =False ,
                reset_camera =False ,
                )
                tr =list (mesh .transform .translation )if mesh .transform else [0 ,0 ,0 ]
                rot =list (mesh .transform .rotation_euler_deg )if mesh .transform else [0 ,0 ,0 ]
                scl =list (mesh .transform .scale )if mesh .transform else [1 ,1 ,1 ]
                tr =(tr +[0 ,0 ,0 ])[:3 ]
                rot =(rot +[0 ,0 ,0 ])[:3 ]
                scl =(scl +[1 ,1 ,1 ])[:3 ]
                if hasattr (actor ,"SetPosition"):
                    actor .SetPosition (*tr )
                if hasattr (actor ,"SetOrientation"):
                    actor .SetOrientation (*rot )
                if hasattr (actor ,"SetScale"):
                    actor .SetScale (*scl )
                self ._mesh_actor_by_id [mesh_id ]=actor 

            self ._plotter .reset_camera ()
            self ._plotter .render ()
        except Exception :
            pass # OpenGL context may be destroyed (e.g. window closed)
        self .mesh_actors_updated .emit (self ._mesh_actor_by_id )

    @property 
    def plotter (self ):
        """PyVista plotter for add_extra_actor, etc."""
        return self ._plotter 

    @property 
    def mesh_actor_by_id (self )->dict [str ,Any ]:
        """Current mesh actors by mesh_id."""
        return self ._mesh_actor_by_id 

    def add_extra_actor (self ,name :str ,actor :Any ,*,already_in_scene :bool =False )->None :
        """Add an actor for custom visualization. Survives refresh_meshes.
        If already_in_scene=True, the actor was already added to the plotter (e.g. via add_mesh).
        """
        if getattr (self ,"_viewport_closed",False ):
            return 
        old =self ._extra_actors .pop (name ,None )
        if old and self ._plotter :
            try :
                self ._plotter .remove_actor (old ,reset_camera =False )
            except Exception :
                pass 
        self ._extra_actors [name ]=actor 
        if self ._plotter and not already_in_scene :
            self ._plotter .add_actor (actor ,reset_camera =False )
        if self ._plotter :
            self ._plotter .render ()

    def remove_extra_actor (self ,name :str )->None :
        """Remove extra actor by name."""
        if getattr (self ,"_viewport_closed",False ):
            return 
        actor =self ._extra_actors .pop (name ,None )
        if actor and self ._plotter :
            try :
                self ._plotter .remove_actor (actor ,reset_camera =False )
                self ._plotter .render ()
            except Exception :
                pass 

    def clear_extra_actors (self )->None :
        """Remove all extra actors."""
        for name in list (self ._extra_actors .keys ()):
            self .remove_extra_actor (name )


class MainViewport (UnifiedMeshViewport ):
    """Main viewport used in the main window. Pickable meshes, selection support."""

    def __init__ (
    self ,
    parent :QWidget |None ,
    get_mesh_data :Callable [[],tuple [dict [str ,Any ],list ]],
    refresh_signals :list |None =None ,
    **kwargs :Any ,
    )->None :
        kwargs .setdefault ("pickable",True )
        kwargs .setdefault ("mesh_color","#9A9A9A")
        super ().__init__ (parent ,get_mesh_data ,refresh_signals ,**kwargs )


class BoundaryConditionsViewport (MainViewport ):
    """
    Viewport for Boundary Conditions window. Extends MainViewport, adds BC primitive visualization.
    """

    def __init__ (
    self ,
    parent :QWidget |None ,
    get_mesh_data :Callable [[],tuple [dict [str ,Any ],list ]],
    get_boundary_conditions :Callable [[],list ],
    refresh_signals :list |None =None ,
    **kwargs :Any ,
    )->None :
        self ._get_boundary_conditions =get_boundary_conditions 
        self ._bc_params_by_id :dict [str ,tuple ]={}# bc_id -> (type, params_tuple)
        kwargs ["pickable"]=False 
        super ().__init__ (parent ,get_mesh_data ,refresh_signals ,**kwargs )

    def refresh_meshes (self )->None :
        """Refresh meshes and BC visualization."""
        super ().refresh_meshes ()
        self ._refresh_boundary_conditions ()

    def _bc_list_changed (self ,bcs :list )->bool :
        """Return True if BC list, types, or parameters changed (requires geometry recreation)."""
        def _bc_signature (bc )->tuple :
            bid =getattr (bc ,"bc_id","")
            btype =getattr (bc ,"bc_type","sphere")
            params =getattr (bc ,"parameters",None )or {}
            return (bid ,btype ,tuple (sorted (params .items ())))
        current_sigs =frozenset (_bc_signature (bc )for bc in bcs )
        stored =getattr (self ,"_bc_params_by_id",{})
        if not stored and current_sigs :
            return True # First run, need to create
        our_sigs =frozenset (
        (bid ,data [0 ],data [1 ])for bid ,data in stored .items ()
        )
        return current_sigs !=our_sigs 

    def _update_bc_transforms (self ,bcs :list )->None :
        """Update BC actor positions only (no geometry recreation)."""
        bc_by_id ={getattr (bc ,"bc_id",str (i )):bc for i ,bc in enumerate (bcs )}
        for bc_id ,actor in list (self ._extra_actors .items ()):
            if not bc_id .startswith ("bc_"):
                continue 
            bid =bc_id [3 :]
            bc =bc_by_id .get (bid )
            if not bc :
                continue 
            tr =list (bc .transform .translation )if bc .transform else [0 ,0 ,0 ]
            rot =list (bc .transform .rotation_euler_deg )if bc .transform else [0 ,0 ,0 ]
            scl =list (bc .transform .scale )if bc .transform else [1 ,1 ,1 ]
            tr =(tr +[0 ,0 ,0 ])[:3 ]
            rot =(rot +[0 ,0 ,0 ])[:3 ]
            scl =(scl +[1 ,1 ,1 ])[:3 ]
            if hasattr (actor ,"SetPosition"):
                actor .SetPosition (*tr )
            if hasattr (actor ,"SetOrientation"):
                actor .SetOrientation (*rot )
            if hasattr (actor ,"SetScale"):
                actor .SetScale (*scl )

    def _refresh_boundary_conditions (self )->None :
        """Update boundary condition primitives in the viewport.
        Recreates only when BC list changes; otherwise only updates actor positions."""
        if getattr (self ,"_viewport_closed",False ):
            return 
        if not self ._plotter or not pv :
            return 
        bcs =self ._get_boundary_conditions ()
        if not self ._bc_list_changed (bcs ):
        # Same BC list: only update transforms
            self ._update_bc_transforms (bcs )
            try :
                if self ._plotter :
                    self ._plotter .render ()
            except Exception :
                pass 
            return 
            # BC list changed: full rebuild
        for name in list (self ._extra_actors .keys ()):
            if name .startswith ("bc_"):
                self .remove_extra_actor (name )
        self ._bc_params_by_id .clear ()
        for bc in bcs :
            try :
                actor =self ._create_bc_primitive (bc )
                if actor :
                    self .add_extra_actor (f"bc_{bc .bc_id }",actor )
                    params =getattr (bc ,"parameters",None )or {}
                    self ._bc_params_by_id [bc .bc_id ]=(
                    getattr (bc ,"bc_type","sphere"),
                    tuple (sorted (params .items ())),
                    )
            except Exception :
                pass 
        try :
            if self ._plotter :
                self ._plotter .render ()
        except Exception :
            pass # OpenGL context may be destroyed (e.g. window closed)

    def _create_bc_primitive (self ,bc )->Any :
        """Create PyVista primitive for a boundary condition."""
        tr =list (bc .transform .translation )if bc .transform else [0 ,0 ,0 ]
        rot =list (bc .transform .rotation_euler_deg )if bc .transform else [0 ,0 ,0 ]
        scl =list (bc .transform .scale )if bc .transform else [1 ,1 ,1 ]
        tr =(tr +[0 ,0 ,0 ])[:3 ]
        rot =(rot +[0 ,0 ,0 ])[:3 ]
        scl =(scl +[1 ,1 ,1 ])[:3 ]
        params =bc .parameters or {}
        bc_type =getattr (bc ,"bc_type","sphere")
        if bc_type =="sphere":
            r =params .get ("radius",1.0 )
            mesh =pv .Sphere (radius =r ,center =(0 ,0 ,0 ))
        elif bc_type =="box":
            sx =params .get ("box_x",1.0 )
            sy =params .get ("box_y",1.0 )
            sz =params .get ("box_z",1.0 )
            mesh =pv .Box (bounds =(-sx /2 ,sx /2 ,-sy /2 ,sy /2 ,-sz /2 ,sz /2 ))
        elif bc_type =="cylinder":
            r =params .get ("cylinder_radius",1.0 )
            h =params .get ("cylinder_height",1.0 )
            mesh =pv .Cylinder (radius =r ,height =h ,center =(0 ,0 ,0 ))
        elif bc_type =="tube":
            length =params .get ("tube_length",10.0 )
            r_inner =params .get ("tube_radius_inner",1.0 )
            r_outer =params .get ("tube_radius_outer",2.0 )
            mesh =pv .CylinderStructured (
            radius =[r_inner ,r_outer ],
            height =length ,
            center =(0 ,0 ,0 ),
            direction =(0 ,0 ,1 ),
            )
        else :
            mesh =pv .Sphere (radius =1.0 ,center =(0 ,0 ,0 ))
        actor =self ._plotter .add_mesh (
        mesh ,
        color ="#4080FF",
        opacity =0.4 ,
        show_edges =True ,
        pickable =False ,
        reset_camera =False ,
        )
        if hasattr (actor ,"SetPosition"):
            actor .SetPosition (*tr )
        if hasattr (actor ,"SetOrientation"):
            actor .SetOrientation (*rot )
        if hasattr (actor ,"SetScale"):
            actor .SetScale (*scl )
        return actor 

    def add_extra_actor (self ,name :str ,actor :Any ,*,already_in_scene :bool =False )->None :
        """BC primitives are added via add_mesh, so always already_in_scene."""
        if name .startswith ("bc_"):
            already_in_scene =True 
        super ().add_extra_actor (name ,actor ,already_in_scene =already_in_scene )


class TopologyViewport (QWidget ):
    'Viewport is only for displaying the generated 3D topology.\n    Meshes are not displayed. The topology is shown as parallelepipeds (boxes).\n    The layer slider hides FEs with a Z center higher than the specified value.'

    layer_range_changed =Signal (float ,float )# (z_min, z_max) when changing topology

    def __init__ (self ,parent :QWidget |None =None )->None :
        super ().__init__ (parent )
        self ._plotter =None 
        self ._interactor =None 
        self ._topology_actor =None 
        self ._viewport_closed =False 
        self ._topology_dict =None 
        self ._layer_cutoff_z =None # None = show all
        self ._lod =5 # Level of detail: 1:N — render every Nth CE (default 1:5)

        if QtInteractor is None or pv is None :
            self ._interactor =ViewportPlaceholder ()
            layout =QVBoxLayout (self )
            layout .addWidget (self ._interactor )
            return 

        self ._plotter =QtInteractor (self )
        self ._plotter .set_background ("#1f1f1f")
        layout =QVBoxLayout (self )
        layout .setContentsMargins (0 ,0 ,0 ,0 )
        layout .addWidget (self ._plotter .interactor )
        self ._interactor =self ._plotter .interactor 
        _setup_lighting (self ._plotter )
        self ._add_scene_basics ()

    def _add_scene_basics (self )->None :
        if self ._plotter :
            self ._plotter .add_axes ()
            self ._plotter .show_grid (color ="#555555")

    def close_viewport (self )->None :
        self ._viewport_closed =True 
        if self ._plotter :
            try :
                self ._plotter .close ()
            except Exception :
                pass 
            self ._plotter =None 

    def set_topology (self ,topology :dict |None )->None :
        'Sets the topology to be displayed.\n        topology: dict with element_position_xyz [n,3], element_size_xyz [n,3]'
        if getattr (self ,"_viewport_closed",False ):
            return 
        if not self ._plotter or not pv :
            return 

        self ._topology_dict =topology 
        self ._layer_cutoff_z =None 
        self ._do_reset_camera =True 
        if topology :
            pos =topology .get ("element_position_xyz")
            if pos is not None and pos .size >0 :
                import numpy as np 
                z =np .asarray (pos ,dtype =np .float64 )[:,2 ]
                self .layer_range_changed .emit (float (np .min (z )),float (np .max (z )))
        self ._render_topology ()

    def set_layer_cutoff (self ,z_max :float |None )->None :
        'Hide FE with center Z > z_max. None = show all.'
        self ._layer_cutoff_z =z_max 
        self ._render_topology ()

    def set_lod (self ,lod :int )->None :
        'Level of detail: 1:N — render every Nth CE (1=all, 5=every 5th).'
        lod =max (1 ,int (lod ))
        if self ._lod !=lod :
            self ._lod =lod 
            self ._render_topology ()

    def _render_topology (self )->None :
        'Construct and draw mesh topologies taking into account layer_cutoff.'
        if getattr (self ,"_viewport_closed",False ):
            return 
        if not self ._plotter or not pv :
            return 

        if self ._topology_actor :
            try :
                self ._plotter .remove_actor (self ._topology_actor ,reset_camera =False )
            except Exception :
                pass 
            self ._topology_actor =None 

        topology =self ._topology_dict 
        if not topology :
            if self ._plotter :
                self ._plotter .render ()
            return 

        pos =topology .get ("element_position_xyz")
        size =topology .get ("element_size_xyz")
        if pos is None or size is None or pos .size ==0 :
            if self ._plotter :
                self ._plotter .render ()
            return 

        try :
            import numpy as np 
            pos_arr =np .asarray (pos ,dtype =np .float64 )
            size_arr =np .asarray (size ,dtype =np .float64 )
            if pos_arr .ndim !=2 or pos_arr .shape [1 ]!=3 :
                return 
            if size_arr .shape !=pos_arr .shape :
                return 

            n_full =pos_arr .shape [0 ]
            # LOD: render every _lod CE (1:1=all, 1:5=every 5th)
            lod =max (1 ,getattr (self ,"_lod",1 ))
            indices =np .arange (0 ,n_full ,lod ,dtype =np .intp )
            # Filter by layer: hide FE with center Z > layer_cutoff
            cutoff =self ._layer_cutoff_z 
            if cutoff is not None :
                z_vals =pos_arr [indices ,2 ]
                indices =indices [z_vals <=cutoff ]
            if len (indices )==0 :
                if self ._plotter :
                    self ._plotter .render ()
                return 
            pos_arr =pos_arr [indices ]
            size_arr =size_arr [indices ]
            boundary_mask =topology .get ("boundary_mask_elements")
            if boundary_mask is not None :
                bnd =np .asarray (boundary_mask ,dtype =np .int32 ).ravel ()
                if bnd .size ==n_full :
                    boundary_mask =bnd [indices ]
                else :
                    boundary_mask =None 

            n =pos_arr .shape [0 ]
            points_list =[]
            cells_list =[]
            voff =0 
            for i in range (n ):
                c =pos_arr [i ]
                h =size_arr [i ]/2.0 
                verts =np .array ([
                [c [0 ]-h [0 ],c [1 ]-h [1 ],c [2 ]-h [2 ]],
                [c [0 ]+h [0 ],c [1 ]-h [1 ],c [2 ]-h [2 ]],
                [c [0 ]+h [0 ],c [1 ]+h [1 ],c [2 ]-h [2 ]],
                [c [0 ]-h [0 ],c [1 ]+h [1 ],c [2 ]-h [2 ]],
                [c [0 ]-h [0 ],c [1 ]-h [1 ],c [2 ]+h [2 ]],
                [c [0 ]+h [0 ],c [1 ]-h [1 ],c [2 ]+h [2 ]],
                [c [0 ]+h [0 ],c [1 ]+h [1 ],c [2 ]+h [2 ]],
                [c [0 ]-h [0 ],c [1 ]+h [1 ],c [2 ]+h [2 ]],
                ],dtype =np .float64 )
                points_list .append (verts )
                cells_list .append ([8 ,voff ,voff +1 ,voff +2 ,voff +3 ,voff +4 ,voff +5 ,voff +6 ,voff +7 ])
                voff +=8 

            pts =np .vstack (points_list )
            cells_flat =np .array ([x for cell in cells_list for x in cell ],dtype =np .int64 )
            cell_types_arr =np .full (n ,12 ,dtype =np .uint8 )
            ug =pv .UnstructuredGrid (cells_flat ,cell_types_arr ,pts )

            # Boundary FE - red, internal - blue
            if boundary_mask is not None and boundary_mask .size ==n :
                ug .cell_data ["boundary"]=boundary_mask 
                self ._topology_actor =self ._plotter .add_mesh (
                ug ,
                scalars ="boundary",
                cmap =["#6B9BD1","#E85D4C"],
                show_edges =True ,
                opacity =1.0 ,
                reset_camera =getattr (self ,"_do_reset_camera",False ),
                )
            else :
                self ._topology_actor =self ._plotter .add_mesh (
                ug ,
                color ="#6B9BD1",
                show_edges =True ,
                opacity =1.0 ,
                reset_camera =getattr (self ,"_do_reset_camera",False ),
                )
            if getattr (self ,"_do_reset_camera",False ):
                self ._plotter .reset_camera ()
                self ._do_reset_camera =False 
            self ._plotter .render ()
        except Exception :
            pass 

    @property 
    def plotter (self ):
        return self ._plotter 
