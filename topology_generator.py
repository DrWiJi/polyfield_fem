# -*- coding: utf-8 -*-
'Generator of volumetric 3D topology from data model meshes.\n- Solid: voxel mesh (PyVista voxelize_rectilinear).\n- Membrane/Sensor: planar generation - the mesh is considered a plane oriented along XY/XZ/YZ.\n  FE thickness = mesh thickness, mesh pitch = element_size_mm from settings. FEs have unequal sizes.\n\nOutput format (compatible with diaphragm_opencl.PlanarDiaphragmOpenCL.set_custom_topology):\n- element_position_xyz: [n, 3] float64 — coordinates of FE centers in the global coordinate system\n- element_size_xyz: [n, 3] float64 — brick half-extents in **global** X,Y,Z (sx,sy,sz), same convention as voxel solids;\n  required by OpenCL face_area_from_size and spring rest lengths (XY/XZ/YZ membranes).\n- neighbors: [n, 6] int32 — FACE_DIRS: +X,-X,+Y,-Y,+Z,-Z; -1 = no neighbor\n- material_index: [n] uint8 — row index in material_props (0=membrane, 4=sensor, ...)\n- boundary_mask_elements: [n] int32 - 1 = boundary (perimeter), 0 = inner\n\nAdditional data for diaphragm_opencl (NOT from topology):\n- material_props: [n_materials, 8] float64 - ..., coupling_recv, acoustic_inject.\n  acoustic_inject>0 on passive layers gives feedback v→p (scattering/"echo" from FE); 0 = energy goes into solid without re-radiation into the grid.\n  External contour of the air grid: Sommerfeld + sponge - weak reflections from the border of the box (analogous to an open field).\n  Must contain rows for all material_index from the topology. Call set_material_library().\n- material_key_to_index: mapping the material_key of the mesh -> index in material_props. Must match\n  with the order of materials in the library (MaterialLibraryModel.materials).'

from __future__ import annotations 

import math 
from pathlib import Path 
from typing import Any 

import numpy as np 

try :
    import pyvista as pv 
except ImportError :
    pv =None 

try :
    import trimesh 
except ImportError :
    trimesh =None 

from project_model import BoundaryCondition ,MeshEntity ,MeshTransform 

# Compatible with diaphragm_opencl
FACE_DIRS =6 # +X, -X, +Y, -Y, +Z, -Z
MAT_MEMBRANE =np .uint8 (0 )
MAT_SENSOR =np .uint8 (4 )
MAT_FOAM_VE3015 =np .uint8 (1 )
MAT_AIR =np .uint8 (10 )

# Acoustic boundary condition kinds for air grid missing-neighbor faces.
# Stored in air_neighbor_absorb_u8 for backwards compatibility (historically 0/1 absorb mask).
AIR_BC_INTERIOR =np .uint8 (0 )# neighbor exists (not a boundary face)
AIR_BC_OPEN =np .uint8 (1 )# radiating/open boundary (outer domain boundary)
AIR_BC_RIGID =np .uint8 (2 )# rigid wall (solid boundary / blocked face)


def _estimate_model_unit_scale (extent_max :float )->float :
    'Heuristic units: CAD meshes are usually in mm. Small extents are treated as meters.'
    return 1.0 if extent_max >1.0 else 1e-3


def _estimate_global_unit_scale_for_solids (
solid_data :list [tuple [np .ndarray ,np .ndarray ,int ,str ,str ,str ]],
)->float :
    'Use one unit scale for all solid meshes to avoid mixed-grid mismatch.'
    if not solid_data :
        return 1.0
    extents =[]
    for verts ,_faces ,_mat_idx ,_name ,_mesh_id ,_role in solid_data :
        if verts is None or np .asarray (verts ).size ==0 :
            continue
        vv =np .asarray (verts ,dtype =np .float64 )
        ext =np .max (vv ,axis =0 )-np .min (vv ,axis =0 )
        extents .append (float (np .max (ext )))
    if not extents :
        return 1.0
    return _estimate_model_unit_scale (float (np .median (np .asarray (extents ,dtype =np .float64 ))))


def _harmonize_solid_grid_and_neighbors (
positions :np .ndarray ,
sizes :np .ndarray ,
neighbors :np .ndarray ,
material_index :np .ndarray ,
mesh_ids :list [str ],
*,
element_size_mm :float ,
log_fn =None ,
)->tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
    'Snap solid voxels to one common grid and rebuild solid neighbors globally.'
    def _log (msg :str )->None :
        if log_fn :
            log_fn (msg )

    if positions .size ==0 :
        return positions ,sizes ,neighbors

    mats =np .asarray (material_index ,dtype =np .int32 ).ravel ()
    solid_mask =(mats !=int (MAT_MEMBRANE ))&(mats !=int (MAT_SENSOR ))
    solid_idx =np .flatnonzero (solid_mask )
    if solid_idx .size ==0 :
        return positions ,sizes ,neighbors

    pos =np .ascontiguousarray (positions .copy (),dtype =np .float64 )
    siz =np .ascontiguousarray (sizes .copy (),dtype =np .float64 )
    nbh =np .ascontiguousarray (neighbors .copy (),dtype =np .int32 )

    mins =pos [solid_idx ]-0.5 *siz [solid_idx ]
    maxs =pos [solid_idx ]+0.5 *siz [solid_idx ]
    ext =np .max (maxs ,axis =0 )-np .min (mins ,axis =0 )
    unit_scale =_estimate_model_unit_scale (float (np .max (ext )))
    step =max (float (element_size_mm )*unit_scale ,1e-12 )

    origin =np .min (mins ,axis =0 )
    corner =pos [solid_idx ]-0.5 *siz [solid_idx ]
    ijk =np .rint ((corner -origin [None ,:])/(step +1e-30 )).astype (np .int64 )
    snapped_corner =origin [None ,:]+ijk .astype (np .float64 )*step
    pos [solid_idx ]=snapped_corner +0.5 *step
    siz [solid_idx ]=step

    key_to_idx ={}
    n_dup =0
    for row ,gidx in enumerate (solid_idx .tolist ()):
        key =(int (ijk [row ,0 ]),int (ijk [row ,1 ]),int (ijk [row ,2 ]))
        if key in key_to_idx :
            n_dup +=1
            continue
        key_to_idx [key ]=int (gidx )

    deltas =((1 ,0 ,0 ),(-1 ,0 ,0 ),(0 ,1 ,0 ),(0 ,-1 ,0 ),(0 ,0 ,1 ),(0 ,0 ,-1 ))
    cross_links =0
    for row ,gidx in enumerate (solid_idx .tolist ()):
        key =(int (ijk [row ,0 ]),int (ijk [row ,1 ]),int (ijk [row ,2 ]))
        mesh_id =mesh_ids [gidx ]if gidx <len (mesh_ids )else ""
        for d ,(dx ,dy ,dz )in enumerate (deltas ):
            ng =key_to_idx .get ((key [0 ]+dx ,key [1 ]+dy ,key [2 ]+dz ),-1 )
            nbh [gidx ,d ]=int (ng )
            if ng >=0 :
                n_mesh =mesh_ids [ng ]if ng <len (mesh_ids )else ""
                if n_mesh !=mesh_id :
                    cross_links +=1

    _log (
        f"  Solid grid harmonized: step={step :.4e} m-equivalent, "
        f"solid={int (solid_idx .size )}, duplicates={int (n_dup )}, "
        f"cross-mesh links={int (cross_links //2 )}"
    )
    return pos ,siz ,nbh


def _generate_regular_air_topology (
solid_positions :np .ndarray ,
solid_sizes :np .ndarray ,
solid_boundary_mask :np .ndarray ,
*,
solid_material_index :np .ndarray |None =None ,
air_material_index :int ,
element_size_mm :float ,
padding_mm :float ,
    air_gap_layers :int =2 ,
max_air_cells :int =1_200_000_000 ,
acoustic_boundary_conditions :list [BoundaryCondition ]|None =None ,
log_fn =None ,
)->dict [str ,np .ndarray ]:
    'Generate a regular 3D air FE grid around solids and build solid↔air connectivity map.'
    def _log (msg :str )->None :
        if log_fn :
            log_fn (msg )

    empty ={
    "air_element_position_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
    "air_element_size_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
    "air_neighbors":np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
    "air_neighbor_absorb_u8":np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 ),
    "air_material_index":np .zeros (0 ,dtype =np .uint8 ),
    "air_boundary_mask_elements":np .zeros (0 ,dtype =np .int32 ),
    "solid_to_air_index":np .full (solid_positions .shape [0 ],-1 ,dtype =np .int32 ),
    "solid_to_air_index_plus":np .full (solid_positions .shape [0 ],-1 ,dtype =np .int32 ),
    "solid_to_air_index_minus":np .full (solid_positions .shape [0 ],-1 ,dtype =np .int32 ),
    "air_grid_shape":np .zeros (3 ,dtype =np .int32 ),
    }
    if solid_positions .size ==0 :
        return empty

    _ =solid_boundary_mask # API: was used for perimeter-only mapping; all solids are mapped now.

    mins =solid_positions -0.5 *solid_sizes
    maxs =solid_positions +0.5 *solid_sizes
    bmin =np .min (mins ,axis =0 )
    bmax =np .max (maxs ,axis =0 )
    ext =bmax -bmin
    ext_max =float (np .max (ext )) if ext .size else 0.0
    unit_scale =_estimate_model_unit_scale (ext_max )

    base_step =float (element_size_mm )*unit_scale
    if base_step <=0.0 :
        base_step =1e-3
    # Use one common step for solids and air so exposed solid faces always sit on the same grid.
    step =base_step

    # Guarantee at least a small air "gap" around the entire topology so that solid outer faces
    # (including ±Z for voxel solids) see air neighbors and do not degenerate into missing-neighbor
    # interface forces. Two layers is usually enough to avoid "no air on one side" pathologies.
    gap_layers =max (0 ,int (air_gap_layers ))
    pad_base =max (float (padding_mm )*unit_scale ,step )
    pad =pad_base +gap_layers *step
    x0 ,y0 ,z0 =(bmin -pad )
    x1 ,y1 ,z1 =(bmax +pad )

    def _dims_for_step (s :float )->tuple [int ,int ,int ]:
        nx =max (1 ,int (np .ceil ((x1 -x0 )/(s +1e-30 ))))
        ny =max (1 ,int (np .ceil ((y1 -y0 )/(s +1e-30 ))))
        nz =max (1 ,int (np .ceil ((z1 -z0 )/(s +1e-30 ))))
        return nx ,ny ,nz

    requested_step =float (step )
    nx ,ny ,nz =_dims_for_step (step )
    est =nx *ny *nz
    if est >max_air_cells :
        scale =(est /max_air_cells )**(1.0 /3.0 )
        step *=scale *1.05
        nx ,ny ,nz =_dims_for_step (step )
        est =nx *ny *nz

    _log (f"--- Stage 4: Air grid generation ---")
    _log (f"  Air bbox: x[{x0 :.4e}, {x1 :.4e}] y[{y0 :.4e}, {y1 :.4e}] z[{z0 :.4e}, {z1 :.4e}]")
    if abs (step -requested_step )>1e-15 :
        _log (
            f"  [air][warn] Requested air step {requested_step :.4e} increased to {step :.4e} "
            f"to satisfy max_air_cells={int (max_air_cells )}."
        )
    _log (f"  Air step: {step :.4e} m-equivalent, grid: {nx}×{ny}×{nz} ({est} cells) before carving solids")

    solid_mask =np .zeros ((nx ,ny ,nz ),dtype =bool )

    inv =1.0 /(step +1e-30 )
    for i in range (solid_positions .shape [0 ]):
        mn =mins [i ]
        mx =maxs [i ]
        ix0 =max (0 ,min (nx -1 ,int (np .floor ((mn [0 ]-x0 )*inv ))))
        iy0 =max (0 ,min (ny -1 ,int (np .floor ((mn [1 ]-y0 )*inv ))))
        iz0 =max (0 ,min (nz -1 ,int (np .floor ((mn [2 ]-z0 )*inv ))))
        ix1 =max (0 ,min (nx -1 ,int (np .floor ((mx [0 ]-x0 )*inv ))))
        iy1 =max (0 ,min (ny -1 ,int (np .floor ((mx [1 ]-y0 )*inv ))))
        iz1 =max (0 ,min (nz -1 ,int (np .floor ((mx [2 ]-z0 )*inv ))))
        solid_mask [ix0 :ix1 +1 ,iy0 :iy1 +1 ,iz0 :iz1 +1 ]=True

    air_mask =~solid_mask
    air_cells =np .argwhere (air_mask )
    if air_cells .size ==0 :
        _log ('  Air grid is empty after solid carving.')
        return empty

    n_air =int (air_cells .shape [0 ])
    _log (f"  Air cells after carving: {n_air}")

    air_map =np .full ((nx ,ny ,nz ),-1 ,dtype =np .int32 )
    air_map [air_cells [:,0 ],air_cells [:,1 ],air_cells [:,2 ]]=np .arange (n_air ,dtype =np .int32 )

    air_pos =np .zeros ((n_air ,3 ),dtype =np .float64 )
    air_pos [:,0 ]=x0 +(air_cells [:,0 ].astype (np .float64 )+0.5 )*step
    air_pos [:,1 ]=y0 +(air_cells [:,1 ].astype (np .float64 )+0.5 )*step
    air_pos [:,2 ]=z0 +(air_cells [:,2 ].astype (np .float64 )+0.5 )*step
    air_size =np .full ((n_air ,3 ),step ,dtype =np .float64 )
    air_neighbors =np .full ((n_air ,FACE_DIRS ),-1 ,dtype =np .int32 )
    air_neighbor_absorb =np .zeros ((n_air ,FACE_DIRS ),dtype =np .uint8 )
    air_boundary =np .zeros (n_air ,dtype =np .int32 )

    deltas =[(1 ,0 ,0 ),(-1 ,0 ,0 ),(0 ,1 ,0 ),(0 ,-1 ,0 ),(0 ,0 ,1 ),(0 ,0 ,-1 )]
    for idx ,(ix ,iy ,iz )in enumerate (air_cells ):
        if ix ==0 or iy ==0 or iz ==0 or ix ==nx -1 or iy ==ny -1 or iz ==nz -1 :
            air_boundary [idx ]=1
        for d ,(dx ,dy ,dz )in enumerate (deltas ):
            ni ,nj ,nk =ix +dx ,iy +dy ,iz +dz
            if ni <0 or ni >=nx or nj <0 or nj >=ny or nk <0 or nk >=nz :
                # Domain outer boundary: open/radiating.
                air_neighbor_absorb [idx ,d ]=AIR_BC_OPEN
            else :
                j =int (air_map [ni ,nj ,nk ])
                air_neighbors [idx ,d ]=j
                if j <0 :
                    # Solid boundary (carved-out neighbor): rigid wall by default.
                    air_neighbor_absorb [idx ,d ]=AIR_BC_RIGID

    # Optional: override air boundary kinds using project BoundaryCondition primitives.
    # Conservative: affects only boundary air cells and only missing-neighbor faces.
    if acoustic_boundary_conditions :
        bcs =list (acoustic_boundary_conditions )
        if bcs :
            _log (f"  Acoustic BCs (air): {len (bcs )} primitives")
        for bc in bcs :
            flags =dict (getattr (bc ,"flags",{})or {})
            bc_kind =None
            if bool (flags .get ("acoustic_open",False )):
                bc_kind =AIR_BC_OPEN
            elif bool (flags .get ("acoustic_rigid",False )):
                bc_kind =AIR_BC_RIGID
            if bc_kind is None :
                continue
            params =dict (getattr (bc ,"parameters",{})or {})
            tr =getattr (bc ,"transform",None )
            if tr is None :
                continue
            bmask =air_boundary .astype (bool ,copy =False )
            if not np .any (bmask ):
                continue
            pts =air_pos [bmask ]
            M_inv =_build_inverse_transform_matrix (
                list (tr .translation ),
                list (tr .rotation_euler_deg ),
                list (tr .scale ),
            )
            local =_apply_transform (pts ,M_inv )
            inside =np .zeros (local .shape [0 ],dtype =bool )
            for i in range (local .shape [0 ]):
                if _point_inside_bc_primitive (local [i ],bc .bc_type ,params ):
                    inside [i ]=True
            if not np .any (inside ):
                continue
            idxs =np .flatnonzero (bmask )[inside ]
            miss =air_neighbors [idxs ]<0
            air_neighbor_absorb [idxs ]=np .where (miss ,bc_kind ,air_neighbor_absorb [idxs ]).astype (np .uint8 ,copy =False )

    # Map only likely-radiating solids to air:
    # - thin layers (membrane/sensor-like) are mapped everywhere;
    # - bulk solids are mapped only near solid-air interfaces (surface-adjacent).
    # This prevents interior bulk from collapsing into a few nearest air voxels and inflating CSR rows.

    n_sol =int (solid_positions .shape [0 ])
    _air_kdtree =None
    try :
        from scipy .spatial import cKDTree 
        _air_kdtree =cKDTree (air_pos )
    except ImportError :
        _log (
        "  [warn] scipy.spatial.cKDTree unavailable; solid↔air NN uses chunked brute-force "
        "(install scipy for large meshes)."
        )

    def _nearest_air_indices_batch (points :np .ndarray )->np .ndarray :
        """Nearest air cell index in 0..n_air-1 for each query point [n,3]."""
        pts =np .ascontiguousarray (points ,dtype =np .float64 )
        if pts .size ==0 :
            return np .zeros ((0 ,),dtype =np .int32 )
        if _air_kdtree is not None :
            try :
                _ ,idx =_air_kdtree .query (pts ,k =1 ,workers =-1 )
            except TypeError :
                _ ,idx =_air_kdtree .query (pts ,k =1 )
            return np .asarray (idx ,dtype =np .int32 ).ravel ()
        # No SciPy: chunked brute-force (install scipy for large models).
        n =pts .shape [0 ]
        out =np .empty (n ,dtype =np .int32 )
        chunk =max (256 ,min (2048 ,max (1 ,50_000_000 //max (1 ,n_air ))))
        for s in range (0 ,n ,chunk ):
            sub =pts [s :s +chunk ]
            d2 =np .sum ((sub [:,None ,:]-air_pos [None ,:,:])**2 ,axis =2 )
            out [s :s +chunk ]=np .argmin (d2 ,axis =1 ).astype (np .int32 )
        return out 

    # Candidate mask for FE->air coupling.
    # "Thin" means strongly anisotropic layers (membrane/sensor-like), not regular voxels.
    min_size =np .min (solid_sizes ,axis =1 )
    max_size =np .max (solid_sizes ,axis =1 )
    thin_ratio =min_size /np .maximum (max_size ,1e-30 )
    thin_mask_geom =thin_ratio <=0.40
    membrane_sensor_mask =np .zeros (n_sol ,dtype =bool )
    if solid_material_index is not None and np .asarray (solid_material_index ).size ==n_sol :
        mats_for_thin =np .asarray (solid_material_index ,dtype =np .int32 ).ravel ()
        membrane_sensor_mask =(mats_for_thin ==int (MAT_MEMBRANE ))|(mats_for_thin ==int (MAT_SENSOR ))
        thin_mask =membrane_sensor_mask |thin_mask_geom
    else :
        membrane_sensor_mask =thin_mask_geom .copy ()
        thin_mask =thin_mask_geom

    # Surface-adjacent bulk detection from the carved occupancy grid.
    ix =np .clip (np .floor ((solid_positions [:,0 ]-x0 )*inv ).astype (np .int64 ),0 ,nx -1 )
    iy =np .clip (np .floor ((solid_positions [:,1 ]-y0 )*inv ).astype (np .int64 ),0 ,ny -1 )
    iz =np .clip (np .floor ((solid_positions [:,2 ]-z0 )*inv ).astype (np .int64 ),0 ,nz -1 )
    interior =solid_mask [ix ,iy ,iz ]
    surf_mask =np .zeros (n_sol ,dtype =bool )
    for dx ,dy ,dz in deltas :
        ni =np .clip (ix +dx ,0 ,nx -1 )
        nj =np .clip (iy +dy ,0 ,ny -1 )
        nk =np .clip (iz +dz ,0 ,nz -1 )
        surf_mask |=(~solid_mask [ni ,nj ,nk ])
    map_mask =(thin_mask |surf_mask )&interior

    # Prefer local surface-adjacent air for mapped solids.
    # Global nearest can collapse many elements from different regions into one air voxel.
    map_rows =np .flatnonzero (map_mask )
    local_map =np .full (map_rows .size ,-1 ,dtype =np .int32 )
    for j ,e_idx in enumerate (map_rows ):
        iix =int (ix [e_idx ]); iiy =int (iy [e_idx ]); iiz =int (iz [e_idx ])
        best_d2 =np .inf
        best_air =-1
        for dx ,dy ,dz in deltas :
            ni ,nj ,nk =iix +dx ,iiy +dy ,iiz +dz
            if ni <0 or ni >=nx or nj <0 or nj >=ny or nk <0 or nk >=nz :
                continue
            a =int (air_map [ni ,nj ,nk ])
            if a <0 :
                continue
            ddx =air_pos [a ,0 ]-solid_positions [e_idx ,0 ]
            ddy =air_pos [a ,1 ]-solid_positions [e_idx ,1 ]
            ddz =air_pos [a ,2 ]-solid_positions [e_idx ,2 ]
            d2 =ddx *ddx +ddy *ddy +ddz *ddz
            if d2 <best_d2 :
                best_d2 =d2
                best_air =a
        local_map [j ]=best_air

    solid_to_air =np .full (n_sol ,-1 ,dtype =np .int32 )
    n_local_hit =0
    n_local_miss =0
    if map_rows .size >0 :
        solid_to_air [map_rows ]=local_map
        miss =(local_map <0 )
        n_local_miss =int (np .sum (miss ))
        n_local_hit =int (map_rows .size -n_local_miss )
        if np .any (miss ):
            fallback =_nearest_air_indices_batch (solid_positions [map_rows [miss ]])
            solid_to_air [map_rows [miss ]]=fallback
    # Keep center mapping only for surface-adjacent or explicitly thin elements.
    # Interior bulk elements do not couple directly to air and stay unmapped (-1).
    n_global_fill =0
    n_unmapped =int (np .sum (solid_to_air <0 ))

    solid_to_air_plus =np .full (n_sol ,-1 ,dtype =np .int32 )
    solid_to_air_minus =np .full (n_sol ,-1 ,dtype =np .int32 )
    # Bilateral coupling for membrane/sensor:
    # find nearest air voxel along +normal and -normal directions from each FE center voxel.
    # Strict rule: every membrane/sensor FE must have two distinct side cells.
    axis =np .argmin (solid_sizes ,axis =1 )
    bnd_mask =np .zeros (n_sol ,dtype =bool )
    if solid_boundary_mask is not None :
        sb =np .asarray (solid_boundary_mask ,dtype =np .int32 ).ravel ()
        if sb .size ==n_sol :
            bnd_mask =(sb !=0 )
    # Strict bilateral requirement applies to active (non-boundary) membrane/sensor elements.
    ms_rows =np .flatnonzero (membrane_sensor_mask &(~bnd_mask ))

    def _search_air_along_axis (iix :int ,iiy :int ,iiz :int ,ax :int ,sgn :int )->int :
        ci ,cj ,ck =iix ,iiy ,iiz
        for _ in range (max (nx ,ny ,nz )):
            if ax ==0 :
                ci +=sgn
            elif ax ==1 :
                cj +=sgn
            else :
                ck +=sgn
            if ci <0 or ci >=nx or cj <0 or cj >=ny or ck <0 or ck >=nz :
                return -1
            a =int (air_map [ci ,cj ,ck ])
            if a >=0 :
                return a
        return -1

    def _search_air_along_axis_with_lateral (
        e_idx :int ,
        iix :int ,
        iiy :int ,
        iiz :int ,
        ax :int ,
        sgn :int ,
        *,
        lateral_radius :int =3 ,
    )->int :
        """Directional local fallback: search nearby rays on the requested side only."""
        best_air =-1
        best_d2 =np .inf
        p0 =solid_positions [e_idx ]
        axes_lat =[d for d in (0 ,1 ,2 )if d !=ax ]
        a0 ,a1 =axes_lat [0 ],axes_lat [1 ]
        for r in range (0 ,max (0 ,int (lateral_radius ))+1 ):
            for du in range (-r ,r +1 ):
                for dv in range (-r ,r +1 ):
                    if r >0 and max (abs (du ),abs (dv ))!=r :
                        continue
                    c =[iix ,iiy ,iiz ]
                    c [a0 ]+=du
                    c [a1 ]+=dv
                    if c [0 ]<0 or c [0 ]>=nx or c [1 ]<0 or c [1 ]>=ny or c [2 ]<0 or c [2 ]>=nz :
                        continue
                    ci ,cj ,ck =int (c [0 ]),int (c [1 ]),int (c [2 ])
                    for _ in range (max (nx ,ny ,nz )):
                        if ax ==0 :
                            ci +=sgn
                        elif ax ==1 :
                            cj +=sgn
                        else :
                            ck +=sgn
                        if ci <0 or ci >=nx or cj <0 or cj >=ny or ck <0 or ck >=nz :
                            break
                        a =int (air_map [ci ,cj ,ck ])
                        if a >=0 :
                            d =air_pos [a ]-p0
                            d2 =float (d [0 ]*d [0 ]+d [1 ]*d [1 ]+d [2 ]*d [2 ])
                            if d2 <best_d2 :
                                best_d2 =d2
                                best_air =a
                            break
            if best_air >=0 :
                break
        return best_air

    n_side_miss =0
    n_side_same =0
    bad_rows =[]
    for e in ms_rows :
        iix =int (ix [e ]); iiy =int (iy [e ]); iiz =int (iz [e ])
        ax =int (axis [e ])
        cp =_search_air_along_axis (iix ,iiy ,iiz ,ax ,+1 )
        cm =_search_air_along_axis (iix ,iiy ,iiz ,ax ,-1 )
        if cp <0 :
            cp =_search_air_along_axis_with_lateral (int (e ),iix ,iiy ,iiz ,ax ,+1 )
        if cm <0 :
            cm =_search_air_along_axis_with_lateral (int (e ),iix ,iiy ,iiz ,ax ,-1 )

        if cp >=0 :
            solid_to_air_plus [e ]=cp
        if cm >=0 :
            solid_to_air_minus [e ]=cm
        if cp <0 or cm <0 :
            n_side_miss +=1
            if len (bad_rows )<24 :
                bad_rows .append ((int (e ),int (cp ),int (cm ),int (ax )))
        elif cp ==cm :
            n_side_same +=1
            if len (bad_rows )<24 :
                bad_rows .append ((int (e ),int (cp ),int (cm ),int (ax )))

    n_linked =int (np .sum (solid_to_air >=0 ))
    n_thin =int (np .sum (thin_mask ))
    n_surf =int (np .sum (surf_mask ))
    _log (
        f"  Solid↔air mapped solid elements: {n_linked}/{solid_positions .shape [0 ]} "
        f"(thin={n_thin}, surface-adjacent={n_surf})"
    )
    _log (
        f"  [map] local-adjacent mapping: hit={n_local_hit}, miss={n_local_miss} "
        f"(fallback={n_local_miss}), global_fill={n_global_fill}, unmapped={n_unmapped}"
    )
    _log (
        f"  [map] membrane/sensor bilateral rows={int (ms_rows .size )}, "
        f"missing_side_pairs={int (n_side_miss )}, same_side_cell={int (n_side_same )}"
    )
    if ms_rows .size >0 :
        cp_idx =solid_to_air_plus [ms_rows ]
        cm_idx =solid_to_air_minus [ms_rows ]
        cp_good =cp_idx [cp_idx >=0 ].astype (np .int64 ,copy =False )
        cm_good =cm_idx [cm_idx >=0 ].astype (np .int64 ,copy =False )
        if cp_good .size >0 :
            cp_bc =np .bincount (cp_good ,minlength =n_air )
            cp_max =int (np .max (cp_bc ))
            cp_top =np .argsort (cp_bc )[::-1 ][:6 ]
            cp_txt =", ".join (f"{int (c )}:{int (cp_bc [c ])}" for c in cp_top if int (cp_bc [c ])>0 )
            _log (f"  [map][ms] plus multiplicity max={cp_max}, top={cp_txt}")
        if cm_good .size >0 :
            cm_bc =np .bincount (cm_good ,minlength =n_air )
            cm_max =int (np .max (cm_bc ))
            cm_top =np .argsort (cm_bc )[::-1 ][:6 ]
            cm_txt =", ".join (f"{int (c )}:{int (cm_bc [c ])}" for c in cm_top if int (cm_bc [c ])>0 )
            _log (f"  [map][ms] minus multiplicity max={cm_max}, top={cm_txt}")
    if n_side_miss >0 or n_side_same >0 :
        sample =", ".join (f"(e={e }, cp={cp }, cm={cm }, ax={ax })" for (e ,cp ,cm ,ax )in bad_rows [:12 ])
        raise RuntimeError (
            "Topology error: membrane/sensor FE must map to two distinct air cells (one per side). "
            f"invalid_pairs={int (n_side_miss +n_side_same )}/{int (ms_rows .size )}; sample={sample }"
        )
    _log (
        f"  [map] thin ratio min/p50/p95/max = "
        f"{float (np .min (thin_ratio )):.3f}/{float (np .median (thin_ratio )):.3f}/"
        f"{float (np .percentile (thin_ratio ,95.0 )):.3f}/{float (np .max (thin_ratio )):.3f}"
    )
    if n_linked >0 :
        center_idx =solid_to_air [solid_to_air >=0 ].astype (np .int64 ,copy =False )
        center_bc =np .bincount (center_idx ,minlength =n_air )
        center_max =int (np .max (center_bc ))
        center_mean =float (np .mean (center_bc [center_bc >0 ])) if np .any (center_bc >0 )else 0.0
        top_c =np .argsort (center_bc )[::-1 ][:8 ]
        top_txt =", ".join (f"{int (c )}:{int (center_bc [c ])}" for c in top_c if int (center_bc [c ])>0 )
        _log (f"  [map] center multiplicity: max={center_max}, mean_nonzero={center_mean:.2f}, top={top_txt}")
        linked_rows =np .flatnonzero (solid_to_air >=0 )
        d =np .linalg .norm (solid_positions [linked_rows ]-air_pos [solid_to_air [linked_rows ]],axis =1 )
        _log (
            f"  [map] solid->air center distance m: min/p50/p95/max = "
            f"{float (np .min (d )):.4e}/{float (np .median (d )):.4e}/"
            f"{float (np .percentile (d ,95.0 )):.4e}/{float (np .max (d )):.4e}"
        )

    # Predict CSR row width from mapping payload to catch topology pathologies early.
    if solid_material_index is not None and np .asarray (solid_material_index ).size ==n_sol :
        mats =np .asarray (solid_material_index ,dtype =np .int32 ).ravel ()
        csr_row =np .zeros (n_air ,dtype =np .int32 )
        thin_mat =(mats ==int (MAT_MEMBRANE ))|(mats ==int (MAT_SENSOR ))
        for e in range (n_sol ):
            if thin_mat [e ]:
                cp =int (solid_to_air_plus [e ]); cm =int (solid_to_air_minus [e ])
                if cp >=0 : csr_row [cp ]+=1
                if cm >=0 : csr_row [cm ]+=1
            else :
                ce =int (solid_to_air [e ])
                if ce >=0 : csr_row [ce ]+=1
        csr_max =int (np .max (csr_row )) if csr_row .size >0 else 0
        top_r =np .argsort (csr_row )[::-1 ][:8 ]
        top_r_txt =", ".join (f"{int (c )}:{int (csr_row [c ])}" for c in top_r if int (csr_row [c ])>0 )
        _log (f"  [csr-precheck] predicted max row nnz={csr_max}; top rows={top_r_txt}")
        if csr_max >1024 :
            _log ("  [warn] High predicted CSR row nnz (>1024). Possible solid->air mapping collapse.")

    return {
    "air_element_position_xyz":air_pos ,
    "air_element_size_xyz":air_size ,
    "air_neighbors":air_neighbors ,
    # NOTE: historically boolean "absorb mask". Now AIR_BC_* enum per missing-neighbor face.
    "air_neighbor_absorb_u8":air_neighbor_absorb ,
    "air_material_index":np .full (n_air ,air_material_index ,dtype =np .uint8 ),
    "air_boundary_mask_elements":air_boundary ,
    "solid_to_air_index":solid_to_air ,
    "solid_to_air_index_plus":solid_to_air_plus ,
    "solid_to_air_index_minus":solid_to_air_minus ,
    "air_grid_shape":np .array ([nx ,ny ,nz ],dtype =np .int32 ),
    }


def _build_transform_matrix (
translation :list [float ],
rotation_deg :list [float ],
scale :list [float ],
)->np .ndarray :
    'Constructs a 4x4 affine transformation matrix from translation, euler deg, scale.'
    tr =(list (translation )+[0.0 ,0.0 ,0.0 ])[:3 ]
    rot =(list (rotation_deg )+[0.0 ,0.0 ,0.0 ])[:3 ]
    scl =(list (scale )+[1.0 ,1.0 ,1.0 ])[:3 ]

    rx ,ry ,rz =math .radians (rot [0 ]),math .radians (rot [1 ]),math .radians (rot [2 ])
    cx ,sx =math .cos (rx ),math .sin (rx )
    cy ,sy =math .cos (ry ),math .sin (ry )
    cz ,sz =math .cos (rz ),math .sin (rz )

    R =np .array ([
    [cz *cy ,cz *sy *sx -sz *cx ,cz *sy *cx +sz *sx ],
    [sz *cy ,sz *sy *sx +cz *cx ,sz *sy *cx -cz *sx ],
    [-sy ,cy *sx ,cy *cx ],
    ],dtype =np .float64 )
    S =np .diag ([scl [0 ],scl [1 ],scl [2 ]])
    M =np .eye (4 )
    M [:3 ,:3 ]=R @S 
    M [:3 ,3 ]=tr 
    return M 


def _apply_transform (points :np .ndarray ,M :np .ndarray )->np .ndarray :
    'Applies a 4x4 matrix to points [n, 3]. Returns [n, 3].'
    n =points .shape [0 ]
    ones =np .ones ((n ,1 ),dtype =np .float64 )
    pts =np .hstack ([points ,ones ])
    return (M @pts .T ).T [:,:3 ]


def _build_inverse_transform_matrix (
translation :list [float ],
rotation_deg :list [float ],
scale :list [float ],
)->np .ndarray :
    'Constructs a 4x4 inverse transformation matrix: global -> local (primitive center in origin).'
    tr =(list (translation )+[0.0 ,0.0 ,0.0 ])[:3 ]
    rot =(list (rotation_deg )+[0.0 ,0.0 ,0.0 ])[:3 ]
    scl =(list (scale )+[1.0 ,1.0 ,1.0 ])[:3 ]
    rx ,ry ,rz =math .radians (rot [0 ]),math .radians (rot [1 ]),math .radians (rot [2 ])
    cx ,sx =math .cos (rx ),math .sin (rx )
    cy ,sy =math .cos (ry ),math .sin (ry )
    cz ,sz =math .cos (rz ),math .sin (rz )
    R =np .array ([
    [cz *cy ,cz *sy *sx -sz *cx ,cz *sy *cx +sz *sx ],
    [sz *cy ,sz *sy *sx +cz *cx ,sz *sy *cx -cz *sx ],
    [-sy ,cy *sx ,cy *cx ],
    ],dtype =np .float64 )
    S_inv =np .diag ([1.0 /scl [0 ],1.0 /scl [1 ],1.0 /scl [2 ]])
    M_inv =np .eye (4 )
    M_inv [:3 ,:3 ]=S_inv @R .T 
    M_inv [:3 ,3 ]=-M_inv [:3 ,:3 ]@np .array (tr ,dtype =np .float64 )
    return M_inv 


def _point_inside_bc_primitive (
p_local :np .ndarray ,
bc_type :str ,
params :dict [str ,float ],
)->bool :
    'Check: a point in the local CS of the primitive (center in origin) inside the figure.\n    Primitives: sphere, box, cylinder, tube.'
    if bc_type =="sphere":
        r =params .get ("radius",1.0 )
        return float (np .dot (p_local ,p_local ))<=r *r +1e-12 
    if bc_type =="box":
        hx =params .get ("box_x",1.0 )*0.5 
        hy =params .get ("box_y",1.0 )*0.5 
        hz =params .get ("box_z",1.0 )*0.5 
        return (
        abs (p_local [0 ])<=hx +1e-12 
        and abs (p_local [1 ])<=hy +1e-12 
        and abs (p_local [2 ])<=hz +1e-12 
        )
    if bc_type =="cylinder":
        r =params .get ("cylinder_radius",1.0 )
        h =params .get ("cylinder_height",1.0 )
        r2 =p_local [0 ]*p_local [0 ]+p_local [1 ]*p_local [1 ]
        return r2 <=r *r +1e-12 and abs (p_local [2 ])<=h *0.5 +1e-12 
    if bc_type =="tube":
        r_in =params .get ("tube_radius_inner",1.0 )
        r_out =params .get ("tube_radius_outer",2.0 )
        length =params .get ("tube_length",10.0 )
        r2 =p_local [0 ]*p_local [0 ]+p_local [1 ]*p_local [1 ]
        return (
        r_in *r_in -1e-12 <=r2 <=r_out *r_out +1e-12 
        and abs (p_local [2 ])<=length *0.5 +1e-12 
        )
    return False 


def _polydata_to_vertices_faces (poly ,transform :MeshTransform )->tuple [np .ndarray ,np .ndarray ]|None :
    'Converts PyVista PolyData to (vertices, faces) with transform applied.'
    if poly is None or not hasattr (poly ,"points"):
        return None 
    pts =np .asarray (poly .points ,dtype =np .float64 )
    M =_build_transform_matrix (
    list (transform .translation ),
    list (transform .rotation_euler_deg ),
    list (transform .scale ),
    )
    pts =_apply_transform (pts ,M )
    cells =None 
    if hasattr (poly ,"faces"):
        cells =poly .faces 
    elif hasattr (poly ,"cells"):
        cells =poly .cells 
    if cells is None or (hasattr (cells ,"size")and cells .size ==0 ):
        return None 
    offset =0 
    faces_list =[]
    while offset <cells .shape [0 ]:
        nv =int (cells [offset ])
        offset +=1 
        if offset +nv >cells .shape [0 ]:
            break 
        idx =cells [offset :offset +nv ]
        offset +=nv 
        if nv ==3 :
            faces_list .append (idx )
        elif nv ==4 :
            faces_list .append ([idx [0 ],idx [1 ],idx [2 ]])
            faces_list .append ([idx [0 ],idx [2 ],idx [3 ]])
    if not faces_list :
        return None 
    faces =np .array (faces_list ,dtype =np .int64 )
    return pts ,faces 


def _build_polydata_cells (faces :np .ndarray )->np .ndarray :
    'Constructs an array of cells for PyVista PolyData from triangular faces [n, 3].'
    n_tri =faces .shape [0 ]
    cells =np .hstack ([np .full ((n_tri ,1 ),3 ,dtype =np .int64 ),faces ]).ravel ()
    return cells 


    # --- Planar topology (membrane, sensor) ---


def _compute_face_normals (vertices :np .ndarray ,faces :np .ndarray )->np .ndarray :
    'Face normals [n_faces, 3], not normalized.'
    v0 =vertices [faces [:,0 ]]
    v1 =vertices [faces [:,1 ]]
    v2 =vertices [faces [:,2 ]]
    n =np .cross (v1 -v0 ,v2 -v0 )
    return n 


def _analyse_planar_mesh (
vertices :np .ndarray ,
faces :np .ndarray ,
unit_scale :float =1.0 ,
)->tuple [dict [str ,Any ]|None ,str |None ]:
    'Analyzes the mesh as one FE layer in the plane.\n    Thickness = the smallest span of the vertices along the axes (min extent).\n    Plane = two axes with a large span (XY, XZ or YZ).'
    if vertices .shape [0 ]<3 or faces .shape [0 ]<1 :
        return None ,'Not enough vertices or faces'

        # Span of vertices along each axis (difference between extreme points)
    vmin =np .min (vertices ,axis =0 )
    vmax =np .max (vertices ,axis =0 )
    extent =np .array ([vmax [i ]-vmin [i ]for i in range (3 )],dtype =np .float64 )

    # Thickness = smallest span (normal axis to plane)
    normal_axis =int (np .argmin (extent ))
    thickness =float (extent [normal_axis ])
    if thickness <1e-12 :
        extent_max =float (np .max (extent ))
        thickness =max (extent_max *1e-6 ,1e-12 )

        # Plane = two axes with large span (u_axis, v_axis)
    if normal_axis ==0 :
        u_axis ,v_axis =1 ,2 # YZ plane
    elif normal_axis ==1 :
        u_axis ,v_axis =0 ,2 # XZ plane
    else :
        u_axis ,v_axis =0 ,1 # XY plane

    centroid =np .mean (vertices ,axis =0 )

    # 2D projection of vertices in the plane
    verts_2d =vertices [:,[u_axis ,v_axis ]].astype (np .float64 )
    u_min ,u_max =float (np .min (verts_2d [:,0 ])),float (np .max (verts_2d [:,0 ]))
    v_min ,v_max =float (np .min (verts_2d [:,1 ])),float (np .max (verts_2d [:,1 ]))

    return {
    "normal_axis":normal_axis ,
    "thickness":thickness ,
    "u_axis":u_axis ,
    "v_axis":v_axis ,
    "bbox_u":(u_min ,u_max ),
    "bbox_v":(v_min ,v_max ),
    "verts_2d":verts_2d ,
    "faces":faces ,
    "vertices":vertices ,
    "centroid":centroid ,
    },None 


def _point_in_triangle_2d (p :np .ndarray ,a :np .ndarray ,b :np .ndarray ,c :np .ndarray )->bool :
    'Check: point p inside triangle abc in 2D (barycentric coordinates).'
    v0 =c -a 
    v1 =b -a 
    v2 =p -a 
    d00 =np .dot (v0 ,v0 )
    d01 =np .dot (v0 ,v1 )
    d11 =np .dot (v1 ,v1 )
    d20 =np .dot (v2 ,v0 )
    d21 =np .dot (v2 ,v1 )
    denom =d00 *d11 -d01 *d01 
    if abs (denom )<1e-20 :
        return False 
    s =(d11 *d20 -d01 *d21 )/denom 
    t =(d00 *d21 -d01 *d20 )/denom 
    return s >=-1e-12 and t >=-1e-12 and (s +t )<=1.0 +1e-12 


def _point_inside_mesh_2d (
p :np .ndarray ,
verts_2d :np .ndarray ,
faces :np .ndarray ,
)->bool :
    'Check: point p (2D) inside the mesh (projection of triangles).'
    for i in range (faces .shape [0 ]):
        a =verts_2d [faces [i ,0 ]]
        b =verts_2d [faces [i ,1 ]]
        c =verts_2d [faces [i ,2 ]]
        if _point_in_triangle_2d (p ,a ,b ,c ):
            return True 
    return False 


def _generate_planar_topology (
vertices :np .ndarray ,
faces :np .ndarray ,
material_index :int ,
element_size_mm :float ,
padding_mm :float ,
unit_scale :float ,
log_fn =None ,
)->tuple [np .ndarray ,np .ndarray ,np .ndarray ,np .ndarray ,np .ndarray ]:
    'Generates one FE layer for a flat mesh (membrane/sensor).\n    - Thickness = the smallest span of the vertices along the axes.\n    - Plane = two axes with a large span (XY, XZ or YZ).\n    - FE size: element_size × element_size × thickness. All FEs are in the same plane.'
    def _log (msg :str )->None :
        if log_fn :
            log_fn (msg )

    empty =(
    np .zeros ((0 ,3 ),dtype =np .float64 ),
    np .zeros ((0 ,3 ),dtype =np .float64 ),
    np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
    np .zeros (0 ,dtype =np .uint8 ),
    np .zeros (0 ,dtype =np .int32 ),
    )

    info ,err =_analyse_planar_mesh (vertices ,faces ,unit_scale )
    if err is not None :
        _log (f"  Error анализа плоскости: {err }")
        return empty 

    _log (f"  Plane: normal axis={info ['normal_axis']}, thickness={info ['thickness']:.4e}, "
    f"bbox_u=[{info ['bbox_u'][0 ]:.3f},{info ['bbox_u'][1 ]:.3f}], "
    f"bbox_v=[{info ['bbox_v'][0 ]:.3f},{info ['bbox_v'][1 ]:.3f}]")

    # FE size in plane = element_size_mm; in thickness direction = thickness
    step =float (element_size_mm )*unit_scale 
    if step <=0 :
        step =1e-3 
    pad =float (padding_mm )*unit_scale 

    u_min ,u_max =info ["bbox_u"]
    v_min ,v_max =info ["bbox_v"]
    u_min -=pad 
    u_max +=pad 
    v_min -=pad 
    v_max +=pad 

    Lu =u_max -u_min 
    Lv =v_max -v_min 
    if Lu <=0 or Lv <=0 :
        _log ('Zero plane size after padding')
        return empty 

        # Mesh: step = element_size (same FE size in plane)
    _MAX_GRID_DIM =10_000 
    try :
        n_u =max (1 ,min (int (np .ceil (Lu /(step +1e-30 ))),_MAX_GRID_DIM ))
        n_v =max (1 ,min (int (np .ceil (Lv /(step +1e-30 ))),_MAX_GRID_DIM ))
    except (OverflowError ,ValueError ):
        n_u ,n_v =_MAX_GRID_DIM ,_MAX_GRID_DIM 
    u_edges =np .linspace (u_min ,u_max ,n_u +1 ,dtype =np .float64 )
    v_edges =np .linspace (v_min ,v_max ,n_v +1 ,dtype =np .float64 )
    nu =len (u_edges )-1 
    nv =len (v_edges )-1 
    _log (f"  Grid: {nu }×{nv } cells, step={step :.4e}, thickness={info ['thickness']:.4e}")

    verts_2d =info ["verts_2d"]
    faces_arr =info ["faces"]
    normal_axis =info ["normal_axis"]
    thickness =info ["thickness"]
    u_axis =info ["u_axis"]
    v_axis =info ["v_axis"]
    centroid =info ["centroid"]

    # We collect cells whose center is inside the mesh. CE size: step×step×thickness
    cells_data =[]
    for i in range (nu ):
        for j in range (nv ):
            cu =(u_edges [i ]+u_edges [i +1 ])*0.5 
            cv =(v_edges [j ]+v_edges [j +1 ])*0.5 
            if _point_inside_mesh_2d (np .array ([cu ,cv ]),verts_2d ,faces_arr ):
                du =step # uniform size in plane
                dv =step 
                cells_data .append ((i ,j ,du ,dv ,cu ,cv ))

    if not cells_data :
        _log ('No cells inside mesh outline')
        return empty 

    n =len (cells_data )
    _log (f"  Cells inside contour: {n } (из {nu *nv } candidates)")
    positions =np .zeros ((n ,3 ),dtype =np .float64 )
    sizes =np .zeros ((n ,3 ),dtype =np .float64 )
    neighbors =np .full ((n ,FACE_DIRS ),-1 ,dtype =np .int32 )
    mat_arr =np .full (n ,material_index ,dtype =np .uint8 )
    boundary =np .zeros (n ,dtype =np .int32 )

    cell_map ={}
    for idx ,(ii ,jj ,du ,dv ,cu ,cv )in enumerate (cells_data ):
        cell_map [(ii ,jj )]=idx 

        # FACE_DIRS: +X=0, -X=1, +Y=2, -Y=3, +Z=4, -Z=5
        # Connections only in the plane of the membrane. The plane axes (u_axis, v_axis) specify the directions.
        # For axis a: + = 2*a, - = 2*a+1. There are no neighbors along the normal_axis.
    dir_plus_u =2 *u_axis 
    dir_minus_u =2 *u_axis +1 
    dir_plus_v =2 *v_axis 
    dir_minus_v =2 *v_axis +1 
    dir_plus_n =2 *normal_axis 
    dir_minus_n =2 *normal_axis +1 

    z_coord =float (centroid [normal_axis ])

    # OpenCL kernel uses element_size_xyz as global (sx,sy,sz): face areas and rest_len per ±X/±Y/±Z neighbor.
    for idx ,(ii ,jj ,du ,dv ,cu ,cv )in enumerate (cells_data ):
        pos_3d =np .zeros (3 )
        pos_3d [u_axis ]=cu 
        pos_3d [v_axis ]=cv 
        pos_3d [normal_axis ]=z_coord 
        positions [idx ]=pos_3d 

        size_3d =np .zeros (3 ,dtype =np .float64 )
        size_3d [u_axis ]=du 
        size_3d [v_axis ]=dv 
        size_3d [normal_axis ]=thickness 
        sizes [idx ]=size_3d 

        # Neighbours
        n_plus_u =cell_map .get ((ii +1 ,jj ))
        n_minus_u =cell_map .get ((ii -1 ,jj ))
        n_plus_v =cell_map .get ((ii ,jj +1 ))
        n_minus_v =cell_map .get ((ii ,jj -1 ))

        neighbors [idx ,dir_plus_u ]=n_plus_u if n_plus_u is not None else -1 
        neighbors [idx ,dir_minus_u ]=n_minus_u if n_minus_u is not None else -1 
        neighbors [idx ,dir_plus_v ]=n_plus_v if n_plus_v is not None else -1 
        neighbors [idx ,dir_minus_v ]=n_minus_v if n_minus_v is not None else -1 
        neighbors [idx ,dir_plus_n ]=-1 
        neighbors [idx ,dir_minus_n ]=-1 

        # Automatic boundary conditions: perimeter = FE without neighbors along at least one face in the plane
        if n_plus_u is None or n_minus_u is None or n_plus_v is None or n_minus_v is None :
            boundary [idx ]=1 

    n_boundary =int (np .sum (boundary ))
    _log (f"  Perimeter: {n_boundary } FE elements (automatic boundary conditions)")

    return positions ,sizes ,neighbors ,mat_arr ,boundary 


def _voxelize_single_mesh (
vertices :np .ndarray ,
faces :np .ndarray ,
element_size_mm :float ,
padding_mm :float ,
material_index :int ,
unit_scale_override :float |None =None ,
)->tuple [np .ndarray ,np .ndarray ,np .ndarray ,np .ndarray ,np .ndarray ]:
    'Voxelization of a single mesh using PyVista voxelize_rectilinear.\n    Returns (positions, sizes, neighbors, material_index, boundary).'
    empty_result =(
    np .zeros ((0 ,3 ),dtype =np .float64 ),
    np .zeros ((0 ,3 ),dtype =np .float64 ),
    np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
    np .zeros (0 ,dtype =np .uint8 ),
    np .zeros (0 ,dtype =np .int32 ),
    )
    if pv is None :
        return empty_result 

    try :
        cells =_build_polydata_cells (faces )
        poly =pv .PolyData (vertices ,cells )
    except Exception :
        return empty_result 

    bmin =np .array (vertices .min (axis =0 ),dtype =np .float64 )
    bmax =np .array (vertices .max (axis =0 ),dtype =np .float64 )
    extent =bmax -bmin 

    # Use one unit scale for all solids when provided.
    if unit_scale_override is not None and unit_scale_override >0.0 :
        unit_scale =float (unit_scale_override )
    else :
        # Units: CAD/STL usually in mm. Small extents are treated as meters.
        extent_max =float (np .max (extent ))
        unit_scale =_estimate_model_unit_scale (extent_max )
    pad =float (padding_mm )*unit_scale 
    bmin =bmin -pad 
    bmax =bmax +pad 
    extent =bmax -bmin 

    dx =float (element_size_mm )*unit_scale 
    if dx <=0 :
        dx =1e-3 

    nx =max (1 ,int (np .ceil (extent [0 ]/dx )))
    ny =max (1 ,int (np .ceil (extent [1 ]/dx )))
    nz =max (1 ,int (np .ceil (extent [2 ]/dx )))

    dx_act =extent [0 ]/nx 
    dy_act =extent [1 ]/ny 
    dz_act =extent [2 ]/nz 
    elem_size =np .array ([dx_act ,dy_act ,dz_act ],dtype =np .float64 )

    try :
        poly =poly .clean ()
        # Do not compute_normals before voxelize: PyVista can add Normals array with
        # wrong length for cell data, causing InvalidMeshWarning in voxelize_rectilinear.
    except Exception :
        pass 

        # Remove Normals if present (from prior ops) to avoid InvalidMeshWarning in voxelize
    try :
        for key in ("Normals","normals"):
            if key in poly .point_data :
                del poly .point_data [key ]
            if key in poly .cell_data :
                del poly .cell_data [key ]
    except Exception :
        pass 

    try :
        vox =poly .voxelize_rectilinear (spacing =(dx_act ,dy_act ,dz_act ))
    except Exception :
        return empty_result 

        # The mask can be in point_data (old versions) or cell_data (PyVista 0.47+)
    if "mask"in vox .cell_data :
        mask_arr =vox .cell_data ["mask"]
    elif "mask"in vox .point_data :
        mask_arr =vox .point_data ["mask"]
    else :
        mask_arr =None 
    if mask_arr is None :
        return empty_result 

    mask =np .asarray (mask_arr ).ravel ()
    dims =np .array (vox .dimensions ,dtype =np .int32 )
    dimx ,dimy ,dimz =dims [0 ],dims [1 ],dims [2 ]

    # dimensions = number of points; cells (nx, ny, nz) = (dimx-1, dimy-1, dimz-1)
    ncx =int (dimx -1 )
    ncy =int (dimy -1 )
    ncz =int (dimz -1 )
    if ncx <=0 or ncy <=0 or ncz <=0 :
        return empty_result 

        # Use Python ints for index arithmetic to avoid overflow (ncx*ncy*ncz can exceed int32)
    dimx_py =int (dimx )
    dimy_py =int (dimy )

    def _collect_voxels (foreground_val :int )->list :
        out =[]
        if mask .size ==vox .n_cells :
            for i in range (ncx ):
                for j in range (ncy ):
                    for k in range (ncz ):
                        cell_idx =i +j *ncx +k *ncx *ncy 
                        if cell_idx <mask .size and mask [cell_idx ]==foreground_val :
                            out .append ((i ,j ,k ))
        else :
            for i in range (ncx ):
                for j in range (ncy ):
                    for k in range (ncz ):
                        pt_idx =(i +1 )+(j +1 )*dimx_py +(k +1 )*dimx_py *dimy_py 
                        if pt_idx <mask .size and mask [pt_idx ]==foreground_val :
                            out .append ((i ,j ,k ))
        return out 

    voxel_list =_collect_voxels (1 )
    if not voxel_list :
        voxel_list =_collect_voxels (0 )
    if not voxel_list :
        return empty_result 

    voxel_to_idx ={v :idx for idx ,v in enumerate (voxel_list )}
    n =len (voxel_list )

    x_coords =np .asarray (vox .x )
    y_coords =np .asarray (vox .y )
    z_coords =np .asarray (vox .z )

    positions =np .zeros ((n ,3 ),dtype =np .float64 )
    sizes =np .tile (elem_size ,(n ,1 ))
    neighbors =np .full ((n ,FACE_DIRS ),-1 ,dtype =np .int32 )
    mat_arr =np .full (n ,material_index ,dtype =np .uint8 )
    boundary =np .zeros (n ,dtype =np .int32 )

    deltas =[(1 ,0 ,0 ),(-1 ,0 ,0 ),(0 ,1 ,0 ),(0 ,-1 ,0 ),(0 ,0 ,1 ),(0 ,0 ,-1 )]

    for idx ,(ix ,iy ,iz )in enumerate (voxel_list ):
        cx =(x_coords [ix ]+x_coords [ix +1 ])*0.5 
        cy =(y_coords [iy ]+y_coords [iy +1 ])*0.5 
        cz =(z_coords [iz ]+z_coords [iz +1 ])*0.5 
        positions [idx ]=[cx ,cy ,cz ]

        for d ,(di ,dj ,dk )in enumerate (deltas ):
            ni ,nj ,nk =ix +di ,iy +dj ,iz +dk 
            nkey =(ni ,nj ,nk )
            if nkey in voxel_to_idx :
                neighbors [idx ,d ]=voxel_to_idx [nkey ]
                # Edge boundaries only for membrane/sensor (planar generation).
                # Solid: boundary only from 3D BC primitives (sphere, box, cylinder, tube).
                # else: boundary[idx] = 1 # disabled for solid

    return positions ,sizes ,neighbors ,mat_arr ,boundary 


def _get_mesh_vertices_faces_list (
meshes :list [MeshEntity ],
polydata_by_id :dict [str ,Any ],
load_mesh_fn ,
material_key_to_index :dict [str ,int ],
log_fn =None ,
)->tuple [list [tuple [np .ndarray ,np .ndarray ,int ,str ,str ,str ]],list [tuple [np .ndarray ,np .ndarray ,int ,str ,str ,str ]]]:
    'Returns (solid_list, planar_list).\n    solid_list: (vertices, faces, material_index, name, mesh_id, role) for voxelization.\n    planar_list: (vertices, faces, material_index, name, mesh_id, role) for planar generation (membrane, sensor).'
    def _log (msg :str )->None :
        if log_fn :
            log_fn (msg )

    def _get_verts_faces (mesh :MeshEntity ):
        verts ,faces =None ,None 
        poly =polydata_by_id .get (mesh .mesh_id )
        if poly is not None :
            vf =_polydata_to_vertices_faces (poly ,mesh .transform )
            if vf is not None :
                verts ,faces =vf 
        if verts is None and load_mesh_fn :
            raw =load_mesh_fn (mesh )
            if raw is not None and hasattr (raw ,"vertices")and hasattr (raw ,"faces"):
                pts =np .asarray (raw .vertices ,dtype =np .float64 )
                M =_build_transform_matrix (
                list (mesh .transform .translation ),
                list (mesh .transform .rotation_euler_deg ),
                list (mesh .transform .scale ),
                )
                pts =_apply_transform (pts ,M )
                f =np .asarray (raw .faces ,dtype =np .int64 )
                if f .shape [1 ]==4 :
                    f3 =np .hstack ([f [:,:3 ],f [:,[0 ,2 ,3 ]]]).reshape (-1 ,3 )
                else :
                    f3 =f 
                verts ,faces =pts ,f3 
        return verts ,faces 

    solid_list =[]
    planar_list =[]

    for mesh in meshes :
        role =(mesh .role or "solid").lower ()
        verts ,faces =_get_verts_faces (mesh )

        if verts is None :
            _log (f"  Mesh '{mesh .name }' ({mesh .mesh_id }): failed to get geometry — skipped")
            continue 

        name =mesh .name or mesh .mesh_id 
        mat_key =(mesh .material_key or "").lower ()
        if role =="membrane":
            mat_idx =material_key_to_index .get (mat_key or "membrane",int (MAT_MEMBRANE ))
        elif role =="sensor":
            mat_idx =material_key_to_index .get (mat_key or "sensor",int (MAT_SENSOR ))
        else :
            mat_idx =material_key_to_index .get (mat_key or "foam_ve3015",int (MAT_FOAM_VE3015 ))

        nv ,nf =verts .shape [0 ],faces .shape [0 ]
        if role in ("membrane","sensor"):
            _log (f"  Mesh '{name }': {nv } verts., {nf } faces. → planar generation (role {role })")
            planar_list .append ((verts ,faces ,mat_idx ,name ,mesh .mesh_id ,role ))
        else :
            _log (f"  Mesh '{name }': {nv } verts., {nf } faces. → voxelization (role {role })")
            solid_list .append ((verts ,faces ,mat_idx ,name ,mesh .mesh_id ,role ))

    return solid_list ,planar_list 


    # BC check is always in the main process. ProcessPoolExecutor with millions of elements
    # copies data to each worker → 50+ GB leak. Chunk processing reduces peak memory.
_BC_CHUNK_SIZE =50_000 


def _apply_boundary_conditions_inprocess (
positions :np .ndarray ,
boundary :np .ndarray ,
mesh_ids :list [str ],
boundary_conditions :list [BoundaryCondition ],
log_fn =None ,
)->None :
    'Marks elements as boundary elements in the main process. Processing in chunks to save memory.'
    def _log (msg :str )->None :
        if log_fn :
            log_fn (msg )

    n =positions .shape [0 ]
    mesh_id_list =list (dict .fromkeys (mesh_ids ))
    mesh_id_to_idx ={mid :i for i ,mid in enumerate (mesh_id_list )}
    mesh_indices =np .array ([mesh_id_to_idx .get (mid ,-1 )for mid in mesh_ids ],dtype =np .int32 )

    n_bcs =len (boundary_conditions )
    n_chunks =(n +_BC_CHUNK_SIZE -1 )//_BC_CHUNK_SIZE 

    for bc_idx ,bc in enumerate (boundary_conditions ):
        bc_type =bc .bc_type 
        scope ='all meshes'if not (bc .mesh_ids or [])else f"meshes {bc .mesh_ids }"
        _log (f"  BC {bc_idx +1 }/{n_bcs } ({bc_type }, {scope })")
        bc_mesh_ids =set (bc .mesh_ids or [])
        apply_to_all =len (bc_mesh_ids )==0 
        params =dict (bc .parameters or {})
        M_inv =_build_inverse_transform_matrix (
        list (bc .transform .translation ),
        list (bc .transform .rotation_euler_deg ),
        list (bc .transform .scale ),
        )
        # Processing in chunks: we do not create arrays of size n, only chunk_size
        last_pct_logged =-1 
        for chunk_idx ,start in enumerate (range (0 ,n ,_BC_CHUNK_SIZE )):
            end =min (start +_BC_CHUNK_SIZE ,n )
            pct =int (100.0 *end /n )if n >0 else 0 
            # Log every 25% or on the last chunk
            if pct >=last_pct_logged +25 or chunk_idx ==n_chunks -1 :
                _log (f"    progress: {pct }% ({end }/{n } elements)")
                last_pct_logged =pct 
            chunk =positions [start :end ]
            n_chunk =chunk .shape [0 ]
            ones =np .ones ((n_chunk ,1 ),dtype =np .float64 )
            pts_h =np .hstack ([chunk ,ones ])
            local =(M_inv @pts_h .T ).T [:,:3 ]
            for i in range (n_chunk ):
                idx =start +i 
                if not apply_to_all :
                    midx =mesh_indices [idx ]
                    if midx <0 or midx >=len (mesh_id_list ):
                        continue 
                    if mesh_id_list [midx ]not in bc_mesh_ids :
                        continue 
                if _point_inside_bc_primitive (local [i ],bc .bc_type ,params ):
                    boundary [idx ]=1 


def _apply_boundary_conditions (
positions :np .ndarray ,
boundary :np .ndarray ,
mesh_ids :list [str ],
boundary_conditions :list [BoundaryCondition ],
log_fn =None ,
)->None :
    'Marks elements as boundary (boundary[elem]=1) if the center of the FE is inside any BC.'
    def _log (msg :str )->None :
        if log_fn :
            log_fn (msg )

    if not boundary_conditions :
        return 

    n =positions .shape [0 ]
    if n ==0 :
        return 

    _log (f"  Mode: main process, chunks of {_BC_CHUNK_SIZE } elements")
    _apply_boundary_conditions_inprocess (
    positions ,boundary ,mesh_ids ,boundary_conditions ,
    log_fn =log_fn ,
    )
    n_bc =int (np .sum (boundary ))
    _log (f"Boundary conditions: {n_bc } FE elements marked as boundary (including perimeter)")


def generate_topology_from_meshes (
meshes :list [MeshEntity ],
polydata_by_id :dict [str ,Any ],
load_mesh_fn ,
*,
element_size_mm :float =0.5 ,
padding_mm :float =0.0 ,
    air_gap_layers :int =2 ,
generate_air_grid :bool =True ,
max_air_cells :int =1_200_000 ,
material_key_to_index :dict [str ,int ]|None =None ,
boundary_conditions :list [BoundaryCondition ]|None =None ,
log_callback =None ,
)->dict [str ,np .ndarray ]:
    "Generates a regular 3D volumetric topology from project meshes.\n\n    Solids are harmonized to one common voxel grid and connected globally,\n    so touching solids from different source meshes are linked by neighbors.\n    Air grid uses the same step as solids.\n\n    element_size_mm: Common finite-element size for solids and air (mm).\n    padding_mm: Padding from topology bbox (mm).\n    boundary_conditions: list of BCs - FEs whose center is inside any figure are marked as boundary.\n\n    Returns a dictionary:\n    - element_position_xyz: [n, 3]\n    - element_size_xyz: [n, 3]\n    - neighbors: [n, FACE_DIRS]\n    - material_index: [n]\n    - boundary_mask_elements: [n]\n    log_callback: Called with (msg: str) to output to the log."
    def _log (msg :str )->None :
        if log_callback :
            log_callback (msg )

    if material_key_to_index is None :
        material_key_to_index ={
        "membrane":int (MAT_MEMBRANE ),
        "foam_ve3015":int (MAT_FOAM_VE3015 ),
        "sensor":int (MAT_SENSOR ),
        "air":int (MAT_AIR ),
        }

    _log ('=== Topology generation ===')
    _log (f"Parameters: element_size={element_size_mm } mm, padding={padding_mm } mm")
    if generate_air_grid :
        _log (f"Air grid: enabled, step={element_size_mm } mm (common), gap_layers={int (air_gap_layers )}, max_cells={int (max_air_cells )}")
    else :
        _log ('Air grid: disabled')
    _log (f"Meshей в проекте: {len (meshes )}")
    _log (f"PolyData in cache: {list (polydata_by_id .keys ())}")
    _log (f"Boundary conditions: {len (boundary_conditions or [])}")
    _log ("")
    _log ('---Step 1: Loading Mesh Geometry ---')

    solid_data ,planar_data =_get_mesh_vertices_faces_list (
    meshes ,polydata_by_id ,load_mesh_fn ,
    material_key_to_index =material_key_to_index ,
    log_fn =_log ,
    )

    _log (f"  Total: {len (solid_data )} мешей for voxelization, {len (planar_data )} for planar generation")
    _log ("")

    if not solid_data and not planar_data :
        _log ('There are no meshes to generate (all are missing or geometry is not available).')
        return {
        "element_position_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "element_size_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "neighbors":np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
        "material_index":np .zeros (0 ,dtype =np .uint8 ),
        "boundary_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        "air_element_position_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "air_element_size_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "air_neighbors":np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
        "air_neighbor_absorb_u8":np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 ),
        "air_material_index":np .zeros (0 ,dtype =np .uint8 ),
        "air_boundary_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        "solid_to_air_index":np .zeros (0 ,dtype =np .int32 ),
        "solid_to_air_index_plus":np .zeros (0 ,dtype =np .int32 ),
        "solid_to_air_index_minus":np .zeros (0 ,dtype =np .int32 ),
        "air_grid_shape":np .zeros (3 ,dtype =np .int32 ),
        "membrane_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        "sensor_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        }

    all_positions =[]
    all_sizes =[]
    all_neighbors =[]
    all_material =[]
    all_boundary =[]
    all_membrane_mask =[]
    all_sensor_mask =[]
    all_mesh_ids =[]
    offset =0 

    # 1. Planar generation (membrane, sensor)
    _log ('--- Stage 2: Planar generation (membrane, sensor) ---')
    for verts ,faces ,mat_idx ,name ,mesh_id ,role in planar_data :
        nv ,nf =verts .shape [0 ],faces .shape [0 ]
        extent_max =float (np .max (np .max (verts ,axis =0 )-np .min (verts ,axis =0 )))
        # CAD/STL is usually in mm. Eardrum ~9mm → extent_max=9. Threshold 10 erroneously gave unit_scale=1e-3
        # and step=0.5µm instead of 0.5mm → n_u≈18000, OOM. Threshold 1: extent>1 → mm (unit_scale=1).
        unit_scale =1.0 if extent_max >1.0 else 1e-3 
        _log (f"  Mesh '{name }': {nv } vertices, {nf } faces, unit_scale={unit_scale }")
        pos ,sizes ,nbh ,mat ,bnd =_generate_planar_topology (
        verts ,faces ,mat_idx ,
        element_size_mm =element_size_mm ,
        padding_mm =padding_mm ,
        unit_scale =unit_scale ,
        log_fn =_log ,
        )
        if len (pos )==0 :
            _log (f"  Mesh '{name }': 0 elements (planar generation) — skipped")
            continue 
        n_bnd =int (np .sum (bnd ))
        _log (f"  Mesh '{name }': {len (pos )} elements, {n_bnd } boundary (периметр)")
        all_positions .append (pos )
        all_sizes .append (sizes )
        nbh_adj =np .where (nbh >=0 ,nbh +offset ,-1 )
        all_neighbors .append (nbh_adj )
        all_material .append (mat )
        all_boundary .append (bnd )
        all_membrane_mask .append (np .full (len (pos ),1 if role =="membrane" else 0 ,dtype =np .int32 ))
        all_sensor_mask .append (np .full (len (pos ),1 if role =="sensor" else 0 ,dtype =np .int32 ))
        all_mesh_ids .extend ([mesh_id ]*len (pos ))
        offset +=len (pos )

        # 2. Voxelization (solid)
    if solid_data :
        _log ("")
        _log ('--- Stage 3: Voxelization (solid) ---')
        if pv is None :
            raise RuntimeError ('PyVista is required for voxelization. Install: pip install pyvista')
        solid_unit_scale =_estimate_global_unit_scale_for_solids (solid_data )
        _log (f"  Common unit_scale for solids: {solid_unit_scale }")
        _log (f"  Вокселизация {len (solid_data )} мешей")
        for task_idx ,(verts ,faces ,mat_idx ,name ,mesh_id ,role )in enumerate (solid_data ):
            pos ,sizes ,nbh ,mat ,bnd =_voxelize_single_mesh (
            verts ,faces ,element_size_mm ,padding_mm ,mat_idx ,
            unit_scale_override =solid_unit_scale ,
            )
            mesh_name =name if name else f"#{task_idx }"
            if len (pos )==0 :
                _log (f"  Mesh '{mesh_name }': 0 elements (voxelization) — skipped")
                continue 
            n_bnd =int (np .sum (bnd ))
            _log (f"  Mesh '{mesh_name }': {len (pos )} elements, {n_bnd } boundary (поверхность)")
            all_positions .append (pos )
            all_sizes .append (sizes )
            nbh_adj =np .where (nbh >=0 ,nbh +offset ,-1 )
            all_neighbors .append (nbh_adj )
            all_material .append (mat )
            all_boundary .append (bnd )
            all_membrane_mask .append (np .zeros (len (pos ),dtype =np .int32 ))
            all_sensor_mask .append (np .zeros (len (pos ),dtype =np .int32 ))
            all_mesh_ids .extend ([mesh_id ]*len (pos ))
            offset +=len (pos )

    if not all_positions :
        _log ("")
        _log ('Total: 0 elements (all meshes gave an empty result).')
        return {
        "element_position_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "element_size_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "neighbors":np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
        "material_index":np .zeros (0 ,dtype =np .uint8 ),
        "boundary_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        "air_element_position_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "air_element_size_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
        "air_neighbors":np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
        "air_neighbor_absorb_u8":np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 ),
        "air_material_index":np .zeros (0 ,dtype =np .uint8 ),
        "air_boundary_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        "solid_to_air_index":np .zeros (0 ,dtype =np .int32 ),
        "solid_to_air_index_plus":np .zeros (0 ,dtype =np .int32 ),
        "solid_to_air_index_minus":np .zeros (0 ,dtype =np .int32 ),
        "air_grid_shape":np .zeros (3 ,dtype =np .int32 ),
        "membrane_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        "sensor_mask_elements":np .zeros (0 ,dtype =np .int32 ),
        }

    total =sum (len (p )for p in all_positions )
    _log ("")
    _log ('---Step 4: Data merging ---')
    _log (f"  Всего elements: {total }")
    _log (f"  Meshей-источников: {len (all_positions )}")

    positions =np .vstack (all_positions )
    sizes =np .vstack (all_sizes )
    neighbors =np .vstack (all_neighbors )
    material =np .concatenate (all_material )
    positions ,sizes ,neighbors =_harmonize_solid_grid_and_neighbors (
        positions ,
        sizes ,
        neighbors ,
        material ,
        mesh_ids =all_mesh_ids ,
        element_size_mm =element_size_mm ,
        log_fn =_log ,
    )
    boundary =np .concatenate (all_boundary )
    membrane_mask =np .concatenate (all_membrane_mask )if all_membrane_mask else np .zeros (positions .shape [0 ],dtype =np .int32 )
    sensor_mask =np .concatenate (all_sensor_mask )if all_sensor_mask else np .zeros (positions .shape [0 ],dtype =np .int32 )
    mesh_ids =all_mesh_ids 

    n_before_bc =int (np .sum (boundary ))
    _log (f"  Boundary before BC (perimeter/surface): {n_before_bc }")

    if boundary_conditions :
        _log ("")
        _log ('--- Step 5: Applying Boundary Conditions ---')
        _log (f"  BC: {len (boundary_conditions )} pcs.")
        _apply_boundary_conditions (
        positions ,boundary ,mesh_ids ,
        boundary_conditions ,
        log_fn =_log ,
        )
    else :
        _log ("")
        _log ('--- Step 5: Boundary Conditions ---')
        _log ('BC not specified, skip')

    n_final_bc =int (np .sum (boundary ))

    air_data ={
    "air_element_position_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
    "air_element_size_xyz":np .zeros ((0 ,3 ),dtype =np .float64 ),
    "air_neighbors":np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 ),
    "air_neighbor_absorb_u8":np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 ),
    "air_material_index":np .zeros (0 ,dtype =np .uint8 ),
    "air_boundary_mask_elements":np .zeros (0 ,dtype =np .int32 ),
    "solid_to_air_index":np .full (positions .shape [0 ],-1 ,dtype =np .int32 ),
    "solid_to_air_index_plus":np .full (positions .shape [0 ],-1 ,dtype =np .int32 ),
    "solid_to_air_index_minus":np .full (positions .shape [0 ],-1 ,dtype =np .int32 ),
    "air_grid_shape":np .zeros (3 ,dtype =np .int32 ),
    }
    if generate_air_grid :
        air_mat_idx =int (material_key_to_index .get ("air",int (MAT_AIR )))
        air_data =_generate_regular_air_topology (
        positions ,
        sizes ,
        boundary ,
        solid_material_index =material ,
        air_material_index =air_mat_idx ,
        element_size_mm =element_size_mm ,
        padding_mm =padding_mm ,
        air_gap_layers =int (air_gap_layers ),
        max_air_cells =int (max_air_cells ),
        acoustic_boundary_conditions =boundary_conditions ,
        log_fn =_log ,
        )

    _log ("")
    _log ('===Completed ===')
    _log (f"  Elements: {total }")
    _log (f"  Boundary (total): {n_final_bc }")
    _log (f"  Air elements: {int (air_data ['air_element_position_xyz'].shape [0 ])}")
    _log (f"  positions size: {positions .nbytes /1024 /1024 :.2f} MB")

    out ={
    "element_position_xyz":positions ,
    "element_size_xyz":sizes ,
    "neighbors":neighbors ,
    "material_index":material ,
    "boundary_mask_elements":boundary ,
    "membrane_mask_elements":membrane_mask ,
    "sensor_mask_elements":sensor_mask ,
    }
    out .update (air_data )
    return out


def generate_procedural_topology_membrane (*args ,**kwargs )->dict [str ,np .ndarray ]:
    'Deprecated: Use generate_topology_from_meshes with mesh role=membrane.'
    raise NotImplementedError (
    'Use generate_topology_from_meshes with mesh role=membrane.'
    'Import a flat mesh and assign it the role of membrane.'
    )


def generate_procedural_topology_sensor (*args ,**kwargs )->dict [str ,np .ndarray ]:
    'Deprecated: Use generate_topology_from_meshes with mesh role=sensor.'
    raise NotImplementedError (
    'Use generate_topology_from_meshes with mesh role=sensor.'
    'Import a flat mesh and assign it the sensor role.'
    )
