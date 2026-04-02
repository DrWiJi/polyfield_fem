# -*- coding: utf-8 -*-
'Model of a diaphragm with a computing kernel for OpenCL.\n\nOne GPU thread = one end element: in one core are considered\nall forces (elastic from neighbors, air resistance, external pressure)\nand integration step RK4 is performed.\n\nRequires: PyOpenCL, OpenCL 1.2+ with cl_khr_fp64 (double) support.'

from __future__ import annotations 

import json 
import os 
import struct 
import time 
import numpy as np 
from typing import Callable ,Optional 

# Correctness criteria for debugging (explosion, 0 Hz, noise on the spectrogram)
MAX_UZ_UM_OK =500.0 # max |uz| (µm): above = "explosion" model
MIN_FREQ_HZ_OK =1.0 # spectrum peak below = "0 Hz", elasticity does not work
MIN_PEAK_PROMINENCE =2.0 # minimum (peak/spectral average): below = noise, not oscillation

try :
    from analytical_diaphragm import (
    analytical_natural_frequencies ,
    natural_frequency_membrane_rect ,
    )
except ImportError :
    analytical_natural_frequencies =None 
    natural_frequency_membrane_rect =None 

    # Avoiding KeyError in PyOpenCL when the kernel build fails (cache checks this variable)
if "PYOPENCL_CACHE_FAILURE_FATAL"not in os .environ :
    os .environ ["PYOPENCL_CACHE_FAILURE_FATAL"]="0"

import matplotlib .pyplot as plt 
from matplotlib .animation import FuncAnimation 

try :
    import pyopencl as cl 
except ImportError :
    cl =None 

    # Layout of the Params structure (must match diaphragm_opencl_kernel.cl).
    # In C, after int use_nonlinear_stiffness, the compiler aligns the next double by 8 bytes - adds 4 bytes of padding.
    # Without this, pre_tension and the remaining fields after int would be read by the kernel at the wrong offset.
_PARAMS_FORMAT =(
"i"*4 # nx, ny, n_elements, n_dof
+"d"*5 # dx, dy, thickness, arm_x, arm_y
+"d"*5 # k_axial_x, k_axial_y, k_shear, k_bending_x, k_bending_y
+"d"*3 # stiffness_transition_center, width, ratio
+"i"# use_nonlinear_stiffness
+"4x"# padding to 8-byte boundary (as in C struct)
+"d"*8 # rho_air, mu_air, Cd, element_area, element_mass, Ixx, Iyy, Izz
+"d"*7 # dt, pre_tension, k_soft, k_stiff, strain_transition, strain_width, k_bend
+"i"*2 # debug_elem, debug_step
)

# Materials:
# [density, E_parallel, E_perp, poisson, Cd, eta_visc, acoustic_impedance, acoustic_inject]
# acoustic_impedance: air->solid interface impedance (Pa·s/m).
# acoustic_inject: contribution to wave injection into the air-field from the FE speed (0 = does not radiate); emitter membrane ~1.
_MATERIAL_PROPS_STRIDE =8 
FACE_DIRS =6 

# Material indexes in the material_props array
MAT_PROP_DENSITY =0 
MAT_PROP_E_PARALLEL =1 
MAT_PROP_E_PERP =2 
MAT_PROP_POISSON =3 
MAT_PROP_CD =4 
MAT_PROP_ETA_VISC =5 
MAT_PROP_ACOUSTIC_IMPEDANCE =6 
MAT_PROP_ACOUSTIC_INJECT =7 

# Laws of interaction of materials
LAW_SOLID_SPRING =np .uint8 (0 )

# Material aliases (for readability and uniform indexes)
MAT_MEMBRANE =np .uint8 (0 )
MAT_FOAM_VE3015 =np .uint8 (1 )
MAT_SHEEPSKIN_LEATHER =np .uint8 (2 )
MAT_HUMAN_EAR_AVG =np .uint8 (3 )
MAT_SENSOR =np .uint8 (4 )
MAT_COTTON_WOOL =np .uint8 (5 )

EXCITATION_MODE_EXTERNAL =0
EXCITATION_MODE_EXTERNAL_FULL_OVERRIDE =1
EXCITATION_MODE_SECOND_ORDER_BOUNDARY_FULL_OVERRIDE =2
EXCITATION_MODE_EXTERNAL_VELOCITY_OVERRIDE =3
_EXCITATION_MODE_TO_KERNEL ={
    "external":EXCITATION_MODE_EXTERNAL ,
    "external_full_override":EXCITATION_MODE_EXTERNAL_FULL_OVERRIDE ,
    "second_order_boundary_full_override":EXCITATION_MODE_SECOND_ORDER_BOUNDARY_FULL_OVERRIDE ,
    "external_velocity_override":EXCITATION_MODE_EXTERNAL_VELOCITY_OVERRIDE ,
}


def _default_acoustic_inject_from_legacy_row (row_index :int )->float :
    'For 7-column libraries: sets acoustic_inject by row index (= material id in standard numbering).\n\n    - Sensor (microphone): 0 - does not radiate into the grid.\n    - Membrane and other solids: 1 - full FE->air source by default.\n    If needed, set column 8 explicitly in the material library.'
    ri =int (row_index )
    if ri ==int (MAT_SENSOR ):
        return 0.0 
    return 1.0


def _expand_material_props_to_stride8 (props :np .ndarray )->np .ndarray :
    'Expands the library to 8 columns (acoustic_inject).\n    Already 8 columns - no changes. Exactly 7 - inject column by row: see _default_acoustic_inject_from_legacy_row.'
    p =np .asarray (props ,dtype =np .float64 )
    if p .ndim !=2 :
        raise ValueError ('material_props should be an array shape[n_materials, n_cols]')
    if p .shape [1 ]==8 :
        if np .any (p [:,7 ]<0.0 ):
            raise ValueError ('acoustic_inject (column 7) must be >= 0')
        return p 
    if p .shape [1 ]!=7 :
        raise ValueError (
        'After normalizing to 7 columns, shape[n, 7] is expected before adding acoustic_inject;'
        f"got {p .shape }"
        )
    inj =np .zeros ((p .shape [0 ],1 ),dtype =np .float64 )
    for i in range (p .shape [0 ]):
        inj [i ,0 ]=_default_acoustic_inject_from_legacy_row (i )
    out =np .hstack ((p ,inj ))
    return out 


def _impedance_from_density_E (density :float ,E_parallel :float )->float :
    rho =max (float (density ),1e-12 )
    E =max (float (E_parallel ),0.0 )
    return float (np .sqrt (rho *E ))


_STOCK_IMPEDANCE_BY_NAME ={
    "membrane":_impedance_from_density_E (1380.0 ,5.0e9 ),
    "foam_ve3015":_impedance_from_density_E (400.0 ,0.08e6 ),
    "sheepskin_leather":_impedance_from_density_E (998.0 ,10.0e6 ),
    "human_ear_avg":_impedance_from_density_E (1080.0 ,1.80e6 ),
    "sensor":_impedance_from_density_E (1380.0 ,5.0e9 ),
    "cotton_wool":_impedance_from_density_E (400.0 ,0.03e6 ),
    "abs_plastic":_impedance_from_density_E (1050.0 ,2.4e9 ),
    "neodymium_magnet":_impedance_from_density_E (7500.0 ,160.0e9 ),
    "stainless_steel_303":_impedance_from_density_E (8000.0 ,195.0e9 ),
    "silicone_30_shore_a":_impedance_from_density_E (1100.0 ,2.5e6 ),
    "air":1.225 *343.0 ,
}
_STOCK_IMPEDANCE_BY_INDEX =[
    _STOCK_IMPEDANCE_BY_NAME ["membrane"],
    _STOCK_IMPEDANCE_BY_NAME ["foam_ve3015"],
    _STOCK_IMPEDANCE_BY_NAME ["sheepskin_leather"],
    _STOCK_IMPEDANCE_BY_NAME ["human_ear_avg"],
    _STOCK_IMPEDANCE_BY_NAME ["sensor"],
    _STOCK_IMPEDANCE_BY_NAME ["cotton_wool"],
]


def _stock_impedance_for_name (name :str )->float :
    key =str (name ).strip ().lower ()
    return float (_STOCK_IMPEDANCE_BY_NAME .get (key ,_STOCK_IMPEDANCE_BY_NAME ["membrane"]))


def _stock_impedance_for_index (idx :int )->float :
    i =int (idx )
    if 0 <=i <len (_STOCK_IMPEDANCE_BY_INDEX ):
        return float (_STOCK_IMPEDANCE_BY_INDEX [i ])
    return float (_STOCK_IMPEDANCE_BY_NAME ["membrane"])


def _pack_params (
nx :int ,ny :int ,
n_elements :int ,
width :float ,height :float ,thickness :float ,
density :float ,E_parallel :float ,E_perp :float ,poisson :float ,
use_nonlinear :bool ,
stiffness_transition_center :float ,stiffness_transition_width :float ,stiffness_ratio :float ,
rho_air :float ,mu_air :float ,Cd :float ,
dt :float ,
pre_tension :float ,
k_soft :float ,k_stiff :float ,
strain_transition :float ,strain_width :float ,
k_bend :float ,
debug_elem :int =-1 ,
debug_step :int =0 ,
)->bytes :
    n_dof =n_elements *6 
    dx =width /nx 
    dy =height /ny 
    element_volume =dx *dy *thickness 
    element_mass =density *element_volume 
    Ixx =element_mass *(dy **2 +thickness **2 )/12.0 
    Iyy =element_mass *(dx **2 +thickness **2 )/12.0 
    Izz =element_mass *(dx **2 +dy **2 )/12.0 
    element_area =dx *dy 
    arm_x =dx /2.0 
    arm_y =dy /2.0 
    k_axial_x =E_parallel *thickness *dy /dx 
    k_axial_y =E_perp *thickness *dx /dy 
    k_shear =E_parallel *thickness /(2.0 *(1.0 +poisson ))
    k_bending_x =E_parallel *thickness **3 *dy /(12.0 *dx **3 )
    k_bending_y =E_perp *thickness **3 *dx /(12.0 *dy **3 )

    return struct .pack (
    _PARAMS_FORMAT ,
    nx ,ny ,n_elements ,n_dof ,
    dx ,dy ,thickness ,arm_x ,arm_y ,
    k_axial_x ,k_axial_y ,k_shear ,k_bending_x ,k_bending_y ,
    stiffness_transition_center ,stiffness_transition_width ,stiffness_ratio ,
    1 if use_nonlinear else 0 ,
    rho_air ,mu_air ,Cd ,element_area ,element_mass ,Ixx ,Iyy ,Izz ,
    dt ,pre_tension ,k_soft ,k_stiff ,strain_transition ,strain_width ,k_bend ,
    debug_elem ,debug_step ,
    )


    # Offsets in the trace (must match the kernel TRACE_BUF_SIZE 127)
_TRACE_STEP =0 
_TRACE_ELASTIC =1 # 42
_TRACE_M_FINAL =43 # 3
_TRACE_POS_ME =46 # 6
_TRACE_VEL_ME =52 # 6
_TRACE_POS_MID =58 # 6
_TRACE_VEL_MID =64 # 6
_TRACE_F =70 # 6
_TRACE_MASS =76 # 6
_TRACE_ACC =82 # 6
_TRACE_X_NEW =88 # 6
_TRACE_V_NEW =94 # 6
_TRACE_ELASTIC_EXTRA =100 # 20: rx,ry,rz, link_len0, strain0, k_eff0, force_mag0, force_local0(3), lever0(3), M0(3), eff_len, rest_len
_TRACE_I =120 # Ixx, Iyy, Izz


def _print_opencl_trace (buf :np .ndarray ,debug_elem :int ,step_idx :int )->None :
    'Printing and validation of the full trace: integration and elasticity (including rotations).'
    n =min (buf .size ,127 )
    if n <123 :
        print (f"\n--- Trace elem={debug_elem } step={step_idx }: buffer too small ({n }) ---")
        return 
    step =int (buf [_TRACE_STEP ])
    pos_me =buf [_TRACE_POS_ME :_TRACE_POS_ME +6 ]
    vel_me =buf [_TRACE_VEL_ME :_TRACE_VEL_ME +6 ]
    pos_mid =buf [_TRACE_POS_MID :_TRACE_POS_MID +6 ]
    vel_mid =buf [_TRACE_VEL_MID :_TRACE_VEL_MID +6 ]
    F =buf [_TRACE_F :_TRACE_F +6 ]
    mass =buf [_TRACE_MASS :_TRACE_MASS +6 ]
    acc =buf [_TRACE_ACC :_TRACE_ACC +6 ]
    x_new =buf [_TRACE_X_NEW :_TRACE_X_NEW +6 ]
    v_new =buf [_TRACE_V_NEW :_TRACE_V_NEW +6 ]
    rx ,ry ,rz =buf [_TRACE_ELASTIC_EXTRA ],buf [_TRACE_ELASTIC_EXTRA +1 ],buf [_TRACE_ELASTIC_EXTRA +2 ]
    link_len0 =buf [_TRACE_ELASTIC_EXTRA +3 ]
    strain0 =buf [_TRACE_ELASTIC_EXTRA +4 ]
    k_eff0 =buf [_TRACE_ELASTIC_EXTRA +5 ]
    force_mag0 =buf [_TRACE_ELASTIC_EXTRA +6 ]
    Ixx ,Iyy ,Izz =buf [_TRACE_I ],buf [_TRACE_I +1 ],buf [_TRACE_I +2 ]
    M_el =buf [4 :7 ]
    F_el =buf [1 :4 ]

    finite =np .all (np .isfinite (buf [1 :n ]))
    rot_ok =np .isfinite (rx )and np .isfinite (ry )and np .isfinite (rz )
    rot_small =abs (rx )<1.0 and abs (ry )<1.0 and abs (rz )<1.0 
    ang_acc =np .array ([F [3 ]/(Ixx +1e-30 ),F [4 ]/(Iyy +1e-30 ),F [5 ]/(Izz +1e-30 )])
    ang_acc_ok =np .all (np .isfinite (ang_acc ))and np .all (np .abs (ang_acc )<1e12 )

    issues =[]
    if not finite :
        issues .append ('not all values \u200b\u200bare finite')
    if not rot_ok :
        issues .append ('rx,ry,rz is not finite')
    if not rot_small :
        issues .append ('rotations come out of [-1,1] rad')
    if not ang_acc_ok :
        issues .append ('angular acceleration is inadequate')
    if not np .all (np .isfinite (x_new )):
        issues .append ('x_new contains NaN/Inf')
    if not np .all (np .isfinite (v_new )):
        issues .append ('v_new contains NaN/Inf')
    if step_idx ==0 and abs (pos_me [0 ])<1e-10 and abs (pos_me [1 ])<1e-10 :
        issues .append ('pos_me (x,y)=0 at step 0 - check grid initialization (rest positions)')
    status ='[ERROR:'+"; ".join (issues )+"]"if issues else ""

    print (f"\n--- Trace elem={debug_elem } step={step_idx } (kernel step={step }){status } ---")
    print ('Integration (RK2 stage2):')
    print (f"    pos_me   = {pos_me }  (x,y,z,rx,ry,rz)")
    print (f"    vel_me   = {vel_me }")
    print (f"    pos_mid  = {pos_mid }")
    print (f"    vel_mid  = {vel_mid }")
    print (f"    F        = {F }  (Fx,Fy,Fz, Mx,My,Mz)")
    print (f"    mass     = {mass }")
    print (f"    acc      = {acc }  (a = F/mass)")
    print (f"    x_new    = {x_new }")
    print (f"    v_new    = {v_new }")
    print ('Rotations (pos_mid[3:6], used in elasticity):')
    print (f"    rx={rx :.6e} ry={ry :.6e} rz={rz :.6e}")
    print ('Elasticity dir0: center_len, strain, k_eff, force_mag:')
    print (f"    {link_len0 :.6e} {strain0 :.6e} {k_eff0 :.6e} {force_mag0 :.6e}")
    print ('Moments: M_elastic_total =',M_el ,"  F_elastic =",F_el )
    print ('Angular acceleration (M/I):',ang_acc )
    print ('Moments of inertia Ixx,Iyy,Izz:',Ixx ,Iyy ,Izz )
    if buf .size >_TRACE_ELASTIC_EXTRA +18 :
        pre_tension_kernel =buf [_TRACE_ELASTIC_EXTRA +18 ]
        print ('pre_tension (read in kernel):',pre_tension_kernel )
    print ()


    # ---------------------------------------------------------------------------
    # Model class (OpenCL)
    # ---------------------------------------------------------------------------
class PlanarDiaphragmOpenCL :
    'Model of a diaphragm with a computing kernel for OpenCL.\n    One time step dt: RK4 for FE; the air mesh uses anisotropic CFL (c*dt_sub*sqrt(1/dx^2+1/dy^2+1/dz^2) <~1) with homogeneous leapfrog substeps; FE→air inject is applied once per FE dt, not every substep.\n\n    Acoustic field: discrete wave equation p_tt = c^2 lap(p); missing-air neighbors use absorbing treatment in the OpenCL air kernel.\n    FE→air injection uses air_inject_dV_dot; CSR uses center air_elem_map for bulk solids, bilateral ±cells for membrane/sensor (thin) so both sides radiate.\n    Air feeds back through pressure traction on the structure. Initial p is uniform (air_initial_uniform_pressure_pa).\n    External pressure_pa in step() is mechanical drive on the membrane, not a direct air initial condition.\n\n    Elasticity: nonlinear BOPET spring between adjacent element faces.\n    Air is advanced once per FE step using FDTD leapfrog on GPU.\n    Air→FE traction uses −∇p·V with a central difference over 2·dx_air (see kernel); tune air_coupling_gain / coupling_recv if you change grid spacing.'

    def __init__ (
    self ,
    width_mm :float =48.0 ,
    height_mm :float =63.0 ,
    nx :int =24 ,
    ny :int =32 ,
    thickness_mm :float =0.012 ,
    density_kg_m3 :float =1380.0 ,
    E_parallel_gpa :float =5.0 ,
    E_perp_gpa :float =3.5 ,
    poisson :float =0.3 ,
    use_nonlinear_stiffness :bool =True ,
    stiffness_transition_center :float =0.002 ,
    stiffness_transition_width :float =0.001 ,
    stiffness_ratio :float =20.0 ,
    rho_air :float =1.2 ,
    mu_air :float =1.81e-5 ,
    Cd :float =1.0 ,
    air_sound_speed_m_s :float =343.0 ,
    air_padding_mm :float |None =None ,
    air_grid_step_mm :float |None =None ,
    air_solver_mode :str ="second_order",
    excitation_mode :str ="external",
    air_coupling_gain :float =1 ,
    air_initial_uniform_pressure_pa :float =0.0 ,
    pre_tension_N_per_m :float =10.0 ,
    k_soft :float |None =None ,
    k_stiff :float |None =None ,
    strain_transition :float =0.002 ,
    strain_width :float =0.0005 ,
    k_bend :float |None =None ,
    platform_index :int =0 ,
    device_index :int =0 ,
    kernel_debug :bool =False ,
    material_props :np .ndarray |None =None ,
    )->None :
        self .width =width_mm *1e-3 
        self .height =height_mm *1e-3 
        self .thickness =thickness_mm *1e-3 
        self .nx =nx 
        self .ny =ny 
        self .n_membrane_elements =nx *ny 
        self .n_layers_total =1 
        self .n_elements =self .n_membrane_elements *self .n_layers_total 
        self .dof_per_element =6 
        self .n_dof =self .n_elements *self .dof_per_element 
        self ._topology_is_rect_grid =True 
        self ._visual_shape =(self .ny ,self .nx )
        self ._visual_element_indices =np .arange (self .n_elements ,dtype =np .int32 )
        self .visualization_enabled =True 

        self .density =density_kg_m3 
        self .E_parallel =E_parallel_gpa *1e9 
        self .E_perp =E_perp_gpa *1e9 
        self .poisson =poisson 
        self .use_nonlinear_stiffness =use_nonlinear_stiffness 
        self .stiffness_transition_center =stiffness_transition_center 
        self .stiffness_transition_width =stiffness_transition_width 
        self .stiffness_ratio =stiffness_ratio 
        self .rho_air =rho_air 
        self .mu_air =mu_air 
        self .Cd =Cd 
        self .air_sound_speed =air_sound_speed_m_s 
        self .air_padding =(air_padding_mm *1e-3 )if air_padding_mm is not None else None 
        self .air_grid_step =(air_grid_step_mm *1e-3 )if air_grid_step_mm is not None else None 
        self .air_solver_mode =str (air_solver_mode ).strip ().lower ()
        self .excitation_mode =str (excitation_mode ).strip ().lower ()
        if self .excitation_mode not in _EXCITATION_MODE_TO_KERNEL :
            raise ValueError (
                "excitation_mode must be one of: external, external_full_override, "
                "second_order_boundary_full_override"
            )
        self ._kernel_excitation_mode =int (_EXCITATION_MODE_TO_KERNEL [self .excitation_mode ])
        if self .excitation_mode =="second_order_boundary_full_override":
            self .air_solver_mode ="second_order"
        self .air_coupling_gain =air_coupling_gain 
        # Uniform acoustic pressure in all cells when resetting/assembling the grid.
        self .air_initial_uniform_pressure_pa =float (air_initial_uniform_pressure_pa )
        self .air_bulk_modulus_pa =float (self .rho_air *(self .air_sound_speed **2 ))

        self .pre_tension =pre_tension_N_per_m 
        dx =self .width /nx 
        dy =self .height /ny 
        dz =self .thickness 
        self .element_size_xyz =np .empty ((self .n_elements ,3 ),dtype =np .float64 )
        self .element_size_xyz [:,0 ]=dx 
        self .element_size_xyz [:,1 ]=dy 
        self .element_size_xyz [:,2 ]=dz 
        k_axial_x =self .E_parallel *self .thickness *dy /dx 
        k_axial_y =self .E_perp *self .thickness *dx /dy 
        self .k_soft =k_soft if k_soft is not None else (k_axial_x +k_axial_y )/2 /stiffness_ratio 
        self .k_stiff =k_stiff if k_stiff is not None else (k_axial_x +k_axial_y )/2 
        self .strain_transition =strain_transition 
        self .strain_width =strain_width 
        k_bend_base =self .E_parallel *self .thickness **3 /12.0 
        self .k_bend =k_bend if k_bend is not None else k_bend_base *(dy /dx +dx /dy )/2 

        if material_props is not None :
            props =np .asarray (material_props ,dtype =np .float64 )
            if props .ndim !=2 or props .shape [1 ]not in (5 ,6 ,7 ,_MATERIAL_PROPS_STRIDE ):
                raise ValueError (
                'material_props must have shape [n_materials, 5..8]'
                '(8: added acoustic_inject for injection into air-field)'
                )
            if props .shape [1 ]==5 :
                props =np .hstack ((props ,np .zeros ((props .shape [0 ],1 )),np .ones ((props .shape [0 ],1 ))))
            elif props .shape [1 ]==6 :
                props =np .hstack ((props ,np .ones ((props .shape [0 ],1 ))))
            if props .shape [0 ]<1 :
                raise ValueError ('material_props must contain at least 1 material')
            self .material_props =_expand_material_props_to_stride8 (props )
        else :
            self .material_props =self ._build_default_material_library ()
        self .material_id_map ={
        "membrane":int (MAT_MEMBRANE ),
        "foam_ve3015":int (MAT_FOAM_VE3015 ),
        "sheepskin_leather":int (MAT_SHEEPSKIN_LEATHER ),
        "human_ear_avg":int (MAT_HUMAN_EAR_AVG ),
        "sensor":int (MAT_SENSOR ),
        "cotton_wool":int (MAT_COTTON_WOOL ),
        }

        self .material_index =np .full (self .n_elements ,MAT_SENSOR ,dtype =np .uint8 )
        self ._sensor_mask =np .zeros (self .n_elements ,dtype =bool )
        self ._update_sensor_mask ()
        self .membrane_mask =np .zeros (self .n_elements ,dtype =np .int32 )
        self .membrane_mask [:self .n_membrane_elements ]=1 
        self ._fe_air_coupling_mask =np .asarray ((self .membrane_mask !=0 )|self ._sensor_mask ,dtype =np .uint8 )
        self .laws =np .full (
        (self .material_props .shape [0 ],self .material_props .shape [0 ]),
        LAW_SOLID_SPRING ,
        dtype =np .uint8 ,
        )
        flat_topology =self .generate_planar_membrane_topology (
        plane ="xy",
        thickness_m =self .thickness ,
        size_u_m =self .width ,
        size_v_m =self .height ,
        )
        self .neighbors =flat_topology ["neighbors"]
        self .boundary_mask_elements =flat_topology ["boundary_mask_elements"]
        self .position =np .zeros (self .n_dof ,dtype =np .float64 )
        self .position [0 ::self .dof_per_element ]=flat_topology ["element_position_xyz"][:,0 ]
        self .position [1 ::self .dof_per_element ]=flat_topology ["element_position_xyz"][:,1 ]
        self .position [2 ::self .dof_per_element ]=flat_topology ["element_position_xyz"][:,2 ]
        self .element_size_xyz =flat_topology ["element_size_xyz"]
        self .nx_air =0 
        self .ny_air =0 
        self .nz_air =0 
        self .n_air_cells =0 
        self .dx_air =0.0 
        self .dy_air =0.0 
        self .dz_air =0.0 
        self .air_origin_x =0.0 
        self .air_origin_y =0.0 
        self .air_origin_z =0.0 
        self .air_map_6 =np .empty ((self .n_elements ,6 ),dtype =np .int32 )
        self .air_elem_map =np .empty ((self .n_elements ,),dtype =np .int32 )
        self .air_inject_cell_plus =np .full (self .n_elements ,-1 ,dtype =np .int32 )
        self .air_inject_cell_minus =np .full (self .n_elements ,-1 ,dtype =np .int32 )
        self .air_elem_face_area =np .empty ((self .n_elements ,3 ),dtype =np .float64 )
        self .air_elem_volume =np .empty ((self .n_elements ,),dtype =np .float64 )
        self .air_neighbors =np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 )
        self .air_neighbor_absorb_u8 =np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 )
        self .air_boundary_mask_elements =np .zeros (0 ,dtype =np .int32 )
        self ._air_topology_from_payload =False
        self ._air_warned_cfl =False
        self ._air_subcycle_note_shown =False
        self ._air_subcycle_cap_warned =False
        self ._air_acoustic_summary_shown =False
        self ._fe_subcycle_summary_shown =False
        # FE subcycling defaults are intentionally moderate (speed/stability balance).
        self .fe_stability_safety =0.75
        self .fe_subcycle_cap = 1 # TODO Choose a better value
        self ._air_slice_duplicate_warned =False
        self ._air_last_acoustic_dt :float |None =None
        self .air_pressure_prev =np .empty ((0 ,),dtype =np .float64 )
        self .air_pressure_curr =np .empty ((0 ,),dtype =np .float64 )
        self .air_pressure_next =np .empty ((0 ,),dtype =np .float64 )
        self ._air_cell_pos_xyz =np .empty ((0 ,3 ),dtype =np .float64 )
        self .velocity =np .zeros (self .n_dof ,dtype =np .float64 )
        self ._velocity_prev =np .zeros (self .n_dof ,dtype =np .float64 )
        self ._velocity_delta =np .zeros (self .n_dof ,dtype =np .float64 )
        self .force_external =np .zeros (self .n_dof ,dtype =np .float64 )
        self ._force_override_active =False
        self ._force_drive_mask_u8 =np .zeros (self .n_elements ,dtype =np .uint8 )
        self ._force_drive_axis_u8 =np .zeros (self .n_elements ,dtype =np .uint8 )
        self ._force_drive_area_n =np .zeros (self .n_elements ,dtype =np .float64 )
        self ._update_center_index ()

        self .history_disp_center :list [float ]=[]
        self ._record_history =False 
        self .history_disp_all :list [np .ndarray ]=[]
        self .history_air_center_xz :list [np .ndarray ]=[]
        self .history_air_pressure_xy_center_z :list [np .ndarray ]=[]
        self .history_air_pressure_step :int =1
        self ._warned_no_external_excitation =False
        self ._warned_no_external_velocity_excitation =False
        self ._warned_no_z0_radiating =False
        self ._air_trace_step_counter =0
        self .kernel_debug =bool (kernel_debug )
        # Air explosion diagnostics (used by _log_air_pressure_metrics; lightweight unless triggered).
        self .air_explosion_abs_pa =1e5
        self .air_explosion_rel_growth =200.0
        self .air_explosion_log_topk_cells =12
        self .air_explosion_log_topk_elems =25
        self ._air_last_fe_dt =None
        self ._air_last_dt_sub =None
        self ._air_history_iy_cached :int |None =None
        # FE→air CSR sanity checks (invalid pickles / old bilateral matrices).
        self .air_inject_csr_validate =True
        self .air_inject_csr_max_nnz_per_row =16384
        self .air_inject_csr_validate_full_every_steps =200
        self ._air_csr_validate_step_counter =0
        self ._air_csr_dirty =True
        self ._static_fe_dirty =True
        self .air_metric_log_every_steps =25
        self .air_explosion_dump_cooldown_steps =200
        self ._air_last_explosion_dump_step =-10**9

        if cl is None :
            raise RuntimeError ('PyOpenCL is not installed. Install: pip install pyopencl')

            # Platform and device
        platforms =cl .get_platforms ()
        if platform_index >=len (platforms ):
            platform_index =0 
        platform =platforms [platform_index ]
        devices =platform .get_devices (cl .device_type .GPU )
        if not devices and platform .get_devices ():
            devices =platform .get_devices ()
        if not devices :
            raise RuntimeError ('OpenCL: no suitable device found')
        if device_index >=len (devices ):
            device_index =0 
        device =devices [device_index ]
        self .ctx =cl .Context ([device ])
        self .queue =cl .CommandQueue (self .ctx )

        # Building the program (kernel from a file next to the script)
        kernel_path =os .path .join (os .path .dirname (os .path .abspath (__file__ )),"diaphragm_opencl_kernel.cl")
        if not os .path .isfile (kernel_path ):
            raise FileNotFoundError (f"Kernel file not found: {kernel_path }")
        with open (kernel_path ,"r",encoding ="utf-8")as f :
            kernel_src =f .read ()

            # Without compiler options: some drivers (including on Windows) give INVALID_COMPILER_OPTIONS to -cl-std / -cl-khr-fp64.
            # Double is enabled in the kernel via #pragma OPENCL EXTENSION cl_khr_fp64 : enable
        build_options =[
            "-DENABLE_DEBUG=1"if self .kernel_debug else "-DENABLE_DEBUG=0",
            f"-DEXCITATION_MODE={self ._kernel_excitation_mode }",
        ]
        try :
            self .prg =cl .Program (self .ctx ,kernel_src ).build (options =build_options )
        except cl .RuntimeError as e :
            raise RuntimeError (f"OpenCL kernel build failed: {e }")from e 

        self ._kernel_acc =self .prg .diaphragm_rk4_acc 
        self ._kernel_rk4_stage_state =self .prg .diaphragm_rk4_stage_state 
        self ._kernel_rk4_finalize =self .prg .diaphragm_rk4_finalize 
        self ._kernel_air_inject_reduce =self .prg .air_inject_reduce_to_pressure
        self ._kernel_air_force_from_p =self .prg .air_pressure_to_fe_force
        self ._kernel_air_acoustic =self .prg .air_acoustic_leapfrog_sommerfeld 
        self ._kernel_air_acoustic_second_order =getattr (self .prg ,"air_pressure_wave_second_order_bc",None )
        # First-order acoustic (p + particle velocity) kernels (optional).
        self ._kernel_air_u =getattr (self .prg ,"air_first_order_update_u",None )
        self ._kernel_air_p =getattr (self .prg ,"air_first_order_update_p",None )
        # Buffers (create once, reuse)
        mf =cl .mem_flags 
        self ._buf_position =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_velocity =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_acc =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_acc_k2 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_acc_k3 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_acc_k4 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_velocity_delta =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_dof *8 )
        self ._buf_force_external =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_force_drive_mask =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements )
        self ._buf_force_drive_axis =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements )
        self ._buf_force_drive_area =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements *8 )
        self ._buf_boundary =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements *4 )
        self ._buf_element_size =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements *3 *8 )
        self ._buf_material_index =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements )
        self ._buf_fe_air_coupling_mask =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements )
        self ._buf_material_props =cl .Buffer (
        self .ctx ,mf .READ_ONLY ,size =self .material_props .size *8 
        )
        self ._buf_neighbors =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_elements *FACE_DIRS *4 )
        self ._buf_laws =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .laws .size )
        self ._buf_position_mid =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_velocity_mid =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_position_0 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_velocity_0 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_velocity_k2 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_velocity_k3 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_velocity_k4 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._buf_first_bad =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =4 )
        self ._buf_first_bad_meta =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =4 )
        self ._first_bad_meta_host =np .array ([0 ],dtype =np .int32 )
        self ._buf_first_bad_neighbor_elem =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =4 )
        self ._first_bad_neighbor_elem_host =np .array ([-1 ],dtype =np .int32 )
        self ._buf_first_bad_interface_dir =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =4 )
        self ._first_bad_interface_dir_host =np .array ([-1 ],dtype =np .int32 )
        self ._buf_params :cl .Buffer |None =None
        self ._DEBUG_BUF_DOUBLES =127 # tracing: step, elastic(42), pos/vel/F/acc/x_new/v_new, trace_elastic(20), I
        self ._buf_debug =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self ._DEBUG_BUF_DOUBLES *8 )
        self ._buf_air_force_external =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )
        self ._air_force_external =np .zeros (self .n_dof ,dtype =np .float64 )
        self ._acc =np .zeros (self .n_dof ,dtype =np .float64 )

        # Group size (local size) and global size
        self ._local_size =min (256 ,self ._get_max_work_group_size ())
        self ._global_size =((self .n_elements +self ._local_size -1 )//self ._local_size )*self ._local_size 

        backend =f"OpenCL ({platform .name }, {device .name })"
        # air_grid below is from the temporary flat shell in __init__ only; after set_custom_topology()
        # rebuild_air_field recomputes nx_air, ny_air, nz_air from the real FE bbox (see topology report).
        print (
        f"PlanarDiaphragmOpenCL: {width_mm }x{height_mm } mm, {nx }x{ny }, membrane={self .n_membrane_elements }, "
        f"DOF={self .n_dof }, backend={backend }, kernel_debug={self .kernel_debug }"
        )

    def _get_max_work_group_size (self )->int :
        try :
            return self ._kernel_acc .get_work_group_info (
            cl .kernel_work_group_info .WORK_GROUP_SIZE ,self .ctx .devices [0 ]
            )
        except Exception :
            return 256 

    def _sync_simulation_buffers (self )->None :
        cl .enqueue_copy (self .queue ,self ._buf_position ,self .position )
        cl .enqueue_copy (self .queue ,self ._buf_velocity ,self .velocity )
        cl .enqueue_copy (self .queue ,self ._buf_velocity_delta ,self ._velocity_delta )
        cl .enqueue_copy (self .queue ,self ._buf_force_external ,self .force_external )
        self ._upload_static_fe_buffers_if_dirty (force =True )
        cl .enqueue_copy (self .queue ,self ._buf_air_force_external ,self ._air_force_external )
        self .queue .finish ()

    def _upload_static_fe_buffers_if_dirty (self ,*,force :bool =False )->None :
        if (not force )and (not getattr (self ,"_static_fe_dirty",True )):
            return
        self ._rebuild_force_drive_geometry ()
        cl .enqueue_copy (self .queue ,self ._buf_boundary ,self .boundary_mask_elements )
        cl .enqueue_copy (self .queue ,self ._buf_element_size ,self .element_size_xyz )
        cl .enqueue_copy (self .queue ,self ._buf_material_index ,self .material_index )
        cl .enqueue_copy (self .queue ,self ._buf_fe_air_coupling_mask ,self ._fe_air_coupling_mask )
        cl .enqueue_copy (self .queue ,self ._buf_material_props ,self .material_props )
        cl .enqueue_copy (self .queue ,self ._buf_neighbors ,self .neighbors )
        cl .enqueue_copy (self .queue ,self ._buf_laws ,self .laws )
        cl .enqueue_copy (self .queue ,self ._buf_force_drive_mask ,self ._force_drive_mask_u8 )
        cl .enqueue_copy (self .queue ,self ._buf_force_drive_axis ,self ._force_drive_axis_u8 )
        cl .enqueue_copy (self .queue ,self ._buf_force_drive_area ,self ._force_drive_area_n )
        self ._static_fe_dirty =False

    def _rebuild_force_drive_geometry (self )->None :
        """Precompute scalar-pressure drive mapping (mask, axis, area) for kernel-side force generation."""
        if self .n_elements <=0 :
            self ._force_drive_mask_u8 =np .zeros (0 ,dtype =np .uint8 )
            self ._force_drive_axis_u8 =np .zeros (0 ,dtype =np .uint8 )
            self ._force_drive_area_n =np .zeros (0 ,dtype =np .float64 )
            return
        n =self .n_elements
        sz_blk =np .asarray (self .element_size_xyz [:n ],dtype =np .float64 )
        normal_axis =np .argmin (sz_blk ,axis =1 ).astype (np .uint8 )
        sx ,sy ,sz =sz_blk [:,0 ],sz_blk [:,1 ],sz_blk [:,2 ]
        area_normal =np .where (
            normal_axis ==0 ,sy *sz ,
            np .where (normal_axis ==1 ,sx *sz ,sx *sy ),
        ).astype (np .float64 )
        if self .boundary_mask_elements .size >=n :
            non_boundary =(self .boundary_mask_elements [:n ]==0 )
        else :
            non_boundary =np .ones (n ,dtype =bool )
        membrane_only =(
            np .asarray (self .membrane_mask [:n ]!=0 ,dtype =bool )
            if hasattr (self ,"membrane_mask")
            else np .asarray (self .material_index [:n ]==MAT_MEMBRANE ,dtype =bool )
        )
        drive_candidates =non_boundary &membrane_only
        z_c =self .position [2 :n *self .dof_per_element :self .dof_per_element ]
        z0_mask =np .isclose (z_c ,0.0 ,atol =1e-12 )
        excite_mask =z0_mask &drive_candidates
        if not np .any (excite_mask )and np .any (drive_candidates ):
            excite_mask =drive_candidates
            if not self ._warned_no_z0_radiating :
                print (
                    "[force][warn] No membrane non-boundary FE on z≈0; "
                    f"fallback excitation layer: {int (np .sum (excite_mask ))} elements."
                )
                self ._warned_no_z0_radiating =True
        forced_count =int (np .sum (excite_mask ))
        if forced_count ==0 and not self ._warned_no_external_excitation :
            print (
                "[force][warn] External pressure drive applied to 0 elements "
                "(no non-boundary candidates in z≈0 layer or thin-layer fallback)."
            )
            self ._warned_no_external_excitation =True
        self ._force_drive_mask_u8 =np .ascontiguousarray (excite_mask .astype (np .uint8 ))
        self ._force_drive_axis_u8 =np .ascontiguousarray (normal_axis .astype (np .uint8 ))
        self ._force_drive_area_n =np .ascontiguousarray (area_normal .astype (np .float64 ))

    def _air_debug_diagnostics (self ,dt :float ,where :str )->None :
        if self .n_air_cells <=0 :
            return
        dx =float (self .dx_air if self .dx_air >0.0 else 1.0 )
        dy =float (self .dy_air if self .dy_air >0.0 else 1.0 )
        dz =float (self .dz_air if self .dz_air >0.0 else 1.0 )
        eh =1e-30
        inv_h2 =1.0 /(dx *dx +eh )+1.0 /(dy *dy +eh )+1.0 /(dz *dz +eh )
        if inv_h2 >0.0 :
            dt_cfl =float (getattr (self ,'_air_last_acoustic_dt',None )or dt )
            nu_eff =float (self .air_sound_speed *dt_cfl *np .sqrt (inv_h2 ))
            if nu_eff >1.0 and not self ._air_warned_cfl :
                print (
                f"[air] CFL instability at {where }: c*dt_air*sqrt(1/dx^2+1/dy^2+1/dz^2)={nu_eff :.3f} > 1. "
                "Reduce FE dt, coarsen the air voxel step, or raise the substep cap."
                )
                self ._air_warned_cfl =True
        if not self .kernel_debug :
            return
        pressure_bad =(
            self .air_pressure_curr .size >0
            and not np .all (np .isfinite (self .air_pressure_curr ))
        )
        if pressure_bad :
            print (f"[air][debug] Non-finite pressure values detected at {where }.")

            try :
                p =np .asarray (self .air_pressure_curr ,dtype =np .float64 )
                bad =~np .isfinite (p )
                if bad .any ():
                    i =int (np .flatnonzero (bad )[0 ])
                    print (f"[air][debug] first_bad_air_cell={i } p={p [i ]!r} "
                           f"(dt_fe={float (dt ):.3e}, dt_sub={float (getattr (self ,'_air_last_dt_sub',dt )):.3e}, "
                           f"dx,dy,dz=({dx :.3e},{dy :.3e},{dz :.3e}))")
                    if hasattr (self ,"air_neighbors")and isinstance (self .air_neighbors ,np .ndarray )and self .air_neighbors .shape [0 ]>i :
                        nb =self .air_neighbors [i ]
                        ab =None
                        if hasattr (self ,"air_neighbor_absorb_u8")and isinstance (self .air_neighbor_absorb_u8 ,np .ndarray )and self .air_neighbor_absorb_u8 .shape [0 ]>i :
                            ab =self .air_neighbor_absorb_u8 [i ]
                        print (f"[air][debug] neighbors={nb .tolist ()} absorb_u8={(ab .tolist () if ab is not None else None)}")
                    # Pull relevant device buffers for a focused stencil dump.
                    if hasattr (self ,"_buf_air_prev")and hasattr (self ,"_buf_air_curr")and hasattr (self ,"_buf_air_plus"):
                        p_prev =np .empty_like (p )
                        p_curr =np .empty_like (p )
                        p_inj =np .empty_like (p )
                        cl .enqueue_copy (self .queue ,p_prev ,self ._buf_air_prev )
                        cl .enqueue_copy (self .queue ,p_curr ,self ._buf_air_curr )
                        cl .enqueue_copy (self .queue ,p_inj ,self ._buf_air_plus )
                        self .queue .finish ()
                        print (f"[air][debug] buffers@i: p_prev={p_prev [i ]!r} p_curr={p_curr [i ]!r} inject_dp={p_inj [i ]!r}")
                        if hasattr (self ,"air_neighbors")and isinstance (self .air_neighbors ,np .ndarray )and self .air_neighbors .shape [0 ]>i :
                            nb =self .air_neighbors [i ].astype (np .int64 ,copy =False )
                            # Dump neighbor values for the Laplacian stencil.
                            def _val (arr ,j ):
                                if j <0 :
                                    return arr [i ]
                                if j >=arr .shape [0 ]:
                                    return float ("nan")
                                return arr [j ]
                            vals_prev =[_val (p_prev ,int (j )) for j in nb ]
                            vals_curr =[_val (p_curr ,int (j )) for j in nb ]
                            print (f"[air][debug] neighbor p_prev={vals_prev }")
                            print (f"[air][debug] neighbor p_curr={vals_curr }")
            except Exception :
                pass
            # Reuse existing detailed dump (top cells + CSR element contributions).
            try :
                self ._dump_air_exploding_cells (
                    int (getattr (self ,"_air_trace_step_counter",-1 )),
                    np .asarray (self .air_pressure_curr ,dtype =np .float64 ),
                    f"debug_{where }",
                )
            except Exception :
                pass

            # If the pressure already diverged, the most likely culprit is an invalid injected
            # delta-pressure term from FE→air communication. Dump inject_dp to pinpoint it.
            if where == "air_one_step" and hasattr (self ,"_buf_air_plus"):
                try :
                    inj =np .empty_like (self .air_pressure_curr ,dtype =np .float64 )
                    cl .enqueue_copy (self .queue ,inj ,self ._buf_air_plus )
                    self .queue .finish ()
                    inj_bad_mask =~np .isfinite (inj )
                    if inj_bad_mask .any ():
                        bad_idx =np .flatnonzero (inj_bad_mask )
                        sample =bad_idx [:min (bad_idx .size ,10 )].tolist ()
                        print (f"[air][debug] Non-finite inject_dp detected at {where}: "
                               f"n_bad={bad_idx .size} sample_idx={sample}")
                except Exception :
                    pass
        if self ._air_force_external .size >0 and not np .all (np .isfinite (self ._air_force_external )):
            print (f"[air][debug] Non-finite air force values detected at {where }.")
            try :
                ff =np .asarray (self ._air_force_external ,dtype =np .float64 )
                badf =~np .isfinite (ff )
                if badf .any ():
                    j =int (np .flatnonzero (badf )[0 ])
                    elem =j // int (self .dof_per_element )
                    comp =j % int (self .dof_per_element )
                    print (f"[air][debug] first_bad_air_force_dof={j } (elem={elem }, comp={comp }) value={ff [j ]!r}")
            except Exception :
                pass

    @staticmethod
    def _air_neighbor_values (p :np .ndarray ,nb :np .ndarray ,fallback :np .ndarray |float |None =None )->np .ndarray :
        """Pressure (or field) at neighbor cells. If nb[i] < 0 (no neighbor), use `fallback`.

        Important: without `fallback`, missing neighbors used `p[clip(nb)]`, i.e. index 0 for all
        -1 entries — wrong for Laplacian stencils and biases the field (e.g. no negative p).
        For the wave step, pass `fallback=p` (same cell) for Neumann ∂p/∂n≈0 at domain faces."""
        n =p .shape [0 ]
        idx =np .clip (nb ,0 ,max (n -1 ,0 ))
        out =p [idx ] if n >0 else np .zeros_like (nb ,dtype =np .float64 )
        if fallback is None :
            fb =p if p .shape ==nb .shape else out
        else :
            fb =np .asarray (fallback ,dtype =np .float64 )
            if fb .shape !=nb .shape :
                fb =np .broadcast_to (fb ,nb .shape )
        out =np .where (nb >=0 ,out ,fb )
        return out

    def _infer_air_neighbor_absorb_from_geometry (self )->np .ndarray :
        'Missing air neighbor: Sommerfeld (1) on all missing-neighbor faces.'
        n =int (self .n_air_cells )
        nb =self .air_neighbors
        if nb .shape !=(n ,FACE_DIRS )or n <=0 :
            return np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 )
        ab =np .zeros ((n ,FACE_DIRS ),dtype =np .uint8 )
        ab [nb <0 ]=1
        return ab

    def _compute_air_force_from_pressure_buffer (self ,p_buf )->None :
        if self .n_air_cells <=0 :
            self ._air_force_external .fill (0.0 )
            cl .enqueue_copy (self .queue ,self ._buf_air_force_external ,self ._air_force_external )
            return
        self ._kernel_air_force_from_p .set_args (
        p_buf ,
        self ._buf_air_map_6 ,
        self ._buf_air_elem_map ,
        self ._buf_element_size ,
        self ._buf_material_index ,
        self ._buf_fe_air_coupling_mask ,
        self ._buf_material_props ,
        np .int32 (self .material_props .shape [0 ]),
        np .int32 (self .n_air_cells ),
        np .int32 (self .n_elements ),
        np .float64 (float (self .air_initial_uniform_pressure_pa )),
        np .float64 (float (self .air_coupling_gain )),
        np .float64 (float (self .dx_air if self .dx_air >0.0 else 1.0 )),
        np .float64 (float (self .dy_air if self .dy_air >0.0 else 1.0 )),
        np .float64 (float (self .dz_air if self .dz_air >0.0 else 1.0 )),
        self ._buf_air_force_external ,
        )
        cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_air_force_from_p ,(self ._global_size ,),(self ._local_size ,))
        if self .kernel_debug :
            try :
                cl .enqueue_copy (self .queue ,self ._air_force_external ,self ._buf_air_force_external )
                self .queue .finish ()
                f3 =self ._air_force_external .reshape (self .n_elements ,self .dof_per_element )[:,:3 ]
                fmag =np .sqrt (np .sum (f3 *f3 ,axis =1 ))
                allowed =np .asarray (self ._fe_air_coupling_mask [:self .n_elements ]!=0 ,dtype =bool )
                leak =(~allowed )&(fmag >0.0 )
                if np .any (leak ):
                    idx =np .flatnonzero (leak )[:8 ]
                    print (
                        "[air][debug] air->FE force leak to non-coupled elements: "
                        f"count={int (np .sum (leak ))}, sample={idx .tolist ()}"
                    )
            except Exception :
                pass

    def _air_wave_step_host (self ,dt :float )->None :
        """Advance air over one FE interval dt with explicit leapfrog substeps when needed.

        CFL uses the anisotropic 7-point bound: c*dt <= 1/sqrt(1/dx^2+1/dy^2+1/dz^2) (times safety).

        Injection is applied **once** with the full FE dt, then only the homogeneous wave update is
        subcycled. Re-injecting every substep was driving the membrane–air interface at the substep
        rate and stacking with Sommerfeld (which uses 1/dt_sub), which made the field look stiffer
        and more “jelly-like/dense”."""
        if self .n_air_cells <=0 :
            return
        self ._air_trace_step_counter +=1
        self ._air_last_fe_dt =float (dt )
        n =self .n_air_cells
        dx =float (self .dx_air if self .dx_air >0.0 else 1.0 )
        dy =float (self .dy_air if self .dy_air >0.0 else 1.0 )
        dz =float (self .dz_air if self .dz_air >0.0 else 1.0 )
        c_sound =float (max (self .air_sound_speed ,1e-12 ))
        _eps_h =1e-30
        inv_h2 =1.0 /(dx *dx +_eps_h )+1.0 /(dy *dy +_eps_h )+1.0 /(dz *dz +_eps_h )
        safety =0.92
        if inv_h2 >1e-60 :
            dt_ac_max =safety /(c_sound *np .sqrt (inv_h2 ))
            n_sub =max (1 ,int (np .ceil (dt /dt_ac_max )))
        else :
            n_sub =1
        _AIR_SUB_CAP =65536
        if n_sub >_AIR_SUB_CAP :
            if not self ._air_subcycle_cap_warned :
                print (
                f"[air] acoustic substep count capped at {_AIR_SUB_CAP } (would need {n_sub }); "
                "wave CFL may still be violated — coarsen FE dt or refine air_grid_step_mm."
                )
                self ._air_subcycle_cap_warned =True
            n_sub =_AIR_SUB_CAP
        dt_sub =dt /float (n_sub )
        self ._air_last_dt_sub =float (dt_sub )
        self ._air_last_acoustic_dt =dt_sub
        if not self ._air_acoustic_summary_shown :
            nu_fe =float (c_sound *dt *np .sqrt (inv_h2 ))
            spc =1.0 /max (nu_fe ,1e-30 )
            print (
            f"[air] acoustic (once): FE dt={dt :.3e} s, voxel (dx,dy,dz)=({dx :.3e},{dy :.3e},{dz :.3e}) m, "
            f"nu=c*dt*sqrt(1/dx^2+..)={nu_fe :.4g}, leapfrog substeps/run={n_sub }. "
            f"~{spc :.0f} FE steps for sound to cross ~one cell. "
            f"If nu<<1, subcycling is off and jelly-like maps are usually not a CFL violation "
            f"(near-field dipole coupling, small domain vs wavelength, or strong air<->FE feedback)."
            )
            self ._air_acoustic_summary_shown =True
        if n_sub >1 and not self ._air_subcycle_note_shown :
            nu_eff =c_sound *dt_sub *np .sqrt (inv_h2 )
            print (
            f"[air] subcycling: {n_sub } homogeneous wave substeps per FE step "
            f"(dt_sub={dt_sub :.3e} s, c*dt_sub*sqrt(1/dx^2+..)≈{nu_eff :.3f}, target≤{safety :.3f}); "
            "inject per acoustic substep."
            )
            self ._air_subcycle_note_shown =True
        gs =int (getattr (self ,'_air_global_size',n ))
        if gs <n :
            gs =((n +self ._local_size -1 )//self ._local_size )*self ._local_size
        self ._validate_air_inject_csr ("air_inject_reduce")
        self ._kernel_air_inject_reduce .set_args (
        self ._buf_air_inject_csr_offsets ,
        self ._buf_air_inject_csr_indices ,
        self ._buf_air_inject_csr_signs ,
        self ._buf_velocity ,
        self ._buf_neighbors ,
        self ._buf_boundary ,
        self ._buf_element_size ,
        self ._buf_material_index ,
        self ._buf_material_props ,
        np .int32 (self .material_props .shape [0 ]),
        np .int32 (n ),
        np .float64 (float (self .rho_air )),
        np .float64 (c_sound ),
        np .float64 (dt_sub ),
        np .float64 (dx ),
        np .float64 (dy ),
        np .float64 (dz ),
        self ._buf_air_plus ,
        )
        cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_air_inject_reduce ,(gs ,),(self ._local_size ,))
        if not hasattr (self ,'_air_zero_inject_np')or self ._air_zero_inject_np .size !=n :
            self ._air_zero_inject_np =np .zeros (n ,dtype =np .float64 )
        z_inj =self ._air_zero_inject_np 
        buf_prev =self ._buf_air_prev
        buf_curr =self ._buf_air_curr
        buf_next =self ._buf_air_next
        for k in range (n_sub ):
            use_first_order =(
                self .air_solver_mode =="first_order"
                and self ._kernel_air_u is not None
                and self ._kernel_air_p is not None
            )
            if use_first_order :
                self ._kernel_air_u .set_args (
                buf_curr ,
                self ._buf_air_ux ,
                self ._buf_air_uy ,
                self ._buf_air_uz ,
                self ._buf_air_neighbors ,
                self ._buf_air_absorb ,
                np .int32 (n ),
                np .float64 (float (self .rho_air )),
                np .float64 (c_sound ),
                np .float64 (dx ),
                np .float64 (dy ),
                np .float64 (dz ),
                np .float64 (dt_sub ),
                )
                cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_air_u ,(gs ,),(self ._local_size ,))
                self ._kernel_air_p .set_args (
                buf_curr ,
                self ._buf_air_ux ,
                self ._buf_air_uy ,
                self ._buf_air_uz ,
                self ._buf_air_plus ,
                self ._buf_air_neighbors ,
                self ._buf_air_absorb ,
                np .int32 (n ),
                np .float64 (float (self .rho_air )),
                np .float64 (c_sound ),
                np .float64 (dx ),
                np .float64 (dy ),
                np .float64 (dz ),
                np .float64 (dt_sub ),
                )
                cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_air_p ,(gs ,),(self ._local_size ,))
            else :
                kernel_2 =(
                    self ._kernel_air_acoustic_second_order
                    if self .air_solver_mode =="second_order"and self ._kernel_air_acoustic_second_order is not None
                    else self ._kernel_air_acoustic
                )
                kernel_2 .set_args (
                buf_prev ,
                buf_curr ,
                self ._buf_air_plus ,
                buf_next ,
                self ._buf_air_neighbors ,
                self ._buf_air_absorb ,
                np .int32 (n ),
                np .float64 (dx ),
                np .float64 (dy ),
                np .float64 (dz ),
                np .float64 (c_sound ),
                np .float64 (dt_sub ),
                )
                cl .enqueue_nd_range_kernel (self .queue ,kernel_2 ,(gs ,),(self ._local_size ,))
                buf_prev ,buf_curr ,buf_next =buf_curr ,buf_next ,buf_prev
            # Keep injection applied at every acoustic substep so the forcing is time-consistent
            # with the wave update step size.
        self ._buf_air_prev ,self ._buf_air_curr ,self ._buf_air_next =buf_prev ,buf_curr ,buf_next
        self ._compute_air_force_from_pressure_buffer (self ._buf_air_curr )
        cl .enqueue_copy (self .queue ,self .air_pressure_curr ,self ._buf_air_curr )
        if self .kernel_debug :
            try :
                p_n =np .empty_like (self .air_pressure_curr )
                p_next =self .air_pressure_curr .copy ()
                p_plus =np .empty_like (self .air_pressure_curr )
                cl .enqueue_copy (self .queue ,p_n ,self ._buf_air_prev )
                cl .enqueue_copy (self .queue ,p_plus ,self ._buf_air_plus )
                do_log =(self ._air_trace_step_counter <=10 )or (self ._air_trace_step_counter %50 ==0 )
                if do_log :
                    print (
                        f"[air][trace] field step={self ._air_trace_step_counter}: "
                        f"max|p_n|={float (np .max (np .abs (p_n ))):.3e} Pa, "
                        f"max|inject_dp|={float (np .max (np .abs (p_plus ))):.3e} Pa, "
                        f"max|p_next|={float (np .max (np .abs (p_next ))):.3e} Pa"
                    )
            except Exception :
                pass

    def _air_one_step_coupling_sources (self ,dt :float )->None :
        'One acoustic time step: inject from FE normal velocity, then leapfrog wave update; air force uses ∇p on the structure.'
        if self .n_air_cells <=0 :
            self ._air_force_external .fill (0.0 )
            return
        self ._air_wave_step_host (dt )
        self ._air_debug_diagnostics (dt ,"air_one_step")

    def _run_air_coupling (self ,dt :float ,pressure_pa :float |np .ndarray )->None :
        _ =pressure_pa
        if self .n_air_cells <=0 :
            self ._air_force_external .fill (0.0 )
            cl .enqueue_copy (self .queue ,self ._buf_air_force_external ,self ._air_force_external )
            return
        self ._update_air_coupling_geometry_from_motion ()
        self ._air_one_step_coupling_sources (dt )
        cl .enqueue_copy (self .queue ,self ._air_force_external ,self ._buf_air_force_external )
        self .queue .finish ()

    def _sync_fe_state_from_gpu_for_air_maps (self )->None :
        'RK4 intermediate stages live on the GPU; KE↔air maps are built on the host using self.position/velocity.'
        cl .enqueue_copy (self .queue ,self .position ,self ._buf_position )
        cl .enqueue_copy (self .queue ,self .velocity ,self ._buf_velocity )
        self .queue .finish ()

    def _run_air_coupling_for_acceleration (
        self ,
        dt :float ,
        *,
        sync_fe_from_gpu :bool =True ,
    )->None :
        """Air forces for RK4 acceleration from current pressure field p(t_n).

        When sync_fe_from_gpu is False, uses host self.position/self.velocity (caller must
        match the GPU buffers). This avoids four GPU→host FE pulls per RK4 step."""
        if self .n_air_cells <=0 :
            self ._air_force_external .fill (0.0 )
            cl .enqueue_copy (self .queue ,self ._buf_air_force_external ,self ._air_force_external )
            return
        if self .air_pressure_curr .size !=self .n_air_cells :
            p0 =float (self .air_initial_uniform_pressure_pa )
            self .air_pressure_curr =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
        if sync_fe_from_gpu :
            self ._sync_fe_state_from_gpu_for_air_maps ()
        self ._update_air_coupling_geometry_from_motion ()
        self ._compute_air_force_from_pressure_buffer (self ._buf_air_curr )
        cl .enqueue_copy (self .queue ,self ._air_force_external ,self ._buf_air_force_external )
        self ._air_debug_diagnostics (dt ,"rk4_acc")

    def _evaluate_acceleration (
        self ,
        buf_params ,
        dt :float ,
        pressure_pa :float ,
        out_buf ,
        *,
        validate_finite :bool =False ,
        refresh_air_coupling :bool =True ,
        acc_stage_id :int =0 ,
    )->None :
        if refresh_air_coupling :
            self ._run_air_coupling_for_acceleration (dt )

        self ._kernel_acc .set_args (
        self ._buf_position ,
        self ._buf_velocity ,
        self ._buf_force_external ,
        np .float64 (pressure_pa ),
        self ._buf_force_drive_mask ,
        self ._buf_force_drive_axis ,
        self ._buf_force_drive_area ,
        self ._buf_air_force_external ,
        self ._buf_boundary ,
        self ._buf_element_size ,
        self ._buf_material_index ,
        self ._buf_material_props ,
        self ._buf_neighbors ,
        self ._buf_laws ,
        np .int32 (self .material_props .shape [0 ]),
        buf_params ,
        out_buf ,
        self ._buf_first_bad ,
        self ._buf_first_bad_meta ,
        self ._buf_first_bad_neighbor_elem ,
        self ._buf_first_bad_interface_dir ,
        np .int32 (acc_stage_id ),
        np .int32 (1 if validate_finite else 0 ),
        )
        cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_acc ,(self ._global_size ,),(self ._local_size ,))

    def _update_center_index (self )->None :
        'Updates the central CE index (for history and diagnostics).\n\n        Priority:\n        1) among the FE material MAT_SENSOR;\n        2) inside one Z-layer (closest to the median Z of sensory FE);\n        3) closest to the center of the layer in the XY plane.\n        If there are no sensory CEs, fallback to the geometric center of all CEs.'
        xyz =self .position .reshape (self .n_elements ,self .dof_per_element )[:,:3 ]
        sensor_idx =np .flatnonzero (self ._sensor_mask )
        if sensor_idx .size >0 :
            z_sensor =xyz [sensor_idx ,2 ]
            z_med =float (np .median (z_sensor ))
            z_layer =float (z_sensor [int (np .argmin (np .abs (z_sensor -z_med )))])
            in_layer_mask =np .isclose (z_sensor ,z_layer ,atol =1e-12 ,rtol =1e-9 )
            layer_idx =sensor_idx [in_layer_mask ]
            if layer_idx .size ==0 :
                layer_idx =sensor_idx 
            xy_layer =xyz [layer_idx ,:2 ]
            xy_center =np .mean (xy_layer ,axis =0 )
            d2 =np .sum ((xy_layer -xy_center )**2 ,axis =1 )
            center_idx =int (layer_idx [int (np .argmin (d2 ))])
        else :
            center =np .mean (xyz ,axis =0 )
            center_idx =int (np .argmin (np .sum ((xyz -center )**2 ,axis =1 )))
        self .center_idx =center_idx 
        self .center_dof =self .center_idx *self .dof_per_element +2 

    def _sync_visualization_flag (self )->None :
        'Disables rendering if the topology is not 2D rectangular.'
        self .visualization_enabled =bool (
        self ._topology_is_rect_grid 
        and self ._visual_shape is not None 
        and self ._visual_element_indices is not None 
        and self ._visual_element_indices .size ==(self ._visual_shape [0 ]*self ._visual_shape [1 ])
        )

    def _allocate_air_buffers (self )->None :
        '(Re)creates air buffers and FE <-> air-grid communication maps.'
        mf =cl .mem_flags 
        self ._buf_air_prev =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_air_cells *8 )
        self ._buf_air_curr =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_air_cells *8 )
        self ._buf_air_next =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_air_cells *8 )
        # Particle velocity (first-order solver); allocated unconditionally for simplicity.
        self ._buf_air_ux =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_air_cells *8 )
        self ._buf_air_uy =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_air_cells *8 )
        self ._buf_air_uz =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_air_cells *8 )
        self ._buf_air_plus =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_air_cells *8 )
        self ._buf_air_neighbors =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_air_cells *FACE_DIRS *4 )
        self ._buf_air_absorb =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .n_air_cells *FACE_DIRS )
        cap_csr =max (1 ,2 *self .n_elements )
        self ._air_inject_csr_offsets =np .zeros (self .n_air_cells +1 ,dtype =np .int32 )
        self ._air_inject_csr_indices =np .zeros (cap_csr ,dtype =np .int32 )
        self ._air_inject_csr_signs =np .ones (cap_csr ,dtype =np .float64 )
        self ._buf_air_inject_csr_offsets =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =(self .n_air_cells +1 )*4 )
        self ._buf_air_inject_csr_indices =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =cap_csr *4 )
        self ._buf_air_inject_csr_signs =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =cap_csr *8 )
        self ._buf_air_force_external =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =self .n_dof *8 )# [n_elements * 6] acoustic pressure forces
        self ._air_force_external =np .zeros (self .n_dof ,dtype =np .float64 )# Separate acoustic force per DOF
        # Create air mapping buffers
        try :
            air_map_size =self .air_map_6 .size *self .air_map_6 .itemsize 
            air_elem_map_size =self .air_elem_map .size *self .air_elem_map .itemsize 
            air_face_size =self .air_elem_face_area .size *self .air_elem_face_area .itemsize 
            air_volume_size =self .air_elem_volume .size *self .air_elem_volume .itemsize 
            self ._buf_air_map_6 =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =air_map_size )# [n_elements, 6] int32
            self ._buf_air_elem_map =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =air_elem_map_size )# [n_elements] int32
            self ._buf_air_elem_face_area =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =air_face_size )# [n_elements, 3] float64
            self ._buf_air_elem_volume =cl .Buffer (self .ctx ,mf .READ_WRITE ,size =air_volume_size )# [n_elements] float64
        except Exception as e :
            raise RuntimeError (f"Failed to create air mapping buffers: {e }. Sizes: map={air_map_size }, elem_map={air_elem_map_size }, face={air_face_size }, volume={air_volume_size }")from e 
        cl .enqueue_copy (self .queue ,self ._buf_air_map_6 ,self .air_map_6 )
        cl .enqueue_copy (self .queue ,self ._buf_air_elem_map ,self .air_elem_map )
        cl .enqueue_copy (self .queue ,self ._buf_air_elem_face_area ,self .air_elem_face_area )
        cl .enqueue_copy (self .queue ,self ._buf_air_elem_volume ,self .air_elem_volume )
        p0 =float (self .air_initial_uniform_pressure_pa )
        self .air_pressure_prev =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
        self .air_pressure_curr =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
        self .air_pressure_next =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
        cl .enqueue_copy (self .queue ,self ._buf_air_prev ,self .air_pressure_prev )
        cl .enqueue_copy (self .queue ,self ._buf_air_curr ,self .air_pressure_curr )
        cl .enqueue_copy (self .queue ,self ._buf_air_next ,self .air_pressure_next )
        z_u =np .zeros (self .n_air_cells ,dtype =np .float64 )
        cl .enqueue_copy (self .queue ,self ._buf_air_ux ,z_u )
        cl .enqueue_copy (self .queue ,self ._buf_air_uy ,z_u )
        cl .enqueue_copy (self .queue ,self ._buf_air_uz ,z_u )
        z_inj =np .zeros (self .n_air_cells ,dtype =np .float64 )
        cl .enqueue_copy (self .queue ,self ._buf_air_plus ,z_inj )
        cl .enqueue_copy (self .queue ,self ._buf_air_neighbors ,np .ascontiguousarray (self .air_neighbors ))
        cl .enqueue_copy (self .queue ,self ._buf_air_absorb ,np .ascontiguousarray (self .air_neighbor_absorb_u8 ))
        self ._rebuild_air_inject_csr ()
        self ._upload_air_inject_csr_buffers ()
        self ._air_force_external .fill (0.0 )
        cl .enqueue_copy (self .queue ,self ._buf_air_force_external ,self ._air_force_external )
        self .queue .finish ()

    def _inject_csr_use_bilateral_for_element (self ,e :int )->bool :
        """Bilateral FE->air coupling is used only for membrane/sensor elements."""
        if e <0 or e >=self .n_elements :
            return False
        if hasattr (self ,'membrane_mask')and self .membrane_mask .size >e and int (self .membrane_mask [e ])!=0 :
            return True
        if hasattr (self ,'_sensor_mask')and self ._sensor_mask .size >e and bool (self ._sensor_mask [e ]):
            return True
        return False

    def _rebuild_air_inject_csr (self )->None :
        """CSR rows = air cells; entries (elem, sign).

        Physically enforced policy:
        - Only **membrane/sensor** FEs are allowed to inject into air.
        - They use bilateral ± air voxels from topology (`air_inject_cell_plus/minus`)
          so both sides couple to air (dipole-like source).
        - All other solid materials are excluded from FE->air injection CSR."""
        n_cells =int (self .n_air_cells )
        n_el =int (self .n_elements )
        cap_csr =max (1 ,2 *n_el )
        if n_cells <=0 :
            self ._air_inject_csr_offsets =np .zeros (1 ,dtype =np .int32 )
            self ._air_inject_csr_indices =np .zeros (1 ,dtype =np .int32 )
            self ._air_inject_csr_signs =np .zeros (1 ,dtype =np .float64 )
            return
        pairs =[[]for _ in range (n_cells )]
        ip =np .asarray (self .air_inject_cell_plus ,dtype =np .int32 ).ravel ()
        im =np .asarray (self .air_inject_cell_minus ,dtype =np .int32 ).ravel ()
        bad_bilateral =[]
        for e in range (n_el ):
            if not self ._inject_csr_use_bilateral_for_element (e ):
                continue
            if e <self .boundary_mask_elements .size and int (self .boundary_mask_elements [e ])!=0 :
                continue
            c_plus =int (ip [e ])if e <ip .size else -1 
            c_minus =int (im [e ])if e <im .size else -1 
            if c_plus <0 or c_minus <0 or c_plus ==c_minus :
                if len (bad_bilateral )<24 :
                    bad_bilateral .append ((int (e ),int (c_plus ),int (c_minus )))
                continue
            if 0 <=c_plus <n_cells :
                pairs [c_plus ].append ((e ,1.0 ))
            if 0 <=c_minus <n_cells :
                pairs [c_minus ].append ((e ,-1.0 ))
        if bad_bilateral :
            sample =", ".join (f"(e={e }, cp={cp }, cm={cm })" for (e ,cp ,cm )in bad_bilateral [:12 ])
            raise RuntimeError (
                "[air] invalid membrane/sensor bilateral mapping: each FE must map to two distinct side air cells. "
                f"bad_count={len (bad_bilateral )} sample={sample }"
            )
        for c in range (n_cells ):
            pairs [c ].sort (key =lambda t :t [0 ])
        offsets =np .zeros (n_cells +1 ,dtype =np .int32 )
        for c in range (n_cells ):
            offsets [c +1 ]=offsets [c ]+len (pairs [c ])
        nnz =int (offsets [-1 ])
        if nnz >cap_csr :
            raise RuntimeError (
            f"air inject CSR nnz={nnz } exceeds buffer cap {cap_csr }; increase n_elements scaling in buffers."
            )
        idx =np .zeros (cap_csr ,dtype =np .int32 )
        sgn =np .zeros (cap_csr ,dtype =np .float64 )
        k =0 
        for c in range (n_cells ):
            for (elem ,s )in pairs [c ]:
                idx [k ]=elem 
                sgn [k ]=s 
                k +=1 
        self ._air_inject_csr_offsets =offsets 
        self ._air_inject_csr_indices =idx 
        self ._air_inject_csr_signs =sgn 
        self ._validate_air_inject_csr ("_rebuild_air_inject_csr")
        self ._air_csr_dirty =True

    def _validate_air_inject_csr (self ,where :str ,full_scan :bool =True )->None :
        """Raise RuntimeError if host-side air injection CSR is inconsistent or unsafe to launch.

        Catches: wrong offset length, out-of-range element indices, buffer overrun, non-monotone
        rows, duplicate (cell,elem) pairs, duplicate global elem (broken old bilateral CSR), or
        absurd row width."""
        if not getattr (self ,'air_inject_csr_validate',True ):
            return
        n_cells =int (self .n_air_cells )
        n_el =int (self .n_elements )
        if n_cells <=0 :
            return
        cap_csr =max (1 ,2 *n_el )
        off =getattr (self ,'_air_inject_csr_offsets',None )
        idx =getattr (self ,'_air_inject_csr_indices',None )
        sgn =getattr (self ,'_air_inject_csr_signs',None )
        if off is None or idx is None or sgn is None :
            raise RuntimeError (f"[air] CSR buffers missing ({where })")
        off =np .asarray (off ,dtype =np .int64 ).ravel ()
        idx =np .asarray (idx ,dtype =np .int64 ).ravel ()
        sgn =np .asarray (sgn ,dtype =np .float64 ).ravel ()
        if off .size !=n_cells +1 :
            raise RuntimeError (
            f"[air] CSR offsets length {off .size } != n_air_cells+1 ({n_cells +1 }) ({where })"
            )
        if int (off [0 ])!=0 :
            raise RuntimeError (f"[air] CSR offsets[0] must be 0, got {off [0 ]} ({where })")
        nnz =int (off [-1 ])
        if nnz <0 :
            raise RuntimeError (f"[air] CSR nnz negative: {nnz } ({where })")
        if nnz >cap_csr :
            raise RuntimeError (
            f"[air] CSR nnz={nnz } exceeds allocation cap {cap_csr } (reallocate air buffers) ({where })"
            )
        if nnz >idx .size or nnz >sgn .size :
            raise RuntimeError (
            f"[air] CSR nnz={nnz } exceeds indices/signs length idx={idx .size } sgn={sgn .size } ({where })"
            )
        d =np .diff (off )
        if np .any (d <0 ):
            raise RuntimeError (f"[air] CSR offsets not non-decreasing ({where })")
        max_row =int (np .max (d )) if d .size >0 else 0
        lim =int (getattr (self ,'air_inject_csr_max_nnz_per_row',16384 ))
        if max_row >lim :
            raise RuntimeError (
            f"[air] CSR max row length {max_row } exceeds limit {lim } (stale bilateral / bad topology). "
            f"Rebuild from air_elem_map only or raise air_inject_csr_max_nnz_per_row . ({where })"
            )
        packed =idx [:nnz ]
        if np .any (packed <0 )or np .any (packed >=n_el ):
            bad =np .flatnonzero ((packed <0 )|(packed >=n_el ))[:12 ]
            raise RuntimeError (
            f"[air] CSR element indices out of [0,{n_el }): pos {bad .tolist ()} "
            f"vals {packed [bad ].tolist ()} ({where })"
            )
        if not np .all (np .isfinite (sgn [:nnz ])):
            raise RuntimeError (f"[air] CSR signs non-finite ({where })")
        if not full_scan :
            return
        # Expensive integrity checks; run periodically instead of every step.
        for c in range (n_cells ):
            s ,e =int (off [c ]),int (off [c +1 ])
            if s >e :
                raise RuntimeError (f"[air] CSR air cell {c } invalid range [{s },{e }) ({where })")
            row =packed [s :e ]
            if row .size >1 and row .size !=np .unique (row ).size :
                raise RuntimeError (
                f"[air] CSR duplicate element index in air cell {c } row [{s },{e }) ({where })"
                )
        if nnz >0 :
            bc =np .bincount (packed .astype (np .int64 ,copy =False ),minlength =n_el )
            if np .any (bc >2 ):
                dup =np .flatnonzero (bc >2 )[:24 ]
                raise RuntimeError (
                f"[air] CSR element index appears >2 times: {dup .tolist ()} ({where })"
                )

    def _upload_air_inject_csr_buffers (self )->None :
        cl .enqueue_copy (self .queue ,self ._buf_air_inject_csr_offsets ,self ._air_inject_csr_offsets )
        cl .enqueue_copy (self .queue ,self ._buf_air_inject_csr_indices ,self ._air_inject_csr_indices )
        cl .enqueue_copy (self .queue ,self ._buf_air_inject_csr_signs ,self ._air_inject_csr_signs )

    def _log_air_csr_hot_rows (self ,where :str ,top_k :int =8 )->None :
        """One-shot CSR row density trace to catch mapping collapse early."""
        try :
            if int (getattr (self ,"n_air_cells",0 ))<=0 :
                return
            off =np .asarray (getattr (self ,"_air_inject_csr_offsets",[]),dtype =np .int64 ).ravel ()
            if off .size !=int (self .n_air_cells )+1 :
                return
            rows =np .diff (off )
            if rows .size ==0 :
                return
            max_row =int (np .max (rows ))
            nz_rows =rows [rows >0 ]
            mean_nz =float (np .mean (nz_rows )) if nz_rows .size >0 else 0.0
            idx =np .argsort (rows )[::-1 ][:max (1 ,int (top_k ))]
            top_txt =", ".join (f"{int (i )}:{int (rows [i ])}" for i in idx if int (rows [i ])>0 )
            print (
                f"[air][csr] {where }: max_row={max_row} mean_nonzero_row={mean_nz:.2f} "
                f"top_rows={top_txt}"
            )
        except Exception :
            pass

    def generate_planar_membrane_topology (
    self ,
    plane :str ,
    thickness_m :float ,
    size_u_m :float ,
    size_v_m :float ,
    )->dict [str ,np .ndarray ]:
        'Generation of a single-layer flat membrane on a regular nx x ny grid.\n\n        plane:\n            Construction plane: "xy", "xz" or "yz".\n        thickness_m:\n            Single layer thickness (m).\n        size_u_m, size_v_m:\n            Dimensions along two axes of the selected plane (m).\n\n        Returns a dictionary:\n        - element_position_xyz: [n_elements, 3]\n        - element_size_xyz: [n_elements, 3]\n        - neighbors: [n_elements, FACE_DIRS]\n        - boundary_mask_elements: [n_elements]'
        pl =str (plane ).strip ().lower ()
        if pl =="xy":
            axis_u ,axis_v ,axis_n =0 ,1 ,2 
        elif pl =="xz":
            axis_u ,axis_v ,axis_n =0 ,2 ,1 
        elif pl =="yz":
            axis_u ,axis_v ,axis_n =1 ,2 ,0 
        else :
            raise ValueError ("plane must be one of: 'xy', 'xz', 'yz'")
        if thickness_m <=0.0 :
            raise ValueError ('thickness_m must be > 0')
        if size_u_m <=0.0 or size_v_m <=0.0 :
            raise ValueError ('size_u_m and size_v_m must be > 0')

        du =float (size_u_m )/float (self .nx )
        dv =float (size_v_m )/float (self .ny )
        n =self .n_elements 
        pos =np .zeros ((n ,3 ),dtype =np .float64 )
        size =np .zeros ((n ,3 ),dtype =np .float64 )
        neighbors =np .full ((n ,FACE_DIRS ),-1 ,dtype =np .int32 )
        boundary =np .zeros (n ,dtype =np .int32 )

        for elem in range (self .n_membrane_elements ):
            ix =elem %self .nx 
            iy =elem //self .nx 
            pos [elem ,axis_u ]=(ix +0.5 )*du 
            pos [elem ,axis_v ]=(iy +0.5 )*dv 
            pos [elem ,axis_n ]=0.0 
            size [elem ,axis_u ]=du 
            size [elem ,axis_v ]=dv 
            size [elem ,axis_n ]=float (thickness_m )
            if ix +1 <self .nx :
                neighbors [elem ,0 ]=elem +1 
            if ix -1 >=0 :
                neighbors [elem ,1 ]=elem -1 
            if iy +1 <self .ny :
                neighbors [elem ,2 ]=elem +self .nx 
            if iy -1 >=0 :
                neighbors [elem ,3 ]=elem -self .nx 
            if ix ==0 or ix ==self .nx -1 or iy ==0 or iy ==self .ny -1 :
                boundary [elem ]=1 

        return {
        "element_position_xyz":pos ,
        "element_size_xyz":size ,
        "neighbors":neighbors ,
        "boundary_mask_elements":boundary ,
        }

    def _set_rest_position (self )->None :
        'Reset the position to the base flat diaphragm in the XY plane.'
        topo =self .generate_planar_membrane_topology (
        plane ="xy",
        thickness_m =self .thickness ,
        size_u_m =self .width ,
        size_v_m =self .height ,
        )
        self .position [0 ::self .dof_per_element ]=topo ["element_position_xyz"][:,0 ]
        self .position [1 ::self .dof_per_element ]=topo ["element_position_xyz"][:,1 ]
        self .position [2 ::self .dof_per_element ]=topo ["element_position_xyz"][:,2 ]

    def _build_neighbors_topology (self )->np .ndarray :
        'Universal connection topology [n_elements, FACE_DIRS].\n        Direction order: +X, -X, +Y, -Y, +Z, -Z.\n        Air FEs are disabled: only the membrane XY mesh is built.'
        return self .generate_planar_membrane_topology (
        plane ="xy",
        thickness_m =self .thickness ,
        size_u_m =self .width ,
        size_v_m =self .height ,
        )["neighbors"]

    def _build_default_material_library (self )->np .ndarray :
        'Built-in material library\n        [density, E_parallel, E_perp, poisson, Cd, eta_visc, acoustic_impedance, acoustic_inject].\n        The values \u200b\u200bfor foam/leather/ear are approximate and are suitable as starting FE parameters.'
        # Memory foam (VE3015, benchmark for viscoelastic PU-foam ranges).
        foam_density =55.0 
        foam_E_parallel =0.08e6 
        foam_E_perp =0.05e6 
        foam_poisson =0.30 
        foam_Cd =1.20 
        foam_eta_visc =150.0 
        foam_impedance =_impedance_from_density_E (foam_density ,foam_E_parallel )

        # Sheep leather (guideline for FE leather models: E ~ 10 MPa, rho ~ 998 kg/m^3).
        leather_density =998.0 
        leather_E_parallel =10.0e6 
        leather_E_perp =7.0e6 
        leather_poisson =0.40 
        leather_Cd =1.05 
        leather_eta_visc =12.0 
        leather_impedance =_impedance_from_density_E (leather_density ,leather_E_parallel )

        # Human ear (averaged, without tissue separation; reference point by auricular cartilage E ~ 1.4..2.1 MPa).
        ear_density =1080.0 
        ear_E_parallel =1.80e6 
        ear_E_perp =1.50e6 
        ear_poisson =0.45 
        ear_Cd =1.10 
        ear_eta_visc =20.0 
        ear_impedance =_impedance_from_density_E (ear_density ,ear_E_parallel )

        # Sensor (microphone): temporarily use the parameters of the PET film as for the membrane.
        sensor_density =self .density 
        sensor_E_parallel =self .E_parallel 
        sensor_E_perp =self .E_perp 
        sensor_poisson =self .poisson 
        sensor_Cd =self .Cd 
        sensor_eta_visc =0.8 
        membrane_eta_visc =0.8 
        sensor_impedance =_impedance_from_density_E (sensor_density ,sensor_E_parallel )
        membrane_impedance =_impedance_from_density_E (self .density ,self .E_parallel )

        # Cotton wool (approximate effective parameters for soft porous filler).
        cotton_density =250.0 
        cotton_E_parallel =0.03e6 
        cotton_E_perp =0.02e6 
        cotton_poisson =0.20 
        cotton_Cd =1.35 
        cotton_eta_visc =220.0 
        cotton_impedance =_impedance_from_density_E (cotton_density ,cotton_E_parallel )

        return np .array (
        [
        [self .density ,self .E_parallel ,self .E_perp ,self .poisson ,self .Cd ,membrane_eta_visc ,membrane_impedance ,1.0 ],
        [
        foam_density ,
        foam_E_parallel ,
        foam_E_perp ,
        foam_poisson ,
        foam_Cd ,
        foam_eta_visc ,
        foam_impedance ,
        _default_acoustic_inject_from_legacy_row (int (MAT_FOAM_VE3015 )),
        ],
        [
        leather_density ,
        leather_E_parallel ,
        leather_E_perp ,
        leather_poisson ,
        leather_Cd ,
        leather_eta_visc ,
        leather_impedance ,
        _default_acoustic_inject_from_legacy_row (int (MAT_SHEEPSKIN_LEATHER )),
        ],
        [
        ear_density ,
        ear_E_parallel ,
        ear_E_perp ,
        ear_poisson ,
        ear_Cd ,
        ear_eta_visc ,
        ear_impedance ,
        _default_acoustic_inject_from_legacy_row (int (MAT_HUMAN_EAR_AVG )),
        ],
        [sensor_density ,sensor_E_parallel ,sensor_E_perp ,sensor_poisson ,sensor_Cd ,sensor_eta_visc ,sensor_impedance ,0.0 ],
        [
        cotton_density ,
        cotton_E_parallel ,
        cotton_E_perp ,
        cotton_poisson ,
        cotton_Cd ,
        cotton_eta_visc ,
        cotton_impedance ,
        _default_acoustic_inject_from_legacy_row (int (MAT_COTTON_WOOL )),
        ],
        ],
        dtype =np .float64 ,
        )

    def _update_sensor_mask (self )->None :
        self ._sensor_mask =(self .material_index ==MAT_SENSOR )
        self ._air_history_iy_cached =None

    def _configure_air_field_grid (
    self ,
    position_xyz :np .ndarray |None =None ,
    size_xyz :np .ndarray |None =None ,
    )->None :
        'Separate 3D acoustic field:\n        - boundaries are calculated based on the extreme coordinates of all FEs (position ± half-size);\n        - air_padding_mm padding is added.\n        position_xyz, size_xyz: if specified, used instead of self.position/self.element_size_xyz\n        (for the correct extent when rebuilding from set_custom_topology).'
        if self .air_grid_step is not None :
            self .dx_air =float (self .air_grid_step )
            self .dy_air =float (self .air_grid_step )
            self .dz_air =float (self .air_grid_step )
        else :
            sz =size_xyz if size_xyz is not None else self .element_size_xyz [:self .n_elements ]
            self .dx_air =float (np .mean (sz [:,0 ]))
            self .dy_air =float (np .mean (sz [:,1 ]))
            dz_geom =float (np .mean (sz [:,2 ]))
            if dz_geom <=0.0 :
                dz_geom =float (min (self .dx_air ,self .dy_air ))
            self .dz_air =dz_geom
        self ._air_history_iy_cached =None
        pad =float (self .air_padding )if self .air_padding is not None else 0.002 # 2 mm default

        # Use all FE on all axes for extent (position + half-size per element).
        if position_xyz is not None and size_xyz is not None :
            pos_xyz =np .asarray (position_xyz ,dtype =np .float64 )
            size_xyz =np .asarray (size_xyz ,dtype =np .float64 )
        else :
            pos_xyz =self .position [:self .n_elements *self .dof_per_element ].reshape (
            self .n_elements ,self .dof_per_element 
            )[:,:3 ]
            size_xyz =self .element_size_xyz [:self .n_elements ]
        x_min_elem =float (np .min (pos_xyz [:,0 ]-0.5 *size_xyz [:,0 ]))
        x_max_elem =float (np .max (pos_xyz [:,0 ]+0.5 *size_xyz [:,0 ]))
        y_min_elem =float (np .min (pos_xyz [:,1 ]-0.5 *size_xyz [:,1 ]))
        y_max_elem =float (np .max (pos_xyz [:,1 ]+0.5 *size_xyz [:,1 ]))
        z_min_elem =float (np .min (pos_xyz [:,2 ]-0.5 *size_xyz [:,2 ]))
        z_max_elem =float (np .max (pos_xyz [:,2 ]+0.5 *size_xyz [:,2 ]))
        z_plane =float (np .mean (pos_xyz [:,2 ]))

        self .air_origin_x =x_min_elem -pad 
        self .air_origin_y =y_min_elem -pad 
        self .air_origin_z =z_min_elem -pad 
        x_max_air =x_max_elem +pad 
        y_max_air =y_max_elem +pad 
        z_max_air =z_max_elem +pad 

        # Air grid size depends only on spatial extent and grid step (air_grid_step_mm).
        self .nx_air =int (np .ceil ((x_max_air -self .air_origin_x )/self .dx_air ))+1 
        self .ny_air =int (np .ceil ((y_max_air -self .air_origin_y )/self .dy_air ))+1 
        self .nz_air =int (np .ceil ((z_max_air -self .air_origin_z )/self .dz_air ))+1 
        self .n_air_cells =int (self .nx_air *self .ny_air *self .nz_air )
        MAX_AIR_CELLS =2 **31 -1 # int32 max for air_map_6 indices
        if self .n_air_cells >MAX_AIR_CELLS :
            raise ValueError (
            f"Air field too large: {self .n_air_cells } cells (max {MAX_AIR_CELLS }). "
            "Decrease air_grid_step_mm or air_padding_mm."
            )
        self .air_z0 =int (np .round ((z_plane -self .air_origin_z )/self .dz_air ))
        self .air_z0 =int (np .clip (self .air_z0 ,1 ,self .nz_air -2 ))

        ids =np .arange (self .n_air_cells ,dtype =np .int32 )
        nxny =self .nx_air *self .ny_air
        ix =ids %self .nx_air
        iy =(ids //self .nx_air )%self .ny_air
        iz =ids //nxny
        self .air_neighbors =np .full ((self .n_air_cells ,FACE_DIRS ),-1 ,dtype =np .int32 )
        self .air_neighbors [:,0 ]=np .where (ix +1 <self .nx_air ,ids +1 ,-1 )
        self .air_neighbors [:,1 ]=np .where (ix -1 >=0 ,ids -1 ,-1 )
        self .air_neighbors [:,2 ]=np .where (iy +1 <self .ny_air ,ids +self .nx_air ,-1 )
        self .air_neighbors [:,3 ]=np .where (iy -1 >=0 ,ids -self .nx_air ,-1 )
        self .air_neighbors [:,4 ]=np .where (iz +1 <self .nz_air ,ids +nxny ,-1 )
        self .air_neighbors [:,5 ]=np .where (iz -1 >=0 ,ids -nxny ,-1 )
        # Missing-neighbor faces are treated as radiating boundaries (Sommerfeld).
        self .air_neighbor_absorb_u8 =np .where (self .air_neighbors <0 ,np .uint8 (1 ),np .uint8 (0 )).astype (np .uint8 )
        self .air_boundary_mask_elements =np .where (
        (ix ==0 )|(ix ==self .nx_air -1 )|(iy ==0 )|(iy ==self .ny_air -1 )|(iz ==0 )|(iz ==self .nz_air -1 ),
        1 ,0
        ).astype (np .int32 )

        # Full 3D coupling: gradient ∇p and velocity v for all axes.
        # air_map_6: [n, 6] indices for +X,-X,+Y,-Y,+Z,-Z (FACE_DIRS order).
        # air_elem_face_area: [n, 3] face areas perpendicular to X,Y,Z (A_x=sy*sz, A_y=sx*sz, A_z=sx*sy).
        size_arr =self .element_size_xyz [:self .n_elements ]
        self .air_map_6 =np .full ((self .n_elements ,6 ),-1 ,dtype =np .int32 )
        self .air_elem_map =np .full (self .n_elements ,-1 ,dtype =np .int32 )# Center-cell mapping for add_air_pressure_force_to_fe
        self .air_inject_cell_plus =np .full (self .n_elements ,-1 ,dtype =np .int32 )
        self .air_inject_cell_minus =np .full (self .n_elements ,-1 ,dtype =np .int32 )
        self .air_elem_face_area =np .zeros ((self .n_elements ,3 ),dtype =np .float64 )
        self .air_elem_face_area [:,0 ]=size_arr [:,1 ]*size_arr [:,2 ]
        self .air_elem_face_area [:,1 ]=size_arr [:,0 ]*size_arr [:,2 ]
        self .air_elem_face_area [:,2 ]=size_arr [:,0 ]*size_arr [:,1 ]
        np .maximum (self .air_elem_face_area ,1e-18 ,out =self .air_elem_face_area )
        self .air_elem_volume =(size_arr [:,0 ]*size_arr [:,1 ]*size_arr [:,2 ]).astype (np .float64 )

        for elem in range (self .n_elements ):
            base =elem *self .dof_per_element 
            x_e =self .position [base +0 ]
            y_e =self .position [base +1 ]
            z_e =self .position [base +2 ]
            ax =int (np .round ((x_e -self .air_origin_x )/self .dx_air ))
            ay =int (np .round ((y_e -self .air_origin_y )/self .dy_air ))
            az =int (np .round ((z_e -self .air_origin_z )/self .dz_air ))
            ax =int (np .clip (ax ,0 ,self .nx_air -1 ))
            ay =int (np .clip (ay ,0 ,self .ny_air -1 ))
            az =int (np .clip (az ,0 ,self .nz_air -1 ))
            ix_p =min (self .nx_air -1 ,ax +1 )
            ix_m =max (0 ,ax -1 )
            iy_p =min (self .ny_air -1 ,ay +1 )
            iy_m =max (0 ,ay -1 )
            iz_p =min (self .nz_air -1 ,az +1 )
            iz_m =max (0 ,az -1 )
            self .air_map_6 [elem ,0 ]=az *(self .nx_air *self .ny_air )+ay *self .nx_air +ix_p 
            self .air_map_6 [elem ,1 ]=az *(self .nx_air *self .ny_air )+ay *self .nx_air +ix_m 
            self .air_map_6 [elem ,2 ]=az *(self .nx_air *self .ny_air )+iy_p *self .nx_air +ax 
            self .air_map_6 [elem ,3 ]=az *(self .nx_air *self .ny_air )+iy_m *self .nx_air +ax 
            self .air_map_6 [elem ,4 ]=iz_p *(self .nx_air *self .ny_air )+ay *self .nx_air +ax 
            self .air_map_6 [elem ,5 ]=iz_m *(self .nx_air *self .ny_air )+ay *self .nx_air +ax 
            self .air_elem_map [elem ]=az *(self .nx_air *self .ny_air )+ay *self .nx_air +ax 
            na =int (np .argmin (size_arr [elem ]))
            nxny =self .nx_air *self .ny_air 
            if na ==0 :
                self .air_inject_cell_plus [elem ]=az *nxny +ay *self .nx_air +ix_p 
                self .air_inject_cell_minus [elem ]=az *nxny +ay *self .nx_air +ix_m 
            elif na ==1 :
                self .air_inject_cell_plus [elem ]=az *nxny +iy_p *self .nx_air +ax 
                self .air_inject_cell_minus [elem ]=az *nxny +iy_m *self .nx_air +ax 
            else :
                self .air_inject_cell_plus [elem ]=iz_p *nxny +ay *self .nx_air +ax 
                self .air_inject_cell_minus [elem ]=iz_m *nxny +ay *self .nx_air +ax 
        p0 =float (self .air_initial_uniform_pressure_pa )
        self .air_pressure_prev =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
        self .air_pressure_curr =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
        self .air_pressure_next =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )

    def _configure_air_topology_payload (
    self ,
    air_element_position_xyz :np .ndarray ,
    air_element_size_xyz :np .ndarray ,
    air_neighbors :np .ndarray |None =None ,
    air_neighbor_absorb_u8 :np .ndarray |None =None ,
    air_boundary_mask_elements :np .ndarray |None =None ,
    solid_to_air_index :np .ndarray |None =None ,
    solid_to_air_index_plus :np .ndarray |None =None ,
    solid_to_air_index_minus :np .ndarray |None =None ,
    air_grid_shape :np .ndarray |None =None ,
    membrane_mask_elements :np .ndarray |None =None ,
    sensor_mask_elements :np .ndarray |None =None ,
    )->None :
        self ._air_history_iy_cached =None
        air_pos =np .asarray (air_element_position_xyz ,dtype =np .float64 )
        air_size =np .asarray (air_element_size_xyz ,dtype =np .float64 )
        if air_pos .ndim !=2 or air_pos .shape [1 ]!=3 :
            raise ValueError ('air_element_position_xyz must have shape [n_air, 3]')
        if air_size .shape !=air_pos .shape :
            raise ValueError ('air_element_size_xyz must have shape [n_air, 3]')
        if np .any (air_size <=0.0 ):
            raise ValueError ('air_element_size_xyz should only contain positive sizes')
        # Topology payload may be in mm while solver state is in meters.
        # Detect large mismatch against FE element sizes and convert payload to meters.
        try :
            fe_size_ref =float (np .mean (np .asarray (self .element_size_xyz [:self .n_elements ],dtype =np .float64 )))
            air_size_ref =float (np .mean (air_size ))
            if fe_size_ref >0.0 and air_size_ref >0.0 and (air_size_ref /fe_size_ref )>50.0 :
                air_pos =air_pos *1e-3
                air_size =air_size *1e-3
                print (
                    "[air][warn] Air topology units look like mm; converted air_element_position/size to meters."
                )
        except Exception :
            pass
        n_air =int (air_pos .shape [0 ])
        self .n_air_cells =n_air
        if n_air <=0 :
            self .nx_air =0 ;self .ny_air =0 ;self .nz_air =0
            self .dx_air =0.0 ;self .dy_air =0.0 ;self .dz_air =0.0
            self .air_neighbors =np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 )
            self .air_boundary_mask_elements =np .zeros (0 ,dtype =np .int32 )
            self .air_elem_map =np .full (self .n_elements ,-1 ,dtype =np .int32 )
            self .air_map_6 =np .full ((self .n_elements ,FACE_DIRS ),-1 ,dtype =np .int32 )
            self .air_elem_face_area =np .zeros ((self .n_elements ,3 ),dtype =np .float64 )
            self .air_elem_volume =np .zeros ((self .n_elements ,),dtype =np .float64 )
            self .air_inject_cell_plus =np .full (self .n_elements ,-1 ,dtype =np .int32 )
            self .air_inject_cell_minus =np .full (self .n_elements ,-1 ,dtype =np .int32 )
            self ._air_cell_pos_xyz =np .empty ((0 ,3 ),dtype =np .float64 )
            self .air_neighbor_absorb_u8 =np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 )
            return

        if air_neighbors is None :
            nb =np .full ((n_air ,FACE_DIRS ),-1 ,dtype =np .int32 )
        else :
            nb =np .asarray (air_neighbors ,dtype =np .int32 )
            if nb .shape !=(n_air ,FACE_DIRS ):
                raise ValueError (f"air_neighbors must have shape [{n_air }, {FACE_DIRS }]")
            if np .any ((nb <-1 )|(nb >=n_air )):
                raise ValueError ('air_neighbors contains indexes outside the range [-1, n_air)')
        if air_boundary_mask_elements is None :
            bnd =np .zeros (n_air ,dtype =np .int32 )
        else :
            bnd =np .asarray (air_boundary_mask_elements ,dtype =np .int32 ).ravel ()
            if bnd .size !=n_air :
                raise ValueError ('air_boundary_mask_elements must be of size n_air')
            bnd =np .where (bnd !=0 ,1 ,0 ).astype (np .int32 )
        self .air_neighbors =nb .copy ()
        self .air_boundary_mask_elements =bnd
        self ._air_cell_pos_xyz =air_pos .copy ()

        d_est =np .mean (air_size ,axis =0 )
        self .dx_air =float (max (d_est [0 ],1e-12 ))
        self .dy_air =float (max (d_est [1 ],1e-12 ))
        self .dz_air =float (max (d_est [2 ],1e-12 ))
        amin =np .min (air_pos -0.5 *air_size ,axis =0 )
        amax =np .max (air_pos +0.5 *air_size ,axis =0 )
        self .air_origin_x =float (amin [0 ])
        self .air_origin_y =float (amin [1 ])
        self .air_origin_z =float (amin [2 ])
        if air_grid_shape is not None :
            shp =np .asarray (air_grid_shape ,dtype =np .int32 ).ravel ()
            if shp .size ==3 and np .all (shp >0 ):
                self .nx_air =int (shp [0 ])
                self .ny_air =int (shp [1 ])
                self .nz_air =int (shp [2 ])
            else :
                raise ValueError ('air_grid_shape must have 3 positive ints')
        else :
            ext =amax -amin
            self .nx_air =max (1 ,int (np .ceil (ext [0 ]/(self .dx_air +1e-30 ))))
            self .ny_air =max (1 ,int (np .ceil (ext [1 ]/(self .dy_air +1e-30 ))))
            self .nz_air =max (1 ,int (np .ceil (ext [2 ]/(self .dz_air +1e-30 ))))

        if air_neighbor_absorb_u8 is None :
            self .air_neighbor_absorb_u8 =self ._infer_air_neighbor_absorb_from_geometry ()
        else :
            ab =np .asarray (air_neighbor_absorb_u8 ,dtype =np .uint8 ).reshape (n_air ,FACE_DIRS )
            if ab .shape !=(n_air ,FACE_DIRS ):
                raise ValueError (f"air_neighbor_absorb_u8 must have shape [{n_air }, {FACE_DIRS }]")
            self .air_neighbor_absorb_u8 =np .where (ab !=0 ,np .uint8 (1 ),np .uint8 (0 )).astype (np .uint8 )

        size_arr =self .element_size_xyz [:self .n_elements ]
        self .air_elem_face_area =np .zeros ((self .n_elements ,3 ),dtype =np .float64 )
        self .air_elem_face_area [:,0 ]=size_arr [:,1 ]*size_arr [:,2 ]
        self .air_elem_face_area [:,1 ]=size_arr [:,0 ]*size_arr [:,2 ]
        self .air_elem_face_area [:,2 ]=size_arr [:,0 ]*size_arr [:,1 ]
        np .maximum (self .air_elem_face_area ,1e-18 ,out =self .air_elem_face_area )
        self .air_elem_volume =(size_arr [:,0 ]*size_arr [:,1 ]*size_arr [:,2 ]).astype (np .float64 )
        thin_axes =np .argmin (np .asarray (size_arr ,dtype =np .float64 ),axis =1 )

        self .air_elem_map =np .full (self .n_elements ,-1 ,dtype =np .int32 )
        used_payload_mapping =False
        if solid_to_air_index is not None :
            s2a =np .asarray (solid_to_air_index ,dtype =np .int32 ).ravel ()
            if s2a .size !=self .n_elements :
                raise ValueError ('solid_to_air_index must be of size n_elements')
            self .air_elem_map [:]=np .where ((s2a >=0 )&(s2a <n_air ),s2a ,-1 )
            used_payload_mapping =True
        else :
            # nearest center fallback (slow, but used only without explicit mapping)
            p =self .position [:self .n_elements *self .dof_per_element ].reshape (self .n_elements ,self .dof_per_element )[:,:3 ]
            for i in range (self .n_elements ):
                dv =air_pos -p [i ]
                d2 =np .sum (dv *dv ,axis =1 )
                self .air_elem_map [i ]=int (np .argmin (d2 ))

        # If payload mapping links only clamped/non-radiating FE, air injection becomes zero.
        # Auto-fallback to geometric nearest-cell mapping for all FE.
        if used_payload_mapping :
            mapped =(self .air_elem_map >=0 )&(self .air_elem_map <n_air )
            non_boundary =(self .boundary_mask_elements ==0 )if self .boundary_mask_elements .size ==self .n_elements else np .ones (self .n_elements ,dtype =bool )
            if int (np .sum (mapped &non_boundary ))==0 :
                mat_idx_all =np .clip (self .material_index .astype (np .int32 ),0 ,self .material_props .shape [0 ]-1 )
                radiating =(self .material_props [mat_idx_all ,MAT_PROP_ACOUSTIC_INJECT ]>0.0 )
                thaw =mapped &radiating
                if np .any (thaw ):
                    self .boundary_mask_elements [thaw ]=0
                    print (
                        "[air][warn] solid_to_air_index maps only boundary FE; "
                        f"released boundary on {int (np .sum (thaw ))} mapped radiating elements."
                    )

        n_el =int (self .n_elements )
        self .air_inject_cell_plus =np .full (n_el ,-1 ,dtype =np .int32 )
        self .air_inject_cell_minus =np .full (n_el ,-1 ,dtype =np .int32 )
        filled_bilateral =False
        if solid_to_air_index_plus is not None and solid_to_air_index_minus is not None :
            sp =np .asarray (solid_to_air_index_plus ,dtype =np .int32 ).ravel ()
            sm =np .asarray (solid_to_air_index_minus ,dtype =np .int32 ).ravel ()
            if sp .size ==n_el and sm .size ==n_el :
                self .air_inject_cell_plus [:]=np .where ((sp >=0 )&(sp <n_air ),sp ,-1 )
                self .air_inject_cell_minus [:]=np .where ((sm >=0 )&(sm <n_air ),sm ,-1 )
                filled_bilateral =True
        if not filled_bilateral :
            na =np .argmin (np .asarray (self .element_size_xyz [:n_el ],dtype =np .float64 ),axis =1 )
            for i in range (n_el ):
                c =int (self .air_elem_map [i ])
                if c <0 or c >=n_air :
                    continue
                ax =int (na [i ])
                nb_c =self .air_neighbors [c ]
                pp =int (nb_c [2 *ax ])
                pm =int (nb_c [2 *ax +1 ])
                if pp >=0 :
                    self .air_inject_cell_plus [i ]=pp
                if pm >=0 :
                    self .air_inject_cell_minus [i ]=pm

        self .air_map_6 =np .full ((self .n_elements ,FACE_DIRS ),-1 ,dtype =np .int32 )
        for i in range (self .n_elements ):
            c =int (self .air_elem_map [i ])
            if c <0 or c >=n_air :
                continue
            self .air_map_6 [i ,:]=c
            nb_c =self .air_neighbors [c ]
            na =int (thin_axes [i ])
            ipp =int (self .air_inject_cell_plus [i ])
            imm =int (self .air_inject_cell_minus [i ])
            for axis in range (3 ):
                p_i =c
                m_i =c
                best_p =0.0
                best_m =0.0
                for nn in nb_c :
                    j =int (nn )
                    if j <0 :
                        continue
                    dd =air_pos [j ,axis ]-air_pos [c ,axis ]
                    if dd >best_p :
                        best_p =dd ;p_i =j
                    if dd <best_m :
                        best_m =dd ;m_i =j
                if axis ==na :
                    if ipp >=0 :
                        p_i =ipp
                    if imm >=0 :
                        m_i =imm
                self .air_map_6 [i ,2 *axis ]=p_i
                self .air_map_6 [i ,2 *axis +1 ]=m_i

        p0 =float (self .air_initial_uniform_pressure_pa )
        self .air_pressure_prev =np .full (n_air ,p0 ,dtype =np .float64 )
        self .air_pressure_curr =np .full (n_air ,p0 ,dtype =np .float64 )
        self .air_pressure_next =np .full (n_air ,p0 ,dtype =np .float64 )

    def _update_air_coupling_geometry_from_motion (self )->None :
        'Updates air_map_6 with the current position of the FE (elements are displaced in the air mesh).'
        if self ._air_topology_from_payload :
            # keep mapping from payload; topology generator already provides solid↔air links
            if hasattr (self ,'_buf_air_map_6'):
                cl .enqueue_copy (self .queue ,self ._buf_air_map_6 ,self .air_map_6 )
            if hasattr (self ,'_buf_air_elem_map'):
                cl .enqueue_copy (self .queue ,self ._buf_air_elem_map ,self .air_elem_map )
            return
        n =self .n_elements 
        pos_xyz =self .position .reshape (n ,self .dof_per_element )[:,:3 ]
        # Rare non-finite at the buffer boundary/after a GPU failure gives NaN→int32 warning; clip to the mesh after casting.
        with np .errstate (invalid ="ignore"):
            ax =np .rint ((pos_xyz [:,0 ]-self .air_origin_x )/self .dx_air ).astype (np .int32 )
            ay =np .rint ((pos_xyz [:,1 ]-self .air_origin_y )/self .dy_air ).astype (np .int32 )
            az =np .rint ((pos_xyz [:,2 ]-self .air_origin_z )/self .dz_air ).astype (np .int32 )
        np .clip (ax ,0 ,self .nx_air -1 ,out =ax )
        np .clip (ay ,0 ,self .ny_air -1 ,out =ay )
        np .clip (az ,0 ,self .nz_air -1 ,out =az )
        ix_p =np .clip (ax +1 ,0 ,self .nx_air -1 )
        ix_m =np .clip (ax -1 ,0 ,self .nx_air -1 )
        iy_p =np .clip (ay +1 ,0 ,self .ny_air -1 )
        iy_m =np .clip (ay -1 ,0 ,self .ny_air -1 )
        iz_p =np .clip (az +1 ,0 ,self .nz_air -1 )
        iz_m =np .clip (az -1 ,0 ,self .nz_air -1 )
        self .air_map_6 [:,0 ]=az *(self .nx_air *self .ny_air )+ay *self .nx_air +ix_p 
        self .air_map_6 [:,1 ]=az *(self .nx_air *self .ny_air )+ay *self .nx_air +ix_m 
        self .air_map_6 [:,2 ]=az *(self .nx_air *self .ny_air )+iy_p *self .nx_air +ax 
        self .air_map_6 [:,3 ]=az *(self .nx_air *self .ny_air )+iy_m *self .nx_air +ax 
        self .air_map_6 [:,4 ]=iz_p *(self .nx_air *self .ny_air )+ay *self .nx_air +ax 
        self .air_map_6 [:,5 ]=iz_m *(self .nx_air *self .ny_air )+ay *self .nx_air +ax 
        self .air_elem_map =az *(self .nx_air *self .ny_air )+ay *self .nx_air +ax 
        nxny =self .nx_air *self .ny_air 
        na =np .argmin (np .asarray (self .element_size_xyz [:n ],dtype =np .float64 ),axis =1 )
        cp =np .zeros (n ,dtype =np .int32 )
        cm =np .zeros (n ,dtype =np .int32 )
        m0 =na ==0 
        cp [m0 ]=(az *nxny +ay *self .nx_air +ix_p )[m0 ]
        cm [m0 ]=(az *nxny +ay *self .nx_air +ix_m )[m0 ]
        m1 =na ==1 
        cp [m1 ]=(az *nxny +iy_p *self .nx_air +ax )[m1 ]
        cm [m1 ]=(az *nxny +iy_m *self .nx_air +ax )[m1 ]
        m2 =na ==2 
        cp [m2 ]=(iz_p *nxny +ay *self .nx_air +ax )[m2 ]
        cm [m2 ]=(iz_m *nxny +ay *self .nx_air +ax )[m2 ]
        self .air_inject_cell_plus =cp 
        self .air_inject_cell_minus =cm 
        cl .enqueue_copy (self .queue ,self ._buf_air_map_6 ,self .air_map_6 )
        if hasattr (self ,'_buf_air_elem_map'):
            cl .enqueue_copy (self .queue ,self ._buf_air_elem_map ,self .air_elem_map )
        self ._rebuild_air_inject_csr ()
        self ._upload_air_inject_csr_buffers ()
    def _reset_air_field (self )->None :
        if self .n_air_cells >0 :
            p0 =float (self .air_initial_uniform_pressure_pa )
            self .air_pressure_prev =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
            self .air_pressure_curr =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
            self .air_pressure_next =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
            if hasattr (self ,'_buf_air_prev'):
                cl .enqueue_copy (self .queue ,self ._buf_air_prev ,self .air_pressure_prev )
            if hasattr (self ,'_buf_air_curr'):
                cl .enqueue_copy (self .queue ,self ._buf_air_curr ,self .air_pressure_curr )
            if hasattr (self ,'_buf_air_next'):
                cl .enqueue_copy (self .queue ,self ._buf_air_next ,self .air_pressure_next )
            self ._warned_no_external_excitation =False
            self ._warned_no_z0_radiating =False
            self ._air_trace_step_counter =0
            self ._air_acoustic_summary_shown =False
            self ._air_warned_cfl =False
            self ._air_slice_duplicate_warned =False
        else :
            self .air_pressure_prev =np .empty ((0 ,),dtype =np .float64 )
            self .air_pressure_curr =np .empty ((0 ,),dtype =np .float64 )
            self .air_pressure_next =np .empty ((0 ,),dtype =np .float64 )
        self ._air_force_external .fill (0.0 )
        cl .enqueue_copy (self .queue ,self ._buf_air_force_external ,self ._air_force_external )
        self .queue .finish ()

    def _air_center_xz_slice_from_flat (self ,flat :np .ndarray )->np .ndarray :
        if self .nx_air <=0 or self .ny_air <=0 or self .nz_air <=0 or flat .size !=(self .nx_air *self .ny_air *self .nz_air ):
            return flat .reshape (1 ,-1 ).copy ()
        'Center cut X-Z (perpendicular to the membrane) through the middle along the Y.\n        Returns an array shape[nz_air, nx_air].'
        p3 =flat .reshape (self .nz_air ,self .ny_air ,self .nx_air )
        iy_sel =self ._air_history_iy_from_sensor_center ()
        return p3 [:,iy_sel ,:].copy ()

    def _air_history_iy_from_sensor_center (self )->int :
        """Y index for recorded air slices: center of all sensor FE in world Y.

        Falls back to geometric center when sensor FE are absent or mapping is unavailable.
        """
        ny =int (getattr (self ,"ny_air",0 ))
        if ny <=0 :
            return 0
        cached =getattr (self ,"_air_history_iy_cached",None )
        if cached is not None :
            return max (0 ,min (int (cached ),ny -1 ))
        iy_geo =max (0 ,min (ny //2 ,ny -1 ))
        try :
            if self .n_elements <=0 :
                self ._air_history_iy_cached =iy_geo
                return iy_geo
            sidx =np .flatnonzero (self ._sensor_mask )
            if sidx .size <=0 :
                self ._air_history_iy_cached =iy_geo
                return iy_geo
            xyz =self .position [:self .n_elements *self .dof_per_element ].reshape (self .n_elements ,self .dof_per_element )[:,:3 ]
            y_center =float (np .mean (xyz [sidx ,1 ]))
            y0 =float (self .air_origin_y )
            dy =float (self .dy_air if self .dy_air >0.0 else 1.0 )
            iy =int (np .floor ((y_center -y0 )/dy ))
            iy =max (0 ,min (iy ,ny -1 ))
            self ._air_history_iy_cached =iy
            return iy
        except Exception :
            self ._air_history_iy_cached =iy_geo
            return iy_geo

    def _air_center_xy_slice_from_flat (self ,flat :np .ndarray )->np .ndarray :
        """XZ plane at the center air voxel row iy = ny_air//2 (pressure vs x,z).

        For carved air topologies, each flat entry is one voxel. The old path used a loose Y tolerance
        and np.add.at **averaging** into (ix,iz) bins, which merged multiple Y-layers into one pixel
        and made propagating waves look like smooth “jelly”. Here we select **one** iy index from
        voxel coordinates aligned with air_origin_* and dx_air,dy_air,dz_air."""
        if flat .size !=self .n_air_cells or self .n_air_cells <=0 :
            return flat .reshape (1 ,-1 ).copy ()
        if self ._air_cell_pos_xyz .shape ==(self .n_air_cells ,3 ):
            pos =self ._air_cell_pos_xyz
            dx =float (self .dx_air if self .dx_air >0.0 else 1.0 )
            dy =float (self .dy_air if self .dy_air >0.0 else 1.0 )
            dz =float (self .dz_air if self .dz_air >0.0 else 1.0 )
            x0 =float (self .air_origin_x )
            y0 =float (self .air_origin_y )
            z0 =float (self .air_origin_z )
            nx_g =int (self .nx_air )
            ny_g =int (self .ny_air )
            nz_g =int (self .nz_air )
            if nx_g >0 and ny_g >0 and nz_g >0 :
                ix =np .floor ((pos [:,0 ]-x0 )/dx ).astype (np .int32 )
                iy =np .floor ((pos [:,1 ]-y0 )/dy ).astype (np .int32 )
                iz =np .floor ((pos [:,2 ]-z0 )/dz ).astype (np .int32 )
                np .clip (ix ,0 ,nx_g -1 ,out =ix )
                np .clip (iy ,0 ,ny_g -1 ,out =iy )
                np .clip (iz ,0 ,nz_g -1 ,out =iz )
                iy_sel =self ._air_history_iy_from_sensor_center ()
                mask =iy ==iy_sel
                if not np .any (mask )and iy .size >0 :
                    iy_pick =int (iy [np .argmin (np .abs (iy -iy_sel ))])
                    mask =iy ==iy_pick
                if np .any (mask ):
                    out =np .full ((nz_g ,nx_g ),np .nan ,dtype =np .float64 )
                    vals =np .asarray (flat [mask ],dtype =np .float64 )
                    iz_m =iz [mask ]
                    ix_m =ix [mask ]
                    for k in range (vals .shape [0 ]):
                        zi ,xi =int (iz_m [k ]),int (ix_m [k ])
                        if not np .isnan (out [zi ,xi ])and not self ._air_slice_duplicate_warned :
                            print (
                            "[air][warn] Multiple carved-air cells map to the same XZ pixel in "
                            "history slice; last value wins (check air grid / origins)."
                            )
                            self ._air_slice_duplicate_warned =True
                        out [zi ,xi ]=vals [k ]
                    return out
        if self .nx_air <=0 or self .ny_air <=0 or self .nz_air <=0 or flat .size !=(self .nx_air *self .ny_air *self .nz_air ):
            return flat .reshape (1 ,-1 ).copy ()
        p3 =flat .reshape (self .nz_air ,self .ny_air ,self .nx_air )
        # Slice at the membrane Z index from rebuild_air_field (not nz//2): asymmetric padding
        # or non-centered mesh makes nz//2 miss the source plane; outgoing waves then look like a
        # "growing positive ring" in an off-plane XY cut.
        z_sel =int (getattr (self ,"air_z0",self .nz_air //2 ))
        z_sel =max (0 ,min (z_sel ,self .nz_air -1 ))
        return p3 [z_sel ,:,:].copy ()

    def _log_air_pressure_metrics (self ,step_idx :int ,flat :np .ndarray ,tag :str )->None :
        """Print integral pressure metrics for each saved air-history frame."""
        if not self .kernel_debug :
            return
        if tag =="history_save":
            every =max (1 ,int (getattr (self ,"air_metric_log_every_steps",1 )))
            if (step_idx %every )!=0 :
                return
        p =np .asarray (flat ,dtype =np .float64 ).ravel ()
        if p .size ==0 :
            print (f"[air][metric] {tag }: step={step_idx }, empty pressure field")
            return
        finite =np .isfinite (p )
        if not np .all (finite ):
            n_bad =int (p .size -np .sum (finite ))
            print (f"[air][metric] {tag }: step={step_idx }, non-finite values={n_bad }/{p .size }")
            # Keep the full array for explosion dump, but filter for integrals below.
            p_full =p
            p =p [finite ]
        else :
            p_full =p
        if p .size ==0 :
            # all values were non-finite; dump context if possible
            self ._dump_air_exploding_cells (step_idx ,p_full ,tag )
            return
        sum_abs =float (np .sum (np .abs (p )))
        sum_signed =float (np .sum (p ))
        mean_abs =float (np .mean (np .abs (p )))
        mean_signed =float (np .mean (p ))
        # Guard against overflow when p is huge.
        with np .errstate (over ="ignore",invalid ="ignore"):
            l2 =float (np .sqrt (np .sum (p *p )))
        p_min =float (np .min (p ))
        p_max =float (np .max (p ))
        cell_vol =float (
        (self .dx_air if self .dx_air >0.0 else 1.0 )
        *(self .dy_air if self .dy_air >0.0 else 1.0 )
        *(self .dz_air if self .dz_air >0.0 else 1.0 )
        )
        integral_abs =sum_abs *cell_vol
        integral_signed =sum_signed *cell_vol
        with np .errstate (over ="ignore",invalid ="ignore"):
            integral_p2 =float (np .sum (p *p )*cell_vol )
        print (
            f"[air][metric] {tag }: step={step_idx } "
            f"sum|p|={sum_abs :.6e} Pa, sum(p)={sum_signed :.6e} Pa, "
            f"mean|p|={mean_abs :.6e} Pa, mean(p)={mean_signed :.6e} Pa, "
            f"L2={l2 :.6e} Pa, min/max=[{p_min :.6e}, {p_max :.6e}] Pa, "
            f"int|p|dV={integral_abs :.6e} Pa*m^3, int(p)dV={integral_signed :.6e} Pa*m^3, "
            f"int(p^2)dV={integral_p2 :.6e} Pa^2*m^3"
        )
        # If the field looks like it is starting to diverge, dump detailed cell context.
        try :
            p_abs_full =np .abs (p_full )
            max_abs =float (np .nanmax (p_abs_full )) if p_abs_full .size >0 else 0.0
            if (not np .isfinite (max_abs ))or (max_abs >float (self .air_explosion_abs_pa )):
                cooldown =max (0 ,int (getattr (self ,"air_explosion_dump_cooldown_steps",0 )))
                last =int (getattr (self ,"_air_last_explosion_dump_step",-10**9 ))
                if step_idx -last >=cooldown :
                    self ._air_last_explosion_dump_step =int (step_idx )
                    self ._dump_air_exploding_cells (step_idx ,p_full ,tag )
        except Exception :
            pass

    def _dump_air_exploding_cells (self ,step_idx :int ,p_full :np .ndarray ,tag :str )->None :
        """Dump detailed info for the most extreme air cells (by |p| and by non-finite entries).

        This is intended for long runs: only triggers when the metric logger detects divergence.
        """
        if not self .kernel_debug :
            return
        p =np .asarray (p_full ,dtype =np .float64 ).ravel ()
        if p .size ==0 :
            return
        n =int (p .size )
        # Identify non-finite indices first.
        bad =~np .isfinite (p )
        bad_idx =np .flatnonzero (bad )
        if bad_idx .size >0 :
            show_bad =bad_idx [:min (bad_idx .size ,self .air_explosion_log_topk_cells )]
            print (f"[air][explode][metric] {tag }: step={step_idx } non-finite sample idx={show_bad .tolist ()}")
        # Top-K by absolute magnitude (ignoring NaNs for ranking).
        p_abs =np .abs (p )
        # Replace non-finite with +inf so they appear in top-K.
        p_abs =np .where (np .isfinite (p_abs ),p_abs ,np .inf )
        topk =min (int (self .air_explosion_log_topk_cells ),n )
        idx =np .argpartition (p_abs ,-topk )[-topk :]
        # Sort descending by |p|.
        idx =idx [np .argsort (p_abs [idx ])[::-1 ]]

        # Try to fetch injection buffer only when we are already in "explode" mode.
        inj =None
        try :
            if hasattr (self ,'_buf_air_plus')and hasattr (self ,'queue')and getattr (self ,'n_air_cells',0 )==n :
                inj =np .empty (n ,dtype =np .float64 )
                cl .enqueue_copy (self .queue ,inj ,self ._buf_air_plus )
                self .queue .finish ()
        except Exception :
            inj =None

        # Local replicas for CSR-based contribution introspection (host-side approximation).
        dt =float (self ._air_last_fe_dt )if getattr (self ,'_air_last_fe_dt',None )is not None else None
        cell_vol =float (
            (self .dx_air if self .dx_air >0.0 else 1.0 )
            *(self .dy_air if self .dy_air >0.0 else 1.0 )
            *(self .dz_air if self .dz_air >0.0 else 1.0 )
        )
        c_sound =float (max (self .air_sound_speed ,1e-12 ))
        bulk_modulus =max (float (self .rho_air )*c_sound *c_sound ,1e-18 )
        n_materials =int (self .material_props .shape [0 ])if hasattr (self ,'material_props')else 0

        AIR_INJECT_THIN_RATIO =0.36
        face_norm =(
            (1.0 ,0.0 ,0.0 ),
            (-1.0 ,0.0 ,0.0 ),
            (0.0 ,1.0 ,0.0 ),
            (0.0 ,-1.0 ,0.0 ),
            (0.0 ,0.0 ,1.0 ),
            (0.0 ,0.0 ,-1.0 ),
        )
        def face_area_from_size(d ,size_e ):
            sx ,sy ,sz =float (size_e [0 ]),float (size_e [1 ]),float (size_e [2 ])
            if d <2 : return sy *sz
            if d <4 : return sx *sz
            return sx *sy
        def air_inject_dV_dot_host(e ,v_e ,size_e ):
            sx ,sy ,sz =float (size_e [0 ]),float (size_e [1 ]),float (size_e [2 ])
            smax =max (sx ,sy ,sz )
            smin =min (sx ,sy ,sz )
            if smin <=0.0 or smax <=0.0 :
                return 0.0
            if smin <=AIR_INJECT_THIN_RATIO *smax :
                thin =0 if (sx <=sy and sx <=sz ) else (1 if (sy <=sz ) else 2 )
                A_n = (sy *sz ) if thin ==0 else ((sx *sz ) if thin ==1 else (sx *sy ))
                v_thin = (v_e [0 ]) if thin ==0 else ((v_e [1 ]) if thin ==1 else (v_e [2 ]))
                return A_n *v_thin
            dV_dot =0.0
            for d in range (6 ):
                nb =int (self .neighbors [e ,d ]) if hasattr (self ,'neighbors')else -1
                if nb >=0 :
                    v_nb =self .velocity [nb *6 +0 :nb *6 +3 ]
                    v_rel =v_nb -v_e
                else :
                    v_rel =-v_e
                nx ,ny ,nz =face_norm [d ]
                v_n =nx *float (v_rel [0 ]) +ny *float (v_rel [1 ]) +nz *float (v_rel [2 ])
                A_face =face_area_from_size (d ,size_e )
                dV_dot +=A_face *v_n
            return dV_dot

        for c in idx .tolist ():
            # Coords if grid shape available.
            ix =iy =iz =-1
            if getattr (self ,'nx_air',0 )>0 and getattr (self ,'ny_air',0 )>0 and getattr (self ,'nz_air',0 )>0 :
                nxny =int (self .nx_air *self .ny_air )
                iz =int (c //nxny )
                rem =int (c -iz *nxny )
                iy =int (rem //self .nx_air )
                ix =int (rem -iy *self .nx_air )
            nb =self .air_neighbors [c ] if hasattr (self ,'air_neighbors')and self .air_neighbors .shape [0 ]==n else None
            missing =int (np .sum (nb <0 )) if nb is not None else -1
            nb_cnt =int (np .sum (nb >=0 )) if nb is not None else -1
            inj_c =float (inj [c ]) if inj is not None else float ("nan")
            print (
                f"[air][explode][metric] {tag }: step={step_idx } cell={c} (ix,iy,iz)=({ix},{iy},{iz}) "
                f"p={p[c ]!r} |p|={p_abs[c ]!r} inj_dp={inj_c :.6e} "
                f"nb_cnt={nb_cnt} missing_faces={missing} nb_idx={(nb .tolist () if nb is not None else None)}"
            )
            # CSR + element contributions (best-effort; requires dt and CSR to exist).
            if dt is None :
                continue
            if not hasattr (self ,'_air_inject_csr_offsets') :
                continue
            try :
                beg =int (self ._air_inject_csr_offsets [c ])
                end =int (self ._air_inject_csr_offsets [c +1 ])
                nnz =max (0 ,end -beg )
                print (f"[air][explode][metric] cell={c} CSR nnz={nnz} dt={dt :.3e} cell_vol={cell_vol :.3e}")
                contrib =[]
                for k in range (beg ,end ):
                    e =int (self ._air_inject_csr_indices [k ])
                    sgn =float (self ._air_inject_csr_signs [k ])
                    if e <0 or e >=self .n_elements :
                        continue
                    if int (self .boundary_mask_elements [e ]) !=0 :
                        continue
                    mat_id =int (self .material_index [e ]) if hasattr (self ,'material_index')else -1
                    if mat_id <0 or mat_id >=n_materials :
                        continue
                    acoustic_inject =float (self .material_props [mat_id ,7 ])
                    if acoustic_inject <=0.0 :
                        continue
                    v_e =self .velocity [e *6 +0 :e *6 +3 ].astype (np .float64 )
                    size_e =self .element_size_xyz [e ,:3 ].astype (np .float64 )
                    dV_dot =air_inject_dV_dot_host (e ,v_e ,size_e )
                    dp_elem =acoustic_inject *bulk_modulus * (dV_dot *dt /max (cell_vol ,1e-18 )) *sgn
                    contrib .append ((abs (dp_elem ),dp_elem,e,sgn,mat_id,acoustic_inject,dV_dot,v_e,size_e))
                contrib .sort (key =lambda t :t [0 ],reverse =True )
                top_e =min (len (contrib ),int (self .air_explosion_log_topk_elems ))
                if top_e >0 :
                    dp_sum =float (sum (t [1] for t in contrib ))
                    print (f"[air][explode][metric] cell={c} predicted dp_sum={dp_sum :.6e} (inj_dp={inj_c :.6e})")
                    for i in range (top_e ):
                        _,dp_elem,e,sgn,mat_id,ac_inj,dV_dot,v_e,size_e =contrib [i ]
                        print (
                            f"  elem={e} sign={sgn:+.1f} mat={mat_id} acoustic_inject={ac_inj:.3e} "
                            f"v_e=({v_e[0]:.3e},{v_e[1]:.3e},{v_e[2]:.3e}) "
                            f"size=({size_e[0]:.3e},{size_e[1]:.3e},{size_e[2]:.3e}) "
                            f"dV_dot={dV_dot:.3e} dp_elem={dp_elem:.3e}"
                        )
            except Exception :
                pass

    def _params_bytes (self ,dt :float ,step_idx :int =0 ,debug_elem :int =-1 )->bytes :
        return _pack_params (
        self .nx ,self .ny ,
        self .n_elements ,
        self .width ,self .height ,self .thickness ,
        self .density ,self .E_parallel ,self .E_perp ,self .poisson ,
        self .use_nonlinear_stiffness ,
        self .stiffness_transition_center ,
        self .stiffness_transition_width ,
        self .stiffness_ratio ,
        self .rho_air ,self .mu_air ,self .Cd ,
        dt ,
        self .pre_tension ,
        self .k_soft ,self .k_stiff ,
        self .strain_transition ,self .strain_width ,
        self .k_bend ,
        debug_elem ,
        step_idx ,
        )

    def _build_force_external (self ,pressure_pa :float |np .ndarray )->None :
        """Uniform-pressure mechanical drive on membrane non-boundary elements.

        Force on translational DOF along the **thinnest** brick dimension (membrane normal), with
        magnitude |F| = |p| × (face area normal to that axis). Sign follows p. Independent of
        acoustic_inject and air_elem_map (those only affect FE→air radiation, not this load).

        Not written to air_pressure_*."""
        self .force_external .fill (0.0 )
        n =self .n_elements 
        sz_blk =np .asarray (self .element_size_xyz [:n ],dtype =np .float64 )
        normal_axis =np .argmin (sz_blk ,axis =1 )
        sx ,sy ,sz =sz_blk [:,0 ],sz_blk [:,1 ],sz_blk [:,2 ]
        area_normal =np .where (
        normal_axis ==0 ,sy *sz ,
        np .where (normal_axis ==1 ,sx *sz ,sx *sy ),
        ).astype (np .float64 )
        if self .boundary_mask_elements .size >=n :
            non_boundary =(self .boundary_mask_elements [:n ]==0 )
        else :
            non_boundary =np .ones (n ,dtype =bool )
        membrane_only =np .asarray (self .membrane_mask [:n ]!=0 ,dtype =bool )if hasattr (self ,"membrane_mask")else np .asarray (self .material_index [:n ]==MAT_MEMBRANE ,dtype =bool )
        drive_candidates =non_boundary &membrane_only
        z_c =self .position [2 :n *self .dof_per_element :self .dof_per_element ]
        z0_mask =np .isclose (z_c ,0.0 ,atol =1e-12 )
        excite_mask =z0_mask &drive_candidates
        if not np .any (excite_mask ):
            candidates =drive_candidates
            if np .any (candidates ):
                # Fallback remains membrane-only: use all membrane non-boundary FEs.
                excite_mask =candidates
                if not self ._warned_no_z0_radiating :
                    print (
                    "[force][warn] No membrane non-boundary FE on z≈0; "
                    f"fallback excitation layer: {int (np .sum (excite_mask ))} elements."
                    )
                    self ._warned_no_z0_radiating =True
        excite_idx =np .flatnonzero (excite_mask )
        n_exc =int (excite_idx .size )
        if np .isscalar (pressure_pa ):
            p =np .full (n_exc ,float (pressure_pa ),dtype =np .float64 )
        else :
            p =np .asarray (pressure_pa ,dtype =np .float64 ).ravel ()
            if p .size ==1 :
                p =np .full (n_exc ,float (p [0 ]),dtype =np .float64 )
            elif p .size ==n :
                p =p [excite_idx ]
            elif p .size !=n_exc :
                raise ValueError ('pressure_pa must be scalar, n_elements, or n_excited_elements')
        forced_count =0
        for local_idx ,elem in enumerate (excite_idx ):
            ax =int (normal_axis [elem ])
            self .force_external [elem *6 +ax ]=p [local_idx ]*float (area_normal [elem ])
            forced_count +=1
        if self .kernel_debug and forced_count >0 :
            try :
                fe =self .force_external .reshape (n ,self .dof_per_element )[:,:3 ]
                fmag =np .sqrt (np .sum (fe *fe ,axis =1 ))
                sensor_mask =np .asarray (self ._sensor_mask [:n ],dtype =bool )if hasattr (self ,"_sensor_mask")else (self .material_index [:n ]==MAT_SENSOR )
                membrane_mask =np .asarray (self .membrane_mask [:n ]!=0 ,dtype =bool )if hasattr (self ,"membrane_mask")else (self .material_index [:n ]==MAT_MEMBRANE )
                sensor_forced =int (np .sum (fmag [sensor_mask ]>0.0 ))
                membrane_forced =int (np .sum (fmag [membrane_mask ]>0.0 ))
                other_forced =int (np .sum (fmag [~(membrane_mask |sensor_mask )]>0.0 ))
                print (
                    f"[force][trace] external pressure drive: elements={forced_count}, "
                    f"max|F|={float (np .max (fmag )):.3e} N, "
                    f"membrane_forced={membrane_forced}, sensor_forced={sensor_forced}, other_forced={other_forced}"
                )
            except Exception :
                pass
        if forced_count ==0 and not self ._warned_no_external_excitation :
            print (
                "[force][warn] External pressure drive applied to 0 elements "
                "(no non-boundary candidates in z≈0 layer or thin-layer fallback)."
            )
            self ._warned_no_external_excitation =True

    def _mode_uses_full_force_override (self )->bool :
        return self .excitation_mode in (
            "external_full_override",
            "second_order_boundary_full_override",
        )

    def _mode_uses_velocity_override (self )->bool :
        return self .excitation_mode =="external_velocity_override"

    def _apply_velocity_external_override (self ,velocity_mps :float |np .ndarray )->None :
        """Override translational velocity for targeted FE along their drive axis."""
        n =self .n_elements
        mask =np .asarray (self ._force_drive_mask_u8 [:n ]!=0 ,dtype =bool )
        excite_idx =np .flatnonzero (mask )
        n_exc =int (excite_idx .size )
        if n_exc <=0 :
            if not self ._warned_no_external_velocity_excitation :
                print ("[velocity][warn] External velocity override targets 0 elements.")
                self ._warned_no_external_velocity_excitation =True
            return
        if np .isscalar (velocity_mps ):
            v =np .full (n_exc ,float (velocity_mps ),dtype =np .float64 )
        else :
            arr =np .asarray (velocity_mps ,dtype =np .float64 ).ravel ()
            if arr .size ==1 :
                v =np .full (n_exc ,float (arr [0 ]),dtype =np .float64 )
            elif arr .size ==n :
                v =arr [excite_idx ]
            elif arr .size ==n_exc :
                v =arr 
            else :
                raise ValueError ("velocity override expects scalar, n_elements, or n_excited_elements")
        axis =np .asarray (self ._force_drive_axis_u8 [:n ],dtype =np .int32 )
        for local_idx ,elem in enumerate (excite_idx ):
            ax =int (axis [elem ])
            self .velocity [elem *6 +ax ]=float (v [local_idx ])

    def set_material_library (self ,material_props :np .ndarray )->None :
        'Updates the materials table.\n        String format:\n        [density, E_parallel, E_perp, poisson, Cd, eta_visc, acoustic_impedance, acoustic_inject].\n        For backward compatibility, the following formats are allowed:\n        [n_materials, 5] -> eta_visc=0, acoustic_impedance=sqrt(rho*E_parallel);\n        [n_materials, 6] -> acoustic_impedance=sqrt(rho*E_parallel);\n        [n_materials, 7] -> acoustic_inject is added (sensor 0, others 1).'
        props =np .asarray (material_props ,dtype =np .float64 )
        if props .ndim !=2 or props .shape [1 ]not in (5 ,6 ,7 ,_MATERIAL_PROPS_STRIDE ):
            raise ValueError (
            'material_props must have shape [n_materials, 5], [6], [7] or [8]'
            )
        if props .shape [1 ]==5 :
            z =np .sqrt (np .maximum (props [:,0 ],1e-12 )*np .maximum (props [:,1 ],0.0 ))[:,None ]
            props =np .hstack (
            (
            props ,
            np .zeros ((props .shape [0 ],1 ),dtype =np .float64 ),
            z ,
            )
            )
        elif props .shape [1 ]==6 :
            z =np .sqrt (np .maximum (props [:,0 ],1e-12 )*np .maximum (props [:,1 ],0.0 ))[:,None ]
            props =np .hstack ((props ,z ))
        props =_expand_material_props_to_stride8 (props )
        if props .shape [0 ]<1 :
            raise ValueError ('material_props must contain at least 1 material')
        if np .any (props [:,6 ]<0.0 ):
            raise ValueError ('acoustic_impedance (column 6) must be >= 0')
        if np .any (props [:,7 ]<0.0 ):
            raise ValueError ('acoustic_inject (column 7) must be >= 0')
        if self .material_index .size >0 and int (self .material_index .max ())>=int (props .shape [0 ]):
            raise ValueError ('The current material_index contains indexes outside the new material library')
        self .material_props =props .copy ()
        self ._static_fe_dirty =True
        n_materials =self .material_props .shape [0 ]
        if self .laws .shape !=(n_materials ,n_materials ):
            self .laws =np .full ((n_materials ,n_materials ),LAW_SOLID_SPRING ,dtype =np .uint8 )
            # We recreate the buffer, because library size may change.
        if hasattr (self ,"ctx"):
            mf =cl .mem_flags 
            self ._buf_material_props =cl .Buffer (
            self .ctx ,mf .READ_ONLY ,size =self .material_props .size *8 
            )
            self ._buf_laws =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .laws .size )

    def set_element_material_index (self ,material_index :np .ndarray )->None :
        'Assigns a material index for each PI (uint8).'
        idx =np .asarray (material_index ,dtype =np .uint8 ).ravel ()
        if idx .size !=self .n_elements :
            raise ValueError ('material_index should be of size n_elements')
        if self .material_props .shape [0 ]>255 :
            raise ValueError ('Material count > 255 is not supported by uint8 index')
        if idx .size >0 and int (idx .max ())>=int (self .material_props .shape [0 ]):
            raise ValueError ('material_index has indexes outside the range of material_props')
        self .material_index =idx .copy ()
        self ._update_sensor_mask ()
        self ._fe_air_coupling_mask =np .asarray ((self .membrane_mask [:self .n_elements ]!=0 )|self ._sensor_mask [:self .n_elements ],dtype =np .uint8 )
        self ._static_fe_dirty =True

    def set_material_laws (self ,laws :np .ndarray )->None :
        'Matrix of interaction laws [n_materials, n_materials], dtype uint8.\n        laws[i, j] specifies the law for the connection (material i, material j).'
        arr =np .asarray (laws ,dtype =np .uint8 )
        n_materials =self .material_props .shape [0 ]
        if arr .shape !=(n_materials ,n_materials ):
            raise ValueError (f"laws must have shape [{n_materials }, {n_materials }]")
        self .laws =arr .copy ()
        self ._static_fe_dirty =True
        if hasattr (self ,"ctx"):
            mf =cl .mem_flags 
            self ._buf_laws =cl .Buffer (self .ctx ,mf .READ_ONLY ,size =self .laws .size )

    def set_neighbors_topology (self ,neighbors :np .ndarray )->None :
        'Explicitly sets the universal topology of neighbors [n_elements, FACE_DIRS].\n        For an absent neighbor, use -1.'
        arr =np .asarray (neighbors ,dtype =np .int32 )
        if arr .shape !=(self .n_elements ,FACE_DIRS ):
            raise ValueError (f"neighbors must have shape [{self .n_elements }, {FACE_DIRS }]")
        self .neighbors =arr .copy ()
        self ._static_fe_dirty =True

    def set_boundary_mask (self ,boundary_mask_elements :np .ndarray )->None :
        'Sets a mask of fixed CIs (1 = fixed, 0 = free).\n        Necessary for mixed scenes (membrane + other materials), where not all edges need to be fixed.'
        arr =np .asarray (boundary_mask_elements ,dtype =np .int32 ).ravel ()
        if arr .size !=self .n_elements :
            raise ValueError ('boundary_mask_elements must be of size n_elements')
        arr =np .where (arr !=0 ,1 ,0 ).astype (np .int32 )
        self .boundary_mask_elements =arr 
        self ._static_fe_dirty =True

    @staticmethod 
    def _validate_topology_payload (
    element_position_xyz :np .ndarray ,
    element_size_xyz :np .ndarray ,
    neighbors :np .ndarray ,
    material_index :np .ndarray |None =None ,
    boundary_mask_elements :np .ndarray |None =None ,
    element_active_mask :np .ndarray |None =None ,
    )->tuple [np .ndarray ,np .ndarray ,np .ndarray ,np .ndarray ,np .ndarray ,np .ndarray ]:
        pos =np .asarray (element_position_xyz ,dtype =np .float64 )
        size =np .asarray (element_size_xyz ,dtype =np .float64 )
        nbh =np .asarray (neighbors ,dtype =np .int32 )
        if pos .ndim !=2 or pos .shape [1 ]!=3 :
            raise ValueError ('element_position_xyz must have shape [n, 3]')
        if size .shape !=pos .shape :
            raise ValueError ('element_size_xyz must have shape [n, 3]')
        if nbh .shape !=(pos .shape [0 ],FACE_DIRS ):
            raise ValueError (f"neighbors must have shape [n, {FACE_DIRS }]")
        if np .any (size <=0.0 ):
            raise ValueError ('element_size_xyz should only contain positive sizes')
        n =pos .shape [0 ]
        if np .any ((nbh <-1 )|(nbh >=n )):
            raise ValueError ('neighbors contains indexes outside the range [-1, n)')

        if material_index is None :
            mat =np .full (n ,int (MAT_SENSOR ),dtype =np .uint8 )
        else :
            mat =np .asarray (material_index ,dtype =np .uint8 ).ravel ()
            if mat .size !=n :
                raise ValueError ('material_index must be of size n')

        if boundary_mask_elements is None :
            bnd =np .zeros (n ,dtype =np .int32 )
        else :
            bnd =np .asarray (boundary_mask_elements ,dtype =np .int32 ).ravel ()
            if bnd .size !=n :
                raise ValueError ('boundary_mask_elements must be of size n')
            bnd =np .where (bnd !=0 ,1 ,0 ).astype (np .int32 )

        if element_active_mask is None :
            active =np .ones (n ,dtype =bool )
        else :
            active =np .asarray (element_active_mask ,dtype =bool ).ravel ()
            if active .size !=n :
                raise ValueError ('element_active_mask must have size n')

        return pos ,size ,nbh ,mat ,bnd ,active 

    @staticmethod 
    def _aabb_overlap (
    p1 :np .ndarray ,s1 :np .ndarray ,p2 :np .ndarray ,s2 :np .ndarray ,tol :float 
    )->bool :
        h1 =0.5 *s1 
        h2 =0.5 *s2 
        for axis in range (3 ):
            if (p1 [axis ]+h1 [axis ])<(p2 [axis ]-h2 [axis ]-tol ):
                return False 
            if (p2 [axis ]+h2 [axis ])<(p1 [axis ]-h1 [axis ]-tol ):
                return False 
        return True 

    @classmethod 
    def merge_topologies (
    cls ,
    primary :dict [str ,np .ndarray ],
    secondary :dict [str ,np .ndarray ],
    primary_is_main :bool =True ,
    overlap_tol_m :float =1e-12 ,
    )->dict [str ,np .ndarray ]:
        'Combines two FE topologies.\n\n        At the intersections (AABB-overlap) elements of the “main” topology remain.\n        Returns a dictionary with keys:\n        - element_position_xyz, element_size_xyz, neighbors,\n        - material_index, boundary_mask_elements.'
        p_pos ,p_size ,p_nbh ,p_mat ,p_bnd ,p_active =cls ._validate_topology_payload (
        primary ["element_position_xyz"],
        primary ["element_size_xyz"],
        primary ["neighbors"],
        primary .get ("material_index"),
        primary .get ("boundary_mask_elements"),
        primary .get ("element_active_mask"),
        )
        s_pos ,s_size ,s_nbh ,s_mat ,s_bnd ,s_active =cls ._validate_topology_payload (
        secondary ["element_position_xyz"],
        secondary ["element_size_xyz"],
        secondary ["neighbors"],
        secondary .get ("material_index"),
        secondary .get ("boundary_mask_elements"),
        secondary .get ("element_active_mask"),
        )

        if overlap_tol_m <0.0 :
            raise ValueError ('overlap_tol_m must be >= 0')

        if primary_is_main :
            main_pos ,main_size ,main_nbh ,main_mat ,main_bnd ,main_active =(
            p_pos ,p_size ,p_nbh ,p_mat ,p_bnd ,p_active 
            )
            aux_pos ,aux_size ,aux_nbh ,aux_mat ,aux_bnd ,aux_active =(
            s_pos ,s_size ,s_nbh ,s_mat ,s_bnd ,s_active 
            )
        else :
            main_pos ,main_size ,main_nbh ,main_mat ,main_bnd ,main_active =(
            s_pos ,s_size ,s_nbh ,s_mat ,s_bnd ,s_active 
            )
            aux_pos ,aux_size ,aux_nbh ,aux_mat ,aux_bnd ,aux_active =(
            p_pos ,p_size ,p_nbh ,p_mat ,p_bnd ,p_active 
            )

        main_idx =np .flatnonzero (main_active )
        aux_idx =np .flatnonzero (aux_active )
        keep_aux =np .ones (aux_idx .size ,dtype =bool )
        for i_main in main_idx :
            pm =main_pos [i_main ]
            sm =main_size [i_main ]
            for j ,i_aux in enumerate (aux_idx ):
                if not keep_aux [j ]:
                    continue 
                if cls ._aabb_overlap (pm ,sm ,aux_pos [i_aux ],aux_size [i_aux ],overlap_tol_m ):
                    keep_aux [j ]=False 

        aux_idx_kept =aux_idx [keep_aux ]
        out_n =int (main_idx .size +aux_idx_kept .size )
        out_pos =np .zeros ((out_n ,3 ),dtype =np .float64 )
        out_size =np .zeros ((out_n ,3 ),dtype =np .float64 )
        out_nbh =np .full ((out_n ,FACE_DIRS ),-1 ,dtype =np .int32 )
        out_mat =np .zeros (out_n ,dtype =np .uint8 )
        out_bnd =np .zeros (out_n ,dtype =np .int32 )

        # Result indexes: first main, then the remaining aux.
        src_entries :list [tuple [str ,int ]]=[("main",int (i ))for i in main_idx ]
        src_entries +=[("aux",int (i ))for i in aux_idx_kept ]
        map_main ={int (src_i ):out_i for out_i ,(src ,src_i )in enumerate (src_entries )if src =="main"}
        map_aux ={int (src_i ):out_i for out_i ,(src ,src_i )in enumerate (src_entries )if src =="aux"}

        for out_i ,(src ,src_i )in enumerate (src_entries ):
            if src =="main":
                out_pos [out_i ]=main_pos [src_i ]
                out_size [out_i ]=main_size [src_i ]
                out_mat [out_i ]=main_mat [src_i ]
                out_bnd [out_i ]=main_bnd [src_i ]
                src_nbh =main_nbh [src_i ]
                idx_map =map_main 
            else :
                out_pos [out_i ]=aux_pos [src_i ]
                out_size [out_i ]=aux_size [src_i ]
                out_mat [out_i ]=aux_mat [src_i ]
                out_bnd [out_i ]=aux_bnd [src_i ]
                src_nbh =aux_nbh [src_i ]
                idx_map =map_aux 

            for d in range (FACE_DIRS ):
                n_old =int (src_nbh [d ])
                out_nbh [out_i ,d ]=idx_map .get (n_old ,-1 )

        return {
        "element_position_xyz":out_pos ,
        "element_size_xyz":out_size ,
        "neighbors":out_nbh ,
        "material_index":out_mat ,
        "boundary_mask_elements":out_bnd ,
        }

    def set_merged_topologies (
    self ,
    primary :dict [str ,np .ndarray ],
    secondary :dict [str ,np .ndarray ],
    primary_is_main :bool =True ,
    overlap_tol_m :float =1e-12 ,
    visual_shape :tuple [int ,int ]|None =None ,
    preserve_velocity :bool =False ,
    rebuild_air :bool =True ,
    air_grid_step_mm :float |None =None ,
    air_padding_mm :float |None =None ,
    )->None :
        'Merges two topologies and applies the result to the current model.\n\n        Important: the number of CIs after merging must match self.n_elements.'
        merged =self .merge_topologies (
        primary =primary ,
        secondary =secondary ,
        primary_is_main =primary_is_main ,
        overlap_tol_m =overlap_tol_m ,
        )
        if merged ["element_position_xyz"].shape [0 ]!=self .n_elements :
            raise ValueError (
            'After merge, the number of PIs does not match the current model.'
            'Recreate the model with the required n_elements or submit topologies with the appropriate size.'
            )
        self .set_custom_topology (
        element_position_xyz =merged ["element_position_xyz"],
        element_size_xyz =merged ["element_size_xyz"],
        neighbors =merged ["neighbors"],
        material_index =merged ["material_index"],
        boundary_mask_elements =merged ["boundary_mask_elements"],
        visual_shape =visual_shape ,
        preserve_velocity =preserve_velocity ,
        rebuild_air =rebuild_air ,
        air_grid_step_mm =air_grid_step_mm ,
        air_padding_mm =air_padding_mm ,
        )

    def rebuild_air_field (
    self ,
    air_grid_step_mm :float |None =None ,
    air_padding_mm :float |None =None ,
    position_xyz :np .ndarray |None =None ,
    size_xyz :np .ndarray |None =None ,
    )->None :
        'Rebuilds the 3D mesh of the acoustic field according to the FE topology.\n\n        air_grid_step_mm:\n            Air mesh pitch in mm. If None, the current mode (auto/previously set) is used.\n        air_padding_mm:\n            Air offset around the FE geometry in mm. If None, the current one is used.\n        position_xyz, size_xyz:\n            Explicit FE coordinates and sizes for extent. If specified, used instead of self.position/self.element_size_xyz.'
        if air_grid_step_mm is not None :
            if air_grid_step_mm <=0.0 :
                raise ValueError ('air_grid_step_mm must be > 0')
            self .air_grid_step =float (air_grid_step_mm )*1e-3
        if air_padding_mm is not None :
            if air_padding_mm <0.0 :
                raise ValueError ('air_padding_mm must be >= 0')
            self .air_padding =float (air_padding_mm )*1e-3

        # If topology came from explicit payload and no new geometry requested, keep payload field.
        if self ._air_topology_from_payload and position_xyz is None and size_xyz is None :
            if self .n_air_cells >0 :
                self ._allocate_air_buffers ()
            return

        self ._air_topology_from_payload =False
        self ._configure_air_field_grid (position_xyz =position_xyz ,size_xyz =size_xyz )
        if self .n_air_cells >0 :
            self ._allocate_air_buffers ()
            self ._air_global_size =((self .n_air_cells +self ._local_size -1 )//self ._local_size )*self ._local_size
        else :
            self .air_neighbors =np .full ((0 ,FACE_DIRS ),-1 ,dtype =np .int32 )
            self .air_neighbor_absorb_u8 =np .zeros ((0 ,FACE_DIRS ),dtype =np .uint8 )
            self .air_boundary_mask_elements =np .zeros (0 ,dtype =np .int32 )
            self ._air_force_external .fill (0.0 )
            cl .enqueue_copy (self .queue ,self ._buf_air_force_external ,self ._air_force_external )
            self .queue .finish ()

    def set_custom_topology (
    self ,
    element_position_xyz :np .ndarray ,
    element_size_xyz :np .ndarray ,
    neighbors :np .ndarray ,
    material_index :np .ndarray |None =None ,
    boundary_mask_elements :np .ndarray |None =None ,
    visual_shape :tuple [int ,int ]|None =None ,
    preserve_velocity :bool =False ,
    rebuild_air :bool =True ,
    air_grid_step_mm :float |None =None ,
    air_padding_mm :float |None =None ,
    air_element_position_xyz :np .ndarray |None =None ,
    air_element_size_xyz :np .ndarray |None =None ,
    air_neighbors :np .ndarray |None =None ,
    air_neighbor_absorb_u8 :np .ndarray |None =None ,
    air_boundary_mask_elements :np .ndarray |None =None ,
    solid_to_air_index :np .ndarray |None =None ,
    solid_to_air_index_plus :np .ndarray |None =None ,
    solid_to_air_index_minus :np .ndarray |None =None ,
    air_grid_shape :np .ndarray |None =None ,
    membrane_mask_elements :np .ndarray |None =None ,
    sensor_mask_elements :np .ndarray |None =None ,
    )->None :
        'Completely sets the topology of the FE from the outside (for all n_elements).\n\n        Restriction: the number of CIs does not change (must be equal to the current self.n_elements).'
        pos =np .asarray (element_position_xyz ,dtype =np .float64 )
        size =np .asarray (element_size_xyz ,dtype =np .float64 )
        nbh =np .asarray (neighbors ,dtype =np .int32 )
        if pos .shape !=(self .n_elements ,3 ):
            raise ValueError (f"element_position_xyz must have shape [{self .n_elements }, 3]")
        if size .shape !=(self .n_elements ,3 ):
            raise ValueError (f"element_size_xyz must have shape [{self .n_elements }, 3]")
        if nbh .shape !=(self .n_elements ,FACE_DIRS ):
            raise ValueError (f"neighbors must have shape [{self .n_elements }, {FACE_DIRS }]")
        if np .any (size <=0.0 ):
            raise ValueError ('element_size_xyz should only contain positive sizes')
        if np .any ((nbh <-1 )|(nbh >=self .n_elements )):
            raise ValueError ('neighbors contains indexes outside the range [-1, n_elements)')

        self .position [0 ::self .dof_per_element ]=pos [:,0 ]
        self .position [1 ::self .dof_per_element ]=pos [:,1 ]
        self .position [2 ::self .dof_per_element ]=pos [:,2 ]
        self .element_size_xyz =size .copy ()
        self ._static_fe_dirty =True
        self .set_neighbors_topology (nbh )

        if material_index is not None :
            self .set_element_material_index (material_index )
        if boundary_mask_elements is not None :
            self .set_boundary_mask (boundary_mask_elements )
        if membrane_mask_elements is not None :
            mm =np .asarray (membrane_mask_elements ,dtype =np .int32 ).ravel ()
            if mm .size !=self .n_elements :
                raise ValueError ('membrane_mask_elements must be of size n_elements')
            self .membrane_mask =(mm !=0 ).astype (np .int32 )
        else :
            self .membrane_mask =(self .material_index [:self .n_elements ].astype (np .int32 )==int (MAT_MEMBRANE )).astype (np .int32 )
        if sensor_mask_elements is not None :
            sm =np .asarray (sensor_mask_elements ,dtype =np .int32 ).ravel ()
            if sm .size !=self .n_elements :
                raise ValueError ('sensor_mask_elements must be of size n_elements')
            self ._sensor_mask =(sm !=0 )
        else :
            self ._update_sensor_mask ()
        self ._fe_air_coupling_mask =np .asarray ((self .membrane_mask [:self .n_elements ]!=0 )|self ._sensor_mask [:self .n_elements ],dtype =np .uint8 )

        if not preserve_velocity :
            self .velocity .fill (0.0 )
            self ._velocity_prev .fill (0.0 )
            self ._velocity_delta .fill (0.0 )

        if visual_shape is not None :
            if len (visual_shape )!=2 :
                raise ValueError ('visual_shape should be tuple (ny, nx)')
            v_ny ,v_nx =int (visual_shape [0 ]),int (visual_shape [1 ])
            sensor_idx =np .flatnonzero (self ._sensor_mask ).astype (np .int32 )
            if v_ny <=0 or v_nx <=0 or v_ny *v_nx !=sensor_idx .size :
                raise ValueError ('visual_shape must contain positive dimensions and ny*nx == n_sensor_elements (MAT_SENSOR)')
            self ._topology_is_rect_grid =True 
            self ._visual_shape =(v_ny ,v_nx )
            self ._visual_element_indices =sensor_idx 
        else :
            self ._topology_is_rect_grid =False 
            self ._visual_shape =None 
            self ._visual_element_indices =None 
        self ._sync_visualization_flag ()
        self ._update_center_index ()

        has_air_payload =(air_element_position_xyz is not None )or (air_element_size_xyz is not None )or (air_neighbors is not None )
        if has_air_payload :
            if air_element_position_xyz is None or air_element_size_xyz is None :
                raise ValueError ('air_element_position_xyz and air_element_size_xyz must be provided together')
            self ._air_topology_from_payload =True
            self ._configure_air_topology_payload (
            air_element_position_xyz =np .asarray (air_element_position_xyz ,dtype =np .float64 ),
            air_element_size_xyz =np .asarray (air_element_size_xyz ,dtype =np .float64 ),
            air_neighbors =air_neighbors ,
            air_neighbor_absorb_u8 =air_neighbor_absorb_u8 ,
            air_boundary_mask_elements =air_boundary_mask_elements ,
            solid_to_air_index =solid_to_air_index ,
            solid_to_air_index_plus =solid_to_air_index_plus ,
            solid_to_air_index_minus =solid_to_air_index_minus ,
            air_grid_shape =air_grid_shape ,
            )
            if self .n_air_cells >0 :
                self ._allocate_air_buffers ()
                self ._air_global_size =((self .n_air_cells +self ._local_size -1 )//self ._local_size )*self ._local_size
                self ._log_air_csr_hot_rows ("set_custom_topology")
        elif rebuild_air :
            self .rebuild_air_field (
            air_grid_step_mm =air_grid_step_mm ,
            air_padding_mm =air_padding_mm ,
            position_xyz =pos ,
            size_xyz =size ,
            )

    def compute_air_force_center (self )->float :
        if self ._air_force_external .size !=self .n_dof :
            return 0.0
        idx =int (self .center_idx )*self .dof_per_element +2
        if idx <0 or idx >=self ._air_force_external .size :
            return 0.0
        return float (self ._air_force_external [idx ])

    def _snapshot_disp_map_frame (self )->np .ndarray :
        'Current frame uz for history_disp_all (same logic as in step for record_history).'
        uz_all =self .position [2 :self .n_elements *6 :6 ]
        if (
        self ._topology_is_rect_grid 
        and self ._visual_shape is not None 
        and self ._visual_element_indices is not None 
        ):
            return uz_all [self ._visual_element_indices ].reshape (self ._visual_shape ).copy ()
        if np .any (self ._sensor_mask ):
            return uz_all [self ._sensor_mask ].copy ()
        return uz_all .copy ()

    def _estimate_required_fe_substeps (self ,dt :float )->tuple [int ,float ]:
        """Estimate FE explicit substeps from c~sqrt(E/rho) and minimal element size."""
        if dt <=0.0 or self .n_elements <=0 :
            return 1 ,float ("inf")
        try :
            free_mask =(np .asarray (self .boundary_mask_elements [:self .n_elements ],dtype =np .int32 )==0 )
            if not np .any (free_mask ):
                return 1 ,float ("inf")
            mats =np .asarray (self .material_index [:self .n_elements ],dtype =np .int32 )[free_mask ]
            if mats .size ==0 :
                return 1 ,float ("inf")
            uniq =np .unique (mats )
            rho =np .asarray (self .material_props [uniq ,MAT_PROP_DENSITY ],dtype =np .float64 )
            e_par =np .asarray (self .material_props [uniq ,MAT_PROP_E_PARALLEL ],dtype =np .float64 )
            e_perp =np .asarray (self .material_props [uniq ,MAT_PROP_E_PERP ],dtype =np .float64 )
            e_max =np .maximum (e_par ,e_perp )
            valid =(rho >1e-12 )&(e_max >0.0 )&np .isfinite (rho )&np .isfinite (e_max )
            if not np .any (valid ):
                return 1 ,float ("inf")
            c_max =float (np .max (np .sqrt (e_max [valid ]/rho [valid ])))
            if not np .isfinite (c_max )or c_max <=0.0 :
                return 1 ,float ("inf")
            # Less aggressive than strict min(size): use characteristic free-element length.
            size_free =np .asarray (self .element_size_xyz [:self .n_elements ],dtype =np .float64 )[free_mask ]
            h_char =np .cbrt (np .maximum (size_free [:,0 ]*size_free [:,1 ]*size_free [:,2 ],1e-30 ))
            h_eff =float (np .percentile (h_char ,5.0 )) if h_char .size >0 else float ("nan")
            if not np .isfinite (h_eff )or h_eff <=0.0 :
                return 1 ,float ("inf")
            dt_max =float (self .fe_stability_safety )*h_eff /c_max
            if not np .isfinite (dt_max )or dt_max <=0.0 :
                return 1 ,float ("inf")
            n_sub =max (1 ,int (np .ceil (dt /dt_max )))
            cap =max (1 ,int (self .fe_subcycle_cap ))
            if n_sub >cap :
                n_sub =cap
            return n_sub ,dt_max
        except Exception :
            return 1 ,float ("inf")

    def _debug_trace_element_host (self ,elem :int ,step_idx :int ,*,label :str )->None :
        """Host-side force decomposition for one FE element (debug-only helper)."""
        if elem <0 or elem >=self .n_elements :
            return
        tiny =1e-20
        center =np .asarray (self .position [elem *6 :elem *6 +3 ],dtype =np .float64 )
        vel =np .asarray (self .velocity [elem *6 :elem *6 +3 ],dtype =np .float64 )
        size =np .asarray (self .element_size_xyz [elem ,:3 ],dtype =np .float64 )
        mat =int (self .material_index [elem ])
        bnd =int (self .boundary_mask_elements [elem ])
        if bnd !=0 :
            return
        force_ext =np .asarray (self .force_external [elem *6 :elem *6 +3 ],dtype =np .float64 )
        force_air =np .asarray (self ._air_force_external [elem *6 :elem *6 +3 ],dtype =np .float64 )
        E_parallel =float (self .material_props [mat ,MAT_PROP_E_PARALLEL ])
        E_perp =float (self .material_props [mat ,MAT_PROP_E_PERP ])
        Cd_me =float (self .material_props [mat ,MAT_PROP_CD ])
        eta_me =float (self .material_props [mat ,MAT_PROP_ETA_VISC ])
        density =float (self .material_props [mat ,MAT_PROP_DENSITY ])
        F_elastic =np .zeros (3 ,dtype =np .float64 )
        F_iface =np .zeros (3 ,dtype =np .float64 )

        def _face_normal (d :int )->np .ndarray :
            if d ==0 : return np .array ([1.0 ,0.0 ,0.0 ],dtype =np .float64 )
            if d ==1 : return np .array ([-1.0 ,0.0 ,0.0 ],dtype =np .float64 )
            if d ==2 : return np .array ([0.0 ,1.0 ,0.0 ],dtype =np .float64 )
            if d ==3 : return np .array ([0.0 ,-1.0 ,0.0 ],dtype =np .float64 )
            if d ==4 : return np .array ([0.0 ,0.0 ,1.0 ],dtype =np .float64 )
            return np .array ([0.0 ,0.0 ,-1.0 ],dtype =np .float64 )

        def _face_area_from_size (d :int ,size_e :np .ndarray )->float :
            sx ,sy ,sz =float (size_e [0 ]),float (size_e [1 ]),float (size_e [2 ])
            if d <2 :
                return sy *sz
            if d <4 :
                return sx *sz
            return sx *sy

        print (
            f"[debug][elem][{label}] step={step_idx } elem={elem } mat={mat } "
            f"pos={center .tolist ()} vel={vel .tolist ()} size={size .tolist ()}"
        )
        for d in range (FACE_DIRS ):
            nb =int (self .neighbors [elem ,d ])
            has_nb =(0 <=nb <self .n_elements )
            mat_nb =int (self .material_index [nb ]) if has_nb else mat
            law =int (self .laws [mat ,mat_nb ]) if has_nb else int (LAW_SOLID_SPRING )
            nrm =_face_normal (d )
            if has_nb and law ==int (LAW_SOLID_SPRING ):
                center_nb =np .asarray (self .position [nb *6 :nb *6 +3 ],dtype =np .float64 )
                vel_nb =np .asarray (self .velocity [nb *6 :nb *6 +3 ],dtype =np .float64 )
                size_nb =np .asarray (self .element_size_xyz [nb ,:3 ],dtype =np .float64 )
                dcc =center -center_nb
                center_len =float (np .linalg .norm (dcc ))
                if center_len >tiny :
                    direction =dcc /center_len
                else :
                    direction =nrm
                if d <2 :
                    rest_len =0.5 *(float (size [0 ])+float (size_nb [0 ]))
                elif d <4 :
                    rest_len =0.5 *(float (size [1 ])+float (size_nb [1 ]))
                else :
                    rest_len =0.5 *(float (size [2 ])+float (size_nb [2 ]))
                strain =((center_len -rest_len )/rest_len ) if rest_len >tiny else 0.0
                k_axial_x =E_parallel *float (size [2 ])*float (size [1 ])/(float (size [0 ])+tiny )
                k_axial_y =E_perp *float (size [2 ])*float (size [0 ])/(float (size [1 ])+tiny )
                k_axial_z =E_perp *float (size [0 ])*float (size [1 ])/(float (size [2 ])+tiny )
                k_axial_dir =k_axial_x if d <2 else (k_axial_y if d <4 else k_axial_z )
                if k_axial_dir >tiny :
                    k_soft_dir =k_axial_dir /(float (self .stiffness_ratio )+tiny )
                    k_stiff_dir =k_axial_dir
                else :
                    k_soft_dir =0.0
                    k_stiff_dir =0.0
                arg =-(strain -float (self .strain_transition ))/(float (self .strain_width )+tiny )
                s =1.0 /(1.0 +np .exp (arg ))
                k_eff =k_soft_dir *(1.0 -s )+k_stiff_dir *s
                force_el =k_eff *(center_len -rest_len )
                force_tension =0.0
                force_mag =force_el +force_tension
                f =-force_mag *direction
                eta_nb =float (self .material_props [mat_nb ,MAT_PROP_ETA_VISC ])
                eta_eff =0.5 *(eta_me +eta_nb )
                if eta_eff >0.0 :
                    face_area =_face_area_from_size (d ,size )
                    c_link =eta_eff *face_area /(rest_len +tiny )
                    v_rel =vel -vel_nb
                    v_rel_n =float (np .dot (v_rel ,direction ))
                    f =f +(-c_link *v_rel_n )*direction
                F_elastic +=f
                print (
                    f"  d={d } spring nb={nb } mat_nb={mat_nb } "
                    f"center_len={center_len :.3e} rest_len={rest_len :.3e} strain={strain :.3e} "
                    f"k_eff={k_eff :.3e} F={f .tolist ()}"
                )
            else :
                vn =float (np .dot (vel ,nrm ))
                if has_nb :
                    vel_nb =np .asarray (self .velocity [nb *6 :nb *6 +3 ],dtype =np .float64 )
                    vn -=float (np .dot (vel_nb ,nrm ))
                face_area =_face_area_from_size (d ,size )
                a_eff =float (np .sqrt (face_area /np .pi ))
                v_abs =abs (vn )
                rho_eff =float (self .material_props [mat_nb ,MAT_PROP_DENSITY ]) if has_nb else float (self .rho_air )
                if rho_eff <tiny :
                    rho_eff =float (self .rho_air )
                cd_nb =float (self .material_props [mat_nb ,MAT_PROP_CD ]) if has_nb else Cd_me
                cd_eff =0.5 *(Cd_me +cd_nb )
                Re =rho_eff *v_abs *(2.0 *a_eff )/(float (self .mu_air )+tiny )
                transition =1.0 /(1.0 +np .exp (-(Re -100.0 )/50.0 ))
                c_linear =6.0 *np .pi *float (self .mu_air )*a_eff
                c_quad =0.5 *rho_eff *cd_eff *face_area *v_abs
                c_eff =(1.0 -transition )*c_linear +transition *c_quad
                f =(-c_eff *vn )*nrm
                F_iface +=f
                print (
                    f"  d={d } iface nb={nb } mat_nb={mat_nb } law={law } "
                    f"vn={vn :.3e} Re={Re :.3e} c_eff={c_eff :.3e} F={f .tolist ()}"
                )
        F_total =force_ext +force_air +F_elastic +F_iface
        mass =max (density *float (size [0 ])*float (size [1 ])*float (size [2 ]),1e-18 )
        acc =F_total /(mass +1e-18 )
        print (f"[debug][elem][{label}] F_ext={force_ext .tolist ()} F_air={force_air .tolist ()}")
        print (f"[debug][elem][{label}] F_elastic={F_elastic .tolist ()} F_iface={F_iface .tolist ()}")
        print (f"[debug][elem][{label}] F_total={F_total .tolist ()} mass={mass :.3e} acc={acc .tolist ()}")

    def step (
    self ,
    dt :float ,
    pressure_pa :float |np .ndarray ,
    step_idx :int =-1 ,
    first_bad_elem_out :Optional [np .ndarray ]=None ,
    debug_elem :int =-1 ,
    debug_buf_out :Optional [np .ndarray ]=None ,
    debug_silent :bool =False ,
    append_history :bool =True ,
    )->None :
        # Early preflight: abort step before any GPU work if CSR topology is invalid.
        if getattr (self ,'n_air_cells',0 )>0 and hasattr (self ,'_air_inject_csr_offsets'):
            self ._air_csr_validate_step_counter +=1
            self ._validate_air_inject_csr (f"step_preflight step={step_idx }",full_scan =False )
            full_every =max (1 ,int (getattr (self ,"air_inject_csr_validate_full_every_steps",1 )))
            need_full =bool (getattr (self ,"_air_csr_dirty",False ))or ((self ._air_csr_validate_step_counter %full_every )==0 )
            if need_full :
                self ._validate_air_inject_csr (f"step_preflight_full step={step_idx }",full_scan =True )
                self ._air_csr_dirty =False
        p_array_full :np .ndarray |None =None
        pressure_scalar =0.0
        use_force_override =self ._mode_uses_full_force_override ()
        use_velocity_override =self ._mode_uses_velocity_override ()
        if np .isscalar (pressure_pa ):
            pressure_scalar =float (pressure_pa )
        else :
            p_arr =np .asarray (pressure_pa ,dtype =np .float64 ).ravel ()
            if p_arr .size ==1 :
                pressure_scalar =float (p_arr [0 ])
            else :
                p_array_full =p_arr
        self ._upload_static_fe_buffers_if_dirty ()
        if use_velocity_override :
            self ._apply_velocity_external_override (p_array_full if p_array_full is not None else pressure_scalar )
            pressure_scalar =0.0
            use_force_override =False
        elif p_array_full is not None :
            # Compatibility path for per-element external pressure vectors.
            self ._build_force_external (p_array_full )
            use_force_override =True
        elif use_force_override :
            self ._build_force_external (pressure_scalar )
        use_debug =self .kernel_debug and (debug_elem >=0 )
        use_validate =first_bad_elem_out is not None
        params_bytes =self ._params_bytes (dt ,step_idx =max (0 ,step_idx ),debug_elem =debug_elem if use_debug else -1 )

        if self ._buf_params is None :
            mf =cl .mem_flags 
            self ._buf_params =cl .Buffer (self .ctx ,mf .READ_ONLY |mf .COPY_HOST_PTR ,hostbuf =params_bytes )
        else :
            cl .enqueue_copy (self .queue ,self ._buf_params ,params_bytes )

        # Copying the input data to the device
        cl .enqueue_copy (self .queue ,self ._buf_position ,self .position )
        cl .enqueue_copy (self .queue ,self ._buf_velocity ,self .velocity )
        if use_force_override :
            cl .enqueue_copy (self .queue ,self ._buf_force_external ,self .force_external )
            self ._force_override_active =True
        elif self ._force_override_active :
            self .force_external .fill (0.0 )
            cl .enqueue_copy (self .queue ,self ._buf_force_external ,self .force_external )
            self ._force_override_active =False
        # RK4: freeze p on GPU; build air coupling force once at (x0,v0) on host (matches GPU after copies above).
        # Stages k2–k4 reuse _buf_air_force_external (no GPU→host FE pulls). After finalize: full air step.

        if use_validate :
            first_bad_init =np .array ([0x7FFFFFFF ],dtype =np .int32 )
            cl .enqueue_copy (self .queue ,self ._buf_first_bad ,first_bad_init )
            meta_init =np .array ([0 ],dtype =np .int32 )
            cl .enqueue_copy (self .queue ,self ._buf_first_bad_meta ,meta_init )
            neighbor_init =np .array ([-1 ],dtype =np .int32 )
            cl .enqueue_copy (self .queue ,self ._buf_first_bad_neighbor_elem ,neighbor_init )
            interface_dir_init =np .array ([-1 ],dtype =np .int32 )
            cl .enqueue_copy (self .queue ,self ._buf_first_bad_interface_dir ,interface_dir_init )

            # RK4 integration fully on device (host only orchestrates kernel launches)
        cl .enqueue_copy (self .queue ,self ._buf_position_0 ,self ._buf_position )
        cl .enqueue_copy (self .queue ,self ._buf_velocity_0 ,self ._buf_velocity )
        self ._run_air_coupling_for_acceleration (dt ,sync_fe_from_gpu =False )

        # Stage 1: (x0, v0) -> a1
        self ._evaluate_acceleration (
        self ._buf_params ,
        dt ,
        pressure_scalar ,
        self ._buf_acc ,
        validate_finite =use_validate ,
        refresh_air_coupling =False ,
        acc_stage_id =1 ,
        )

        # Stage 2 state: x1 = x0 + 0.5*dt*v0, v1 = v0 + 0.5*dt*a1
        self ._kernel_rk4_stage_state .set_args (
        self ._buf_position_0 ,
        self ._buf_velocity_0 ,
        self ._buf_velocity_0 ,
        self ._buf_acc ,
        self ._buf_boundary ,
        np .int32 (self .n_elements ),
        np .float64 (dt ),
        np .float64 (0.5 ),
        self ._buf_position ,
        self ._buf_velocity ,
        self ._buf_velocity_k2 ,
        )
        cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_rk4_stage_state ,(self ._global_size ,),(self ._local_size ,))

        # Stage 2 acceleration: a2
        self ._evaluate_acceleration (
        self ._buf_params ,
        dt ,
        pressure_scalar ,
        self ._buf_acc_k2 ,
        validate_finite =use_validate ,
        refresh_air_coupling =False ,
        acc_stage_id =2 ,
        )

        # Stage 3 state: x2 = x0 + 0.5*dt*v1, v2 = v0 + 0.5*dt*a2
        self ._kernel_rk4_stage_state .set_args (
        self ._buf_position_0 ,
        self ._buf_velocity_0 ,
        self ._buf_velocity_k2 ,
        self ._buf_acc_k2 ,
        self ._buf_boundary ,
        np .int32 (self .n_elements ),
        np .float64 (dt ),
        np .float64 (0.5 ),
        self ._buf_position ,
        self ._buf_velocity ,
        self ._buf_velocity_k3 ,
        )
        cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_rk4_stage_state ,(self ._global_size ,),(self ._local_size ,))

        # Stage 3 acceleration: a3
        self ._evaluate_acceleration (
        self ._buf_params ,
        dt ,
        pressure_scalar ,
        self ._buf_acc_k3 ,
        validate_finite =use_validate ,
        refresh_air_coupling =False ,
        acc_stage_id =3 ,
        )

        # Stage 4 state: x3 = x0 + dt*v2, v3 = v0 + dt*a3
        self ._kernel_rk4_stage_state .set_args (
        self ._buf_position_0 ,
        self ._buf_velocity_0 ,
        self ._buf_velocity_k3 ,
        self ._buf_acc_k3 ,
        self ._buf_boundary ,
        np .int32 (self .n_elements ),
        np .float64 (dt ),
        np .float64 (1.0 ),
        self ._buf_position ,
        self ._buf_velocity ,
        self ._buf_velocity_k4 ,
        )
        cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_rk4_stage_state ,(self ._global_size ,),(self ._local_size ,))

        # Stage 4 acceleration: a4
        self ._evaluate_acceleration (
        self ._buf_params ,
        dt ,
        pressure_scalar ,
        self ._buf_acc_k4 ,
        validate_finite =use_validate ,
        refresh_air_coupling =False ,
        acc_stage_id =4 ,
        )

        # Final RK4 combine
        self ._kernel_rk4_finalize .set_args (
        self ._buf_position_0 ,
        self ._buf_velocity_0 ,
        self ._buf_velocity_k2 ,
        self ._buf_velocity_k3 ,
        self ._buf_velocity_k4 ,
        self ._buf_acc ,
        self ._buf_acc_k2 ,
        self ._buf_acc_k3 ,
        self ._buf_acc_k4 ,
        self ._buf_boundary ,
        np .int32 (self .n_elements ),
        np .float64 (dt ),
        self ._buf_position ,
        self ._buf_velocity ,
        )
        cl .enqueue_nd_range_kernel (self .queue ,self ._kernel_rk4_finalize ,(self ._global_size ,),(self ._local_size ,))

        # Final (x,v) on GPU → host so that FE↔air maps and full wave step match t_{n+1}.
        self ._sync_fe_state_from_gpu_for_air_maps ()
        self ._run_air_coupling (dt ,pressure_pa )

        # Copy the result back
        cl .enqueue_copy (self .queue ,self .position ,self ._buf_position )
        cl .enqueue_copy (self .queue ,self .velocity ,self ._buf_velocity )
        self ._velocity_delta [:]=self .velocity -self ._velocity_prev 
        self ._velocity_prev [:]=self .velocity 
        if use_validate :
            cl .enqueue_copy (self .queue ,first_bad_elem_out ,self ._buf_first_bad )
            cl .enqueue_copy (self .queue ,self ._first_bad_meta_host ,self ._buf_first_bad_meta )
            cl .enqueue_copy (self .queue ,self ._first_bad_neighbor_elem_host ,self ._buf_first_bad_neighbor_elem )
            cl .enqueue_copy (self .queue ,self ._first_bad_interface_dir_host ,self ._buf_first_bad_interface_dir )
        if use_debug and debug_buf_out is not None :
            cl .enqueue_copy (self .queue ,debug_buf_out ,self ._buf_debug )
        self .queue .finish ()
        if use_validate and first_bad_elem_out [0 ]>=self .n_elements :
            first_bad_elem_out [0 ]=-1 
        if use_debug and debug_buf_out is not None and not debug_silent :
            _print_opencl_trace (debug_buf_out ,debug_elem ,step_idx )

        if append_history :
            self .history_disp_center .append (float (self .position [self .center_dof ]))
            if self ._record_history :
                self .history_disp_all .append (self ._snapshot_disp_map_frame ())

    def simulate (
    self ,
    pressure_profile :np .ndarray ,
    dt :float ,
    record_history :bool =False ,
    check_air_resistance :bool =False ,
    validate_steps :bool =True ,
    reset_state :bool =True ,
    show_progress :bool =True ,
    progress_every_pct :float =5.0 ,
    air_pressure_history_every_steps :int =1 ,
    stop_check :Callable [[],bool ]|None =None ,
    )->np .ndarray :
        'reset_state: if True, resets the position/velocity at the beginning.\n        If False, uses the current position/velocity (needed for natural frequency testing:\n        get_numerical_natural_frequency sets the rest position and momentum; without reset_state they are not overwritten).\n        Previously, erasure gave a start from (0,0,0) for all elements -> “explosion” and 0 Hz on the spectrogram.\n\n        When record_history=True, frame t=0 is added to the beginning (before the first step): zero pressure\n        air on the host after reset and current offsets so that frame 0 does not coincide with the state after the first dt.'
        if pressure_profile .ndim not in (1 ,2 ):
            raise ValueError ('pressure_profile should be 1D [n_steps] or 2D [n_steps, n_elements]')
        n_steps =pressure_profile .shape [0 ]
        if progress_every_pct <=0.0 :
            raise ValueError ('progress_every_pct must be > 0')
        if reset_state :
            self .velocity .fill (0.0 )
            self ._velocity_prev .fill (0.0 )
            self ._velocity_delta .fill (0.0 )
            self ._reset_air_field ()
        self .force_external .fill (0.0 )
        self .history_disp_center =[]
        self .history_disp_all =[]
        self .history_air_center_xz =[]
        self .history_air_pressure_xy_center_z =[]
        self .history_air_pressure_step =max (1 ,int (air_pressure_history_every_steps ))
        last_air_pressure_history_step =-1
        self ._record_history =record_history 
        self ._last_max_uz_um =0.0 
        if record_history :
            self .history_disp_center .append (float (self .position [self .center_dof ]))
            self .history_disp_all .append (self ._snapshot_disp_map_frame ())
            if self .n_air_cells >0 :
                if reset_state :
                    p0 =float (self .air_initial_uniform_pressure_pa )
                    flat0 =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
                    self .history_air_pressure_xy_center_z .append (
                    self ._air_center_xy_slice_from_flat (flat0 )
                    )
                    self ._log_air_pressure_metrics (0 ,flat0 ,"history_save_t0")
                else :
                    cl .enqueue_copy (self .queue ,self .air_pressure_curr ,self ._buf_air_curr )
                    self .queue .finish ()
                    flat0 =self .air_pressure_curr .copy ()
                    self .history_air_pressure_xy_center_z .append (
                    self ._air_center_xy_slice_from_flat (flat0 )
                    )
                    self ._log_air_pressure_metrics (0 ,flat0 ,"history_save_t0")
                last_air_pressure_history_step =0
            if self .n_air_cells >0 :
                if reset_state :
                    p0 =float (self .air_initial_uniform_pressure_pa )
                    flat0 =np .full (self .n_air_cells ,p0 ,dtype =np .float64 )
                    self .history_air_center_xz .append (
                    self ._air_center_xz_slice_from_flat (flat0 )
                    )
                else :
                    cl .enqueue_copy (self .queue ,self .air_pressure_curr ,self ._buf_air_curr )
                    self .queue .finish ()
                    self .history_air_center_xz .append (
                    self ._air_center_xz_slice_from_flat (self .air_pressure_curr .copy ())
                    )
        if validate_steps and not self .kernel_debug :
            print ('[validate] Omission: NaN/Inf checking in the kernel is only available when kernel_debug=True.')
            validate_steps =False 

        first_bad =np .array ([-1 ],dtype =np .int32 )if validate_steps else None 
        self ._first_bad_elem =-1 
        self ._first_bad_step =-1 
        first_huge_vel_reported =False
        hard_trace_elem =int (getattr (self ,"debug_trace_elem",-1 )) if self .kernel_debug else -1
        hard_trace_from =int (getattr (self ,"debug_trace_step_from",0 )) if self .kernel_debug else 0
        hard_trace_to =int (getattr (self ,"debug_trace_step_to",-1 )) if self .kernel_debug else -1
        hard_trace_buf =np .zeros (self ._DEBUG_BUF_DOUBLES ,dtype =np .float64 ) if hard_trace_elem >=0 else None
        fe_substeps ,fe_dt_max =self ._estimate_required_fe_substeps (dt )
        dt_fe =dt /float (fe_substeps )
        if fe_substeps >1 and not self ._fe_subcycle_summary_shown :
            msg =(
                f"[fe] subcycling: {fe_substeps } explicit FE substeps per step "
                f"(dt_fe={dt_fe :.3e} s; estimated stable dt<=~{fe_dt_max :.3e} s, safety={self .fe_stability_safety :.3f})."
            )
            if fe_substeps >=int (self .fe_subcycle_cap ):
                msg +=" [cap reached]"
            print (msg )
            self ._fe_subcycle_summary_shown =True
        report_step =max (1 ,int (np .ceil (n_steps *progress_every_pct /100.0 )))if n_steps >0 else 1 
        next_report =report_step 
        sim_t0 =time .perf_counter ()
        executed_steps =0 
        if show_progress and n_steps >0 :
            print (f"[simulate] start: steps={n_steps }, dt={dt :.3e} s")
        for step_idx in range (n_steps ):
            if stop_check and stop_check ():
                if show_progress and n_steps >0 :
                    print (f"[simulate] stopped by user at {step_idx }/{n_steps } steps")
                break 
            if pressure_profile .ndim ==1 :
                p =pressure_profile [step_idx ]
            else :
                p =pressure_profile [step_idx ]
            use_hard_trace =(
                hard_trace_elem >=0
                and hard_trace_buf is not None
                and hard_trace_from <=step_idx <=hard_trace_to
            )
            if use_hard_trace :
                self ._debug_trace_element_host (hard_trace_elem ,step_idx ,label ="pre")
            cancelled =False
            for sub_idx in range (fe_substeps ):
                append_hist =(sub_idx ==fe_substeps -1 )
                try :
                    self .step (
                        dt_fe ,
                        p ,
                        step_idx =step_idx ,
                        first_bad_elem_out =first_bad ,
                        debug_elem =(hard_trace_elem if use_hard_trace else -1 ),
                        debug_buf_out =(hard_trace_buf if use_hard_trace else None ),
                        debug_silent =False ,
                        append_history =append_hist ,
                    )
                except RuntimeError as e :
                    msg =str (e )
                    if "CSR"in msg or "[air]"in msg or "air inject"in msg :
                        print (f"[simulate] cancelled at step {step_idx }: {msg }")
                        cancelled =True
                        break
                    raise
                if validate_steps and first_bad is not None and first_bad [0 ]>=0 :
                    break
            if cancelled :
                break
            if use_hard_trace :
                self ._debug_trace_element_host (hard_trace_elem ,step_idx ,label ="post")
            executed_steps =step_idx +1 
            if (executed_steps %self .history_air_pressure_step )==0 :
                if self .n_air_cells >0 :
                    flat_hist =self .air_pressure_curr .copy ()
                    self .history_air_pressure_xy_center_z .append (
                    self ._air_center_xy_slice_from_flat (flat_hist )
                    )
                    self ._log_air_pressure_metrics (executed_steps ,flat_hist ,"history_save")
                    last_air_pressure_history_step =executed_steps
            if show_progress and (executed_steps >=next_report or executed_steps ==n_steps ):
                elapsed =time .perf_counter ()-sim_t0 
                progress =100.0 *executed_steps /max (n_steps ,1 )
                eta =elapsed *(n_steps -executed_steps )/max (executed_steps ,1 )
                print (
                f"[simulate] {executed_steps }/{n_steps } ({progress :5.1f}%) "
                f"elapsed={elapsed :7.2f}s eta={eta :7.2f}s"
                )
                next_report +=report_step 
                # Criterion "explosion": max |uz| for all elements (µm)
            uz_all =self .position [2 :self .n_elements *6 :6 ]
            if uz_all .size >0 and np .all (np .isfinite (uz_all )):
                max_uz_um =float (np .max (np .abs (uz_all ))*1e6 )
                if max_uz_um >self ._last_max_uz_um :
                    self ._last_max_uz_um =max_uz_um 
            if self .kernel_debug :
                try :
                    v3 =self .velocity [:self .n_elements *6 ].reshape (self .n_elements ,6 )[:,:3 ]
                    v_abs =np .abs (v3 )
                    vmax =float (np .nanmax (v_abs )) if v_abs .size >0 else 0.0
                    if (not first_huge_vel_reported )and np .isfinite (vmax )and vmax >1e100 :
                        flat =v_abs .reshape (-1 )
                        k =int (np .argmax (flat ))
                        elem_h =k //3
                        comp_h =k %3
                        first_huge_vel_reported =True
                        print (
                            f"[debug][fe] huge finite velocity at step {step_idx }: "
                            f"max|v|={vmax :.3e} m/s, elem={elem_h }, comp={comp_h }, "
                            f"mat={int (self .material_index [elem_h ])}, bnd={int (self .boundary_mask_elements [elem_h ])}"
                        )
                        print (f"[debug][fe] v(elem)={v3 [elem_h ].tolist ()}")
                        if hasattr (self ,"neighbors")and self .neighbors .shape [0 ]>elem_h :
                            nb =self .neighbors [elem_h ]
                            print (f"[debug][fe] neighbors={nb .tolist ()}")
                except Exception :
                    pass
            if validate_steps and first_bad is not None and first_bad [0 ]>=0 :
                elem =int (first_bad [0 ])
                self ._first_bad_elem =elem 
                self ._first_bad_step =step_idx 
                ix ,iy =elem %self .nx ,elem //self .nx 
                pos =self .position [elem *6 :elem *6 +6 ]
                vel =self .velocity [elem *6 :elem *6 +6 ]
                meta =int (self ._first_bad_meta_host [0 ]) if hasattr (self ,'_first_bad_meta_host') else 0
                stage_id =(meta >>16 )&0xFFFF 
                reason_bits =meta &0xFFFF 
                neighbor_elem =int (self ._first_bad_neighbor_elem_host [0 ]) if hasattr (self ,'_first_bad_neighbor_elem_host') else -1
                interface_dir =int (self ._first_bad_interface_dir_host [0 ]) if hasattr (self ,'_first_bad_interface_dir_host') else -1
                # reason_bits encoding:
                #  pos_me[d] -> bit (0+d), vel_me[d] -> bit (3+d), F[d] -> bit (6+d), a[d] -> bit (9+d)
                reason_parts =[]
                for d in range (3 ):
                    if reason_bits &(1 << (0 +d )): reason_parts .append (f"pos[{d}]")
                    if reason_bits &(1 << (3 +d )): reason_parts .append (f"vel[{d}]")
                    if reason_bits &(1 << (6 +d )): reason_parts .append (f"F[{d}]")
                    if reason_bits &(1 << (9 +d )): reason_parts .append (f"a[{d}]")
                for d in range (3 ):
                    if reason_bits &(1 << (12 +d )): reason_parts .append (f"airF[{d}]")
                if reason_bits &(1 << 15 ): reason_parts .append ("interface_term")
                reason_s =", ".join (reason_parts )
                print (f"  [validate] NaN/Inf at step {step_idx }, элемент {elem } (ix={ix }, iy={iy })")
                print (f"    position: {pos }")
                print (f"    velocity: {vel }")
                if meta !=0 :
                    print (f"    acc_stage_id={stage_id } reason={reason_s } neighbor_elem={neighbor_elem } interface_dir={interface_dir } (meta=0x{meta &0xFFFFFFFF:08X})")
                break 
            if check_air_resistance and step_idx %100 ==0 :
                v_z =float (self .velocity [self .center_dof ])
                f_air =self .compute_air_force_center ()
                print (f"  step {step_idx }: v_center={v_z :.2e} m/s, F_air={f_air :.2e} N")
        if check_air_resistance :
            f_end =self .compute_air_force_center ()
            print (f"  (end) F_air_center={f_end :.2e} N")
        if show_progress and n_steps >0 :
            elapsed_total =time .perf_counter ()-sim_t0 
            print (f"[simulate] done: {executed_steps }/{n_steps } steps in {elapsed_total :.2f}s")
        if self .n_air_cells >0 and executed_steps >0 and last_air_pressure_history_step !=executed_steps :
            flat_final =self .air_pressure_curr .copy ()
            self .history_air_pressure_xy_center_z .append (
            self ._air_center_xy_slice_from_flat (flat_final )
            )
            self ._log_air_pressure_metrics (executed_steps ,flat_final ,"history_save_final")

        return np .asarray (self .history_disp_center ,dtype =np .float64 )

    def get_numerical_natural_frequency (
    self ,
    dt :float =1e-7 ,
    duration :float =0.01 ,
    impulse_velocity_z :float =0.01 ,
    freq_min_hz :float =1.0 ,
    freq_max_hz :float =50_000.0 ,
    refine_peak :bool =True ,
    )->tuple [float ,np .ndarray ,np .ndarray ]:
        'Excitation of the initial center velocity, recording u_center(t), FFT and spectrum peak.\n\n        impulse_velocity_z — initial z velocity of the central element (m/s).\n        refine_peak: if True, refine the peak frequency by parabolic interpolation (sub-bin).\n        Frequency resolution FFT: df = 1/duration; at short duration the peak is quantized into bins.\n        Returns (f_peak_Hz, freq_axis, magnitude_spectrum).'
        self ._set_rest_position ()
        self .velocity .fill (0.0 )
        self .velocity [self .center_dof ]=impulse_velocity_z 
        n_steps =int (round (duration /dt ))
        pressure =np .zeros (n_steps ,dtype =np .float64 )
        hist =self .simulate (
        pressure ,dt =dt ,
        record_history =False ,check_air_resistance =False ,validate_steps =False ,
        reset_state =False ,
        )
        if hist .size <4 :
            return np .nan ,np .array ([]),np .array ([])
        freq =np .fft .rfftfreq (hist .size ,dt )
        spec =np .abs (np .fft .rfft (hist ))
        mask =(freq >=freq_min_hz )&(freq <=freq_max_hz )
        if not np .any (mask ):
            return np .nan ,freq ,spec 
        idx_in_masked =np .argmax (spec [mask ])
        idx =np .where (mask )[0 ][idx_in_masked ]
        f_peak =float (freq [idx ])
        if refine_peak and idx >0 and idx <len (freq )-1 :
        # Parabolic interpolation for sub-bin peak frequency estimation
            y0 ,y1 ,y2 =spec [idx -1 ],spec [idx ],spec [idx +1 ]
            denom =y0 -2 *y1 +y2 
            if abs (denom )>1e-30 :
                delta =0.5 *(y0 -y2 )/denom 
                delta =np .clip (delta ,-0.5 ,0.5 )
                df_bin =freq [1 ]-freq [0 ]if len (freq )>1 else 1.0 /(hist .size *dt )
                f_peak =float (freq [idx ]+delta *df_bin )
        return f_peak ,freq ,spec 

    def plot_time_and_spectrum (
    self ,
    history :np .ndarray |None =None ,
    dt :float =1e-6 ,
    max_freq_hz :float =20_000.0 ,
    )->None :
        hist =np .asarray (history if history is not None else self .history_disp_center ,dtype =float )
        if hist .size ==0 :
            print ('Visualization skipped: no story. Run simulate() before rendering.')
            return 
        t =np .arange (len (hist ))*dt *1e3 

        fig ,(ax1 ,ax2 )=plt .subplots (1 ,2 ,figsize =(12 ,4 ))
        ax1 .plot (t ,hist *1e6 )
        ax1 .set_xlabel ("Time, ms")
        ax1 .set_ylabel ("Center displacement, um")
        ax1 .set_title ("Diaphragm center displacement (OpenCL)")
        ax1 .grid (True ,alpha =0.3 )

        if len (hist )>4 :
            freq =np .fft .fftfreq (len (hist ),dt )
            spec =np .fft .fft (hist )
            mask =(freq >0 )&(freq <=max_freq_hz )
            f_plot =freq [mask ]
            amp =np .abs (spec [mask ])
            amp_norm =amp /(np .max (amp )+1e-30 )
            ax2 .loglog (f_plot ,np .maximum (amp_norm ,1e-10 ))
        ax2 .set_xlim (1.0 ,max_freq_hz )
        ax2 .set_xlabel ("Frequency, Hz")
        ax2 .set_ylabel ("Amplitude (norm.)")
        ax2 .set_title ("Spectrum")
        ax2 .grid (True ,alpha =0.3 ,which ="both")
        plt .tight_layout ()
        plt .show ()

    def plot_displacement_map (self ,scale_um :bool =True )->None :
        if not self .visualization_enabled :
            print (
            'Visualization disabled: the topology does not support a 2D map.'
            'Pass visual_shape to set_custom_topology().'
            )
            return 
        if self ._visual_element_indices is None :
            print ('Visualization disabled: the list of elements to be visualized is not specified.')
            return 
        uz_all =self .position [2 :self .n_elements *6 :6 ]
        disp_map =uz_all [self ._visual_element_indices ].reshape (self ._visual_shape ).astype (float )
        if scale_um :
            disp_map *=1e6 
        extent =[0.0 ,self .width *1e3 ,0.0 ,self .height *1e3 ]
        plt .figure (figsize =(5 ,5 ))
        im =plt .imshow (
        disp_map ,
        cmap ="RdBu",
        origin ="lower",
        extent =extent ,
        aspect ="auto",
        )
        plt .xlabel ("X, mm")
        plt .ylabel ("Y, mm")
        unit ="um"if scale_um else "m"
        plt .title (f"Displacement uz ({unit })")
        plt .colorbar (im ,label =f"uz, {unit }")
        plt .tight_layout ()
        plt .show ()

    def animate (
    self ,
    history_disp_all :list [np .ndarray ]|None =None ,
    dt :float =1e-6 ,
    skip :int =1 ,
    interval_ms :int =50 ,
    scale_um :bool =True ,
    cmap :str ="RdBu",
    )->FuncAnimation |None :
        if not self .visualization_enabled :
            print (
            'Rendering disabled: The topology does not support 2D animation.'
            'Pass visual_shape to set_custom_topology().'
            )
            return None 
        frames =history_disp_all if history_disp_all is not None else self .history_disp_all 
        if not frames :
            raise ValueError (
            "No animation data. Run simulate(record_history=True) or pass history_disp_all."
            )
        if np .asarray (frames [0 ]).ndim !=2 :
            print (
            'Visualization disabled: non-2D frames received.'
            'Pass visual_shape to set_custom_topology().'
            )
            return None 
        extent =[0.0 ,self .width *1e3 ,0.0 ,self .height *1e3 ]
        scale =1e6 if scale_um else 1.0 
        vmin =min (np .min (f )for f in frames )*scale 
        vmax =max (np .max (f )for f in frames )*scale 
        vabs =max (abs (vmin ),abs (vmax ),1e-12 )
        vmin ,vmax =-vabs ,vabs 

        fig ,ax =plt .subplots (figsize =(6 ,5 ))
        im =ax .imshow (
        frames [0 ]*scale ,
        cmap =cmap ,
        origin ="lower",
        extent =extent ,
        aspect ="auto",
        vmin =vmin ,
        vmax =vmax ,
        )
        ax .set_xlabel ("X, mm")
        ax .set_ylabel ("Y, mm")
        unit ="um"if scale_um else "m"
        ax .set_title ("t = 0.00 ms")
        plt .colorbar (im ,ax =ax ,label =f"uz, {unit }")
        indices =list (range (0 ,len (frames ),skip ))

        def update (i :int )->tuple :
            idx =indices [i ]
            im .set_data (frames [idx ]*scale )
            ax .set_title (f"t = {idx *dt *1e3 :.2f} ms")
            return (im ,)

        ani =FuncAnimation (fig ,update ,frames =len (indices ),interval =interval_ms ,repeat =True )
        plt .tight_layout ()
        return ani 

    def animate_air_pressure_center_plane (
    self ,
    history_air_center_xz :list [np .ndarray ]|None =None ,
    dt :float =1e-6 ,
    skip :int =1 ,
    interval_ms :int =50 ,
    cmap :str ="RdBu",
    symmetric :bool =True ,
    )->FuncAnimation |None :
        frames =history_air_center_xz if history_air_center_xz is not None else self .history_air_center_xz 
        if not frames :
            print (
            'Visualization skipped: no air-slice data.'
            'Run simulate(record_history=True) or pass history_air_center_xz.'
            )
            return None 

        x0_mm =(self .air_origin_x -0.5 *self .dx_air )*1e3 
        z0_mm =(self .air_origin_z -0.5 *self .dz_air )*1e3 
        x1_mm =(self .air_origin_x +(self .nx_air -0.5 )*self .dx_air )*1e3 
        z1_mm =(self .air_origin_z +(self .nz_air -0.5 )*self .dz_air )*1e3 
        extent =[x0_mm ,x1_mm ,z0_mm ,z1_mm ]

        vmin =min (float (np .min (f ))for f in frames )
        vmax =max (float (np .max (f ))for f in frames )
        if symmetric :
            vabs =max (abs (vmin ),abs (vmax ),1e-12 )
            vmin ,vmax =-vabs ,vabs 
        else :
            if abs (vmax -vmin )<1e-12 :
                vmax =vmin +1e-12 

        fig ,ax =plt .subplots (figsize =(6 ,5 ))
        im =ax .imshow (
        frames [0 ],
        cmap =cmap ,
        origin ="lower",
        extent =extent ,
        aspect ="auto",
        vmin =vmin ,
        vmax =vmax ,
        )
        ax .set_xlabel ("X, mm")
        ax .set_ylabel ("Z, mm")
        ax .set_title ("Air pressure center slice (X-Z), t = 0.00 ms")
        plt .colorbar (im ,ax =ax ,label ="p, Pa")
        indices =list (range (0 ,len (frames ),skip ))

        def update (i :int )->tuple :
            idx =indices [i ]
            im .set_data (frames [idx ])
            ax .set_title (f"Air pressure center slice (X-Z), t = {idx *dt *1e3 :.2f} ms")
            return (im ,)

        ani =FuncAnimation (fig ,update ,frames =len (indices ),interval =interval_ms ,repeat =True )
        plt .tight_layout ()
        return ani 


def _spectrum_peak_prominence (freq :np .ndarray ,spec :np .ndarray ,freq_max_hz :float =50_000.0 )->float :
    'Peak isolation: max(spec)/mean(spec) for positive frequencies up to freq_max_hz.'
    mask =(freq >0 )&(freq <=freq_max_hz )
    if not np .any (mask ):
        return 0.0 
    s =np .asarray (spec [mask ],dtype =float )
    s =s [np .isfinite (s )]
    if s .size ==0 :
        return 0.0 
    return float (np .max (s )/(np .mean (s )+1e-40 ))


def validate_natural_frequencies (
model :"PlanarDiaphragmOpenCL",
dt :float =2e-7 ,
duration :float =0.02 ,
impulse_velocity_z :float =0.01 ,
)->dict [str ,float ]:
    'Comparison of the natural frequency of the numerical model with the analytical one (membrane).\n\n    Algorithm: rest + velocity impulse along z in the center → simulate() → u_center(t) →\n    FFT → search for peak at [1, 50k] Hz; the peak frequency is refined by parabolic interpolation.\n    FFT resolution: df = 1/duration (duration=0.02 s → df=50 Hz; at 0.005 s it was 200 Hz -\n    the peak was quantized into bins of 0, 200, 400... Hz, which is why the numerical f did not respond to tension).\n    Returns a dictionary: numerical_f11_Hz, membrane_f11_Hz, err_membrane_pct, ...'
    out :dict [str ,float ]={}
    if analytical_natural_frequencies is None :
        print ('validate_natural_frequencies: analytical_diaphragm module not found, skipping.')
        return out 

    if model .kernel_debug :
    # Checking: what is packed into the buffer and what is actually read by the kernel (one step without trace output)
        dt_check =2e-7 
        params_bytes =model ._params_bytes (dt_check ,0 ,-1 )
        unpacked =struct .unpack (_PARAMS_FORMAT ,params_bytes )
        pre_tension_packed =unpacked [27 ]
        model ._set_rest_position ()
        model .velocity .fill (0.0 )
        debug_buf =np .zeros (model ._DEBUG_BUF_DOUBLES ,dtype =np .float64 )
        model .step (dt_check ,0.0 ,step_idx =0 ,debug_elem =0 ,debug_buf_out =debug_buf ,debug_silent =True )
        pre_tension_in_kernel =float (debug_buf [_TRACE_ELASTIC_EXTRA +18 ])if debug_buf .size >_TRACE_ELASTIC_EXTRA +18 else float ("nan")
        print ('--- Checking the transmission of pre_tension to the kernel ---')
        print (f"  Python (model.pre_tension):     {model .pre_tension } N/m")
        print (f"  Упаковано в буфер:             {pre_tension_packed } N/m")
        print (f"  Прочитано в ядре (trace):      {pre_tension_in_kernel } N/m")
        if abs (pre_tension_in_kernel -model .pre_tension )>0.01 :
            print ('ATTENTION: the kernel receives a different value - the layout of the Params structure (OpenCL alignment) may be incorrect.')
        else :
            print ('The transfer to the kernel is correct. If the numerical f does not change with T, the dynamics are determined by the mesh stiffness (k_eff), and not just by the tension.')
        print ()
    else :
        print ('--- Checking the transmission of pre_tension to the kernel ---')
        print ('Skip: Kernel tracing is disabled (kernel_debug=False).')
        print ()

    analytical =analytical_natural_frequencies (
    model .width ,model .height ,model .thickness ,
    model .density ,model .E_parallel ,model .poisson ,
    model .pre_tension ,
    )
    f_mem =analytical ["membrane_f11_Hz"]

    f_num ,freq_axis ,spec =model .get_numerical_natural_frequency (
    dt =dt ,duration =duration ,impulse_velocity_z =impulse_velocity_z ,refine_peak =True ,
    )
    df_fft =1.0 /duration 
    max_uz_um =getattr (model ,"_last_max_uz_um",0.0 )
    prominence =_spectrum_peak_prominence (freq_axis ,spec )if spec .size >0 else 0.0 

    is_explosion =max_uz_um >MAX_UZ_UM_OK 
    is_zero_hz =not (np .isfinite (f_num )and f_num >=MIN_FREQ_HZ_OK )
    is_noisy =prominence <MIN_PEAK_PROMINENCE if prominence ==prominence else True 

    out ["numerical_f11_Hz"]=f_num 
    out ["membrane_f11_Hz"]=f_mem 
    out ["max_uz_um"]=max_uz_um 
    out ["peak_prominence"]=prominence 
    out ["is_explosion"]=float (is_explosion )
    out ["is_zero_hz"]=float (is_zero_hz )
    out ["is_noisy"]=float (is_noisy )

    print ('--- Validation: natural frequency (membrane, mode 1.1) ---')
    print (f"  Tension: pre_tension (числ.) = T (аналит.) = {model .pre_tension :.2f} N/m")
    print (f"  Numerical: duration={duration } s, разрешение FFT df = {df_fft :.1f} Hz, пик уточнён параболой.")
    print (f"  Analytical (мембрана, натяжение T): f = {f_mem :.2f} Hz")
    print (f"  Numerical (пик FFT центра):        f = {f_num :.2f} Hz")

    if np .isfinite (f_num )and np .isfinite (f_mem )and f_mem >0 :
        err_mem =100.0 *(f_num -f_mem )/f_mem 
        out ["err_membrane_pct"]=err_mem 
        print (f"  Относительная ошибка:              {err_mem :+.1f} %")
    else :
        out ["err_membrane_pct"]=np .nan 

    print ('--- Correctness criteria (explosion / 0 Hz / noise debugging) ---')
    print (f"  max |uz| по сетке:     {max_uz_um :.2f} um  (порог OK: <={MAX_UZ_UM_OK :.0f} um)")
    print (f"  Peak prominence:    {prominence :.2f}  (порог OK: >={MIN_PEAK_PROMINENCE :.1f})")
    print (f"  Взрыв (uz > порог):   {'YES - inadequate displacements'if is_explosion else 'No'}")
    print (f"  0 Hz (пик < 1 Hz):    {'YES - does not hold elasticity'if is_zero_hz else 'No'}")
    print (f"  Шум (пик не выделен): {'YES - noise spectrum'if is_noisy else 'No'}")
    if is_explosion or is_zero_hz or is_noisy :
        print ('Reasons for the explosion/0 Hz/noise: 1) simulate() erased position/velocity (fixed: reset_state=False).')
        print ('2) The boundary is fixed in the kernel (position/velocity is not updated). 3) If necessary, reduce impulse_velocity_z or dt.')
    print ()
    return out 


def _parse_cli_args (argv :list [str ]):
    'A single parser for CLI launch arguments.'
    import argparse 
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--no-plot",action ="store_true",dest ="no_plot")
    parser .add_argument ("--uniform",action ="store_true",dest ="uniform_pressure")
    parser .add_argument ("--debug",action ="store_true",dest ="debug_m_total")
    parser .add_argument ("--validate",action ="store_true",dest ="do_validate")
    parser .add_argument (
    "--force-shape",
    choices =("impulse","uniform","sine","square","chirp","white_noise"),
    default ="impulse",
    dest ="force_shape",
    help ='External pressure form: impulse|uniform|sine|square|chirp|white_noise',
    )
    parser .add_argument (
    "--excitation-mode",
    choices =tuple (_EXCITATION_MODE_TO_KERNEL .keys ()),
    default ="external",
    dest ="excitation_mode",
    help =(
        "Mechanical excitation mode: "
        "external (kernel-generated pressure force), "
        "external_full_override (host force override), "
        "external_velocity_override (host velocity override), "
        "second_order_boundary_full_override "
        "(host force override + second-order air boundary solver)."
    ),
    )
    parser .add_argument (
    "--force-amplitude",
    type =float ,
    default =10.0 ,
    dest ="force_amplitude",
    help ='Excitation amplitude: Pa for pressure modes, m/s for external_velocity_override',
    )
    parser .add_argument (
    "--force-offset",
    type =float ,
    default =0.0 ,
    dest ="force_offset",
    help ='Excitation offset: Pa for pressure modes, m/s for external_velocity_override',
    )
    parser .add_argument (
    "--force-freq",
    type =float ,
    default =1000.0 ,
    dest ="force_freq",
    help ='Frequency (Hz) for sine/square and start frequency for chirp (ignored for white_noise)',
    )
    parser .add_argument (
    "--force-freq-end",
    type =float ,
    default =5000.0 ,
    dest ="force_freq_end",
    help ='End frequency (Hz) for chirp',
    )
    parser .add_argument (
    "--force-phase-deg",
    type =float ,
    default =0.0 ,
    dest ="force_phase_deg",
    help ='Initial phase (degrees) for sine/square/chirp (ignored for white_noise)',
    )
    parser .add_argument ("--pre-tension","--pre_tension",type =float ,default =10.0 ,dest ="pre_tension")
    parser .add_argument ("--dt",type =float ,default =1e-6 )
    parser .add_argument ("--duration",type =float ,default =None )
    parser .add_argument (
    "--air-grid-step-mm",
    "--air_grid_step_mm",
    type =float ,
    default =None ,
    dest ="air_grid_step_mm",
    help ='The pitch of the air acoustic field grid, mm (if not specified, is taken from the pitch of the FE membrane).',
    )
    parser .add_argument (
    "--air-pressure-history-every-steps",
    "--air_pressure_history_every_steps",
    type =int ,
    default =10 ,
    dest ="air_pressure_history_every_steps",
    help ='Save air pressure history point every N simulation steps (N >= 1).',
    )
    parser .add_argument (
    "--material-library-file",
    type =str ,
    default =None ,
    dest ="material_library_file",
    help ='Path to the material library JSON file (rows of 7–8 numbers: +acoustic_inject).',
    )
    parser .add_argument (
    "--sim-file",
    type =str ,
    default =None ,
    dest ="sim_file",
    help =(
    'Load saved data: results (.json wire / .pkl) or launch case'
    '(.pkl, schema fe_sim_run_v1) to repeat the simulation without UI.'
    ),
    )
    parser .add_argument (
    "--plot-sim-file",
    action ="store_true",
    dest ="plot_sim_file",
    help ='For results only mode (--sim-file): show matplotlib plots (time + spectrum).',
    )
    args ,_ =parser .parse_known_args (argv [1 :])
    return args 


def print_parsed_simulation_parameters (
args ,
*,
pre_tension_effective_N_per_m :float ,
material_source :str ,
material_props_n_rows :int |None ,
)->None :
    """Log all CLI / parsed simulation parameters (goes to stdout; captured by simulation server)."""
    lines =[
    "",
    "========== Parsed simulation parameters (OpenCL kernel driver) ==========",
    ]
    for key in sorted (vars (args )):
        val =getattr (args ,key ,None )
        lines .append (f"  {key }: {val !r }")
    lines .append (f"  PRE_TENSION effective (after env): {pre_tension_effective_N_per_m :.6g} N/m")
    lines .append (f"  material_library source: {material_source }")
    if material_props_n_rows is not None :
        lines .append (f"  material_props rows: {material_props_n_rows }")
    else :
        lines .append ("  material_props: (default built-in library)")
    lines .append ("=========================================================================")
    print ("\n".join (lines ))


def print_topology_report (
model :"PlanarDiaphragmOpenCL",
*,
source :str ="model",
)->None :
    """Print bounding box, sizes, materials, air grid from all FE (after topology applied)."""
    n =model .n_elements 
    pos =model .position [:n *model .dof_per_element ].reshape (n ,model .dof_per_element )[:,:3 ]
    sz =model .element_size_xyz [:n ]
    mn =(pos -0.5 *sz ).min (axis =0 )
    mx =(pos +0.5 *sz ).max (axis =0 )
    ext =mx -mn 
    ext_mm =ext *1e3 
    sz_mm =sz *1e3 
    mat =model .material_index 
    uniq ,counts =np .unique (mat ,return_counts =True )
    n_bnd =int (np .sum (model .boundary_mask_elements [:n ]!=0 ))
    lines =[
    "",
    f"========== Topology report ({source }) ==========",
    f"  n_elements: {n }  DOF: {model .n_dof }  dof_per_element: {model .dof_per_element }",
    f"  Model grid (constructor): nx={model .nx } ny={model .ny } n_membrane_elements={model .n_membrane_elements }",
    "  Bounding box (all FE, centers +/- half size), mm:",
    f"    X: [{mn [0 ]*1e3 :.6g}, {mx [0 ]*1e3 :.6g}]  span {ext_mm [0 ]:.6g}",
    f"    Y: [{mn [1 ]*1e3 :.6g}, {mx [1 ]*1e3 :.6g}]  span {ext_mm [1 ]:.6g}",
    f"    Z: [{mn [2 ]*1e3 :.6g}, {mx [2 ]*1e3 :.6g}]  span {ext_mm [2 ]:.6g}",
    "  Element size (min / max / mean), mm:",
    f"    sx: {sz_mm [:,0 ].min ():.6g} / {sz_mm [:,0 ].max ():.6g} / {sz_mm [:,0 ].mean ():.6g}",
    f"    sy: {sz_mm [:,1 ].min ():.6g} / {sz_mm [:,1 ].max ():.6g} / {sz_mm [:,1 ].mean ():.6g}",
    f"    sz: {sz_mm [:,2 ].min ():.6g} / {sz_mm [:,2 ].max ():.6g} / {sz_mm [:,2 ].mean ():.6g}",
    f"  Boundary elements (mask != 0): {n_bnd } ({100.0 *n_bnd /max (n ,1 ):.2f}%)",
    "  material_index (uint8) counts:",
    ]
    for u ,c in zip (uniq ,counts ,strict =True ):
        lines .append (f"    index {int (u )}: {int (c )} elements")
    lines .append (f"  MAT_SENSOR elements: {int (np .sum (mat ==MAT_SENSOR ))}")
    lines .append (
    f"  Air grid (simulation): nx_air={model .nx_air } ny_air={model .ny_air } nz_air={model .nz_air }  n_cells={model .n_air_cells }"
    )
    pad_eff =float (model .air_padding )if model .air_padding is not None else 0.002 
    exp_nx =int (np .ceil ((ext [0 ]+2 *pad_eff )/model .dx_air ))+1 if model .dx_air >0 else -1 
    exp_ny =int (np .ceil ((ext [1 ]+2 *pad_eff )/model .dy_air ))+1 if model .dy_air >0 else -1 
    exp_nz =int (np .ceil ((ext [2 ]+2 *pad_eff )/model .dz_air ))+1 if model .dz_air >0 else -1 
    lines .append (
    f"  Air cells check (bbox span + 2*padding, same formula as _configure_air_field_grid): "
    f"expect ~ {exp_nx }x{exp_ny }x{exp_nz }, actual {model .nx_air }x{model .ny_air }x{model .nz_air }"
    )
    lines .append (
    f"  Air step (m): dx_air={model .dx_air :.6g} dy_air={model .dy_air :.6g} dz_air={model .dz_air :.6g}"
    )
    lines .append (
    f"  Air origin (m): ({model .air_origin_x :.6g}, {model .air_origin_y :.6g}, {model .air_origin_z :.6g})"
    )
    air_ext_m =np .array (
    [
    model .nx_air *model .dx_air ,
    model .ny_air *model .dy_air ,
    model .nz_air *model .dz_air ,
    ]
    )
    lines .append (
    f"  Air box extent (approx, m): X={air_ext_m [0 ]:.6g} Y={air_ext_m [1 ]:.6g} Z={air_ext_m [2 ]:.6g}  (mm: {air_ext_m [0 ]*1e3 :.4g} x {air_ext_m [1 ]*1e3 :.4g} x {air_ext_m [2 ]*1e3 :.4g})"
    )
    lines .append (f"  air_padding (m): {getattr (model ,'air_padding',None )}  air_grid_step (m): {getattr (model ,'air_grid_step',None )}")
    lines .append (f"  kernel_debug: {model .kernel_debug }")
    lines .append ("=================================================")
    print ("\n".join (lines ))


def _load_material_library_from_file (path :str )->np .ndarray :
    'Loads a material library from a JSON file.\n    Format: rows [[density, E_parallel, E_perp, poisson, Cd, eta_visc, acoustic_impedance], ...]\n    or object {"materials": [{"density": ..., "E_parallel": ..., ...}, ...]}.\n    Legacy keys coupling_gain/coupling_recv are accepted and replaced with stock acoustic_impedance values.'
    with open (path ,"r",encoding ="utf-8")as f :
        data =json .load (f )
    rows =None 
    if isinstance (data ,list ):
        rows =data 
    elif isinstance (data ,dict ):
        if "rows"in data :
            rows =data ["rows"]
        elif "materials"in data :
            mats =data ["materials"]
            rows =[]
            for m in mats :
                if isinstance (m ,(list ,tuple )):
                    rows .append (list (m ))
                else :
                    rows .append ([
                    float (m .get ("density",1000 )),
                    float (m .get ("E_parallel",1e9 )),
                    float (m .get ("E_perp",1e9 )),
                    float (m .get ("poisson",0.3 )),
                    float (m .get ("Cd",1.0 )),
                    float (m .get ("eta_visc",1.0 )),
                    float (
                        m .get (
                            "acoustic_impedance",
                            _stock_impedance_for_name (m .get ("name",""))
                        )
                    ),
                    float (m .get ("acoustic_inject",0.0 )),
                    ])
    if rows is None or len (rows )==0 :
        raise ValueError (f"Файл {path }: не найден массив материалов (rows или materials)")
    arr =np .array (rows ,dtype =np .float64 )
    # Backward compatibility for list-row files: old column 6 was coupling gain [0..1].
    if arr .ndim ==2 and arr .shape [1 ]>=7 :
        col6 =np .asarray (arr [:,6 ],dtype =np .float64 )
        if col6 .size >0 and float (np .nanmax (col6 ))<=2.5 :
            for i in range (arr .shape [0 ]):
                arr [i ,6 ]=_stock_impedance_for_index (i )
    return arr


def _build_test_topology_with_cotton_layer (
model :"PlanarDiaphragmOpenCL",
gap_mm :float =1.0 ,
)->dict [str ,object ]:
    'Generates a test topology from two parallel layers:\n    1) sensory layer, 2) cotton wool layer.\n\n    Restrictions/Conventions:\n    - the total number of CE does not change (model.n_elements is used);\n    - connections are only internal within each layer (+X, -X, +Y, -Y);\n    - fixation around the perimeter of each layer;\n    - the cotton wool layer is shifted by gap_mm from the touch layer along +Z (the gap between the surfaces).'
    n_total =int (model .n_elements )
    if n_total <2 or (n_total %2 )!=0 :
        raise ValueError ('Test 2-layer topology requires even n_elements >= 2')
    n_layer =n_total //2 
    target_ratio =float (model .width /model .height )if model .height >0.0 else 1.0 

    # We select (nx_layer, ny_layer) so that:
    # 1) nx_layer * ny_layer == n_layer
    # 2) the shape was close to the geometry of the model (almost square FE).
    best_nx =1 
    best_ny =n_layer 
    best_score =float ("inf")
    for nx_layer in range (1 ,int (np .sqrt (n_layer ))+1 ):
        if (n_layer %nx_layer )!=0 :
            continue 
        ny_layer =n_layer //nx_layer 
        ratio =float (nx_layer /ny_layer )
        score =abs (np .log ((ratio +1e-12 )/(target_ratio +1e-12 )))
        if score <best_score :
            best_score =score 
            best_nx =nx_layer 
            best_ny =ny_layer 

            # We guarantee nx >= ny for a more expected mesh orientation.
    nx_layer ,ny_layer =(best_nx ,best_ny )if best_nx >=best_ny else (best_ny ,best_nx )
    if nx_layer *ny_layer !=n_layer :
        raise ValueError ('Failed to build regular mesh for layer')

    sx =float (model .width /nx_layer )
    sy =float (model .height /ny_layer )
    # “Cubic or close”: select the thickness according to the smaller step in the plane.
    sz =float (min (sx ,sy ))
    gap =float (gap_mm )*1e-2 

    pos =np .zeros ((n_total ,3 ),dtype =np .float64 )
    size =np .zeros ((n_total ,3 ),dtype =np .float64 )
    neighbors =np .full ((n_total ,FACE_DIRS ),-1 ,dtype =np .int32 )
    material_index =np .full (n_total ,MAT_SENSOR ,dtype =np .uint8 )
    boundary_mask =np .zeros (n_total ,dtype =np .int32 )

    z_mem_center =0.0 
    z_cotton_center =z_mem_center +sz +gap 

    def idx_local (i :int ,j :int )->int :
        return j *nx_layer +i 

    for layer in range (2 ):
        layer_offset =layer *n_layer 
        zc =z_mem_center if layer ==0 else z_cotton_center 
        mat_id =MAT_SENSOR if layer ==0 else MAT_COTTON_WOOL 
        for j in range (ny_layer ):
            for i in range (nx_layer ):
                local =idx_local (i ,j )
                idx =layer_offset +local 
                x =(i +0.5 )*sx -0.5 *model .width 
                y =(j +0.5 )*sy -0.5 *model .height 
                pos [idx ,0 ]=x 
                pos [idx ,1 ]=y 
                pos [idx ,2 ]=zc 
                size [idx ,0 ]=sx 
                size [idx ,1 ]=sy 
                # membrane thickness is 10 microns, cotton thickness is sz microns
                size [idx ,2 ]=10e-6 if mat_id ==MAT_SENSOR else sz 
                material_index [idx ]=mat_id 

                # Internal connections are only in their own plane.
                if i +1 <nx_layer :
                    neighbors [idx ,0 ]=layer_offset +idx_local (i +1 ,j )# +X
                if i -1 >=0 :
                    neighbors [idx ,1 ]=layer_offset +idx_local (i -1 ,j )# -X
                if j +1 <ny_layer :
                    neighbors [idx ,2 ]=layer_offset +idx_local (i ,j +1 )# +Y
                if j -1 >=0 :
                    neighbors [idx ,3 ]=layer_offset +idx_local (i ,j -1 )# -Y
                    # ±Z intentionally without connections.

                if i ==0 or i ==nx_layer -1 or j ==0 or j ==ny_layer -1 :
                    boundary_mask [idx ]=1 

    return {
    "element_position_xyz":pos ,
    "element_size_xyz":size ,
    "neighbors":neighbors ,
    "material_index":material_index ,
    "boundary_mask_elements":boundary_mask ,
    "visual_shape":(ny_layer ,nx_layer ),
    }

def run_cli_simulation (
parsed_args ,
stop_check :Callable [[],bool ]|None =None ,
topology :dict |None =None ,
material_library_rows :list |np .ndarray |None =None ,
)->tuple ["PlanarDiaphragmOpenCL",np .ndarray ]:
    'Helper function: Performs simulation based on CLI arguments.\n\n    Takes the object returned by _parse_cli_args(), creates a model,\n    executes simulate() and returns (model, hist_center).\n    stop_check: if set, every step is called; if True, the simulation is interrupted.\n    topology: if specified, used instead of the test topology (from Topology Generator).\n    material_library_rows: material library rows (8 numbers: +acoustic_inject for air-field injection)\n        transmitted directly (UI/server). Take precedence over --material-library-file.'
    args =parsed_args 
    no_plot =args .no_plot 
    uniform_pressure =args .uniform_pressure 
    debug_m_total =args .debug_m_total 
    do_validate =args .do_validate 
    pre_tension =float (args .pre_tension )
    # Environment variable overrides arguments (useful when running from IDE/launch without passing --pre-tension)
    if os .environ .get ("PRE_TENSION")not in (None ,""):
        try :
            pre_tension =float (os .environ ["PRE_TENSION"].strip ())
        except ValueError :
            pass 

    material_props =None 
    mat_file =getattr (args ,"material_library_file",None )
    material_source ="default built-in"
    material_n_rows :int |None =None 
    if material_library_rows is not None :
        arr =np .asarray (material_library_rows ,dtype =np .float64 )
        if arr .size >0 :
            if arr .ndim !=2 or arr .shape [1 ]not in (5 ,6 ,7 ,_MATERIAL_PROPS_STRIDE ):
                raise ValueError (
                'material_library_rows: expected shape [n_materials, 5..8],'
                f"got {arr .shape }"
                )
            material_props =arr 
            material_source ="inline rows (UI / simulation server payload)"
            material_n_rows =int (arr .shape [0 ])
    if material_props is None and mat_file :
        material_props =_load_material_library_from_file (mat_file )
        material_source =f"CLI file {mat_file !r }"
        material_n_rows =int (material_props .shape [0 ])
    print_parsed_simulation_parameters (
    args ,
    pre_tension_effective_N_per_m =pre_tension ,
    material_source =material_source ,
    material_props_n_rows =material_n_rows ,
    )

    if topology is not None :
        pos =np .asarray (topology ["element_position_xyz"],dtype =np .float64 )
        size =np .asarray (topology ["element_size_xyz"],dtype =np .float64 )
        n_elements =pos .shape [0 ]
        if n_elements <1 :
            raise ValueError ("topology must have at least 1 element")
        extent =np .max (pos ,axis =0 )-np .min (pos ,axis =0 )
        if np .max (np .abs (pos ))>1.0 or np .max (extent )>1.0 :
            pos =pos *1e-3 
            size =size *1e-3 
        extent =np .max (pos ,axis =0 )-np .min (pos ,axis =0 )
        width_mm =float (np .maximum (extent [0 ],1e-6 ))*1e3 
        height_mm =float (np .maximum (extent [1 ],1e-6 ))*1e3 
        nx =1 
        ny =n_elements 
        model =PlanarDiaphragmOpenCL (
        width_mm =width_mm ,
        height_mm =height_mm ,
        nx =nx ,
        ny =ny ,
        pre_tension_N_per_m =pre_tension ,
        air_grid_step_mm =args .air_grid_step_mm ,
        excitation_mode =args .excitation_mode ,
        kernel_debug =debug_m_total ,
        material_props =material_props ,
        )
        mat_idx =np .asarray (topology ["material_index"],dtype =np .uint8 )
        n_sensor =int (np .sum (mat_idx ==MAT_SENSOR ))
        vs =topology .get ("visual_shape")
        if vs is not None and len (vs )==2 :
            v_ny ,v_nx =int (vs [0 ]),int (vs [1 ])
            if v_ny >0 and v_nx >0 and v_ny *v_nx ==n_sensor :
                visual_shape =(v_ny ,v_nx )
            else :
                visual_shape =None 
        else :
            visual_shape =None 
        air_pos_topo =np .asarray (topology .get ("air_element_position_xyz",np .zeros ((0 ,3 ))),dtype =np .float64 )
        n_air_topo =int (air_pos_topo .shape [0 ])
        ab_raw =topology .get ("air_neighbor_absorb_u8")
        air_abs_pass =None
        if ab_raw is not None and n_air_topo >0 :
            ab_flat =np .asarray (ab_raw ,dtype =np .uint8 ).ravel ()
            if ab_flat .size ==n_air_topo *FACE_DIRS :
                air_abs_pass =ab_flat .reshape (n_air_topo ,FACE_DIRS )
        custom_topo ={
        "element_position_xyz":pos ,
        "element_size_xyz":size ,
        "neighbors":np .asarray (topology ["neighbors"],dtype =np .int32 ),
        "material_index":mat_idx ,
        "boundary_mask_elements":np .asarray (topology ["boundary_mask_elements"],dtype =np .int32 ),
        "air_element_position_xyz":air_pos_topo ,
        "air_element_size_xyz":np .asarray (topology .get ("air_element_size_xyz",np .zeros ((0 ,3 ))),dtype =np .float64 ),
        "air_neighbors":np .asarray (topology .get ("air_neighbors",np .full ((0 ,FACE_DIRS ),-1 )),dtype =np .int32 ),
        "air_boundary_mask_elements":np .asarray (topology .get ("air_boundary_mask_elements",np .zeros (0 )),dtype =np .int32 ),
        "solid_to_air_index":np .asarray (topology .get ("solid_to_air_index",np .full (n_elements ,-1 )),dtype =np .int32 ),
        "solid_to_air_index_plus":np .asarray (
        topology .get ("solid_to_air_index_plus",np .full (n_elements ,-1 )),
        dtype =np .int32 ,
        ),
        "solid_to_air_index_minus":np .asarray (
        topology .get ("solid_to_air_index_minus",np .full (n_elements ,-1 )),
        dtype =np .int32 ,
        ),
        "air_grid_shape":np .asarray (topology .get ("air_grid_shape",np .zeros (3 )),dtype =np .int32 ),
        "membrane_mask_elements":np .asarray (topology .get ("membrane_mask_elements",np .zeros (n_elements )),dtype =np .int32 ),
        "sensor_mask_elements":np .asarray (topology .get ("sensor_mask_elements",np .zeros (n_elements )),dtype =np .int32 ),
        "visual_shape":visual_shape ,
        }
        model .set_custom_topology (
        element_position_xyz =custom_topo ["element_position_xyz"],
        element_size_xyz =custom_topo ["element_size_xyz"],
        neighbors =custom_topo ["neighbors"],
        material_index =custom_topo ["material_index"],
        boundary_mask_elements =custom_topo ["boundary_mask_elements"],
        visual_shape =custom_topo ["visual_shape"],
        preserve_velocity =False ,
        rebuild_air =True ,
        air_grid_step_mm =args .air_grid_step_mm ,
        air_element_position_xyz =custom_topo ["air_element_position_xyz"],
        air_element_size_xyz =custom_topo ["air_element_size_xyz"],
        air_neighbors =custom_topo ["air_neighbors"],
        air_neighbor_absorb_u8 =air_abs_pass ,
        air_boundary_mask_elements =custom_topo ["air_boundary_mask_elements"],
        solid_to_air_index =custom_topo ["solid_to_air_index"],
        solid_to_air_index_plus =custom_topo ["solid_to_air_index_plus"],
        solid_to_air_index_minus =custom_topo ["solid_to_air_index_minus"],
        air_grid_shape =custom_topo ["air_grid_shape"],
        membrane_mask_elements =custom_topo ["membrane_mask_elements"],
        sensor_mask_elements =custom_topo ["sensor_mask_elements"],
        )
        print_topology_report (model ,source ="external topology (Topology Generator / project)")
    else :
        model =PlanarDiaphragmOpenCL (
        nx =24 *4 ,
        ny =32 *4 ,
        pre_tension_N_per_m =pre_tension ,
        air_grid_step_mm =args .air_grid_step_mm ,
        excitation_mode =args .excitation_mode ,
        kernel_debug =debug_m_total ,
        material_props =material_props ,
        )
        test_topology =_build_test_topology_with_cotton_layer (model ,gap_mm =1.0 )
        model .set_custom_topology (
        element_position_xyz =test_topology ["element_position_xyz"],
        element_size_xyz =test_topology ["element_size_xyz"],
        neighbors =test_topology ["neighbors"],
        material_index =test_topology ["material_index"],
        boundary_mask_elements =test_topology ["boundary_mask_elements"],
        visual_shape =test_topology ["visual_shape"],
        preserve_velocity =False ,
        rebuild_air =True ,
        air_grid_step_mm =args .air_grid_step_mm ,
        )
        print_topology_report (model ,source ="built-in test topology (cotton layer)")
    if debug_m_total :
        trace_elem_raw =os .environ .get ("DEBUG_TRACE_ELEM")
        if trace_elem_raw not in (None ,""):
            try :
                model .debug_trace_elem =int (trace_elem_raw )
                model .debug_trace_step_from =int (os .environ .get ("DEBUG_TRACE_STEP_FROM","0"))
                model .debug_trace_step_to =int (os .environ .get ("DEBUG_TRACE_STEP_TO","20"))
                print (
                    f"[debug] hard trace enabled: elem={model .debug_trace_elem } "
                    f"steps={model .debug_trace_step_from }..{model .debug_trace_step_to }"
                )
            except Exception :
                model .debug_trace_elem =-1
        else :
            model .debug_trace_elem =-1
    dt =float (args .dt )
    duration =0.05 if args .duration is None else float (args .duration )# 50 ms
    force_shape =str (args .force_shape )
    if uniform_pressure :
        force_shape ="uniform"
    if force_shape =="uniform"and args .duration is None :
        duration =0.0005 
    if dt <=0.0 or duration <=0.0 :
        raise ValueError ('--dt and --duration must be > 0')
    if args .force_freq <0.0 or args .force_freq_end <0.0 :
        raise ValueError ('--force-freq and --force-freq-end must be >= 0')
    if str (args .excitation_mode )=="external_velocity_override"and float (args .force_amplitude )<=0.0 :
        raise ValueError ("external_velocity_override requires --force-amplitude > 0 (m/s)")
    if args .air_grid_step_mm is not None and args .air_grid_step_mm <=0.0 :
        raise ValueError ('--air-grid-step-mm must be > 0')
    if int (args .air_pressure_history_every_steps )<1 :
        raise ValueError ('--air-pressure-history-every-steps must be >= 1')
    n_steps =int (duration /dt )
    t =np .arange (n_steps ,dtype =np .float64 )*dt 
    amp =float (args .force_amplitude )
    off =float (args .force_offset )
    amp_units ="m/s"if str (args .excitation_mode )=="external_velocity_override"else "Pa"
    f0 =float (args .force_freq )
    f1 =float (args .force_freq_end )
    phase =np .deg2rad (float (args .force_phase_deg ))
    print (
    "\n========== Resolved run parameters (before simulate) ==========\n"
    f"  dt: {dt :.6g} s\n"
    f"  duration: {duration :.6g} s\n"
    f"  n_steps: {n_steps }\n"
    f"  force_shape: {force_shape }\n"
    f"  excitation_mode: {args .excitation_mode } (kernel={model ._kernel_excitation_mode })\n"
    f"  force_amplitude: {amp :.6g} {amp_units }  force_offset: {off :.6g} {amp_units }\n"
    f"  force_freq: {f0 :.6g} Hz  force_freq_end: {f1 :.6g} Hz  force_phase_deg: {float (args .force_phase_deg ):.6g}\n"
    f"  uniform_pressure: {uniform_pressure }  no_plot: {no_plot }  debug: {debug_m_total }  validate: {do_validate }\n"
    "=================================================================\n"
    )
    # Build normalized base waveform in [-1, 1], then scale to exact amplitude over the full timeline.
    base =np .zeros (n_steps ,dtype =np .float64 )
    if force_shape =="uniform":
        base .fill (1.0 )
    elif force_shape =="sine":
        base =np .sin (2.0 *np .pi *f0 *t +phase )
    elif force_shape =="square":
        base =np .where (np .sin (2.0 *np .pi *f0 *t +phase )>=0.0 ,1.0 ,-1.0 )
    elif force_shape =="chirp":
        if duration <=0.0 :
            raise ValueError ('For chirp duration must be > 0')
        k =(f1 -f0 )/duration
        phase_t =2.0 *np .pi *(f0 *t +0.5 *k *t *t )+phase
        base =np .sin (phase_t )
    elif force_shape =="white_noise":
        rng =np .random .default_rng ()
        base =rng .uniform (-1.0 ,1.0 ,size =n_steps ).astype (np .float64 )
    else :
        # impulse: historical one-sample pulse.
        if n_steps >0 :
            base [0 ]=1.0
    if base .size >0 :
        peak =float (np .max (np .abs (base )))
        if peak >1e-30 :
            base =base /peak
    pressure =off +amp *base
    if pressure .size >0 :
        rel =pressure -off
        rel_peak =float (np .max (np .abs (rel )))
        print (
            f"[excitation] realized peak amplitude over timeline: {rel_peak :.6g} {amp_units } "
            f"(target={abs (amp ):.6g} {amp_units })"
        )

    if do_validate :
        print ('--- Validation of natural frequencies ---')
        print (f"Numerical модель и аналитика: pre_tension = {model .pre_tension } N/m")
    else :
        print ("\n--- Simulate (OpenCL) ---")
        print (f"External force shape: {force_shape }")
        print (f"Excitation mode: {args .excitation_mode }")
        print (
        f"Paраметры силы: amp={amp :.6g} {amp_units }, offset={off :.6g} {amp_units }, "
        f"f0={f0 :.6g} Hz, f1={f1 :.6g} Hz, phase={float (args .force_phase_deg ):.3f} deg"
        )
    print (f"Pre-tension pre_tension = {pre_tension } N/m")
    first_bad =np .array ([-1 ],dtype =np .int32 )
    if not do_validate :
    # Important: history_disp_all is also needed in --no-plot mode (for GUI/debugging),
    # therefore record_history is always enabled, regardless of no_plot.
        hist =model .simulate (
        pressure ,
        dt =dt ,
        record_history =True ,
        check_air_resistance =True ,
        validate_steps =True ,
        show_progress =True ,
        progress_every_pct =5.0 ,
        air_pressure_history_every_steps =int (args .air_pressure_history_every_steps ),
        stop_check =stop_check ,
        )
        if np .any (~np .isfinite (hist )):
            print ('OpenCL: NaN/Inf in history')
        else :
            print (f"OpenCL: max |u_center| = {np .max (np .abs (hist ))*1e6 :.4f} um")
    else :
        hist =np .array ([])

    if debug_m_total :
        debug_elem =getattr (model ,"_first_bad_elem",model .center_idx )
        if debug_elem <0 :
            debug_elem =64 *4 +1 
        if getattr (model ,"_first_bad_step",-1 )>=0 :
            print (f"\n--- M_total debug for element {debug_elem } (first_bad) ---")
        else :
            print (f"\n--- M_total debug for center element {debug_elem } (first steps) ---")
        model ._set_rest_position ()
        model .velocity .fill (0.0 )
        debug_buf =np .zeros (model ._DEBUG_BUF_DOUBLES ,dtype =np .float64 )
        n_debug_steps =min (35 ,n_steps )
        for step_idx in range (n_debug_steps ):
            p =pressure [step_idx ]if pressure .ndim ==1 else pressure [step_idx ]
            model .step (dt ,p ,step_idx =step_idx ,debug_elem =debug_elem ,debug_buf_out =debug_buf )
            if step_idx >=30 and not np .any (np .isfinite (debug_buf [1 :34 ])):
                break 

    if do_validate :
        validate_natural_frequencies (model ,dt =2e-7 ,duration =0.02 ,impulse_velocity_z =0.01 )

    if not no_plot and hist .size >0 and np .all (np .isfinite (hist )):
        model .plot_time_and_spectrum (dt =dt )
        model .plot_displacement_map ()
        ani =model .animate (dt =dt ,skip =20 ,interval_ms =50 )
        ani_air =model .animate_air_pressure_center_plane (dt =dt ,skip =20 ,interval_ms =50 )
        plt .show ()

    return model ,hist 


def _cli_dump_results_from_packed (packed :dict ,*,plot :bool )->int :
    """Print summary of packed results; optional matplotlib plots."""
    hc =packed .get ("history_disp_center")
    if hc is None :
        print ('There is no history_disp_center in the file.')
        return 1 
    a =np .asarray (hc ,dtype =np .float64 ).ravel ()
    dt =float (packed .get ("dt",1e-6 ))
    n =int (a .size )
    print (
    f"Results: n_samples={n }, dt={dt } s, duration ~ {max (0 ,n -1 )*dt :.6g} s\n"
    f"  width_mm={packed .get ('width_mm')}, height_mm={packed .get ('height_mm')}, "
    f"air_extent={packed .get ('air_extent')}"
    )
    if n >0 and np .all (np .isfinite (a )):
        print (f"  center disp (m): min={a .min ():.6g} max={a .max ():.6g}")
    hda =packed .get ("history_disp_all")or []
    hac =packed .get ("history_air_center_xz")or []
    print (f"  history_disp_all frames: {len (hda )}  history_air_center_xz: {len (hac )}")
    if plot and n >1 and np .all (np .isfinite (a )):
        import matplotlib .pyplot as plt 

        t =np .arange (n ,dtype =np .float64 )*dt 
        fig ,axes =plt .subplots (2 ,1 ,figsize =(9 ,6 ),sharex =False )
        axes [0 ].plot (t *1e3 ,a *1e6 )
        axes [0 ].set_xlabel ("t (ms)")
        axes [0 ].set_ylabel ("u_center (um)")
        axes [0 ].set_title ('Moving the center')
        w =np .hanning (n )
        aw =a *w 
        spec =np .abs (np .fft .rfft (aw ))
        freqs =np .fft .rfftfreq (n ,d =dt )
        m =int (max (1 ,spec .size //2 ))
        axes [1 ].plot (freqs [1 :m ],spec [1 :m ])
        axes [1 ].set_xlabel ("f (Hz)")
        axes [1 ].set_ylabel ("|FFT| (arb.)")
        axes [1 ].set_title ('Spectrum (with Hann window)')
        plt .tight_layout ()
        plt .show ()
    return 0 


def run_cli_sim_file (args )->int :
    '--sim-file processing: view results or repeat run-case from .pkl.'
    from pathlib import Path 

    from simulation_io import (
    RUN_CASE_SCHEMA ,
    argv_from_ui_params ,
    load_run_case_pickle ,
    load_simulation_results_file ,
    prepare_material_library_rows ,
    )

    path =Path (args .sim_file )
    if not path .is_file ():
        print (f"--sim-file: file not found: {path }")
        return 1 

    suffix =path .suffix .lower ()
    if suffix ==".pkl":
        import pickle 

        with open (path ,"rb")as f :
            head =pickle .load (f )
        if isinstance (head ,dict )and head .get ("schema")==RUN_CASE_SCHEMA :
            import sys as _sys 

            params ,material_library ,topology =load_run_case_pickle (path )
            argv =argv_from_ui_params (params ,no_plot =args .no_plot )
            parsed =_parse_cli_args (argv )
            parsed_sys =_parse_cli_args (_sys .argv )
            # argv_from_ui_params does not see --debug, --validate, --dt/--duration overrides, etc.
            parsed .debug_m_total =parsed_sys .debug_m_total 
            parsed .do_validate =parsed_sys .do_validate 
            parsed .material_library_file =parsed_sys .material_library_file 
            parsed .no_plot =parsed_sys .no_plot 
            parsed .uniform_pressure =parsed_sys .uniform_pressure 
            parsed .plot_sim_file =parsed_sys .plot_sim_file 
            for i ,a in enumerate (_sys .argv ):
                if a =="--dt"and i +1 <len (_sys .argv ):
                    try :
                        parsed .dt =float (_sys .argv [i +1 ])
                    except ValueError :
                        pass 
                if a in ("--duration",)and i +1 <len (_sys .argv ):
                    try :
                        parsed .duration =float (_sys .argv [i +1 ])
                    except ValueError :
                        pass 
                if a in ("--pre-tension","--pre_tension")and i +1 <len (_sys .argv ):
                    try :
                        parsed .pre_tension =float (_sys .argv [i +1 ])
                    except ValueError :
                        pass 
                if a in ("--air-grid-step-mm","--air_grid_step_mm")and i +1 <len (_sys .argv ):
                    try :
                        parsed .air_grid_step_mm =float (_sys .argv [i +1 ])
                    except ValueError :
                        pass 
            rows =prepare_material_library_rows (material_library ,topology )
            run_cli_simulation (parsed ,topology =topology ,material_library_rows =rows )
            return 0 

    try :
        packed =load_simulation_results_file (path )
    except Exception as e :
        print (f"Failed to load as simulation results: {e }")
        return 1 
    return _cli_dump_results_from_packed (packed ,plot =args .plot_sim_file )


if __name__ =="__main__":
    import sys 
    if "--server"in sys .argv :
        from simulation_server import main 
        sys .exit (main ())
    cli_args =_parse_cli_args (sys .argv )
    if cli_args .sim_file :
        sys .exit (run_cli_sim_file (cli_args ))
    run_cli_simulation (cli_args )
