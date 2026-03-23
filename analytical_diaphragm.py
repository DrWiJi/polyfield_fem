# -*- coding: utf-8 -*-
'Analytical model of the natural frequency of the diaphragm (membrane) for validation of the numerical model.\n\nMembrane: stretched film with fixed edges, dominant tension T (N/m).\n\nCompliance with numerical model (OpenCL):\n  In the core, pretension is specified as pre_tension (N/m). A connection extension is added:\n    pre_elong = (pre_tension * edge_length) / k_soft,\n  then at rest the force in the connection F = k_eff * pre_elong ≈ k_soft * pre_elong = pre_tension * edge_length,\n  i.e. tension along the edge length T_eff = F / edge_length = pre_tension.\n  Thus, numerical pre_tension and analytical T (tension per unit length) are the same value in N/m; the comparison is correct.\n  Note: with nonlinear stiffness, k_eff at the operating point may differ from k_soft, then the actual tension differs slightly from pre_tension.'
from __future__ import annotations 

import numpy as np 


def natural_frequency_membrane_rect (
Lx :float ,
Ly :float ,
tension_per_unit_length :float ,
rho_surface :float ,
m :int =1 ,
n :int =1 ,
)->float :
    'Natural frequency f_mn of a rectangular membrane with fixed edges.\n\n    Equation: T * (d²w/dx² + d²w/dy²) = ρ_s * d²w/dt².\n    Solution: w ~ sin(m*π*x/Lx)*sin(n*π*y/Ly)*cos(ω*t),\n    ω_mn = π * sqrt(T/ρ_s) * sqrt((m/Lx)² + (n/Ly)²).\n\n    Parameters:\n        Lx, Ly—dimensions in x and y (m).\n        tension_per_unit_length — tension T (N/m).\n        rho_surface — surface density ρ_s (kg/m²) = ρ * h.\n        m, n - mode numbers (1,1 - first mode).\n\n    Returns f_mn in Hz.'
    if tension_per_unit_length <=0 or rho_surface <=0 :
        return np .nan 
    c_sq =tension_per_unit_length /(rho_surface +1e-30 )
    omega =np .pi *np .sqrt (c_sq *((m /Lx )**2 +(n /Ly )**2 ))
    return float (omega /(2.0 *np .pi ))


def analytical_natural_frequencies (
width_m :float ,
height_m :float ,
thickness_m :float ,
density_kg_m3 :float ,
E_parallel_pa :float ,
poisson :float ,
pre_tension_N_per_m :float ,
)->dict [str ,float ]:
    'Natural frequency of the membrane (mode 1.1).\n\n    Returns a dictionary with the key membrane_f11_Hz.'
    rho_s =density_kg_m3 *thickness_m 
    f_mem =natural_frequency_membrane_rect (
    width_m ,height_m ,
    pre_tension_N_per_m ,rho_s ,
    m =1 ,n =1 ,
    )
    return {"membrane_f11_Hz":f_mem }
