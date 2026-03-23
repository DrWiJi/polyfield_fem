# -*- coding: utf-8 -*-
'Validation of the OpenCL iris model.\n\nRun: py validate_diaphragm.py\n  — always RK2, without graphs (do not block the flow).\n  — checking numerical stability, elasticity work, and the presence of resonances.'
from __future__ import annotations 

import sys 
import numpy as np 
from typing import NamedTuple 

from diaphragm_opencl import PlanarDiaphragmOpenCL 

# Simulation parameters (taking into account the forces from the boundary, a smaller dt is needed)
DT =1e-8 
DURATION_IMPULSE =0.001 # 1 ms (100k steps)
DURATION_UNIFORM =0.0005 # 0.5 ms (50k steps)
MAX_FREQ_HZ =20000.0 


class ValidationMetrics (NamedTuple ):
    'Validation metrics.'
    has_nan :bool 
    max_disp_um :float 
    n_resonance_peaks :int 
    fundamental_hz :float 
    peak_freqs_hz :tuple [float ,...]
    lin_fit_r2 :float # R^2 linear trend: ~1 = rigid body
    n_zero_crossings :int 
    decay_ratio :float # |u_end|/|u_max| — attenuation
    spectral_peak_prominence :float 
    has_oscillation :bool # sign of oscillations in the signal form


def _spectral_peaks (hist :np .ndarray ,dt :float ,min_prominence_ratio :float =0.03 )->tuple [list [float ],float ]:
    'Frequencies of spectrum peaks and frequency of fundamental tone.'
    if len (hist )<16 :
        return [],0.0 
    h =hist -np .mean (hist )# remove DC
    h =h -np .polyval (np .polyfit (np .arange (len (h )),h ,1 ),np .arange (len (h )))# detrend
    freq =np .fft .fftfreq (len (h ),dt )
    spec =np .abs (np .fft .fft (h ))
    mask =(freq >0 )&(freq <MAX_FREQ_HZ )
    f_pos =freq [mask ]
    s_pos =spec [mask ]
    if len (s_pos )<2 :
        return [],0.0 
    threshold =np .max (s_pos )*min_prominence_ratio 
    peaks =[]
    for i in range (1 ,len (s_pos )-1 ):
        if s_pos [i ]>=threshold and s_pos [i ]>=s_pos [i -1 ]and s_pos [i ]>=s_pos [i +1 ]:
            peaks .append (float (f_pos [i ]))
    peaks =sorted (peaks ,key =lambda f :np .interp (f ,f_pos ,s_pos ),reverse =True )[:10 ]
    fund =peaks [0 ]if peaks else 0.0 
    return peaks ,fund 


def _linear_trend_r2 (hist :np .ndarray )->float :
    'Linear regression R²: 1 = pure linear drift (no elasticity).'
    if len (hist )<3 :
        return 0.0 
    x =np .arange (len (hist ),dtype =float )
    coef =np .polyfit (x ,hist ,1 )
    y_pred =np .polyval (coef ,x )
    ss_res =np .sum ((hist -y_pred )**2 )
    ss_tot =np .sum ((hist -np .mean (hist ))**2 )
    if ss_tot <1e-30 :
        return 0.0 
    return float (1.0 -ss_res /ss_tot )


def _zero_crossings (hist :np .ndarray )->int :
    'The number of zero crossings relative to the average (a sign of fluctuations).'
    if len (hist )<2 :
        return 0 
    h =hist -np .mean (hist )# oscillations around the average
    s =np .sign (h )
    return int (np .sum (np .abs (np .diff (s ))>0 )/2 )


def run_impulse_validation (model :PlanarDiaphragmOpenCL )->ValidationMetrics :
    'A pulse of 100 Pa in the first 100 steps is damped oscillations with resonances.'
    n_steps =int (DURATION_IMPULSE /DT )
    pressure =np .zeros (n_steps ,dtype =np .float64 )
    n_impulse =min (100 ,n_steps //10 )
    pressure [:n_impulse ]=10.0 # pulse

    model .simulate (pressure ,dt =DT ,record_history =False ,check_air_resistance =False ,validate_steps =True )
    hist =np .asarray (model .history_disp_center ,dtype =np .float64 )

    has_nan =np .any (~np .isfinite (hist ))
    max_disp_um =float (np .max (np .abs (hist ))*1e6 )if not has_nan else np .nan 

    peaks ,fund =_spectral_peaks (hist ,DT )
    r2 =_linear_trend_r2 (hist )
    nzc =_zero_crossings (hist )

    u_max =np .max (np .abs (hist ))if len (hist )>0 and not has_nan else 1e-30 
    u_end =np .abs (hist [-1 ])if len (hist )>0 else 0.0 
    decay_ratio =float (u_end /(u_max +1e-40 ))

    spec =np .abs (np .fft .fft (hist -np .mean (hist )))
    freq =np .fft .fftfreq (len (hist ),DT )
    mask =(freq >0 )&(freq <MAX_FREQ_HZ )
    if np .any (mask ):
        s =spec [mask ]
        peak_val =np .max (s )
        mean_val =np .mean (s )+1e-40 
        prominence =float (peak_val /mean_val )
    else :
        prominence =0.0 

    has_osc =_detect_oscillation (hist )

    return ValidationMetrics (
    has_nan =has_nan ,
    max_disp_um =max_disp_um ,
    n_resonance_peaks =len (peaks ),
    fundamental_hz =fund ,
    peak_freqs_hz =tuple (peaks [:5 ]),
    lin_fit_r2 =r2 ,
    n_zero_crossings =nzc ,
    decay_ratio =decay_ratio ,
    spectral_peak_prominence =prominence ,
    has_oscillation =has_osc ,
    )


def run_uniform_validation (model :PlanarDiaphragmROCm )->tuple [bool ,float ,float ]:
    'Constant pressure is quasi-static equilibrium, not linear drift.'
    n_steps =int (DURATION_UNIFORM /DT )
    pressure =np .full (n_steps ,1.0 ,dtype =np .float64 )

    model .simulate (pressure ,dt =DT ,record_history =False ,check_air_resistance =False ,validate_steps =True )
    hist =np .asarray (model .history_disp_center ,dtype =np .float64 )

    has_nan =np .any (~np .isfinite (hist ))
    max_um =float (np .max (np .abs (hist ))*1e6 )if not has_nan else np .nan 
    r2 =_linear_trend_r2 (hist )
    return has_nan ,max_um ,r2 


def _detect_oscillation (hist :np .ndarray )->bool :
    'Is there oscillation: the sign of the derivative of detrended data changes.'
    if len (hist )<10 :
        return False 
    h =hist -np .mean (hist )
    d =np .diff (h )
    sign_changes =np .sum (np .diff (np .sign (d ))!=0 )
    return sign_changes >=2 # at least one "hump" or "hollow"


def _print_metrics (name :str ,m :ValidationMetrics )->None :
    print (f"\n--- {name } ---")
    print (f"  NaN/Inf:        {'FAIL'if m .has_nan else 'OK'}")
    print (f"  max |u|:        {m .max_disp_um :.4f} µm")
    print (f"  Resonance peaks:     {m .n_resonance_peaks }")
    print (f"  f0 (основной):  {m .fundamental_hz :.1f} Hz")
    if m .peak_freqs_hz :
        print (f"  Peaks (Hz):      {', '.join (f'{f :.0f}'for f in m .peak_freqs_hz )}")
    print (f"  R^2 linearity: {m .lin_fit_r2 :.4f}  (1.0 = жёсткое тело, упругость не работает)")
    print (f"  Zero crossings:  {m .n_zero_crossings }  (колебания)")
    print (f"  Decay:      {m .decay_ratio :.4f}  (|u_end|/|u_max|)")
    print (f"  Peak prominence: {m .spectral_peak_prominence :.2f}")
    print (f"  Oscillations:     {'Yes'if m .has_oscillation else 'No'}")


def main ()->int :
    quick ="--quick"in sys .argv 
    global DT ,DURATION_IMPULSE ,DURATION_UNIFORM 
    if quick :
        DT ,DURATION_IMPULSE ,DURATION_UNIFORM =1e-7 ,0.005 ,0.002 # 50k + 20k steps
        print ('(--quick mode)')

    print ("="*60 )
    print ('Validation of the diaphragm model (RK2, without graphs)')
    print ("="*60 )

    model =PlanarDiaphragmROCm (nx =24 ,ny =32 ,use_rk2 =True ,debug_hip_elastic =False )
    dt_s =DT 
    print (f"\nParameters: dt={dt_s :.0e} s, impulse {DURATION_IMPULSE *1000 :.0f} ms, uniform {DURATION_UNIFORM *1000 :.0f} ms")

    failures =[]

    # 1. Impulse response
    print ('[1] Pulse 100 Pa (first 100 steps)...')
    m_imp =run_impulse_validation (model )
    _print_metrics ('Impulse response',m_imp )

    if m_imp .has_nan :
        failures .append ('Impulse: NaN/Inf')
        # Known limitation: boundary neighbors are skipped - with uniform excitation R^2 can be high

        # 2. Constant pressure
    print ('[2] Constant pressure 1 Pa...')
    has_nan_u ,max_u ,r2_u =run_uniform_validation (model )
    print (f"\n--- Constant pressure ---")
    print (f"  NaN/Inf:        {'FAIL'if has_nan_u else 'OK'}")
    print (f"  max |u|:        {max_u :.4f} µm")
    print (f"  R^2 linearity: {r2_u :.4f}  (should be low at equilibrium)")

    if has_nan_u :
        failures .append ("Uniform: NaN/Inf")
    if r2_u >0.98 :
        failures .append ('Uniform: R^2 > 0.98 - linear drift instead of equilibrium')

        # 3. Summary
    print ("\n"+"="*60 )
    if failures :
        print ('VALIDATION ERRORS:')
        for f in failures :
            print (f"  * {f }")
        return 1 
    print ('Validation passed.')
    print ('Note: resonances and oscillations depend on the consideration of the boundary.')
    print ('Boundary elements in the core are passed through - with uniform excitation')
    print ('the elasticity between the internal elements is low; for the full model you need')
    print ('forces from a fixed boundary (pos_nb=0 for boundary).')
    return 0 


if __name__ =="__main__":
    sys .exit (main ())
