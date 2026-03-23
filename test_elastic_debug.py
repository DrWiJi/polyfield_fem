# -*- coding: utf-8 -*-
'Script for checking the calculation and application of elastic force.\nRun: python test_elastic_debug.py\n\nWith uniform pressure, the membrane moves like a rigid body - d_vec_z=0, F_el=0.\nPressure only on the center creates a displacement gradient → elastic force in z.'
import numpy as np 
from diaphragm_opencl import PlanarDiaphragmOpenCL 

def main ():
    model =PlanarDiaphragmOpenCL (nx =24 ,ny =32 ,debug_hip_elastic =True )
    dt =1e-8 # lower dt for stability at local pressure
    n_steps =200 
    # Pressure only on the central element - creates a gradient uz and elastic force
    pressure =np .zeros ((n_steps ,model .n_elements ),dtype =np .float64 )
    pressure [0 ,model .center_idx ]=1.0 # 1 Pa per center

    print ('===Validation of elastic force (pressure on the center) ===')
    hist =model .simulate (
    pressure ,dt =dt ,record_history =True ,
    check_air_resistance =False ,
    validate_steps =True ,
    )

    print ('=== Result ===')
    if np .any (~np .isfinite (hist )):
        print ('NaN/Inf in history (stop at first) - reduce dt')
    else :
        print (f"max |u_center| = {np .max (np .abs (hist ))*1e6 :.4f} µm")
        ok =np .max (np .abs (hist ))>1e-9 
        print ('OK: elastic force is applied'if ok else 'ERROR: offset ~0 - elasticity does not work')

if __name__ =="__main__":
    main ()
