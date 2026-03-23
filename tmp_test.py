from diaphragm_opencl import PlanarDiaphragmOpenCL
import numpy as np

m = PlanarDiaphragmOpenCL(nx=24, ny=32, pre_tension_N_per_m=10.0, kernel_debug=False)
print('created')
pressure = np.ones(10) * 1000.0
m.simulate(pressure, dt=1e-6, record_history=False, validate_steps=False, show_progress=True, progress_every_pct=100)
print('max u', np.max(np.abs(m.position[2::6])) * 1e6, 'um')
print('max p', np.max(np.abs(m.air_pressure_curr)))
print('center air', m.compute_air_force_center())
