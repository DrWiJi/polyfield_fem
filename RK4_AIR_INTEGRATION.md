# RK4 + Air Integration (Archive Note)

This document previously tracked an intermediate coupling refactor.

## Current status (supersedes older notes)

- Mechanical integration is RK4 (`diaphragm_rk4_*` kernels).
- Air solver default is now second-order pressure wave:
  - `air_pressure_wave_second_order_bc`
- Optional first-order `p+u` kernels remain available for experimentation:
  - `air_first_order_update_u`
  - `air_first_order_update_p`
- FE->air and air->FE coupling are active through:
  - `air_inject_reduce_to_pressure`
  - `air_pressure_to_fe_force`

## Why archived

Older sections in this file referenced kernels and migration steps that are no longer authoritative.

For up-to-date behavior use:

- `PROJECT_DOCUMENTATION.md`
- `README.md`
- code in `diaphragm_opencl.py` and `diaphragm_opencl_kernel.cl`
