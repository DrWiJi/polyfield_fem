# Implementation Guide (Current Baseline)

This guide summarizes the code paths that are currently authoritative.

## 1) Mechanical integration

RK4 kernels:

- `diaphragm_rk4_acc`
- `diaphragm_rk4_stage_state`
- `diaphragm_rk4_finalize`

## 2) Acoustic integration

- kernel: `air_pressure_wave_second_order_bc` (fallback: `air_acoustic_leapfrog_sommerfeld`)
- uses `air_neighbors` + face boundary-kind codes from `air_neighbor_absorb_u8`

Boundary-kind values:

- `0` interior
- `1` open/radiating
- `2` rigid

## 3) FE <-> air coupling kernels

- `air_inject_reduce_to_pressure`
- `air_pressure_to_fe_force`

Host orchestration is in `PlanarDiaphragmOpenCL._air_wave_step_host()` and `_compute_air_force_from_pressure_buffer()`.

## 4) Material row format

`[density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_recv, acoustic_inject]`

Compatibility:

- input with fewer columns is expanded by loader helpers to active stride 8.

## 5) Topology/boundary workflow

`topology_generator.py` produces:

- FE mesh arrays (`positions`, `neighbors`, `material_index`, `boundary_mask_elements`)
- air grid arrays (`air_neighbors`, `air_neighbor_absorb_u8`, maps FE<->air)

Optional acoustic boundary overrides from `BoundaryCondition.flags`:

- `acoustic_open`
- `acoustic_rigid`

## 6) What to update first when extending physics

1. `diaphragm_opencl_kernel.cl` kernels + constants
2. `diaphragm_opencl.py` kernel handles + buffer lifecycle
3. topology metadata (`topology_generator.py`)
4. material model (`fe_ui/material_library_model.py`)
5. docs (`README.md`, `PROJECT_DOCUMENTATION.md`, this file)
