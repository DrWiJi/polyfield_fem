# Project Documentation (Current)

## 1) Scope

`diaphragm_opencl` is a GPU-accelerated FE + acoustics simulation stack for diaphragm-like structures.

- FE dynamics: OpenCL RK4 with nonlinear spring-based interactions.
- Air acoustics: FDTD pressure field with FE<->air coupling.
- Desktop workflow: PySide6 GUI (`fe_ui`) with project model + topology generation.
- Optional client/server execution via `simulation_server.py`.

## 2) Current core numerics

### Mechanical solver

- DOF per element: 6 (`x, y, z, rx, ry, rz`).
- Time integration: RK4 kernels:
  - `diaphragm_rk4_acc`
  - `diaphragm_rk4_stage_state`
  - `diaphragm_rk4_finalize`
- Boundary-fixed elements (`boundary_mask_elements != 0`) are not advanced.

### Acoustic solver

Two air modes exist:

1. **Second-order pressure wave** (default):
   - kernel: `air_pressure_wave_second_order_bc`
   - update: `p_tt = c^2 lap(p)` + bounded boundary damping
   - boundary-kind aware ghosts:
     - open: Sommerfeld/Mur-style
     - rigid: Neumann-like `p_ghost = p_i`

2. **First-order experimental**:
   - kernels: `air_first_order_update_u`, `air_first_order_update_p`
   - collocated `p, ux, uy, uz` path (kept for experimentation).

`PlanarDiaphragmOpenCL(..., air_solver_mode="second_order")` is the default.

## 3) FE <-> Air coupling model

### FE -> Air

- Kernel: `air_inject_reduce_to_pressure`
- Source term:
  - computes volumetric flux rate from FE motion (`dV_dot`)
  - injects pressure increment into mapped air cells:
    - scaled by `rho_air * c_sound^2`
    - scaled per material by `acoustic_inject` (column 7)

### Air -> FE

- Kernel: `air_pressure_to_fe_force`
- Applies pressure-gradient-derived force using FE voxel map (`air_map_6`, `air_elem_map`)
- Coupling gain per material:
  - `coupling_recv` (column 6)
- Additional global scale:
  - `air_coupling_gain`
- Coupling is masked by `fe_air_coupling_mask` (membrane/sensor-focused traction path).

## 4) Material library format (active)

Material row stride is **8**:

`[density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_recv, acoustic_inject]`

Notes:

- `coupling_recv`: air->FE traction scaling (dimensionless model coefficient).
- `acoustic_inject`: FE->air source scaling (dimensionless model coefficient).
- This is a practical engineering model, not a full complex impedance model `Z(omega)`.

## 5) Boundary conditions (FE + air)

Boundary conditions are defined in project data (`BoundaryCondition`) and applied during topology generation.

### FE boundary conditions (mechanical fixation)

- FE fixation is stored in `boundary_mask_elements`:
  - `0` free element,
  - non-zero fixed element.
- In the mechanical solver, fixed elements are not advanced.
- BC primitives are evaluated by element center inclusion in primitive local coordinates.
- Supported primitive types:
  - `sphere`
  - `box`
  - `cylinder`
  - `tube`
- Scope:
  - applies to all meshes when `mesh_ids` is empty,
  - or only selected meshes when `mesh_ids` is provided.
- Mechanical fixation flag:
  - `flags.fix_position = true` marks FE inside the primitive as fixed.

### Air boundary conditions (acoustic)

Topology generator output includes:

- `air_neighbors` (sparse 6-neighbor table),
- `air_neighbor_absorb_u8` (per-face boundary kind code),
- `air_boundary_mask_elements`.

Per-face acoustic boundary-kind codes in `air_neighbor_absorb_u8`:

- `0` interior (neighbor exists),
- `1` open/radiating boundary,
- `2` rigid wall boundary.

Default assignment:

- outer domain faces are `open`,
- carved solid-side missing-neighbor faces are `rigid`.

Optional acoustic overrides via `BoundaryCondition.flags`:

- `acoustic_open: true`
- `acoustic_rigid: true`

Override behavior:

- only boundary air cells are considered,
- only missing-neighbor faces are overridden,
- interior neighbor links are never modified.

## 6) Public interfaces used most often

- `set_material_library(...)`
- `set_custom_topology(...)`
- `rebuild_air_field(...)`
- `step(...)`
- `simulate(...)`

## 7) Physical interpretation and limits

What is good:

- Relative trend studies (geometry/material/tension sweeps).
- Modal and qualitative wave behavior.
- FE-air interaction directionality and feedback loops.

Known limits:

- Material acoustic behavior uses scalar gains (`coupling_recv`, `acoustic_inject`), not full frequency-dependent impedance/absorption.
- Open boundary treatment is practical/robust, but not a full PML.
- Quantitative SPL/absorption metrics require calibration and benchmark validation.

## 8) Related docs

- `README.md` - quick project orientation.
- `QUICK_START.md` - run commands.
- `PROJECT_OVERVIEW.md` - high-level architecture.
- `fe_ui/FE_UI_STRUCTURE.md` - GUI package structure.
- Historical notes: `FORCE_BASED_COUPLING.md`, `IMPLEMENTATION_GUIDE.md`, `OLD_VS_NEW_COMPARISON.md`, `RK4_AIR_INTEGRATION.md`.
