# Coupling Comparison (Archive)

This file contained a planning-era comparison between historical coupling variants.

## Current practical comparison

### Air solver options

1. `second_order` (default)
   - pressure-only wave equation
   - stable and visually cleaner for current topology/coupling path

2. `first_order` (optional)
   - collocated `p + u` update
   - useful for experiments, but may show grid-scale artifacts

### Coupling path in active code

- FE -> air: pressure injection from FE volumetric flux (`air_inject_reduce_to_pressure`)
- Air -> FE: pressure-gradient force (`air_pressure_to_fe_force`)
- Material controls:
  - `coupling_recv`
  - `acoustic_inject`

## Recommendation

Use `second_order` for production studies unless you are explicitly testing first-order formulations.

For full current architecture, see `PROJECT_DOCUMENTATION.md`.
