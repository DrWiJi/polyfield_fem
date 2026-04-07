# Coupling Comparison (Archive)

This file contained a planning-era comparison between historical coupling variants.

## Current practical comparison

### Air solver

- Second-order pressure-only wave equation (`air_pressure_wave_second_order_bc`), with leapfrog fallback in the driver if required.

### Coupling path in active code

- FE -> air: pressure injection from FE volumetric flux (`air_inject_reduce_to_pressure`)
- Air -> FE: pressure-gradient force (`air_pressure_to_fe_force`)
- Material controls:
  - `coupling_recv`
  - `acoustic_inject`

For full current architecture, see `PROJECT_DOCUMENTATION.md`.
