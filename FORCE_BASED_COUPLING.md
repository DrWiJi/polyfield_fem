# FE-Air Coupling Notes (Current + Limits)

## Implemented model

The active model is a practical two-way coupling:

- FE -> air: injected pressure increment from FE motion (`dV_dot`-based source).
- Air -> FE: pressure-gradient-derived force with per-material receive scaling.

Key material coefficients:

- `coupling_recv` (air -> FE traction scale)
- `acoustic_inject` (FE -> air source scale)

## Interpretation

This is physically inspired and useful for engineering trends, but it is not a full impedance boundary model:

- no explicit complex impedance `Z(omega)` at interfaces,
- no frequency-dependent porous absorption law in material rows,
- no strict derivation of reflection coefficient by interface impedance per frequency/angle.

## When to trust results

- Good: comparative studies, mode trends, qualitative wave/coupling behavior.
- Needs calibration: absolute SPL, damping/Q, absorption and reflection magnitudes.

## Related

See `PROJECT_DOCUMENTATION.md` for the current end-to-end architecture.
