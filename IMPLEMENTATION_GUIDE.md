# Python Integration Guide: Force-Based Coupling

## Summary of Changes

This guide shows **exact code changes** needed to integrate force-based coupling into `diaphragm_opencl.py` and material library handling.

---

## File 1: diaphragm_opencl.py - Constants & Setup

### Change 1A: Update Material Property Stride

```python
# BEFORE:
_MATERIAL_PROPS_STRIDE = 8
FACE_DIRS = 6

# AFTER:
_MATERIAL_PROPS_STRIDE = 10  # Added acoustic_impedance_z, acoustic_damping_alpha
FACE_DIRS = 6

# Add new column definitions
MAT_PROP_ACOUSTIC_IMPEDANCE_Z = 8    # [Pa·s/m] = ρ·c
MAT_PROP_ACOUSTIC_DAMPING_ALPHA = 9  # [1/s] energy loss rate
```

### Change 1B: Update Material Library Expansion Function

```python
def _expand_material_props_to_stride10(props: np.ndarray) -> np.ndarray:
    """
    Expand material properties to 10 columns (force-based acoustic coupling).
    
    New columns 8-9:
    - acoustic_impedance_z: Acoustic wave impedance Z = ρ*c [Pa·s/m]
    - acoustic_damping_alpha: Material loss rate [1/s]
    """
    p = np.asarray(props, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("material_props must be 2D array")
    
    # Already in new format
    if p.shape[1] == 10:
        if np.any(p[:, 8] <= 0.0):
            raise ValueError("acoustic_impedance_z (col 8) must be > 0")
        return p
    
    # Legacy format: add new columns
    if p.shape[1] == 8:
        new_cols = np.zeros((p.shape[0], 2), dtype=np.float64)
        
        for i in range(p.shape[0]):
            # Column 8: Acoustic impedance Z = ρ*c
            # Default: air impedance (412 Pa·s/m @ 20°C)
            # This is used for FE → air coupling
            new_cols[i, 0] = 412.0
            
            # Column 9: Acoustic damping rate [1/s]
            # Controls energy dissipation in the frequency domain
            # Higher = more loss (heavier material, more absorption)
            if i == int(MAT_MEMBRANE):
                new_cols[i, 1] = 20.0    # Light viscous damping
            elif i == int(MAT_FOAM_VE3015):
                new_cols[i, 1] = 150.0   # Heavy porous absorption
            elif i == int(MAT_SHEEPSKIN_LEATHER):
                new_cols[i, 1] = 80.0
            elif i == int(MAT_HUMAN_EAR_AVG):
                new_cols[i, 1] = 100.0   # Biological tissue loss
            elif i == int(MAT_SENSOR):
                new_cols[i, 1] = 50.0    # Moderate for microphone
            elif i == int(MAT_COTTON_WOOL):
                new_cols[i, 1] = 200.0   # Very absorbing
            else:
                new_cols[i, 1] = 50.0    # Default fallback
        
        out = np.hstack((p, new_cols))
        return out
    
    raise ValueError(
        f"After normalization, expected 8 or 10 columns; got {p.shape[1]}"
    )


def _expand_material_props_if_needed(props: np.ndarray) -> np.ndarray:
    """Dispatch to appropriate expansion based on input size."""
    p = np.asarray(props, dtype=np.float64)
    if p.shape[1] < 8:
        # Even older format - handle separately if needed
        raise ValueError(f"Unsupported material format: {p.shape[1]} columns")
    elif p.shape[1] == 8:
        return _expand_material_props_to_stride10(p)
    elif p.shape[1] == 10:
        return _expand_material_props_to_stride10(p)  # Already correct
    else:
        raise ValueError(f"Unsupported material format: {p.shape[1]} columns")
```

### Change 1C: Update _build_default_material_library

```python
def _build_default_material_library(self) -> np.ndarray:
    """
    Built-in material library with force-based coupling parameters.
    [density, E_parallel, E_perp, poisson, Cd, eta_visc, 
     coupling_recv, acoustic_inject, acoustic_impedance_z, acoustic_damping_alpha]
    """
    # Existing parameters...
    return np.array(
        [
            # MAT_MEMBRANE (index 0)
            [
                self.density,              # density [kg/m³]
                self.E_parallel,           # E_parallel [Pa]
                self.E_perp,               # E_perp [Pa]
                self.poisson,              # poisson
                self.Cd,                   # Cd [dimensionless]
                membrane_eta_visc,         # eta_visc [Pa·s]
                membrane_coupling_gain,    # coupling_recv [0-1]
                1.0,                       # acoustic_inject [0-1]
                412.0,                     # acoustic_impedance_z = ρ*c [Pa·s/m]
                20.0,                      # acoustic_damping_alpha [1/s]
            ],
            # MAT_FOAM_VE3015 (index 1)
            [
                foam_density,
                foam_E_parallel,
                foam_E_perp,
                foam_poisson,
                foam_Cd,
                foam_eta_visc,
                foam_coupling_gain,        # ~0.25
                0.2,                       # acoustic_inject (~0.55*coupling_recv)
                150.0,                     # Z for foam (stiffer than air)
                150.0,                     # alpha - high damping
            ],
            # MAT_SHEEPSKIN_LEATHER (index 2)
            [
                leather_density,
                leather_E_parallel,
                leather_E_perp,
                leather_poisson,
                leather_Cd,
                leather_eta_visc,
                leather_coupling_gain,     # ~0.60
                0.3,                       # acoustic_inject
                412.0,                     # Z (air reference)
                80.0,                      # alpha
            ],
            # MAT_HUMAN_EAR_AVG (index 3)
            [
                ear_density,
                ear_E_parallel,
                ear_E_perp,
                ear_poisson,
                ear_Cd,
                ear_eta_visc,
                ear_coupling_gain,         # ~0.50
                0.2,                       # acoustic_inject
                412.0,
                100.0,
            ],
            # MAT_SENSOR (index 4) - microphone (listens only)
            [
                sensor_density,
                sensor_E_parallel,
                sensor_E_perp,
                sensor_poisson,
                sensor_Cd,
                sensor_eta_visc,
                sensor_coupling_gain,      # 1.0 (full pressure coupling)
                0.0,                       # acoustic_inject = 0 (does NOT radiate)
                412.0,
                50.0,
            ],
            # MAT_COTTON_WOOL (index 5)
            [
                cotton_density,
                cotton_E_parallel,
                cotton_E_perp,
                cotton_poisson,
                cotton_Cd,
                cotton_eta_visc,
                cotton_coupling_gain,      # ~0.30
                0.15,                      # acoustic_inject (weak re-radiation)
                200.0,                     # Z (porous)
                200.0,                     # alpha (very absorbing)
            ],
        ],
        dtype=np.float64,
    )
```

---

## File 2: diaphragm_opencl.py - Kernel Management

### Change 2A: Update Kernel References in __init__

```python
def __init__(self, ...):
    # In the kernel loading section:
    
    # OLD kernels (remove or deprecate):
    # self._kernel_air_inject_reduce = self.prg.air_inject_membrane_velocity
    # self._kernel_air_inject_direct = self.prg.air_inject_membrane_velocity_direct
    # self._kernel_air_to_force = self.prg.add_air_velocity_to_force_external
    
    # NEW kernels (force-based):
    self._kernel_air_inject_force_reduce = self.prg.air_inject_from_fe_velocity_force_reduce
    self._kernel_air_inject_force = self.prg.air_inject_from_fe_velocity_force
    self._kernel_air_force_to_fe = self.prg.add_air_pressure_force_to_fe
    
    # Flag to choose injection mode
    self.air_inject_use_reduce = True  # or from config
```

### Change 2B: Create Element ↔ Air Cell Mapping

Add this method to `PlanarDiaphragmOpenCL` class:

```python
def _build_air_element_map(self) -> np.ndarray:
    """
    Build mapping: FE element index → closest air cell index.
    
    Each FE element is mapped to its central air cell for:
    - Pressure gradient calculation (air → FE coupling)
    - Velocity accumulation (FE → air coupling)
    
    Returns: [n_elements] array of air cell indices
    """
    air_map = np.full(self.n_elements, -1, dtype=np.int32)
    
    for elem_idx in range(self.n_elements):
        # Get FE position
        base = elem_idx * self.dof_per_element
        x_fe = self.position[base + 0]
        y_fe = self.position[base + 1]
        z_fe = self.position[base + 2]
        
        # Map to air grid cell
        ix = int(np.floor((x_fe - self.air_origin_x) / self.dx_air + 0.5))
        iy = int(np.floor((y_fe - self.air_origin_y) / self.dy_air + 0.5))
        iz = int(np.floor((z_fe - self.air_origin_z) / self.dz_air + 0.5))
        
        # Clamp to valid range
        ix = np.clip(ix, 0, self.nx_air - 1)
        iy = np.clip(iy, 0, self.ny_air - 1)
        iz = np.clip(iz, 0, self.nz_air - 1)
        
        # Compute linear cell index
        cell_idx = iz * (self.nx_air * self.ny_air) + iy * self.nx_air + ix
        air_map[elem_idx] = cell_idx
    
    return air_map
```

Call this in `_allocate_air_buffers()`:

```python
def _allocate_air_buffers(self) -> None:
    """(Re)creates air field buffers and element-cell mappings."""
    # ... existing code ...
    
    # NEW: Build element-to-cell mapping
    self.air_map = self._build_air_element_map()
    self._buf_air_map = cl.Buffer(self.ctx, mf.READ_ONLY, size=self.air_map.nbytes)
    cl.enqueue_copy(self.queue, self._buf_air_map, self.air_map)
    
    # ... rest of existing code ...
```

---

## File 3: diaphragm_opencl.py - Air Coupling Loop Update

### Change 3A: Update _run_air_coupling Method

```python
def _run_air_coupling(self, dt: float, pressure_pa: float | np.ndarray) -> None:
    """
    Run air-FE coupling with force-based algorithm.
    
    3-stage process:
    1. Sync FE velocity to GPU
    2. FE velocity → air force (monopole sources)
    3. Air pressure → FE force (restoring force)
    """
    self._build_force_external(pressure_pa)
    self._sync_simulation_buffers()
    self._update_air_coupling_geometry_from_motion()
    
    n_air_substeps, dt_air = self._get_air_substeps(dt)
    
    # Get acoustic impedance (use membrane material)
    z_acoustic = self.material_props[int(MAT_MEMBRANE), MAT_PROP_ACOUSTIC_IMPEDANCE_Z]
    
    # Main coupling loop (multiple substeps for numerical stability)
    for substep in range(n_air_substeps):
        # ========== STAGE 1: FE Velocity → Air Force ==========
        # Compute acoustic pressure equivalent: p = Z * v_n
        # Then distribute force to air cells
        
        if self.air_inject_use_reduce:
            # Reduce mode: accumulate velocity deltas, then apply
            self._kernel_air_inject_force_reduce.set_args(
                self._buf_air_velocity_delta,           # Output
                self._buf_velocity,                      # Input: FE velocity
                self._buf_boundary,                      # Boundary mask
                self._buf_material_index,                # Material ID per element
                self._buf_material_props,                # Material parameters
                self._buf_air_elem_face_area,           # Contact area [n_elem, 3]
                self._buf_air_elem_volume,              # Element volume [n_elem]
                np.int32(self.n_elements),
                np.float64(self.rho_air),               # ρ [kg/m³]
                np.float64(self.air_sound_speed),       # c [m/s]
                np.float64(z_acoustic),                 # Z = ρ·c [Pa·s/m] NEW
                np.float64(dt_air),                     # Δt_air [s] NEW
            )
        else:
            # Direct mode: apply directly
            self._kernel_air_inject_force.set_args(
                self._buf_air_velocity_delta,
                self._buf_velocity,
                self._buf_boundary,
                self._buf_material_index,
                self._buf_material_props,
                self._buf_air_elem_face_area,
                self._buf_air_elem_volume,
                np.int32(self.n_elements),
                np.float64(self.rho_air),
                np.float64(self.air_sound_speed),
                np.float64(z_acoustic),
                np.float64(dt_air),
            )
        
        cl.enqueue_nd_range_kernel(
            self.queue,
            self._kernel_air_inject_force_reduce if self.air_inject_use_reduce 
                else self._kernel_air_inject_force,
            (self._global_size,),
            (self._local_size,),
        )
        
        # Reduce step (if needed - sum contributions):
        if self.air_inject_use_reduce:
            self._reduce_air_velocity_from_elements()
        
        # ========== STAGE 2: Air Wave Equation (unchanged) ==========
        self._kernel_air_step.set_args(
            self._buf_air_prev,
            self._buf_air_curr,
            self._buf_air_next,
            np.int32(self.nx_air),
            np.int32(self.ny_air),
            np.int32(self.nz_air),
            np.float64(self.dx_air),
            np.float64(self.dy_air),
            np.float64(self.dz_air),
            np.float64(dt_air),
            np.float64(self.air_sound_speed),
            np.float64(self.air_bulk_damping),
            np.float64(self.air_boundary_damping),
            np.int32(self.air_sponge_cells),
            np.float64(self.air_pressure_clip_pa),
        )
        cl.enqueue_nd_range_kernel(
            self.queue,
            self._kernel_air_step,
            (self._air_global_size,),
            (self._local_size,),
        )
        
        # Rotate buffers: curr → prev, next → curr
        self._buf_air_prev, self._buf_air_curr, self._buf_air_next = (
            self._buf_air_curr,
            self._buf_air_next,
            self._buf_air_prev,
        )
    
    # ========== STAGE 3: Air Pressure → FE Force ==========
    # Compute ∇p from air field and apply restoring force to FE
    
    self._kernel_air_force_to_fe.set_args(
        self._buf_force_external,              # Output: accumulate forces
        self._buf_air_curr,                    # Input: current pressure field
        self._buf_air_map,                     # Element → cell mapping NEW
        self._buf_boundary,                    # Boundary mask
        self._buf_material_index,              # Material ID per element
        self._buf_material_props,              # Material parameters
        self._buf_air_elem_volume,             # Element volume for force
        np.int32(self.n_elements),
        np.int32(self.nx_air),                # Grid size for gradient
        np.int32(self.ny_air),
        np.int32(self.nz_air),
        np.float64(self.dx_air),               # Grid spacing for ∇p
        np.float64(self.dy_air),
        np.float64(self.dz_air),
        np.float64(1.0),                       # contact_area_scale
    )
    cl.enqueue_nd_range_kernel(
        self.queue,
        self._kernel_air_force_to_fe,
        (self._global_size,),
        (self._local_size,),
    )
    
    self.queue.finish()
    
    # Copy updated forces back to CPU
    cl.enqueue_copy(self.queue, self.force_external, self._buf_force_external)
    self.queue.finish()
```

### Change 3B: Update Buffer Synchronization

```python
def _sync_simulation_buffers(self) -> None:
    """Sync CPU arrays → GPU buffers."""
    # ... existing code ...
    
    # NEW: Sync element-to-air-cell mapping
    cl.enqueue_copy(self.queue, self._buf_air_map, self.air_map)
    
    # ... rest of existing code ...
```

---

## File 4: Minor Updates

### Change 4A: Update force_external Initialization

```python
def __init__(self, ...):
    # In initial setup:
    self.force_external = np.zeros(self.n_dof, dtype=np.float64)
    
    # Ensure it's cleared before each coupling step
    # (Add to _run_air_coupling):
    self.force_external[:] = 0.0
```

### Change 4B: Add Acoustic Impedance Property

```python
@property
def acoustic_impedance(self) -> float:
    """Air-side acoustic impedance Z = ρ·c [Pa·s/m]."""
    return self.rho_air * self.air_sound_speed

@property
def acoustic_impedance_membrane(self) -> float:
    """FE membrane acoustic impedance (from material library)."""
    return self.material_props[
        int(MAT_MEMBRANE), MAT_PROP_ACOUSTIC_IMPEDANCE_Z
    ]
```

---

## Testing Checklist

After integration, verify:

- [ ] Material library loads with 10 columns
- [ ] `acoustic_impedance_z` values are > 0
- [ ] `acoustic_damping_alpha` values are reasonable (10-200 1/s)
- [ ] Air element mapping is created (check bounds)
- [ ] New kernels compile without errors
- [ ] `_run_air_coupling()` executes without GPU errors
- [ ] Force values are in reasonable range (check force_external array)
- [ ] Energy conservation: pressure should decay exponentially
- [ ] Frequency response shows membrane resonance peak

---

## Performance Notes

**Expected overhead:**
- New gradient computation: ~5-10% slower than old velocity approach
- Better numerical stability allows larger timesteps (+10-20%)
- Net effect: ~5-15% overall simulation time increase, but better physics

**Memory overhead:**
- Air element mapping: `n_elements * 4 bytes` (negligible)
- No new large buffers needed

---

## Backward Compatibility

Old 8-column material libraries are **automatically expanded** to 10 columns with default values. No manual migration needed for existing projects.

To explicitly control new parameters, either:
1. Edit `.fe_lib` library file (add 2 columns)
2. Create new library with all 10 columns from scratch

---

**Last Updated:** 2026-03-21  
**Ready for:** Integration into main codebase
