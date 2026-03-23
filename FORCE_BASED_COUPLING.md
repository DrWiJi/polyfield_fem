# Force-Based Air-FE Coupling: Physics-Correct Algorithm

## Overview

The refactored coupling uses **force** as the primary variable instead of velocity, enabling physically correct acoustic modeling. This approach:

- ✅ Conserves energy through acoustic impedance
- ✅ Properly models acoustic radiation impedance  
- ✅ Integrates physical units directly (Pa·s/m, kg/m³, m/s)
- ✅ Enables broadband frequency response (impulse response evaluation)
- ✅ Separates FE dynamics from air field consistently

---

## Mathematical Foundation

### Key Parameters

**Acoustic Impedance** (formerly "abstract coupling gain"):
$$Z = \rho \cdot c \quad \text{[Pa·s/m] or [kg/(m²·s)]}$$

Where:
- ρ = air density ≈ 1.2 kg/m³
- c = sound speed ≈ 343 m/s  
- **Z_air ≈ 412 kg/(m²·s)**

### Physics-Based Relations

#### 1. **Sound Pressure Level (SPL) from FE Velocity**

When a membrane vibrates with normal velocity $v_n$, it generates acoustic pressure:

$$p = Z \cdot v_n = \rho \cdot c \cdot v_n$$

This is the **plane wave approximation** in free field (far field). Near field and boundary effects modify this.

#### 2. **Force Injection: FE → Air**

From Newton's laws at the FE-air interface:

$$F_{\text{air}} = p \cdot A = Z \cdot v_n \cdot A \quad \text{[N]}$$

Where:
- A = contact surface area [m²]
- $v_n$ = normal velocity component [m/s]

This force accelerates the air: $a = F / (\rho \cdot V)$ where V is air cell volume.

#### 3. **Force Reception: Air → FE**

By Newton's 3rd law (action-reaction), pressure creates a **normal stress** on the FE surface:

$$\sigma_n = p = Z \cdot v_n \quad \text{[Pa]}$$

The net force from pressure gradient:

$$F_{\text{FE}} = -\nabla p \cdot V = -\left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}, \frac{\partial p}{\partial z}\right) \cdot V \quad \text{[N]}$$

This represents the **restoring force** from compressed air on a deforming structure.

#### 4. **Energy Conservation**

Acoustic power transmitted through interface:

$$P = F \cdot v = Z \cdot v_n^2 \cdot A \quad \text{[W]}$$

From FE perspective:
$$P_{\text{FE}} = -F_{\text{air}} \cdot v_{\text{FE}} = -Z \cdot v_n^2 \cdot A$$

This is **power out** (negative sign means energy leaves FE and enters air). Over a cycle, it represents radiated acoustic energy.

From air perspective, this energy becomes acoustic waves: $E_{\text{acoustic}} \propto \int p^2 \, dV$.

---

## Implementation Architecture

### Layer 1: Material Properties

Extend the material library from 8 to **10 columns**:

```python
# Old format (8 columns):
[density, E_parallel, E_perp, poisson, Cd, eta_visc, 
 coupling_recv, acoustic_inject]

# New format (10 columns):
[density, E_parallel, E_perp, poisson, Cd, eta_visc,
 coupling_recv, acoustic_inject,
 acoustic_impedance_z,      # New (9): ρ*c [Pa·s/m]
 acoustic_damping_alpha]    # New (10): frequency-dependent loss [1/s]
```

**Material Property Column Definitions:**

| Column | Name | Unit | Meaning |
|--------|------|------|---------|
| 0 | density | kg/m³ | Material mass |
| 1 | E_parallel | Pa | Young's modulus (parallel) |
| 2 | E_perp | Pa | Young's modulus (perpendicular) |
| 3 | poisson | - | Poisson's ratio |
| 4 | Cd | - | Drag coefficient (air) |
| 5 | eta_visc | Pa·s | Viscous damping (internal) |
| 6 | **coupling_recv** | [0,1] | **Contact conductance** (air → FE) |
| 7 | **acoustic_inject** | [0,1] | **Radiation factor** (FE → air) |
| 8 | **acoustic_impedance_z** | Pa·s/m | **Z = ρ·c** acoustic impedance |
| 9 | **acoustic_damping_alpha** | 1/s | **Energy loss rate** in material |

### Layer 2: OpenCL Kernels (Force-Based)

#### Kernel A: `air_inject_from_fe_velocity_force`

**Purpose:** Transfer FE velocity → acoustic force → air velocity

**Algorithm:**
```
For each FE element:
  1. Read FE velocity v_fe [m/s]
  2. Compute normal component: v_n = v_fe · n̂
  3. Compute acoustic pressure: p = Z · v_n [Pa]
  4. Compute force on air: F_air = p · A [N]
  5. Apply contact conductance: F_air *= coupling_recv
  6. Distribute to air cells: Δv_air = F_air / (ρ · V_air) [m/s]
```

**Key Formula:**
```
p_equiv = Z * v_n = ρ * c * v_n

F_air = contact_conductance * p_equiv * A_contact

Δv_air = F_air * Δt / (ρ * V_air)
```

**Physical Meaning:**
- `contact_conductance` (column 6) scales the coupling efficiency
  - 1.0 = perfect contact (rigid interface)
  - 0.5 = 50% acoustic impedance mismatch
  - ≈ area_contact / area_acoustic

#### Kernel B: `add_air_pressure_force_to_fe`

**Purpose:** Apply air pressure gradient → FE force

**Algorithm:**
```
For each FE element:
  1. Find corresponding air cell (from mapping)
  2. Compute pressure gradient ∇p = (∂p/∂x, ∂p/∂y, ∂p/∂z)
     using finite differences from neighboring cells
  3. Compute force reaction: F_FE = -∇p * V_elem
  4. Scale by contact factor: F_FE *= coupling_recv
  5. Add to external force accumulator
```

**Key Formula:**
```
∇p = (p[i+1,j,k] - p[i-1,j,k]) / (2·Δx) ... [finite difference]

F_FE = -∇p * V_elem * coupling_recv

// This is the net acoustic restoring force on FE
```

**Physical Meaning:**
- `coupling_recv` (column 6) now represents the fraction of pressure that acts on the FE
  - 1.0 = full acoustic coupling (all pressure acts on FE)
  - 0.0 = no coupling (e.g., for sensors that only listen)
  - Can be < 1.0 for absorbing foam layers

---

## Parameter Guidance

### Coupling Parameters (Material Library, Column 6-7)

#### `coupling_recv` - Contact Conductance (Column 6)

Controls how strongly pressure from air affects FE motion.

**Default values by material:**

```python
# Membrane (primary driver)
coupling_recv = 1.0  # Full acoustic coupling

# Foam (absorbing)
coupling_recv = 0.3  # Partial coupling, mostly absorbs

# Leather (ear simulator)
coupling_recv = 0.6  # Moderate coupling

# Sensor/Microphone
coupling_recv = 1.0  # Measures pressure, doesn't emit
```

**Adjustment strategy:**
- If simulation is too stiff (pressure bounces back): **decrease**
- If pressure dissipates too slowly: **increase**
- For absorbing materials: set to ~0.3-0.5

#### `acoustic_inject` - Radiation Factor (Column 7)

Controls how FE velocity couples into air waves.

**Default values by material:**

```python
# Membrane (sound source)
acoustic_inject = 1.0  # Radiates monopole

# Foam (re-radiates from internal motion)
acoustic_inject = 0.2  # Some re-radiation

# Sensor (listens only)
acoustic_inject = 0.0  # Does NOT radiate

# Absorbing materials
acoustic_inject = 0.15  # Weak re-radiation
```

**Adjustment strategy:**
- If radiated sound is too weak: **increase**
- If simulation is unstable (feedback): **decrease**
- Foam `acoustic_inject ≈ 0.55 * coupling_recv` (energy balance)

### Acoustic Impedance (Column 8)

**Auto-calculated from material density and frequency:**

```python
# For air at 20°C
Z_air = rho_air * c_air
      = 1.2 [kg/m³] * 343 [m/s]
      = 412 [Pa·s/m]

# For other materials (default to air)
impedance_z = material_props[material_id, 8]
            = Z_air  # or specific value if known
```

**Never manually set unless you have measured data for:**
- Porous media (foam): Z ≈ 50-200 Pa·s/m (flow resistivity)
- Membrane: use air impedance (controls radiation, not material property)
- Head/ear tissue: Z ≈ 1.5 × 10⁶ Pa·s/m (much stiffer than air)

---

## Integration Steps (Python Side)

### Step 1: Update Material Library Loader

```python
# In diaphragm_opencl.py

_MATERIAL_PROPS_STRIDE = 10  # Changed from 8

def _expand_material_props_to_stride10(props: np.ndarray) -> np.ndarray:
    """Expand legacy material properties to 10 columns (force-based coupling)."""
    p = np.asarray(props, dtype=np.float64)
    
    if p.shape[1] == 10:
        return p  # Already in new format
    
    if p.shape[1] != 8:
        raise ValueError(f"Expected 8 or 10 columns, got {p.shape[1]}")
    
    # Add columns 8-9: acoustic_impedance_z, acoustic_damping_alpha
    new_cols = np.zeros((p.shape[0], 2), dtype=np.float64)
    
    for i in range(p.shape[0]):
        # Column 8: acoustic_impedance_z
        # Default: air impedance (works for radiating surfaces)
        new_cols[i, 0] = 412.0  # ρ * c for air at 20°C
        
        # Column 9: acoustic_damping_alpha
        # Frequency-independent loss rate [1/s]
        # Typical: 10-100 s⁻¹ (controls decay rate)
        if i == MAT_MEMBRANE:
            new_cols[i, 1] = 20.0   # Light damping for membrane
        elif i == MAT_FOAM_VE3015:
            new_cols[i, 1] = 150.0  # Heavy damping for foam
        elif i == MAT_SENSOR:
            new_cols[i, 1] = 50.0   # Moderate for sensor
        else:
            new_cols[i, 1] = 100.0  # Default
    
    return np.hstack((p, new_cols))
```

### Step 2: Update Kernel Launch

```python
# In diaphragm_opencl.py class

def _run_air_coupling(self, dt: float, pressure_pa: float | np.ndarray) -> None:
    # ... existing setup code ...
    
    # Get acoustic impedance
    z_acoustic = self.material_props[0, 8]  # From membrane material
    
    # Stage 1: FE velocity → Air force injection
    if self.air_inject_use_reduce:
        self._kernel_air_inject_force_reduce.set_args(
            self._buf_air_velocity_delta,
            self._buf_velocity,  # Now use full velocity (cols 0-2)
            self._buf_boundary,
            self._buf_material_index,
            self._buf_material_props,
            self._buf_air_elem_face_area,
            self._buf_air_elem_volume,  # NEW: volume needed
            np.int32(self.n_elements),
            np.float64(self.rho_air),
            np.float64(self.air_sound_speed),
            np.float64(z_acoustic),      # NEW: pass impedance
            np.float64(dt_air),           # NEW: air timestep
        )
    else:
        self._kernel_air_inject_force.set_args(...)
    
    # ... air_step_3d (unchanged) ...
    
    # Stage 2: Air pressure → FE force application
    self._kernel_pressure_to_force.set_args(
        self._buf_force_external,
        self._buf_air_curr,
        self._buf_air_elem_map,          # NEW: element to cell mapping
        self._buf_boundary,
        self._buf_material_index,
        self._buf_material_props,
        self._buf_air_elem_volume,
        np.int32(self.n_elements),
        np.int32(self.nx_air),
        np.int32(self.ny_air),
        np.int32(self.nz_air),
        np.float64(self.dx_air),
        np.float64(self.dy_air),
        np.float64(self.dz_air),
        np.float64(1.0),                 # contact_area_scale
    )
    cl.enqueue_nd_range_kernel(...)
```

### Step 3: Update Stability Parameters

```python
PlanarDiaphragmOpenCL.__init__():
    
    # CFL condition for air field coupling
    dt_max_air = 0.5 * min(self.dx_air, self.dy_air, self.dz_air) / self.air_sound_speed
    dt_air = min(dt, dt_max_air)
    
    # Acoustic impedance matching (prevents reflection)
    z_acoustic = self.rho_air * self.air_sound_speed
    
    # Energy dissipation rate (needed for realistic decay)
    alpha_damping = 50.0  # [1/s], tune based on measured Q-factor
```

---

## Validation & Testing

### 1. **Energy Conservation Check**

Monitor that total acoustic energy dissipates at expected rate:

```python
def measure_acoustic_energy(pressure, velocity, rho, c):
    """Total acoustic energy = kinetic + potential."""
    E_kinetic = 0.5 * rho * np.sum(velocity**2)
    E_potential = 0.5 * np.sum(pressure**2) / (rho * c**2)
    return E_kinetic + E_potential
```

Expected: Energy should **decay exponentially** with time constant ~1/(2·alpha_damping).

### 2. **Frequency Response (Impulse Response)**

Apply unit velocity impulse to membrane, measure:
- **SPL vs frequency** (should show membrane resonance)
- **Phase vs frequency** (should be smooth, continuous)
- **Phase lag** (should match ~90° at resonance)

### 3. **Physical Plausibility Checks**

✅ Magnitude of pressure = Z × v_n  
✅ Pressure decays away from source (∝ 1/r²)  
✅ Radiation impedance has correct frequency dependence  
✅ Absorption increases with foam thickness

---

## Tuning Guide for Headphone Simulation

### Goal: Simulate On-Head Response (HRTF)

**Key parameters:**

1. **Membrane (pinna simulator)**
   - `coupling_recv = 1.0` (full acoustic contact)
   - `acoustic_inject = 1.0` (primary radiator)
   - Material: BOET film (existing)

2. **Ear Canal Simulator**
   - Add layer with:
     - `coupling_recv = 0.8` (good acoustic contact)
     - `acoustic_inject = 0.1` (minimal re-radiation)
     - Absorption ∝ `acoustic_damping_alpha = 100-200 [1/s]`

3. **Head/Pinna Material**
   - `coupling_recv = 0.5` (partial coupling, some reflections)
   - `acoustic_inject = 0.2` (weak re-radiation from scattering)
   - Impedance mismatch with air creates reflections (HRTF spectral features)
   - `acoustic_damping_alpha = 50 [1/s]` (biological tissue loss)

4. **Air Cavity Between Membrane and Ear**
   - Critical for acoustic response
   - Use `air_padding = 5-10 mm` (realistic ear-device gap)
   - Fine mesh in this region: `air_grid_step = 1-2 mm`

### Tuning Workflow

```python
def tune_for_hrtf(model):
    """Iterative tuning for realistic headphone-on-head response."""
    
    # 1. Measure baseline (membrane only)
    impulse_response = model.run_simulation_with_impulse()
    freq_response = np.abs(np.fft.rfft(impulse_response))
    
    # Check: Resonance should be in 200-5000 Hz range
    resonance_freq = freq_response.argmax()
    
    # 2. Add ear simulator
    model.add_ear_layer(thickness=2e-3, damping=150.0)
    
    # 3. Verify phase response is smooth (no reflections in interior)
    phase = np.unwrap(np.angle(np.fft.rfft(impulse_response)))
    
    # 4. Compare with measured HRTF if available
    # Adjust air_padding, ear_layer properties, absorption
```

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Pressure explodes** | `z_acoustic` too large or CFL violated | Decrease `air_coupling_gain`, increase `air_grid_step` |
| **No sound output** | `acoustic_inject = 0` or `coupling_recv = 0` | Set both to 1.0 for membrane |
| **Ringing/Oscillation** | Impedance mismatch at interface | Gradually transition Z, add damping layer |
| **Energy grows** | Unstable coupling | Use force model (this refactor!), reduce `dt_air` |
| **Phase wrap** | High pressure, non-linearity | Add damping, reduce source amplitude |

---

## References

- **Acoustic Impedance**: ISO 3382-3 (room acoustics)
- **Radiation Impedance**: Morse & Ingard, "Theoretical Acoustics" (1968)
- **Finite Differences**: Strikwerda, "Finite Difference Schemes" (2004)
- **HRTF**: ITU-R BS.1534 (loudness, listening tests)

---

## Implementation Checklist

- [ ] Add columns 8-9 to material library loader
- [ ] Implement `air_inject_from_fe_velocity_force` kernel
- [ ] Implement `add_air_pressure_force_to_fe` kernel  
- [ ] Create FE ↔ air cell mapping (`air_elem_map`)
- [ ] Update kernel launching code in `_run_air_coupling()`
- [ ] Add validation tests (energy, frequency response)
- [ ] Tune material properties for target application
- [ ] Validate impulse response vs. measured HRTF (if available)

---

**Last Updated:** 2026-03-21  
**Status:** Ready for Integration  
**Compatibility:** Python 3.9+, PyOpenCL, OpenCL 1.2+
