# Quick Start (Current)

This is the practical quick-reference for the current project state.

## 1) Install

```bash
pip install -r requirements.txt
```

OpenCL prerequisites:
- an OpenCL-capable driver/device is available,
- `cl_khr_fp64` is supported.

## 2) Run from CLI

```bash
py -3 diaphragm_opencl.py --dt 1e-6 --duration 0.01
```

Useful variants:

```bash
# Sine excitation
py -3 diaphragm_opencl.py --force-shape sine --force-amplitude 3 --force-freq 800 --duration 0.03

# Validation mode
py -3 diaphragm_opencl.py --validate --pre-tension 10

# Load and plot saved result/run-case
py -3 diaphragm_opencl.py --sim-file results/sim_results_YYYYMMDD_HHMMSS.pkl --plot-sim-file
```

## 3) Start GUI

```bash
py -3 -m fe_ui
```

With startup parameters:

```bash
py -3 -m fe_ui --project projects/priboy_1_project.fe_project --auto-run
```

## 4) Start simulation server for GUI

```bash
py -3 simulation_server.py --server --host 127.0.0.1 --port 8765
```

In GUI, you can use:
- local server mode (auto-start),
- remote server mode (manual connect).

## 5) Current solver/kernel notes

- OpenCL integrator is **RK4**.
- Active kernels in `diaphragm_opencl_kernel.cl`:
  - `diaphragm_rk4_acc`
  - `diaphragm_rk4_stage_state`
  - `diaphragm_rk4_finalize`
- Material row stride is **8**:
  `[density, E_parallel, E_perp, poisson, Cd, eta_visc, coupling_recv, acoustic_inject]`.

## 6) Important docs

- `README.md` — project overview.
- `PROJECT_DOCUMENTATION.md` — technical documentation.
- `fe_ui/FE_UI_STRUCTURE.md` — GUI architecture.

---

### Archive docs

`FORCE_BASED_COUPLING.md`, `IMPLEMENTATION_GUIDE.md`, `OLD_VS_NEW_COMPARISON.md`, and `RK4_AIR_INTEGRATION.md` describe previous/experimental coupling plans and are not fully aligned with current implementation.
