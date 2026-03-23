# Polyfield

OpenCL/GPU finite-element diaphragm simulation with a modular desktop GUI for project editing, topology generation, and simulation runs via local or remote server.

## Current capabilities

- **FE mechanics:** nonlinear elasticity, pre-tension, boundary constraints, material library.
- **Integrator:** **RK4** on OpenCL (`diaphragm_rk4_acc`, `diaphragm_rk4_stage_state`, `diaphragm_rk4_finalize`).
- **CLI workflow:** direct simulation runs, validation mode, replay/plot from saved run/result files (`--sim-file`).
- **GUI workflow (PySide6):** project editor, mesh import, topology generator, boundary conditions, results panel.
- **Server mode:** network backend in `simulation_server.py` used by the GUI.

> Note: legacy markdown files about historical force-coupling experiments are kept for reference and treated as archive material.

## Requirements

- Python 3.10+
- OpenCL 1.2+ with `cl_khr_fp64`
- Dependencies from `requirements.txt`:
  - `pyopencl`, `numpy`, `matplotlib`
  - `PySide6`, `pyvista`, `pyvistaqt`, `trimesh`, `gmsh`

Install:

```bash
pip install -r requirements.txt
```

## Quick start (CLI)

```bash
# Basic impulse run
py -3 diaphragm_opencl.py --dt 1e-6 --duration 0.01

# Sinusoidal excitation
py -3 diaphragm_opencl.py --force-shape sine --force-amplitude 3 --force-freq 800 --duration 0.03

# Natural-frequency validation
py -3 diaphragm_opencl.py --validate --pre-tension 10

# Load and plot a saved result/run-case
py -3 diaphragm_opencl.py --sim-file results/sim_results_YYYYMMDD_HHMMSS.pkl --plot-sim-file
```

See all arguments:

```bash
py -3 diaphragm_opencl.py --help
```

## GUI

```bash
py -3 -m fe_ui
```

GUI CLI options:

```bash
py -3 -m fe_ui --help
```

Main options:
- `--project/-p` load a project on startup.
- `--material-library/-m` replace the default material library.
- `--auto-run` auto-start simulation when a saved topology exists.

## Simulation server

```bash
py -3 simulation_server.py --server --host 127.0.0.1 --port 8765
```

The GUI can auto-start a local server or connect to a remote one.

## Project structure

- `diaphragm_opencl.py` — core model, CLI, post-processing.
- `diaphragm_opencl_kernel.cl` — RK4 OpenCL kernels.
- `simulation_server.py` / `simulation_io.py` — network backend and file/wire formats.
- `project_model.py` — project data model (dataclasses + JSON).
- `fe_ui/` — GUI application package.

## Documentation

- [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) — technical architecture and APIs.
- [QUICK_START.md](QUICK_START.md) — practical run scenarios.
- [fe_ui/FE_UI_STRUCTURE.md](fe_ui/FE_UI_STRUCTURE.md) — GUI module architecture.
