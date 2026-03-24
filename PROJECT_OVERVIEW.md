# General idea of this project

This repository is a **research and simulation toolchain for thin structures such as headphone diaphragms**: a **finite-element mechanical model** runs largely on the **GPU via OpenCL**, with optional **coupling to a 3D acoustic pressure field** so membrane motion and air can interact in one time-stepping loop.

## What problem it addresses

Engineering headphones (and similar transducers) requires understanding how a **membrane or composite stack** moves under **pre-tension, nonlinear elasticity, boundary fixation, and external or acoustic loading**. The project packages that physics into a **numerical experiment** you can repeat, validate against simple analytics where possible, and drive from either a **command line** or a **desktop UI**.

## Core technical idea

- **Discretization:** The diaphragm is represented as a mesh of elements with **six degrees of freedom per node** (translations and small rotations), with neighbor connectivity and material properties driving internal forces.
- **Compute model:** **One OpenCL work-item per element** (or equivalent parallel layout): elastic forces, damping, external pressure, and **time integration** are executed on the device. The host (Python) sets up buffers, applies boundary logic, orchestrates substeps, and handles I/O and plotting.
- **Time integration:** The simulation advances with a **Runge–Kutta integrator** on the GPU (see `README.md` for the current stage kernels); this keeps the heavy inner loop on the accelerator.
- **Acoustics (optional path):** A **3D Cartesian grid** can hold a wave-style pressure field; **bidirectional coupling** maps membrane normal motion and area to injection into the air grid, and pressure differences back to forces on the membrane—so you can study **structure–air interaction** in addition to pure structural response.

## How the pieces fit together

| Piece | Role |
|--------|------|
| `diaphragm_opencl.py` + `diaphragm_opencl_kernel.cl` | Physics, OpenCL orchestration, CLI entry, validation hooks, plotting. |
| `topology_generator.py` | Build or adjust mesh-like topology for non-trivial layouts. |
| `project_model.py` | Serialize **projects**: meshes, materials, settings, and **run history** in a single portable format. |
| `fe_ui/` (PySide6, PyVista) | **Desktop GUI**: edit projects, assign materials and boundaries, launch runs, inspect results. |
| `simulation_server.py` + `simulation_io.py` | **Network backend** so the GUI can offload a run to a **local or remote** Python process that owns the GPU. |

## Typical workflows

1. **CLI:** Run `diaphragm_opencl.py` with time step, duration, forcing, and optional validation or replay from saved result files—good for batch studies and scripting.
2. **GUI + server:** Start the simulation server if you want separation from the UI machine, then `python -m fe_ui` to **author a project**, generate topology, configure materials from a **material library**, and **visualize** displacements and summaries after a run.

## Relation to documentation in the repo

- **`README.md`** — practical name (**Polyfield**), requirements, and up-to-date **quick start** for CLI and GUI.
- **`PROJECT_DOCUMENTATION.md`** — deeper architecture and physics notes; treat kernel/integrator **names and step counts** as subject to change; prefer the code and `README.md` for the exact current integrator.
- Other markdown files (e.g. coupling notes, comparisons) are **supplementary** or historical context for specific experiments.

In short: **this project is a GPU-accelerated finite-element diaphragm dynamics simulator with optional coupled air, wrapped in project-oriented tools and a optional client–server GUI** for interactive work and publication-style numerical studies.
