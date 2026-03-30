# FE UI — Module Structure

Modular desktop GUI for FE project preparation and simulation control.

## Dependency layout (minimal coupling)

```text
constants     — no heavy dependencies
viewport      — PySide6, optional pyvista + pyvistaqt
mesh_list     — PySide6
mesh_editor   — PySide6
simulation    — PySide6
results       — PySide6 + matplotlib
main_window   — panels + project_model (+ optional trimesh/pyvista)
app           — AppController + QApplication
```

## Package structure

```text
fe_ui/
├── __init__.py
├── __main__.py
├── app.py
├── app_controller.py
├── app_model.py
├── constants.py
├── viewport.py
├── mesh_list_panel.py
├── mesh_editor_panel.py
├── simulation_panel.py
├── results_panel.py
├── boundary_conditions_panel.py
├── topology_generator_panel.py
├── material_library_model.py
├── material_library_window.py
└── FE_UI_STRUCTURE.md
```

## Core design principles

1. **State flows through AppModel/AppController**
   - UI panels do not directly own persistence logic.
   - Main window coordinates panel state and project state.

2. **Signals/slots over tight coupling**
   - Panels emit events (`selection_changed`, `apply_clicked`, `run_clicked`, etc.).
   - `FeMainWindow` binds these events to model mutations and backend calls.

3. **Optional rendering/import dependencies are isolated**
   - `pyvista/pyvistaqt` and `trimesh` are used in viewport/import paths.
   - App remains operational in reduced mode when unavailable.

4. **Client/server simulation workflow**
   - `SimulationPanel` triggers actions.
   - `SimulationClientBridge` handles socket communication.
   - `simulation_server.py` executes runs and returns packed results.
   - Topology payload now carries air boundary kinds (`open`/`rigid`) produced by `topology_generator.py`.

5. **Material library is shared across windows**
   - `AppController` manages one shared `MaterialLibraryModel`.
   - Multiple project windows can be opened concurrently.

## Entry points

```bash
python -m fe_ui
# or
python fe_ui/app.py
# or
from fe_ui import run_app; run_app()
```
