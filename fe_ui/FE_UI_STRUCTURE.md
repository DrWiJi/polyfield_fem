# FE UI — модульная структура

Модульный GUI для подготовки проектов FE.

## Зависимости модулей (минимальны)

```
constants     — нет зависимостей
viewport      — PySide6, опционально pyvista + pyvistaqt
mesh_list     — PySide6
mesh_editor   — PySide6
simulation    — PySide6
main_window   — панели + project_model (+ опционально trimesh, pyvista)
app           — main_window + QApplication
```

## Структура пакета

```
fe_ui/
├── __init__.py       # from .app import run_app
├── __main__.py       # точка входа: python -m fe_ui
├── app.py            # run_app(), расширение sys.path для project_model
├── constants.py      # ROLES, FORCE_SHAPES
├── viewport.py       # ViewportPlaceholder, create_viewport(), has_pyvista()
├── mesh_list_panel.py   # MeshListPanel — список мешей, поиск, Add/Remove
├── mesh_editor_panel.py # MeshEditorPanel — вкладки Identity/Material/Membrane/Transform/Boundary
├── simulation_panel.py  # SimulationPanel — solver, excitation, run/stop, console
├── main_window.py    # FeMainWindow — оркестрация, Project, viewport, импорт
└── FE_UI_STRUCTURE.md
```

## Принципы

1. **Панели не знают project_model** — работают с `list[tuple[str,int]]` (mesh_list) и `dict` (mesh_editor, simulation). main_window переводит данные в/из модели.

2. **Сигналы вместо прямых вызовов** — панели эмитят `selection_changed`, `apply_clicked`, `run_clicked` и т.д.; main_window подключает слоты.

3. **Опциональные зависимости изолированы** — trimesh и pyvista используются только в main_window/viewport, при отсутствии приложение работает (placeholder viewport, без импорта мешей).

4. **Запуск**:
   ```bash
   python -m fe_ui
   # или
   python fe_ui/app.py
   # или
   from fe_ui import run_app; run_app()
   ```
