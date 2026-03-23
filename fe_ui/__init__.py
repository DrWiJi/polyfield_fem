# -*- coding: utf-8 -*-
'FE UI - modular GUI for FE workflow.\n\nModules and dependencies:\n  constants - no dependencies\n  viewport - PySide6, optional pyvista/pyvistaqt\n  mesh_list - PySide6\n  mesh_editor - PySide6\n  simulation - PySide6\n  main_window - all panels + project_model\n  app - main_window + QApplication'

from .app import run_app 

__all__ =["run_app"]
