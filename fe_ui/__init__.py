# -*- coding: utf-8 -*-
"""
FE UI — модульный GUI для FE workflow.

Модули и зависимости:
  constants    — нет зависимостей
  viewport     — PySide6, опционально pyvista/pyvistaqt
  mesh_list    — PySide6
  mesh_editor  — PySide6
  simulation   — PySide6
  main_window  — все панели + project_model
  app          — main_window + QApplication
"""

from .app import run_app

__all__ = ["run_app"]
