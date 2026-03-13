# -*- coding: utf-8 -*-
"""
Application entry point.
Depends: main_window, QApplication.
"""

from __future__ import annotations

import os
import sys

# Ensure project_model is importable (may be in parent of fe_ui)
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
_parent = os.path.dirname(_project_root)
for p in (_project_root, _parent):
    if p not in sys.path:
        sys.path.insert(0, p)

from PySide6.QtWidgets import QApplication

from .app_controller import AppController


def run_app() -> int:
    """Create application, controller, first window, run event loop."""
    app = QApplication(sys.argv)
    controller = AppController()
    controller.new_window()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_app())
