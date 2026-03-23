# -*- coding: utf-8 -*-
"""
Application entry point.
Depends: main_window, QApplication.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project_model is importable (may be in parent of fe_ui)
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
_parent = os.path.dirname(_project_root)
for p in (_project_root, _parent):
    if p not in sys.path:
        sys.path.insert(0, p)

from PySide6.QtWidgets import QApplication

from .app_controller import AppController


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FE UI: diaphragm simulation with mesh editor and topology generator.",
    )
    parser.add_argument(
        "--project", "-p",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to project file to load on startup.",
    )
    parser.add_argument(
        "--material-library", "-m",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to material library file (.fe_lib or .json) to load on startup (replaces default).",
    )
    parser.add_argument(
        "--auto-run",
        action="store_true",
        help="Automatically start simulation if project is loaded and has saved topology.",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug logging (mesh import, etc.).",
    )
    return parser.parse_args()


def run_app() -> int:
    """Create application, controller, first window, run event loop."""
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )
    app = QApplication(sys.argv)
    controller = AppController(material_library_path=args.material_library)
    controller.new_window(
        load_path=args.project,
        auto_run_simulation=args.auto_run,
    )
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run_app())
