# -*- coding: utf-8 -*-
"""
Material Library window: table UI, add/edit/delete, import/export, reset to stock.
Depends: PySide6, material_library_model.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .material_library_model import MaterialEntry, MaterialLibraryModel


_COLUMNS = [
    ("name", "Name", None),
    ("density", "Density (kg/m³)", "density"),
    ("E_parallel", "E_parallel (Pa)", "E_parallel"),
    ("E_perp", "E_perp (Pa)", "E_perp"),
    ("poisson", "Poisson", "poisson"),
    ("Cd", "Cd", "Cd"),
    ("eta_visc", "η_visc (Pa·s)", "eta_visc"),
    ("coupling_gain", "Coupling gain", "coupling_gain"),
]


class MaterialEditDialog(QDialog):
    """Dialog for adding or editing a material entry."""

    def __init__(self, parent=None, entry: MaterialEntry | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Material" if entry else "Add Material")
        layout = QFormLayout(self)

        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("e.g. membrane, foam_ve3015")
        layout.addRow("Name", self.ed_name)

        self.sp_density = QDoubleSpinBox()
        self.sp_density.setRange(1.0, 50000.0)
        self.sp_density.setDecimals(1)
        layout.addRow("Density (kg/m³)", self.sp_density)

        self.sp_E_parallel = QDoubleSpinBox()
        self.sp_E_parallel.setRange(1e3, 2e12)
        self.sp_E_parallel.setDecimals(0)
        self.sp_E_parallel.setValue(5e9)
        layout.addRow("E_parallel (Pa)", self.sp_E_parallel)

        self.sp_E_perp = QDoubleSpinBox()
        self.sp_E_perp.setRange(1e3, 2e12)
        self.sp_E_perp.setDecimals(0)
        self.sp_E_perp.setValue(3.5e9)
        layout.addRow("E_perp (Pa)", self.sp_E_perp)

        self.sp_poisson = QDoubleSpinBox()
        self.sp_poisson.setRange(0.0, 0.499)
        self.sp_poisson.setSingleStep(0.01)
        self.sp_poisson.setValue(0.30)
        layout.addRow("Poisson", self.sp_poisson)

        self.sp_Cd = QDoubleSpinBox()
        self.sp_Cd.setRange(0.5, 2.0)
        self.sp_Cd.setSingleStep(0.05)
        self.sp_Cd.setValue(1.0)
        layout.addRow("Cd", self.sp_Cd)

        self.sp_eta_visc = QDoubleSpinBox()
        self.sp_eta_visc.setRange(0.0, 1000.0)
        self.sp_eta_visc.setDecimals(2)
        self.sp_eta_visc.setValue(0.8)
        layout.addRow("η_visc (Pa·s)", self.sp_eta_visc)

        self.sp_coupling_gain = QDoubleSpinBox()
        self.sp_coupling_gain.setRange(0.0, 1.0)
        self.sp_coupling_gain.setSingleStep(0.05)
        self.sp_coupling_gain.setValue(0.9)
        layout.addRow("Coupling gain", self.sp_coupling_gain)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

        if entry:
            self.ed_name.setText(entry.name)
            self.sp_density.setValue(entry.density)
            self.sp_E_parallel.setValue(entry.E_parallel)
            self.sp_E_perp.setValue(entry.E_perp)
            self.sp_poisson.setValue(entry.poisson)
            self.sp_Cd.setValue(entry.Cd)
            self.sp_eta_visc.setValue(entry.eta_visc)
            self.sp_coupling_gain.setValue(entry.coupling_gain)

    def get_entry(self) -> MaterialEntry:
        return MaterialEntry(
            name=self.ed_name.text().strip() or "unnamed",
            density=self.sp_density.value(),
            E_parallel=self.sp_E_parallel.value(),
            E_perp=self.sp_E_perp.value(),
            poisson=self.sp_poisson.value(),
            Cd=self.sp_Cd.value(),
            eta_visc=self.sp_eta_visc.value(),
            coupling_gain=self.sp_coupling_gain.value(),
        )


class MaterialLibraryWindow(QWidget):
    """Standalone window for material library: table, add/edit/delete, import/export, reset."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Material Library")
        self.resize(900, 400)
        self._library = MaterialLibraryModel()

        layout = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_edit = QPushButton("Edit")
        self.btn_remove = QPushButton("Remove")
        self.btn_import = QPushButton("Import...")
        self.btn_export = QPushButton("Export...")
        self.btn_reset = QPushButton("Reset to Stock")
        self.btn_add.clicked.connect(self._action_add)
        self.btn_edit.clicked.connect(self._action_edit)
        self.btn_remove.clicked.connect(self._action_remove)
        self.btn_import.clicked.connect(self._action_import)
        self.btn_export.clicked.connect(self._action_export)
        self.btn_reset.clicked.connect(self._action_reset)
        toolbar.addWidget(self.btn_add)
        toolbar.addWidget(self.btn_edit)
        toolbar.addWidget(self.btn_remove)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.btn_import)
        toolbar.addWidget(self.btn_export)
        toolbar.addSpacing(20)
        toolbar.addWidget(self.btn_reset)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[1] for c in _COLUMNS])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self._action_edit)
        layout.addWidget(self.table)

        self._refresh_table()

    def _refresh_table(self) -> None:
        entries = self._library.materials
        self.table.setRowCount(len(entries))
        for row, e in enumerate(entries):
            for col, (_, _, key) in enumerate(_COLUMNS):
                if key is None:
                    item = QTableWidgetItem(e.name)
                else:
                    val = getattr(e, key)
                    item = QTableWidgetItem(f"{val:.6g}" if isinstance(val, float) else str(val))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row, col, item)

    def _selected_row(self) -> int:
        return self.table.currentRow()

    def _action_add(self) -> None:
        dlg = MaterialEditDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._library.add(dlg.get_entry())
            self._refresh_table()

    def _action_edit(self) -> None:
        row = self._selected_row()
        if row < 0:
            QMessageBox.information(self, "Edit", "Select a row first.")
            return
        entry = self._library.materials[row]
        dlg = MaterialEditDialog(self, entry)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._library.update(row, dlg.get_entry())
            self._refresh_table()

    def _action_remove(self) -> None:
        row = self._selected_row()
        if row < 0:
            QMessageBox.information(self, "Remove", "Select a row first.")
            return
        if QMessageBox.question(
            self, "Remove Material",
            f"Remove material '{self._library.materials[row].name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) == QMessageBox.Yes:
            self._library.remove(row)
            self._refresh_table()

    def _action_import(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Material Library",
            "", "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            merged = self._library.import_and_merge(path)
            self._refresh_table()
            QMessageBox.information(self, "Import", f"Merged {merged} material(s).")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    def _action_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Material Library",
            "", "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            self._library.save_json(path)
            QMessageBox.information(self, "Export", "Library exported.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _action_reset(self) -> None:
        if QMessageBox.question(
            self, "Reset to Stock",
            "Replace library with stock defaults? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        ) == QMessageBox.Yes:
            self._library.clear_and_reset_to_stock()
            self._refresh_table()

    def get_library(self) -> MaterialLibraryModel:
        return self._library
