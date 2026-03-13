# -*- coding: utf-8 -*-
"""
Mesh list dock panel.
Depends: PySide6 only. No project_model — operates on display data.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDockWidget,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MeshListPanel(QDockWidget):
    """Dock with search, mesh list, add/remove buttons."""

    selection_changed = Signal(object)  # model index (int) or None
    search_changed = Signal(str)  # filter text for main_window to refresh
    add_clicked = Signal()
    remove_clicked = Signal()
    isolate_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Mesh List", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        panel = QWidget()
        layout = QVBoxLayout(panel)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search mesh...")
        self.search.textChanged.connect(self._on_search_changed)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_current_row_changed)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("Add")
        self.btn_remove = QPushButton("Remove")
        self.btn_isolate = QPushButton("Isolate")
        self.btn_add.clicked.connect(self.add_clicked.emit)
        self.btn_remove.clicked.connect(self.remove_clicked.emit)
        self.btn_isolate.clicked.connect(self.isolate_clicked.emit)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addWidget(self.btn_isolate)

        layout.addWidget(self.search)
        layout.addWidget(self.list_widget, 1)
        layout.addLayout(btn_row)

        self.setWidget(panel)
        self._search_filter = ""
        self._model_indices: list[int] = []  # list_widget row -> model index

    def set_meshes(self, display_items: list[tuple[str, int]]) -> None:
        """display_items: [(display_text, model_index), ...]"""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self._model_indices.clear()
        for text, idx in display_items:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, idx)
            self.list_widget.addItem(item)
            self._model_indices.append(idx)
        self.list_widget.blockSignals(False)
        if self.list_widget.count() > 0 and self.list_widget.currentRow() < 0:
            self.list_widget.setCurrentRow(0)

    def get_selected_index(self) -> int | None:
        """Return model index of selected item, or None."""
        item = self.list_widget.currentItem()
        if item is None:
            return None
        idx = item.data(Qt.UserRole)
        return int(idx) if idx is not None else None

    def set_selection_by_model_index(self, model_index: int) -> None:
        """Select list row that corresponds to model_index."""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == model_index:
                self.list_widget.blockSignals(True)
                self.list_widget.setCurrentRow(i)
                self.list_widget.blockSignals(False)
                break

    def set_selection_by_row(self, row: int) -> None:
        """Select by list row (for mock pick). Use row=-1 to clear selection."""
        if row >= 0 and row < self.list_widget.count():
            self.list_widget.setCurrentRow(row)
        elif row < 0:
            self.list_widget.setCurrentRow(-1)

    def count(self) -> int:
        return self.list_widget.count()

    def get_search_filter(self) -> str:
        return self.search.text().strip().lower()

    def _on_search_changed(self, _: str) -> None:
        self.search_changed.emit(self.get_search_filter())

    def _on_current_row_changed(self, row: int) -> None:
        idx = None if row < 0 else self.get_selected_index()
        self.selection_changed.emit(idx)
