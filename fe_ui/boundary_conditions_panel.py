# -*- coding: utf-8 -*-
"""
Boundary Conditions editor dock widget.
Depends: PySide6 only. Uses project_model.BoundaryCondition.
"""

from __future__ import annotations

from uuid import uuid4

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from .viewport import BoundaryConditionsViewport
from .widgets import ScientificDoubleSpinBox
from project_model import BoundaryCondition, MeshTransform


class BoundaryConditionsPanel(QDockWidget):
    """Boundary conditions editor dock. Emits signals for BC management."""

    bc_selected = Signal(str)  # bc_id
    bc_created = Signal(object)  # bc_data: dict
    bc_deleted = Signal(str)  # bc_id
    bc_updated = Signal(str, object)  # bc_id, bc_data: dict

    def __init__(self, parent=None) -> None:
        super().__init__("Boundary Conditions", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        widget = QWidget()
        main_layout = QHBoxLayout(widget)

        # Viewport (extends main, adds BC visualization)
        get_mesh_data = self._get_mesh_data_provider(parent)
        get_boundary_conditions = self._get_boundary_conditions_provider(parent)
        refresh_signals = self._get_refresh_signals(parent)
        self._mesh_viewport = BoundaryConditionsViewport(
            self,
            get_mesh_data=get_mesh_data,
            get_boundary_conditions=get_boundary_conditions,
            refresh_signals=refresh_signals,
        )
        self._plotter = self._mesh_viewport.plotter
        self._mesh_viewport.setMinimumSize(320, 240)
        main_layout.addWidget(self._mesh_viewport, 1)

        # Right side: BC list and properties
        splitter = QSplitter(Qt.Vertical)

        # BC list
        self.bc_table = QTableWidget()
        self.bc_table.setColumnCount(3)
        self.bc_table.setHorizontalHeaderLabels(["Name", "Type", "Meshes"])
        self.bc_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bc_table.setSelectionMode(QTableWidget.SingleSelection)
        self.bc_table.itemSelectionChanged.connect(self._on_bc_selection_changed)

        # Control buttons
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("+ Add")
        self.btn_remove = QPushButton("- Remove")
        self.btn_add.clicked.connect(self._on_add_bc)
        self.btn_remove.clicked.connect(self._on_remove_bc)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addStretch()

        # Properties editor
        self.properties_widget = self._build_properties_editor()

        bc_section = QWidget()
        bc_layout = QVBoxLayout(bc_section)
        bc_layout.addWidget(QLabel("Boundary Conditions:"))
        bc_layout.addWidget(self.bc_table)
        bc_layout.addLayout(btn_layout)
        bc_layout.addWidget(QLabel("Properties:"))
        bc_layout.addWidget(self.properties_widget, 1)

        splitter.addWidget(bc_section)
        splitter.setFixedWidth(320)
        main_layout.addWidget(splitter, 0)

        self.setWidget(widget)
        self._current_bc_id: str | None = None
        self._main_window = parent
        self._is_loading_bc = False
        self._set_editor_enabled(False)

    def closeEvent(self, event: QCloseEvent) -> None:
        if hasattr(self._mesh_viewport, "close_viewport"):
            self._mesh_viewport.close_viewport()
        super().closeEvent(event)
        if self._main_window and hasattr(self._main_window, "_on_boundary_conditions_window_closed"):
            self._main_window._on_boundary_conditions_window_closed()

    def _get_mesh_data_provider(self, parent):
        """Return get_mesh_data callable from main window if available."""
        if parent is not None and hasattr(parent, "_mesh_polydata_by_id") and hasattr(parent, "_app"):
            return lambda: (parent._mesh_polydata_by_id.copy(), parent._app.project.source_data.meshes)
        return lambda: ({}, [])

    def _get_boundary_conditions_provider(self, parent):
        """Return get_boundary_conditions callable from main window if available."""
        if parent is not None and hasattr(parent, "_app"):
            return lambda: parent._app.project.source_data.boundary_conditions
        return lambda: []

    def _get_meshes(self):
        """Return list of meshes from project for mesh selection."""
        if self._main_window and hasattr(self._main_window, "_app"):
            return self._main_window._app.project.source_data.meshes
        return []

    def _get_refresh_signals(self, parent) -> list:
        """Return signals to connect for viewport refresh."""
        if parent is not None and hasattr(parent, "mesh_viewport_changed") and hasattr(parent, "_app"):
            signals = [
                parent._app.project_changed,
                parent._app.transform_changed,
                parent.mesh_viewport_changed,
            ]
            if hasattr(parent, "bc_changed"):
                signals.append(parent.bc_changed)
            return signals
        return []

    def _build_properties_editor(self) -> QWidget:
        """Build the boundary condition properties editor."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # General group
        general_group = QGroupBox("General")
        general_layout = QFormLayout(general_group)
        
        self.ed_name = QLineEdit()
        self.cb_type = QComboBox()
        self.cb_type.addItems(["sphere", "box", "cylinder", "plane"])
        self.cb_type.currentTextChanged.connect(self._on_bc_type_changed)
        self.cb_type.currentTextChanged.connect(self._apply_changes)
        self.ed_name.textChanged.connect(self._apply_changes)
        general_layout.addRow("Name", self.ed_name)
        general_layout.addRow("Type", self.cb_type)

        # Meshes group - select which meshes this BC applies to
        meshes_group = QGroupBox("Meshes")
        meshes_layout = QVBoxLayout(meshes_group)
        self.lw_meshes = QListWidget()
        self.lw_meshes.setSelectionMode(QListWidget.ExtendedSelection)
        self.lw_meshes.itemSelectionChanged.connect(self._apply_changes)
        meshes_layout.addWidget(QLabel("Select meshes this BC applies to:"))
        meshes_layout.addWidget(self.lw_meshes)
        layout.addWidget(meshes_group)

        # Transform group
        transform_group = QGroupBox("Transform")
        transform_layout = QFormLayout(transform_group)
        
        # Position
        pos_layout = QHBoxLayout()
        self.sp_pos_x = ScientificDoubleSpinBox()
        self.sp_pos_x.setRange(-1e6, 1e6)
        self.sp_pos_x.setDecimals(6)
        self.sp_pos_x.setValue(0.0)
        self.sp_pos_x.setSuffix(" mm")
        
        self.sp_pos_y = ScientificDoubleSpinBox()
        self.sp_pos_y.setRange(-1e6, 1e6)
        self.sp_pos_y.setDecimals(6)
        self.sp_pos_y.setValue(0.0)
        self.sp_pos_y.setSuffix(" mm")
        
        self.sp_pos_z = ScientificDoubleSpinBox()
        self.sp_pos_z.setRange(-1e6, 1e6)
        self.sp_pos_z.setDecimals(6)
        self.sp_pos_z.setValue(0.0)
        self.sp_pos_z.setSuffix(" mm")
        
        pos_layout.addWidget(QLabel("X:"))
        pos_layout.addWidget(self.sp_pos_x)
        pos_layout.addWidget(QLabel("Y:"))
        pos_layout.addWidget(self.sp_pos_y)
        pos_layout.addWidget(QLabel("Z:"))
        pos_layout.addWidget(self.sp_pos_z)
        
        # Rotation
        rot_layout = QHBoxLayout()
        self.sp_rot_x = ScientificDoubleSpinBox()
        self.sp_rot_x.setRange(-360, 360)
        self.sp_rot_x.setDecimals(2)
        self.sp_rot_x.setValue(0.0)
        self.sp_rot_x.setSuffix("°")
        
        self.sp_rot_y = ScientificDoubleSpinBox()
        self.sp_rot_y.setRange(-360, 360)
        self.sp_rot_y.setDecimals(2)
        self.sp_rot_y.setValue(0.0)
        self.sp_rot_y.setSuffix("°")
        
        self.sp_rot_z = ScientificDoubleSpinBox()
        self.sp_rot_z.setRange(-360, 360)
        self.sp_rot_z.setDecimals(2)
        self.sp_rot_z.setValue(0.0)
        self.sp_rot_z.setSuffix("°")
        
        rot_layout.addWidget(QLabel("X:"))
        rot_layout.addWidget(self.sp_rot_x)
        rot_layout.addWidget(QLabel("Y:"))
        rot_layout.addWidget(self.sp_rot_y)
        rot_layout.addWidget(QLabel("Z:"))
        rot_layout.addWidget(self.sp_rot_z)
        
        # Scale
        scale_layout = QHBoxLayout()
        self.sp_scale_x = ScientificDoubleSpinBox()
        self.sp_scale_x.setRange(0, 1e6)
        self.sp_scale_x.setDecimals(6)
        self.sp_scale_x.setValue(1.0)
        self.sp_scale_x.setSuffix(" mm")
        
        self.sp_scale_y = ScientificDoubleSpinBox()
        self.sp_scale_y.setRange(0, 1e6)
        self.sp_scale_y.setDecimals(6)
        self.sp_scale_y.setValue(1.0)
        self.sp_scale_y.setSuffix(" mm")
        
        self.sp_scale_z = ScientificDoubleSpinBox()
        self.sp_scale_z.setRange(0, 1e6)
        self.sp_scale_z.setDecimals(6)
        self.sp_scale_z.setValue(1.0)
        self.sp_scale_z.setSuffix(" mm")
        
        scale_layout.addWidget(QLabel("X:"))
        scale_layout.addWidget(self.sp_scale_x)
        scale_layout.addWidget(QLabel("Y:"))
        scale_layout.addWidget(self.sp_scale_y)
        scale_layout.addWidget(QLabel("Z:"))
        scale_layout.addWidget(self.sp_scale_z)
        
        transform_layout.addRow("Position", pos_layout)
        transform_layout.addRow("Rotation", rot_layout)
        transform_layout.addRow("Scale", scale_layout)

        # Flags group
        flags_group = QGroupBox("Flags")
        flags_layout = QFormLayout(flags_group)
        
        self.cb_fix_position = QCheckBox("Fix position (fix_position)")
        
        flags_layout.addRow(self.cb_fix_position)

        # Parameters group
        parameters_group = QGroupBox("Parameters")
        parameters_layout = QFormLayout(parameters_group)
        
        # Primitive-specific parameters
        self.sp_radius = ScientificDoubleSpinBox()
        self.sp_radius.setRange(0, 1e6)
        self.sp_radius.setDecimals(6)
        self.sp_radius.setValue(1.0)
        self.sp_radius.setSuffix(" mm")
        self.sp_radius.setVisible(False)
        
        self.sp_box_x = ScientificDoubleSpinBox()
        self.sp_box_x.setRange(0, 1e6)
        self.sp_box_x.setDecimals(6)
        self.sp_box_x.setValue(1.0)
        self.sp_box_x.setSuffix(" mm")
        self.sp_box_x.setVisible(False)
        
        self.sp_box_y = ScientificDoubleSpinBox()
        self.sp_box_y.setRange(0, 1e6)
        self.sp_box_y.setDecimals(6)
        self.sp_box_y.setValue(1.0)
        self.sp_box_y.setSuffix(" mm")
        self.sp_box_y.setVisible(False)
        
        self.sp_box_z = ScientificDoubleSpinBox()
        self.sp_box_z.setRange(0, 1e6)
        self.sp_box_z.setDecimals(6)
        self.sp_box_z.setValue(1.0)
        self.sp_box_z.setSuffix(" mm")
        self.sp_box_z.setVisible(False)
        
        self.sp_cylinder_radius = ScientificDoubleSpinBox()
        self.sp_cylinder_radius.setRange(0, 1e6)
        self.sp_cylinder_radius.setDecimals(6)
        self.sp_cylinder_radius.setValue(1.0)
        self.sp_cylinder_radius.setSuffix(" mm")
        self.sp_cylinder_radius.setVisible(False)
        
        self.sp_cylinder_height = ScientificDoubleSpinBox()
        self.sp_cylinder_height.setRange(0, 1e6)
        self.sp_cylinder_height.setDecimals(6)
        self.sp_cylinder_height.setValue(1.0)
        self.sp_cylinder_height.setSuffix(" mm")
        self.sp_cylinder_height.setVisible(False)
        
        parameters_layout.addRow("Radius", self.sp_radius)
        parameters_layout.addRow("Box X", self.sp_box_x)
        parameters_layout.addRow("Box Y", self.sp_box_y)
        parameters_layout.addRow("Box Z", self.sp_box_z)
        parameters_layout.addRow("Cylinder Radius", self.sp_cylinder_radius)
        parameters_layout.addRow("Cylinder Height", self.sp_cylinder_height)

        layout.addWidget(general_group)
        layout.addWidget(transform_group)
        layout.addWidget(flags_group)
        layout.addWidget(parameters_group)
        layout.addStretch()

        # Connect value changes for immediate apply
        for sp in (
            self.sp_pos_x, self.sp_pos_y, self.sp_pos_z,
            self.sp_rot_x, self.sp_rot_y, self.sp_rot_z,
            self.sp_scale_x, self.sp_scale_y, self.sp_scale_z,
            self.sp_radius, self.sp_box_x, self.sp_box_y, self.sp_box_z,
            self.sp_cylinder_radius, self.sp_cylinder_height,
        ):
            sp.valueChanged.connect(self._apply_changes)
        self.cb_fix_position.stateChanged.connect(self._apply_changes)

        return widget

    def _on_bc_selection_changed(self) -> None:
        """Handle BC selection change - load selected BC into editor."""
        row = self.bc_table.currentRow()
        if row >= 0:
            name_item = self.bc_table.item(row, 0)
            if name_item:
                bc_id = name_item.data(Qt.UserRole)
                if bc_id:
                    self._current_bc_id = bc_id
                    self.bc_selected.emit(bc_id)
                    self._load_bc_to_editor(bc_id)
                    self._set_editor_enabled(True)
                    return
        self._current_bc_id = None
        self._clear_editor()
        self._set_editor_enabled(False)

    def _set_editor_enabled(self, enabled: bool) -> None:
        """Enable or disable all editor fields (when no BC selected, they are disabled)."""
        self.ed_name.setEnabled(enabled)
        self.cb_type.setEnabled(enabled)
        self.sp_pos_x.setEnabled(enabled)
        self.sp_pos_y.setEnabled(enabled)
        self.sp_pos_z.setEnabled(enabled)
        self.sp_rot_x.setEnabled(enabled)
        self.sp_rot_y.setEnabled(enabled)
        self.sp_rot_z.setEnabled(enabled)
        self.sp_scale_x.setEnabled(enabled)
        self.sp_scale_y.setEnabled(enabled)
        self.sp_scale_z.setEnabled(enabled)
        self.cb_fix_position.setEnabled(enabled)
        self.sp_radius.setEnabled(enabled)
        self.sp_box_x.setEnabled(enabled)
        self.sp_box_y.setEnabled(enabled)
        self.sp_box_z.setEnabled(enabled)
        self.sp_cylinder_radius.setEnabled(enabled)
        self.sp_cylinder_height.setEnabled(enabled)
        self.lw_meshes.setEnabled(enabled)

    def _on_bc_type_changed(self, bc_type: str) -> None:
        """Handle boundary condition type change - show/hide parameters."""
        # Hide all parameters
        self.sp_radius.setVisible(False)
        self.sp_box_x.setVisible(False)
        self.sp_box_y.setVisible(False)
        self.sp_box_z.setVisible(False)
        self.sp_cylinder_radius.setVisible(False)
        self.sp_cylinder_height.setVisible(False)
        
        # Show relevant parameters
        if bc_type == "sphere":
            self.sp_radius.setVisible(True)
        elif bc_type == "box":
            self.sp_box_x.setVisible(True)
            self.sp_box_y.setVisible(True)
            self.sp_box_z.setVisible(True)
        elif bc_type == "cylinder":
            self.sp_cylinder_radius.setVisible(True)
            self.sp_cylinder_height.setVisible(True)
        # plane has no specific parameters

    def _on_add_bc(self) -> None:
        """Add a new boundary condition."""
        new_bc = BoundaryCondition(
            name="New Boundary Condition",
            bc_type="sphere",
            mesh_ids=[]
        )
        bc_data = {
            "bc_id": new_bc.bc_id,
            "name": new_bc.name,
            "bc_type": new_bc.bc_type,
            "translation": list(new_bc.transform.translation),
            "rotation_euler_deg": list(new_bc.transform.rotation_euler_deg),
            "scale": list(new_bc.transform.scale),
            "mesh_ids": list(new_bc.mesh_ids),
            "flags": dict(new_bc.flags),
            "parameters": dict(new_bc.parameters),
        }
        self.bc_created.emit(bc_data)

    def _apply_changes(self) -> None:
        """Apply current editor values to the selected boundary condition (immediate apply)."""
        if self._is_loading_bc:
            return
        bc = self.get_current_bc_data()
        if bc and self._current_bc_id:
            bc_data = {
                "name": bc.name,
                "bc_type": bc.bc_type,
                "translation": list(bc.transform.translation),
                "rotation_euler_deg": list(bc.transform.rotation_euler_deg),
                "scale": list(bc.transform.scale),
                "mesh_ids": list(bc.mesh_ids),
                "flags": dict(bc.flags),
                "parameters": dict(bc.parameters),
            }
            self.bc_updated.emit(self._current_bc_id, bc_data)

    def _on_remove_bc(self) -> None:
        """Remove the selected boundary condition."""
        if self._current_bc_id:
            self.bc_deleted.emit(self._current_bc_id)

    def set_boundary_conditions(self, bcs: list[BoundaryCondition], select_bc_id: str | None = None) -> None:
        """Set the list of boundary conditions in the table. Optionally select a BC by id."""
        self.bc_table.setRowCount(len(bcs))
        
        for i, bc in enumerate(bcs):
            # Name (bc_id stored in UserRole for selection lookup)
            name_item = QTableWidgetItem(bc.name)
            name_item.setData(Qt.UserRole, bc.bc_id)
            self.bc_table.setItem(i, 0, name_item)
            
            # Type
            type_item = QTableWidgetItem(bc.bc_type)
            self.bc_table.setItem(i, 1, type_item)
            
            # Meshes
            meshes = self._get_meshes()
            mesh_id_to_name = {m.mesh_id: m.name for m in meshes}
            mesh_names = ", ".join(mesh_id_to_name.get(mid, mid) for mid in bc.mesh_ids) if bc.mesh_ids else "None"
            mesh_item = QTableWidgetItem(mesh_names)
            self.bc_table.setItem(i, 2, mesh_item)

        if select_bc_id:
            for row in range(self.bc_table.rowCount()):
                item = self.bc_table.item(row, 0)
                if item and item.data(Qt.UserRole) == select_bc_id:
                    self.bc_table.selectRow(row)
                    break

    def _populate_meshes_list(self) -> None:
        """Populate mesh list from project for BC mesh selection."""
        self.lw_meshes.clear()
        for mesh in self._get_meshes():
            item = QListWidgetItem(mesh.name)
            item.setData(Qt.UserRole, mesh.mesh_id)
            self.lw_meshes.addItem(item)

    def _load_bc_to_editor(self, bc_id: str) -> None:
        """Load boundary condition data into the editor."""
        self._is_loading_bc = True
        try:
            bcs = self._get_boundary_conditions_provider(self._main_window)()
            bc = next((b for b in bcs if b.bc_id == bc_id), None)
            if not bc:
                return

            # Populate meshes list and set selection
            self._populate_meshes_list()
            for i in range(self.lw_meshes.count()):
                item = self.lw_meshes.item(i)
                if item.data(Qt.UserRole) in bc.mesh_ids:
                    item.setSelected(True)

            # General
            self.ed_name.setText(bc.name)
            self.cb_type.setCurrentText(bc.bc_type)

            # Transform (MeshTransform uses translation, rotation_euler_deg, scale)
            tr = bc.transform.translation
            rot = bc.transform.rotation_euler_deg
            scl = bc.transform.scale
            self.sp_pos_x.setValue(tr[0] if len(tr) > 0 else 0.0)
            self.sp_pos_y.setValue(tr[1] if len(tr) > 1 else 0.0)
            self.sp_pos_z.setValue(tr[2] if len(tr) > 2 else 0.0)
            self.sp_rot_x.setValue(rot[0] if len(rot) > 0 else 0.0)
            self.sp_rot_y.setValue(rot[1] if len(rot) > 1 else 0.0)
            self.sp_rot_z.setValue(rot[2] if len(rot) > 2 else 0.0)
            self.sp_scale_x.setValue(scl[0] if len(scl) > 0 else 1.0)
            self.sp_scale_y.setValue(scl[1] if len(scl) > 1 else 1.0)
            self.sp_scale_z.setValue(scl[2] if len(scl) > 2 else 1.0)

            # Flags
            self.cb_fix_position.setChecked(bc.flags.get("fix_position", False))

            # Parameters
            self.sp_radius.setValue(bc.parameters.get("radius", 1.0))
            self.sp_box_x.setValue(bc.parameters.get("box_x", 1.0))
            self.sp_box_y.setValue(bc.parameters.get("box_y", 1.0))
            self.sp_box_z.setValue(bc.parameters.get("box_z", 1.0))
            self.sp_cylinder_radius.setValue(bc.parameters.get("cylinder_radius", 1.0))
            self.sp_cylinder_height.setValue(bc.parameters.get("cylinder_height", 1.0))

            # Show/hide parameters based on type
            self._on_bc_type_changed(bc.bc_type)
        finally:
            self._is_loading_bc = False

    def _clear_editor(self) -> None:
        """Clear the editor."""
        self.ed_name.clear()
        self.cb_type.setCurrentIndex(0)
        self.sp_pos_x.setValue(0.0)
        self.sp_pos_y.setValue(0.0)
        self.sp_pos_z.setValue(0.0)
        self.sp_rot_x.setValue(0.0)
        self.sp_rot_y.setValue(0.0)
        self.sp_rot_z.setValue(0.0)
        self.sp_scale_x.setValue(1.0)
        self.sp_scale_y.setValue(1.0)
        self.sp_scale_z.setValue(1.0)
        self.cb_fix_position.setChecked(False)
        self.lw_meshes.clear()
        self.sp_radius.setVisible(False)
        self.sp_box_x.setVisible(False)
        self.sp_box_y.setVisible(False)
        self.sp_box_z.setVisible(False)
        self.sp_cylinder_radius.setVisible(False)
        self.sp_cylinder_height.setVisible(False)

    def get_current_bc_data(self) -> BoundaryCondition | None:
        """Get current boundary condition data from editor."""
        if not self._current_bc_id:
            return None

        # Get selected mesh IDs from list
        mesh_ids = []
        for item in self.lw_meshes.selectedItems():
            mid = item.data(Qt.UserRole)
            if mid:
                mesh_ids.append(mid)

        bc = BoundaryCondition(
            bc_id=self._current_bc_id,
            name=self.ed_name.text(),
            bc_type=self.cb_type.currentText(),
            transform=MeshTransform(
                translation=[
                    self.sp_pos_x.value(),
                    self.sp_pos_y.value(),
                    self.sp_pos_z.value()
                ],
                rotation_euler_deg=[
                    self.sp_rot_x.value(),
                    self.sp_rot_y.value(),
                    self.sp_rot_z.value()
                ],
                scale=[
                    self.sp_scale_x.value(),
                    self.sp_scale_y.value(),
                    self.sp_scale_z.value()
                ]
            ),
            mesh_ids=mesh_ids,
            flags={
                "fix_position": self.cb_fix_position.isChecked()
            },
            parameters={
                "radius": self.sp_radius.value() if self.sp_radius.isVisible() else 1.0,
                "box_x": self.sp_box_x.value() if self.sp_box_x.isVisible() else 1.0,
                "box_y": self.sp_box_y.value() if self.sp_box_y.isVisible() else 1.0,
                "box_z": self.sp_box_z.value() if self.sp_box_z.isVisible() else 1.0,
                "cylinder_radius": self.sp_cylinder_radius.value() if self.sp_cylinder_radius.isVisible() else 1.0,
                "cylinder_height": self.sp_cylinder_height.value() if self.sp_cylinder_height.isVisible() else 1.0,
            }
        )
        return bc
