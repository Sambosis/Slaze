#!/usr/bin/env python3
"""
Control panel implementation for the Mandelbrot Set Viewer application.
Provides a widget with various controls for customizing the Mandelbrot visualization.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox,
    QPushButton, QGroupBox, QSpinBox, QCheckBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from mandelbrot.colormap import ColorMapper

class ControlPanel(QWidget):
    """
    Control panel widget for the Mandelbrot Set Viewer.
    Provides UI controls for adjusting visualization parameters.
    """

    # Custom signals for communication with main window
    color_changed = pyqtSignal(str)
    iterations_changed = pyqtSignal(int)
    quality_changed = pyqtSignal(str)
    reset_clicked = pyqtSignal()
    save_clicked = pyqtSignal()
    continuous_coloring_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_mapper = ColorMapper()
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the control panel UI components."""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Title label
        title_label = QLabel("Mandelbrot Controls")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Add separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)

        # Add control groups
        self.add_color_controls(main_layout)
        self.add_iteration_controls(main_layout)
        self.add_quality_controls(main_layout)
        self.add_action_buttons(main_layout)

        # Add stretch to push content to top
        main_layout.addStretch(1)

        self.setLayout(main_layout)
        self.setMinimumWidth(250)
        self.setMaximumWidth(300)

    def add_color_controls(self, layout):
        """Add color mapping controls to the layout."""
        color_group = QGroupBox("Color Settings")
        color_layout = QVBoxLayout()

        # Colormap selection
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(sorted(self.color_mapper.get_available_colormaps()))
        self.colormap_combo.setCurrentText('viridis')

        # Continuous color mapping checkbox
        self.continuous_checkbox = QCheckBox("Continuous Coloring")
        self.continuous_checkbox.setChecked(True)

        color_layout.addWidget(QLabel("Colormap:"))
        color_layout.addWidget(self.colormap_combo)
        color_layout.addWidget(self.continuous_checkbox)
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

    def add_iteration_controls(self, layout):
        """Add iteration count controls to the layout."""
        iter_group = QGroupBox("Iteration Settings")
        iter_layout = QVBoxLayout()

        # Max iterations slider
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_slider.setRange(50, 10000)
        self.iter_slider.setValue(1000)
        self.iter_slider.setTickInterval(1000)
        self.iter_slider.setTickPosition(QSlider.TicksBelow)

        # Max iterations spinbox
        self.iter_spinbox = QSpinBox()
        self.iter_spinbox.setRange(50, 10000)
        self.iter_spinbox.setValue(1000)
        self.iter_spinbox.setSingleStep(100)

        # Connect slider and spinbox
        self.iter_slider.valueChanged.connect(self.iter_spinbox.setValue)
        self.iter_spinbox.valueChanged.connect(self.iter_slider.setValue)

        iter_layout.addWidget(QLabel("Max Iterations:"))
        iter_layout.addWidget(self.iter_slider)
        iter_layout.addWidget(self.iter_spinbox)
        iter_group.setLayout(iter_layout)
        layout.addWidget(iter_group)

    def add_quality_controls(self, layout):
        """Add rendering quality controls to the layout."""
        quality_group = QGroupBox("Rendering Quality")
        quality_layout = QVBoxLayout()

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low (400x300)", "Medium (800x600)", "High (1200x900)"])
        self.quality_combo.setCurrentText("Medium (800x600)")

        quality_layout.addWidget(QLabel("Resolution:"))
        quality_layout.addWidget(self.quality_combo)
        quality_group.setLayout(quality_layout)
        layout.addWidget(quality_group)

    def add_action_buttons(self, layout):
        """Add action buttons to the layout."""
        button_layout = QHBoxLayout()

        self.reset_button = QPushButton("Reset View")
        self.reset_button.setToolTip("Reset to default view and parameters")

        self.save_button = QPushButton("Save Image")
        self.save_button.setToolTip("Save current view as PNG image")

        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.save_button)
        layout.addLayout(button_layout)

    def setup_connections(self):
        """Set up signal connections between widgets."""
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed)
        self.continuous_checkbox.stateChanged.connect(self.on_continuous_changed)
        self.iter_slider.valueChanged.connect(self.on_iterations_changed)
        self.quality_combo.currentTextChanged.connect(self.on_quality_changed)
        self.reset_button.clicked.connect(self.reset_clicked)
        self.save_button.clicked.connect(self.save_clicked)

    def on_colormap_changed(self, colormap_name):
        """Handle colormap selection changes."""
        self.color_changed.emit(colormap_name)

    def on_continuous_changed(self, state):
        """Handle continuous color mapping checkbox changes."""
        continuous = state == Qt.Checked
        self.continuous_coloring_changed.emit(continuous)

    def on_iterations_changed(self, value):
        """Handle iteration count changes."""
        self.iterations_changed.emit(value)

    def on_quality_changed(self, quality):
        """Handle quality preset changes."""
        quality_level = quality.split()[0]
        self.quality_changed.emit(quality_level)

    def reset_controls(self):
        """Reset all controls to default values."""
        self.colormap_combo.setCurrentText('viridis')
        self.continuous_checkbox.setChecked(True)
        self.iter_slider.setValue(1000)
        self.iter_spinbox.setValue(1000)
        self.quality_combo.setCurrentText("Medium (800x600)")

    def set_max_iterations(self, iterations):
        """Update the maximum iterations spinbox value."""
        self.iter_spinbox.setValue(iterations)

    def set_color_map(self, color_map):
        """Update the selected color map."""
        index = self.colormap_combo.findText(color_map)
        if index >= 0:
            self.colormap_combo.setCurrentIndex(index)

    def set_quality(self, quality):
        """Update the selected quality setting."""
        for i in range(self.quality_combo.count()):
            item = self.quality_combo.itemText(i)
            if item.startswith(quality):
                self.quality_combo.setCurrentIndex(i)
                break

    def set_continuous_coloring(self, enabled):
        """Update continuous coloring checkbox state."""
        self.continuous_checkbox.setChecked(enabled)
