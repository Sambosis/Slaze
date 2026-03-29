#!/usr/bin/env python3
"""
Main window implementation for the Mandelbrot Set Viewer application.
Handles UI layout, event handling, and integration with the MandelbrotViewer class.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QHBoxLayout, QVBoxLayout, QStatusBar,
    QMessageBox, QFileDialog, QMenuBar, QMenu, QAction
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage, QIcon

from mandelbrot.viewer import MandelbrotViewer
from ui.controls import ControlPanel

class MandelbrotCalculationThread(QThread):
    """Worker thread for Mandelbrot calculations to prevent UI freezing."""
    calculation_finished = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int)

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.running = False

    def run(self):
        """Execute Mandelbrot calculation in background thread."""
        self.running = True
        try:
            self.progress_updated.emit(10)
            self.viewer.update_display()
            self.progress_updated.emit(90)
            image = self.viewer.get_current_image()
            self.progress_updated.emit(100)
            if self.running:
                self.calculation_finished.emit(image)
        except Exception as e:
            print(f"Calculation error: {e}")
            self.progress_updated.emit(-1)
        finally:
            self.running = False

    def stop(self):
        """Stop the calculation thread."""
        self.running = False

class MandelbrotCanvas(QLabel):
    """Custom canvas widget for displaying Mandelbrot set with mouse interaction."""
    zoom_requested = pyqtSignal(float, float, float)
    pan_requested = pyqtSignal(float, float)
    mouse_moved = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black; border: 1px solid #333;")
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)
        self.last_mouse_pos = None
        self.is_panning = False
        self.is_dragging = False
        self.selection_start = None
        self.mouse_x = 0
        self.mouse_y = 0

    def mousePressEvent(self, event):
        """Handle mouse press events for panning and selection."""
        if event.button() == Qt.LeftButton:
            self.last_mouse_pos = event.pos()
            self.is_panning = True
        elif event.button() == Qt.RightButton:
            self.selection_start = event.pos()
            self.is_dragging = True

    def mouseMoveEvent(self, event):
        """Handle mouse move events for panning and coordinate display."""
        if self.pixmap():
            pixmap_rect = self.pixmap().rect()
            label_rect = self.contentsRect()
            x_offset = (label_rect.width() - pixmap_rect.width()) // 2
            y_offset = (label_rect.height() - pixmap_rect.height()) // 2
            pixmap_x = event.x() - x_offset
            pixmap_y = event.y() - y_offset

            if 0 <= pixmap_x < pixmap_rect.width() and 0 <= pixmap_y < pixmap_rect.height():
                self.mouse_x = pixmap_x / pixmap_rect.width()
                self.mouse_y = pixmap_y / pixmap_rect.height()
                self.mouse_moved.emit(self.mouse_x, self.mouse_y)

        if self.is_panning and self.last_mouse_pos:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()
            self.last_mouse_pos = event.pos()
            self.pan_requested.emit(dx, dy)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.last_mouse_pos = None
        elif event.button() == Qt.RightButton and self.is_dragging:
            selection_end = event.pos()
            if self.selection_start and selection_end:
                width = abs(selection_end.x() - self.selection_start.x())
                height = abs(selection_end.y() - self.selection_start.y())
                if width > 10 and height > 10:
                    center_x = (self.selection_start.x() + selection_end.x()) // 2
                    center_y = (self.selection_start.y() + selection_end.y()) // 2
                    factor = min(self.width() / width, self.height() / height)
                    self.zoom_requested.emit(factor, center_x, center_y)
            self.is_dragging = False
            self.selection_start = None

    def wheelEvent(self, event):
        """Handle wheel events for zooming."""
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.zoom_requested.emit(factor, event.x(), event.y())

class MainWindow(QMainWindow):
    """Main application window for the Mandelbrot Set Viewer."""
    def __init__(self):
        super().__init__()
        self.viewer = MandelbrotViewer(width=800, height=600)
        self.calculation_thread = None
        self.update_timer = QTimer(self)
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self._perform_update)
        self.calculating = False

        self.init_ui()
        self.init_menu()
        self.init_status_bar()
        self._perform_update()

    def init_ui(self):
        """Initialize the main UI components."""
        self.setWindowTitle("Mandelbrot Set Viewer")
        self.setGeometry(100, 100, 1200, 800)

        try:
            self.setWindowIcon(QIcon("assets/icon.png"))
        except:
            pass

        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.canvas = MandelbrotCanvas()
        self.control_panel = ControlPanel()

        main_layout.addWidget(self.canvas, 3)
        main_layout.addWidget(self.control_panel, 1)

        self.setCentralWidget(central_widget)

        self.control_panel.color_changed.connect(self.handle_color_change)
        self.control_panel.iterations_changed.connect(self.handle_iterations_change)
        self.control_panel.quality_changed.connect(self.handle_quality_change)
        self.control_panel.reset_clicked.connect(self.handle_reset)
        self.control_panel.save_clicked.connect(self.handle_save)
        self.control_panel.continuous_coloring_changed.connect(self.handle_continuous_change)

        self.canvas.zoom_requested.connect(self.handle_zoom)
        self.canvas.pan_requested.connect(self.handle_pan)
        self.canvas.mouse_moved.connect(self.update_coordinate_display)

    def init_menu(self):
        """Initialize the menu bar."""
        menubar = QMenuBar()

        file_menu = menubar.addMenu("File")
        save_action = QAction("Save Image", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.handle_save)
        file_menu.addAction(save_action)

        reset_action = QAction("Reset View", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.handle_reset)
        file_menu.addAction(reset_action)

        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("View")
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(lambda: self.handle_zoom(1.5, self.canvas.width()//2, self.canvas.height()//2))
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(lambda: self.handle_zoom(1/1.5, self.canvas.width()//2, self.canvas.height()//2))
        view_menu.addAction(zoom_out_action)

        self.setMenuBar(menubar)

    def init_status_bar(self):
        """Initialize the status bar."""
        self.status_bar = QStatusBar()
        self.coord_label = QLabel("Coordinates: (-0.500000, 0.000000)")
        self.coord_label.setMinimumWidth(200)
        self.iter_label = QLabel(f"Iterations: {self.viewer.max_iter}")
        self.iter_label.setMinimumWidth(120)
        self.zoom_label = QLabel(f"Zoom: {self.viewer.zoom:.2f}")
        self.zoom_label.setMinimumWidth(80)
        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(100)

        self.status_bar.addWidget(self.coord_label, 1)
        self.status_bar.addWidget(self.iter_label)
        self.status_bar.addWidget(self.zoom_label)
        self.status_bar.addWidget(self.progress_label)
        self.setStatusBar(self.status_bar)

    def update_coordinate_display(self, norm_x, norm_y):
        """Update coordinate display based on mouse position."""
        bounds = self.viewer.current_bounds
        x_coord = bounds['x_min'] + norm_x * (bounds['x_max'] - bounds['x_min'])
        y_coord = bounds['y_min'] + norm_y * (bounds['y_max'] - bounds['y_min'])
        self.coord_label.setText(f"Mouse: ({x_coord:.6f}, {y_coord:.6f})")

    def _perform_update(self):
        """Perform Mandelbrot calculation in background thread."""
        if self.calculation_thread and self.calculation_thread.isRunning():
            self.calculation_thread.stop()
            self.calculation_thread.wait()

        self.calculating = True
        self.progress_label.setText("Calculating...")

        self.calculation_thread = MandelbrotCalculationThread(self.viewer)
        self.calculation_thread.calculation_finished.connect(self.update_display)
        self.calculation_thread.progress_updated.connect(self.update_progress)
        self.calculation_thread.start()

    def update_progress(self, percentage):
        """Update progress display."""
        if percentage == -1:
            self.progress_label.setText("Error")
            self.calculating = False
        elif percentage == 100:
            self.progress_label.setText("Complete")
            self.calculating = False
        else:
            self.progress_label.setText(f"Calculating... {percentage}%")

    def update_display(self, image_data):
        """Update the canvas with new Mandelbrot image."""
        if image_data is None or image_data.size == 0:
            return

        try:
            height, width = image_data.shape[:2]
            bytes_per_line = 3 * width
            q_image = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.canvas.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.canvas.setPixmap(scaled_pixmap)

            self.update_status_bar()
        except Exception as e:
            print(f"Error updating display: {e}")

    def update_status_bar(self):
        """Update status bar with current information."""
        center_x, center_y = self.viewer.get_current_center()
        self.coord_label.setText(f"Center: ({center_x:.6f}, {center_y:.6f})")
        self.iter_label.setText(f"Iterations: {self.viewer.max_iter}")
        self.zoom_label.setText(f"Zoom: {self.viewer.zoom:.2f}")

    def handle_zoom(self, factor, x, y):
        """Handle zoom operations."""
        if self.calculating:
            return

        self.viewer.handle_zoom(factor, x, y)
        self.update_timer.stop()
        self.update_timer.start(100)

    def handle_pan(self, dx, dy):
        """Handle panning operations."""
        if self.calculating:
            return

        self.viewer.handle_pan(dx, dy)
        self.update_timer.stop()
        self.update_timer.start(50)

    def handle_color_change(self, color_map):
        """Handle color map changes."""
        if self.calculating:
            return

        self.viewer.set_color_map(color_map)
        self.update_timer.stop()
        self.update_timer.start(50)

    def handle_iterations_change(self, iterations):
        """Handle iteration count changes."""
        if self.calculating:
            return

        try:
            self.viewer.set_max_iterations(iterations)
            self.update_timer.stop()
            self.update_timer.start(100)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Value", str(e))

    def handle_quality_change(self, quality):
        """Handle quality setting changes."""
        if self.calculating:
            return

        if quality == "Low":
            width, height = 400, 300
        elif quality == "Medium":
            width, height = 800, 600
        else:
            width, height = 1200, 900

        self.viewer.set_resolution(width, height)
        self.update_timer.stop()
        self.update_timer.start(150)

    def handle_continuous_change(self, enabled):
        """Handle continuous coloring checkbox changes."""
        if self.calculating:
            return

        self.viewer.color_mapper.set_continuous(enabled)
        self.update_timer.stop()
        self.update_timer.start(50)

    def handle_reset(self):
        """Reset view to default parameters."""
        if self.calculating:
            return

        self.viewer.reset_view()
        self.control_panel.reset_controls()
        self._perform_update()

    def handle_save(self):
        """Save current view as image file."""
        if self.calculating:
            QMessageBox.warning(self, "Busy", "Please wait for calculation to complete.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Mandelbrot Image",
            "",
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )

        if filepath:
            try:
                self.viewer.save_image(filepath)
                QMessageBox.information(self, "Success", "Image saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save image: {str(e)}")

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        if self.viewer and not self.calculating:
            new_width = max(100, self.canvas.width())
            new_height = max(100, self.canvas.height())
            self.viewer.set_resolution(new_width, new_height)
            self.update_timer.stop()
            self.update_timer.start(200)

    def closeEvent(self, event):
        """Handle window close event."""
        if self.calculation_thread and self.calculation_thread.isRunning():
            self.calculation_thread.stop()
            self.calculation_thread.wait(5000)
        event.accept()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)