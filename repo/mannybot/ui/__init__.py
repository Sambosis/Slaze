"""
UI Package for Mandelbrot Set Viewer

This package contains the user interface components for the Mandelbrot Set Viewer
application, built using PyQt5. It provides the main application window and
control panel widgets for interactive visualization of the Mandelbrot set.

Main Components:
- MainWindow: The primary application window containing the Mandelbrot display and controls
- ControlPanel: Widget containing various controls for customizing the visualization

The UI is designed to be responsive and provide real-time feedback during
interactive zooming, panning, and parameter adjustments.
"""

from .main_window import MainWindow
from .controls import ControlPanel

__version__ = "1.0.0"
__author__ = "Mandelbrot Viewer Team"
__description__ = "PyQt5-based UI components for Mandelbrot Set Viewer"

__all__ = [
    'MainWindow',
    'ControlPanel'
]