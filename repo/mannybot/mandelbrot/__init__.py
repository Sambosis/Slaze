"""
Mandelbrot Set Viewer Package

This package provides a high-performance desktop application for visualizing
the Mandelbrot set with interactive zooming, panning, and color customization.

Main Components:
- MandelbrotViewer: Core class for Mandelbrot calculation and display
- ColorMapper: Handles color mapping for visualization

The package is optimized using NumPy for vectorized operations and Numba
for JIT compilation to ensure real-time performance.
"""

from .viewer import MandelbrotViewer
from .colormap import ColorMapper

__version__ = "1.0.0"
__author__ = "Mandelbrot Viewer Team"
__description__ = "High-performance Mandelbrot set viewer with PyQt5 GUI"

__all__ = [
    'MandelbrotViewer',
    'ColorMapper'
]