import numpy as np
from numba import jit
from typing import Tuple, Optional
from matplotlib import cm
from matplotlib.image import imsave
from .colormap import ColorMapper

@jit(nopython=True)
def mandelbrot_pixel(c_real: float, c_imag: float, max_iter: int) -> int:
    """
    Calculate the Mandelbrot convergence iteration for a single pixel.
    """
    z_real, z_imag = 0.0, 0.0
    iterations = 0

    while iterations < max_iter:
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag

        if z_real_sq + z_imag_sq > 4.0:
            break

        z_imag = 2 * z_real * z_imag + c_imag
        z_real = z_real_sq - z_imag_sq + c_real
        iterations += 1

    return iterations

@jit(nopython=True, parallel=True)
def calculate_mandelbrot_vectorized(
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    width: int, height: int,
    max_iter: int
) -> np.ndarray:
    """
    Calculate the Mandelbrot set for the given coordinate range with vectorized operations.
    """
    iterations_array = np.zeros((height, width), dtype=np.int32)
    x_step = (x_max - x_min) / (width - 1)
    y_step = (y_max - y_min) / (height - 1)

    for y in range(height):
        for x in range(width):
            c_real = x_min + x * x_step
            c_imag = y_min + y * y_step
            iterations_array[y, x] = mandelbrot_pixel(c_real, c_imag, max_iter)

    return iterations_array

class MandelbrotViewer:
    """
    Main class for calculating and displaying the Mandelbrot set.
    Handles zooming, panning, and rendering operations with optimized performance.
    """

    def __init__(self, width: int = 800, height: int = 600, max_iter: int = 1000,
                 center_x: float = -0.5, center_y: float = 0.0, zoom: float = 1.0,
                 color_map: str = 'viridis'):
        """
        Initialize the Mandelbrot viewer with default parameters.
        """
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.center_x = center_x
        self.center_y = center_y
        self.zoom = zoom
        self.color_map = color_map
        self.color_mapper = ColorMapper(color_map=color_map)

        self._default_bounds = {
            'x_min': -2.5,
            'x_max': 1.0,
            'y_min': -1.25,
            'y_max': 1.25
        }

        self._current_data: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None
        self._last_calculated_bounds: dict = {}

    @property
    def current_bounds(self) -> dict:
        """
        Calculate the current viewing bounds based on zoom and center.
        """
        zoom = self.zoom
        x_range = (self._default_bounds['x_max'] - self._default_bounds['x_min']) / zoom
        y_range = (self._default_bounds['y_max'] - self._default_bounds['y_min']) / zoom

        return {
            'x_min': self.center_x - x_range / 2,
            'x_max': self.center_x + x_range / 2,
            'y_min': self.center_y - y_range / 2,
            'y_max': self.center_y + y_range / 2
        }

    def calculate_mandelbrot(self, x_min: float, x_max: float,
                            y_min: float, y_max: float) -> np.ndarray:
        """
        Compute the Mandelbrot set for the given bounds using vectorized operations.
        """
        self._last_calculated_bounds = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max
        }

        result = calculate_mandelbrot_vectorized(
            x_min, x_max, y_min, y_max,
            self.width, self.height,
            self.max_iter
        )

        self._current_data = result
        return result

    def update_display(self) -> None:
        """
        Update the display with current Mandelbrot data.
        """
        bounds = self.current_bounds
        self.calculate_mandelbrot(**bounds)
        self._current_image = self.color_mapper.apply_colormap(self._current_data)

    def handle_zoom(self, factor: float, x: float, y: float) -> None:
        """
        Handle zoom operations centered at the given coordinates.
        """
        bounds = self.current_bounds
        x_frac = x / (self.width - 1)
        y_frac = y / (self.height - 1)

        zoom_center_x = bounds['x_min'] + x_frac * (bounds['x_max'] - bounds['x_min'])
        zoom_center_y = bounds['y_min'] + y_frac * (bounds['y_max'] - bounds['y_min'])

        self.zoom *= factor
        self.zoom = np.clip(self.zoom, 0.1, 100000.0)

        new_center_x = bounds['x_min'] + x_frac * (bounds['x_max'] - bounds['x_min'])
        new_center_y = bounds['y_min'] + y_frac * (bounds['y_max'] - bounds['y_min'])

        dx = zoom_center_x - new_center_x
        dy = zoom_center_y - new_center_y

        self.center_x += dx
        self.center_y += dy

    def handle_pan(self, dx: float, dy: float) -> None:
        """
        Handle panning operations.
        """
        bounds = self.current_bounds
        x_scale = (bounds['x_max'] - bounds['x_min']) / self.width
        y_scale = (bounds['y_max'] - bounds['y_min']) / self.height

        self.center_x -= dx * x_scale
        self.center_y -= dy * y_scale

    def reset_view(self) -> None:
        """
        Reset the view to default bounds and settings.
        """
        self.center_x = -0.5
        self.center_y = 0.0
        self.zoom = 1.0
        self.max_iter = 1000

    def get_current_data(self) -> np.ndarray:
        """
        Get the current Mandelbrot calculation data.
        """
        if self._current_data is None:
            raise RuntimeError("No Mandelbrot data available. Call calculate_mandelbrot() or update_display() first.")
        return self._current_data

    def get_current_image(self) -> np.ndarray:
        """
        Get the current colored image data.
        """
        if self._current_image is None:
            raise RuntimeError("No image data available. Call update_display() first.")
        return self._current_image

    def set_resolution(self, width: int, height: int) -> None:
        """
        Set the viewport resolution and invalidate cached data.
        """
        self.width = width
        self.height = height
        self._current_data = None
        self._current_image = None

    def set_max_iterations(self, iterations: int) -> None:
        """
        Set the maximum number of iterations for calculations.
        """
        if iterations < 1:
            raise ValueError("Iterations must be at least 1")
        elif iterations > 10000:
            raise ValueError("Iterations cannot exceed 10000")

        self.max_iter = iterations

    def set_color_map(self, color_map: str) -> None:
        """
        Set the color map for visualization.
        """
        self.color_map = color_map
        self.color_mapper = ColorMapper(color_map=color_map)

    def save_image(self, filepath: str) -> None:
        """
        Save the current Mandelbrot view as a PNG image.
        """
        if self._current_image is None:
            raise ValueError("No image data available to save")

        imsave(filepath, self._current_image)

    def get_current_bounds_tuple(self) -> Tuple[float, float, float, float]:
        """
        Get the current bounds as a tuple.
        """
        bounds = self.current_bounds
        return (bounds['x_min'], bounds['x_max'], bounds['y_min'], bounds['y_max'])

    def get_current_center(self) -> Tuple[float, float]:
        """
        Get the current center coordinates.
        """
        return (self.center_x, self.center_y)