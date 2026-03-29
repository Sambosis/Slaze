"""
Color mapping module for Mandelbrot Set Viewer.

This module provides the ColorMapper class for handling color mapping of Mandelbrot data.
It supports predefined colormaps from matplotlib and custom color schemes.
"""

import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt

class ColorMapper:
    """
    Handles color mapping for Mandelbrot visualization.

    This class provides functionality to apply various colormaps to Mandelbrot
    iteration data, including both predefined matplotlib colormaps and custom
    color schemes.

    Attributes:
        colormap (str): Name of the current colormap
        custom_colors (Optional[List]): List of custom colors if using custom colormap
        continuous (bool): Whether to use continuous color mapping
        _current_cmap: Current matplotlib colormap object
    """

    def __init__(self, color_map: str = 'viridis',
                 custom_colors: Optional[Union[List[str], List[Tuple[float, float, float]]]] = None,
                 continuous: bool = True):
        """
        Initialize the ColorMapper with a specified colormap.

        Args:
            color_map (str): Name of the colormap to use. Default is 'viridis'.
            custom_colors (Optional[Union[List[str], List[Tuple[float, float, float]]]]):
                List of custom colors for custom colormap. Can be hex strings or RGB tuples.
                If provided, overrides the color_map parameter.
            continuous (bool): Whether to use continuous color mapping. Default is True.

        Raises:
            ValueError: If the specified colormap is not available and no custom colors provided.
        """
        self.colormap = color_map
        self.custom_colors = custom_colors
        self.continuous = continuous
        self._current_cmap = None

        # Set up the colormap
        self._setup_colormap()

    def _setup_colormap(self) -> None:
        """
        Set up the matplotlib colormap based on current settings.

        This method creates the appropriate colormap object based on whether
        custom colors are provided or a standard matplotlib colormap is used.
        """
        if self.custom_colors is not None:
            # Create custom colormap
            try:
                if len(self.custom_colors) < 2:
                    raise ValueError("Custom colormap requires at least 2 colors")

                if isinstance(self.custom_colors[0], str):
                    # Convert hex strings to RGB tuples
                    rgb_colors = []
                    for color in self.custom_colors:
                        if not color.startswith('#') or len(color) != 7:
                            raise ValueError(f"Invalid hex color format: {color}")
                        r = int(color[1:3], 16) / 255.0
                        g = int(color[3:5], 16) / 255.0
                        b = int(color[5:7], 16) / 255.0
                        rgb_colors.append((r, g, b))
                    self._current_cmap = LinearSegmentedColormap.from_list('custom', rgb_colors)
                else:
                    # Assume RGB tuples in 0-1 range
                    self._current_cmap = LinearSegmentedColormap.from_list('custom', self.custom_colors)
            except Exception as e:
                raise ValueError(f"Invalid custom colors: {e}")
        else:
            # Use standard matplotlib colormap
            try:
                self._current_cmap = cm.get_cmap(self.colormap)
            except Exception as e:
                # Fallback to viridis if colormap not found
                self._current_cmap = cm.get_cmap('viridis')
                self.colormap = 'viridis'

    def apply_colormap(self, data: np.ndarray,
                      max_iter: Optional[int] = None) -> np.ndarray:
        """
        Apply the selected colormap to Mandelbrot iteration data.

        Args:
            data (np.ndarray): 2D array of Mandelbrot iteration counts
            max_iter (int, optional): Maximum iterations for normalization.
                                     If None, uses max value in data.

        Returns:
            np.ndarray: RGB image array (uint8) with shape (height, width, 3)

        Raises:
            ValueError: If input data is not a 2D numpy array
            RuntimeError: If no colormap is set
        """
        # Validate input
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data must be a 2D numpy array")

        if self._current_cmap is None:
            raise RuntimeError("No colormap is set. Call _setup_colormap() first.")

        # Normalize data
        if max_iter is None:
            max_iter = np.max(data)

        if max_iter == 0:
            # All pixels escaped immediately
            normalized_data = np.zeros_like(data, dtype=np.float32)
        else:
            if self.continuous:
                # Log scaling for better visualization of high iteration areas
                with np.errstate(invalid='ignore', divide='ignore'):
                    normalized_data = np.nan_to_num(
                        np.log(data.astype(np.float32) + 1) / np.log(max_iter + 1),
                        nan=0.0
                    )
            else:
                # Linear scaling
                normalized_data = data.astype(np.float32) / max_iter

            normalized_data = np.clip(normalized_data, 0.0, 1.0)

        # Apply colormap
        try:
            # Use the colormap to convert data to RGBA
            rgba_image = self._current_cmap(normalized_data)

            # Convert to RGB (0-255 range) and remove alpha channel
            rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)
            return rgb_image
        except Exception as e:
            raise RuntimeError(f"Error applying colormap: {e}")

    def set_colormap(self, color_map: str) -> None:
        """
        Set a new colormap.

        Args:
            color_map (str): Name of the new colormap

        Raises:
            ValueError: If the specified colormap is not available
        """
        self.colormap = color_map
        self.custom_colors = None
        self._setup_colormap()

    def set_custom_colors(self, colors: Union[List[str], List[Tuple[float, float, float]]]) -> None:
        """
        Set custom colors for the colormap.

        Args:
            colors (Union[List[str], List[Tuple[float, float, float]]]):
                List of colors for the custom colormap. Can be hex strings or RGB tuples.

        Raises:
            ValueError: If the colors list is invalid
        """
        self.custom_colors = colors
        self.colormap = "custom"
        self._setup_colormap()

    def get_available_colormaps(self) -> List[str]:
        """
        Get the list of available colormap names.

        Returns:
            List[str]: List of available colormap names
        """
        try:
            return sorted(plt.colormaps())
        except:
            # Fallback list if matplotlib.pyplot is not available
            return [
                'viridis', 'plasma', 'inferno', 'magma', 'cividis',
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                'hot', 'afmhot', 'gist_heat', 'copper',
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                'twilight', 'twilight_shifted', 'hsv'
            ]

    def get_current_colormap_name(self) -> str:
        """
        Get the name of the current colormap.

        Returns:
            str: Name of the current colormap
        """
        if self.custom_colors is not None:
            return "custom"
        return self.colormap

    def get_current_colormap(self):
        """
        Get the current matplotlib colormap object.

        Returns:
            matplotlib.colors.Colormap: Current colormap object
        """
        return self._current_cmap

    def set_continuous(self, continuous: bool) -> None:
        """
        Set whether to use continuous color mapping.

        Args:
            continuous (bool): True for continuous (log-scaled) mapping,
                             False for discrete (linear) mapping
        """
        self.continuous = continuous