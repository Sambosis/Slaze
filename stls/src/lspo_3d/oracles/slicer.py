# -*- coding: utf-8 -*-
"""
A wrapper for the PrusaSlicer CLI.

This module provides the SlicerOracle class, which is responsible for running
the PrusaSlicer command-line tool, parsing its output, and returning key
printability metrics for a given .stl file.
"""

import dataclasses
import logging
import pathlib
import re
import subprocess
from typing import Match, Optional

# These imports are included as per the project structure specification,
# even if not all are directly used in this specific implementation.
# For example, a shared logger might be configured in `utils`.
from src.lspo_3d import config
from src.lspo_3d import utils


@dataclasses.dataclass
class SlicerMetrics:
    """
    A data structure to hold the parsed results from the PrusaSlicer CLI.

    Attributes:
        is_sliceable (bool): True if the slicer completed successfully, False otherwise.
        total_filament_volume_mm3 (float): The total volume of filament required
            for the print, in cubic millimeters. Defaults to 0.0.
        support_material_volume_mm3 (float): The volume of filament used for
            support material, in cubic millimeters. Defaults to 0.0.
        estimated_print_time_s (float): The estimated print time in seconds.
            Defaults to 0.0.
    """
    is_sliceable: bool = False
    total_filament_volume_mm3: float = 0.0
    support_material_volume_mm3: float = 0.0
    estimated_print_time_s: float = 0.0


class SlicerOracle:
    """
    A wrapper for the PrusaSlicer command-line interface (CLI).

    This class provides a clean interface to slice a 3D model (.stl file)
    and extract key printability metrics from the slicer's output. It is
    initialized with paths to the slicer executable and a specific configuration
    profile.

    Attributes:
        slicer_path (pathlib.Path): The absolute file path to the PrusaSlicer executable.
        slicer_config_path (pathlib.Path): The absolute file path to the .ini
            configuration profile for slicing.
        logger (logging.Logger): A logger for recording events and errors.
    """

    def __init__(self, slicer_path: pathlib.Path, slicer_config_path: pathlib.Path):
        """
        Initializes the SlicerOracle with necessary paths.

        Args:
            slicer_path (pathlib.Path): The path to the PrusaSlicer executable.
            slicer_config_path (pathlib.Path): The path to the slicer
                configuration (.ini) file to use for all slicing operations.
        """
        if not slicer_path.is_file():
            raise FileNotFoundError(f"Slicer executable not found at: {slicer_path}")
        if not slicer_config_path.is_file():
            raise FileNotFoundError(f"Slicer config not found at: {slicer_config_path}")

        self.slicer_path = slicer_path
        self.slicer_config_path = slicer_config_path
        self.logger = logging.getLogger(__name__)

        # Regex patterns to parse PrusaSlicer's --info output.
        # e.g., "total filament used [mm3] = 4130.49"
        self.TOTAL_FILAMENT_RE = re.compile(r"total filament used \[mm3\]\s*=\s*(\d+\.?\d*)")
        # e.g., "support material [mm3] = 157.51"
        self.SUPPORT_FILAMENT_RE = re.compile(r"support material \[mm3\]\s*=\s*(\d+\.?\d*)")
        # e.g., "estimated printing time (normal mode) = 1h 23m 45s"
        self.ESTIMATED_TIME_RE = re.compile(r"estimated printing time .*=\s*(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?")

    def slice_and_evaluate(self, stl_file_path: pathlib.Path) -> SlicerMetrics:
        """
        Slices a given .stl file and evaluates its printability metrics.

        This is the main public method of the class. It orchestrates the process of
        running the slicer as a subprocess and parsing its output to generate
        a structured report of print metrics.

        Args:
            stl_file_path (pathlib.Path): The path to the .stl file to be evaluated.

        Returns:
            SlicerMetrics: A SlicerMetrics object containing the parsed
                printability data. If slicing fails or output parsing fails,
                the `is_sliceable` attribute will be False.
        """
        result = self._run_slicer_subprocess(stl_file_path)

        if result.returncode != 0:
            self.logger.warning(
                f"Slicing failed for {stl_file_path.name} with exit code {result.returncode}.\n"
                f"Stderr: {result.stderr.strip()}"
            )
            return SlicerMetrics(is_sliceable=False)

        parsed_metrics = self._parse_slicer_output(result.stdout)
        if parsed_metrics is None:
            self.logger.error(
                f"Failed to parse slicer output for {stl_file_path.name}.\n"
                f"Stdout: {result.stdout}"
            )
            return SlicerMetrics(is_sliceable=False)

        return parsed_metrics

    def _run_slicer_subprocess(self, stl_file_path: pathlib.Path) -> subprocess.CompletedProcess:
        """
        Executes the PrusaSlicer CLI as a subprocess.

        Constructs the command with the appropriate flags to slice the model
        using the stored configuration and output slicing information without
        generating a G-code file (`--info` flag).

        Args:
            stl_file_path (pathlib.Path): The path to the input .stl file.

        Returns:
            subprocess.CompletedProcess: The object returned by `subprocess.run`,
                containing the exit code, stdout, and stderr of the process.
        """
        command = [
            str(self.slicer_path),
            "--load", str(self.slicer_config_path),
            "--info", str(stl_file_path)
        ]
        self.logger.debug(f"Running slicer command: {' '.join(command)}")
        return subprocess.run(command, capture_output=True, text=True, check=False)

    def _convert_time_to_seconds(self, time_match: Match[str]) -> float:
        """Converts a regex match of (hours, minutes, seconds) to total seconds."""
        hours_str, minutes_str, seconds_str = time_match.groups()
        hours = int(hours_str) if hours_str else 0
        minutes = int(minutes_str) if minutes_str else 0
        seconds = int(seconds_str) if seconds_str else 0
        return float(hours * 3600 + minutes * 60 + seconds)

    def _parse_slicer_output(self, slicer_output: str) -> Optional[SlicerMetrics]:
        """
        Parses the text output from the PrusaSlicer --info command.

        Uses regular expressions to find and extract key metrics like total filament,
        support material volume, and estimated print time from the raw string
        output of the slicer.

        Args:
            slicer_output (str): The stdout string from the PrusaSlicer process.

        Returns:
            Optional[SlicerMetrics]: A SlicerMetrics object populated with the
                extracted data if all required metrics are found. Returns None if
                the output is malformed or missing expected information.
        """
        try:
            total_filament_match = self.TOTAL_FILAMENT_RE.search(slicer_output)
            support_filament_match = self.SUPPORT_FILAMENT_RE.search(slicer_output)
            time_match = self.ESTIMATED_TIME_RE.search(slicer_output)

            # Total filament and time are required for a successful parse.
            if not total_filament_match or not time_match:
                self.logger.warning("Could not find total filament or time in slicer output.")
                return None

            total_volume = float(total_filament_match.group(1))
            # Support material is optional; if not found, it's 0.
            support_volume = float(support_filament_match.group(1)) if support_filament_match else 0.0
            print_time_s = self._convert_time_to_seconds(time_match)

            return SlicerMetrics(
                is_sliceable=True,
                total_filament_volume_mm3=total_volume,
                support_material_volume_mm3=support_volume,
                estimated_print_time_s=print_time_s
            )
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing slicer output: {e}\nOutput:\n{slicer_output}")
            return None