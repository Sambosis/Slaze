# /3d_lspo/oracles/slicer_verifier.py

"""
A wrapper module for interfacing with a command-line 3D slicer.

This module provides functionality to take a 3D model file (e.g., .stl),
run it through a slicer like PrusaSlicer, and parse the output to
extract key manufacturability metrics. These metrics are crucial for the
reinforcement learning environment to calculate a reward score for a
generated design.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import NamedTuple, Optional, Tuple


class SlicerMetrics(NamedTuple):
    """
    A data structure to hold the metrics extracted from the slicer output.

    This structure standardizes the result of a slicing operation, making it
    easy to consume by the reinforcement learning environment.

    Attributes:
        slicing_successful (bool): True if the slicing process completed without
            errors, False otherwise.
        print_time_seconds (Optional[float]): Estimated print time in seconds.
            None if slicing failed or this metric could not be parsed.
        support_material_volume_mm3 (Optional[float]): Volume of support material
            in cubic millimeters. None if slicing failed or not parsed.
        filament_volume_mm3 (Optional[float]): Total volume of filament used
            (including supports) in cubic millimeters. None if slicing failed or
            not parsed.
        raw_output (str): The raw stdout and stderr from the slicer command,
            useful for debugging.
    """
    slicing_successful: bool
    print_time_seconds: Optional[float]
    support_material_volume_mm3: Optional[float]
    filament_volume_mm3: Optional[float]
    raw_output: str


def _parse_slicer_output(
    output: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Parses the console output from PrusaSlicer to extract key metrics.

    This helper function uses regular expressions to find relevant information
    such as print time, support material, and total filament usage from the
    raw text output generated by the slicer's command-line interface.

    Args:
        output (str): The raw string captured from the slicer's stdout/stderr.

    Returns:
        A tuple containing:
            - (Optional[float]): Estimated print time in seconds.
            - (Optional[float]): Volume of support material used in mm^3.
            - (Optional[float]): Total volume of filament used in mm^3.
        Each value is None if it could not be parsed from the output.
    """
    print_time_sec = None
    support_volume_mm3 = None
    filament_volume_mm3 = None

    # Parse estimated print time. Example: "Estimated printing time: 1h 23m 45s"
    # This regex captures optional hours, minutes, and seconds.
    time_match = re.search(r"Estimated printing time.*?(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?", output)
    if time_match:
        hours = int(time_match.group(1)) if time_match.group(1) else 0
        minutes = int(time_match.group(2)) if time_match.group(2) else 0
        seconds = int(time_match.group(3)) if time_match.group(3) else 0
        print_time_sec = float(hours * 3600 + minutes * 60 + seconds)

    # Parse total filament volume. Example: "Used filament: 123.45cm3"
    # Note: 1 cm^3 = 1000 mm^3
    filament_match = re.search(r"Used filament:\s*([\d\.]+)\s*cm3", output)
    if filament_match:
        filament_volume_cm3 = float(filament_match.group(1))
        filament_volume_mm3 = filament_volume_cm3 * 1000.0

    # Parse support material volume. Example: "Support material: 12.34cm3"
    # This might also appear as "support_material_used = ..." in G-code comments.
    support_match = re.search(r"Support material:\s*([\d\.]+)\s*cm3", output)
    if support_match:
        support_volume_cm3 = float(support_match.group(1))
        support_volume_mm3 = support_volume_cm3 * 1000.0

    return print_time_sec, support_volume_mm3, filament_volume_mm3


def get_slicer_metrics(
    stl_file_path: Path,
    slicer_executable_path: Path,
    slicer_config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> SlicerMetrics:
    """
    Runs a command-line slicer on an STL file and returns printability metrics.

    This function invokes a slicer (e.g., PrusaSlicer) as a subprocess,
    directs it to slice the given STL file, and captures its output. It then
    parses this output to determine if the slice was successful and to extract
    metrics like estimated print time and material usage.

    Args:
        stl_file_path (Path): The absolute path to the input .stl file.
        slicer_executable_path (Path): The absolute path to the slicer's
            executable (e.g., 'prusa-slicer-console.exe' on Windows or
            'prusa-slicer' on Linux).
        slicer_config_path (Optional[Path]): The path to a slicer configuration
            file (.ini). If None, slicer defaults will be used.
        output_dir (Optional[Path]): The directory to save the output .gcode file.
            If None, the G-code file is saved to the same directory as the STL
            file and cleaned up after the operation.

    Returns:
        SlicerMetrics: An object containing the results of the slicing operation.
                       If the slicer command fails to execute (e.g., file not
                       found), returns a SlicerMetrics object with
                       slicing_successful=False and the error message in raw_output.
    """
    # 1. Validate that input paths exist.
    if not stl_file_path.is_file():
        return SlicerMetrics(False, None, None, None, f"STL file not found at: {stl_file_path}")
    if not slicer_executable_path.is_file():
        return SlicerMetrics(False, None, None, None, f"Slicer executable not found at: {slicer_executable_path}")

    # 2. Define the output path for the G-code.
    # If no output_dir is given, the gcode file is temporary.
    is_temp_gcode = output_dir is None
    effective_output_dir = output_dir or stl_file_path.parent
    gcode_output_path = effective_output_dir / stl_file_path.with_suffix('.gcode').name
    
    # 3. Construct the command-line arguments.
    command = [
        str(slicer_executable_path),
        "--slice",
        str(stl_file_path),
        "--output",
        str(gcode_output_path),
        # This flag is crucial for getting detailed info in the output
        "--gcode-comments", 
        # Export metrics to console even if not slicing to gcode
        "--info" 
    ]
    if slicer_config_path:
        if slicer_config_path.is_file():
            command.extend(["--load", str(slicer_config_path)])
        else:
            return SlicerMetrics(False, None, None, None, f"Slicer config file not found at: {slicer_config_path}")

    # 4. Use subprocess.run() to execute the command.
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
            encoding='utf-8',
            errors='ignore'
        )
        raw_output = result.stdout + "\n" + result.stderr
        
        # 5. Check the return code. A non-zero code indicates failure.
        slicing_successful = result.returncode == 0
        
    except subprocess.TimeoutExpired:
        return SlicerMetrics(False, None, None, None, "Slicer process timed out after 120 seconds.")
    except Exception as e:
        return SlicerMetrics(False, None, None, None, f"An unexpected error occurred while running the slicer: {e}")

    # 6. If successful, parse the output for metrics.
    print_time = None
    support_volume = None
    filament_volume = None
    if slicing_successful:
        print_time, support_volume, filament_volume = _parse_slicer_output(raw_output)
    
    # 8. Clean up the generated G-code file if it was temporary.
    if is_temp_gcode and gcode_output_path.exists():
        try:
            gcode_output_path.unlink()
        except OSError as e:
            # Append cleanup error to raw_output for debugging, but don't fail the whole operation
            raw_output += f"\nWarning: Could not clean up temporary G-code file: {e}"

    # 7. Construct and return the SlicerMetrics NamedTuple.
    return SlicerMetrics(
        slicing_successful=slicing_successful,
        print_time_seconds=print_time,
        support_material_volume_mm3=support_volume,
        filament_volume_mm3=filament_volume,
        raw_output=raw_output,
    )