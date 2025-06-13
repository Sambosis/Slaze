"""
utils.py

A collection of utility functions for the 3D-LSPO project.

This module provides common helper functions for tasks such as setting up logging,
managing file I/O operations, and interacting with external command-line
tools like OpenSCAD.
"""

import logging
import subprocess
import pathlib
import sys
from typing import Union, List, Tuple, Optional

# The config module is expected to hold constants like executable paths.
from src.lspo_3d import config


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[Union[str, pathlib.Path]] = None
) -> None:
    """Configures the root logger for the project.

    Sets up a basic logging configuration that outputs messages to the console
    and, optionally, to a specified log file.

    Args:
        log_level (int): The minimum logging level to capture (e.g., logging.INFO,
            logging.DEBUG). Defaults to logging.INFO.
        log_file (Optional[Union[str, pathlib.Path]]): If provided, the path to a
            file where logs will be written. Defaults to None.
    """
    root_logger = logging.getLogger()
    # Clear existing handlers to prevent duplicate logs
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
        
    root_logger.setLevel(log_level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler
    if log_file:
        ensure_directory_exists(log_file)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    logging.info("Logging configured.")


def ensure_directory_exists(path: Union[str, pathlib.Path]) -> None:
    """Ensures that the directory for a given path exists.

    If the path is a file, it ensures the parent directory exists. If the path
    is a directory, it ensures that directory exists.

    Args:
        path (Union[str, pathlib.Path]): The file or directory path.

    Raises:
        OSError: If there is an issue creating the directory.
    """
    p = pathlib.Path(path)
    # If the path has a suffix, it's treated as a file; get its parent directory.
    # Otherwise, it's treated as a directory path itself.
    directory = p.parent if p.suffix else p
    
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Failed to create directory {directory}: {e}")
        raise


def save_text_to_file(filepath: Union[str, pathlib.Path], content: str) -> None:
    """Saves a string of text content to a file.

    This function will overwrite the file if it already exists. It also ensures
    the parent directory exists before writing.

    Args:
        filepath (Union[str, pathlib.Path]): The full path to the file to be saved.
        content (str): The string content to write to the file.
    """
    ensure_directory_exists(filepath)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        logging.error(f"Error writing to file {filepath}: {e}")
        raise


def load_text_from_file(filepath: Union[str, pathlib.Path]) -> str:
    """Loads text content from a specified file.

    Args:
        filepath (Union[str, pathlib.Path]): The path to the file to read.

    Returns:
        str: The content of the file as a string.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there is another issue reading the file.
    """
    p = pathlib.Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"File not found at path: {filepath}")

    try:
        with open(p, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        logging.error(f"Error reading from file {filepath}: {e}")
        raise


def run_openscad_cli(
    input_scad_path: Union[str, pathlib.Path],
    output_stl_path: Union[str, pathlib.Path]
) -> Tuple[bool, str]:
    """Executes the OpenSCAD command-line interface to convert .scad to .stl.

    This function constructs and runs a subprocess command to invoke OpenSCAD for
    model compilation. It captures the output and errors to determine success.
    The path to the OpenSCAD executable is retrieved from the project config.

    Args:
        input_scad_path (Union[str, pathlib.Path]): Path to the input .scad script.
        output_stl_path (Union[str, pathlib.Path]): Path where the output .stl
            mesh file will be saved.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: True if the compilation was successful, False otherwise.
            - str: The captured stdout and stderr from the subprocess, useful for
                   debugging errors.
    """
    if not config.OPENSCAD_PATH or not pathlib.Path(config.OPENSCAD_PATH).is_file():
        error_msg = f"OpenSCAD executable not found or not configured at: {config.OPENSCAD_PATH}"
        logging.error(error_msg)
        return False, error_msg

    # Ensure output directory exists before attempting to write the file there.
    ensure_directory_exists(output_stl_path)

    command: List[str] = [
        str(config.OPENSCAD_PATH),
        '-o', str(output_stl_path),
        str(input_scad_path)
    ]
    
    logging.debug(f"Running OpenSCAD command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.OPENSCAD_TIMEOUT,
            check=False  # Do not raise exception for non-zero exit codes
        )
    except FileNotFoundError:
        error_msg = f"Failed to execute command. Is '{config.OPENSCAD_PATH}' the correct path?"
        logging.error(error_msg)
        return False, error_msg
    except subprocess.TimeoutExpired:
        error_msg = f"OpenSCAD command timed out after {config.OPENSCAD_TIMEOUT} seconds."
        logging.warning(error_msg)
        return False, error_msg
    
    output_log = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"

    if result.returncode != 0:
        logging.warning(f"OpenSCAD failed for {input_scad_path} with exit code {result.returncode}.")
        logging.debug(f"OpenSCAD output log:\n{output_log}")
        return False, output_log

    # Check that the output file was actually created
    if not pathlib.Path(output_stl_path).is_file() or pathlib.Path(output_stl_path).stat().st_size == 0:
        error_msg = f"OpenSCAD ran successfully but the output file '{output_stl_path}' is missing or empty."
        logging.warning(error_msg)
        logging.debug(f"OpenSCAD output log:\n{output_log}")
        return False, f"{error_msg}\n\n{output_log}"

    logging.debug(f"Successfully compiled {input_scad_path} to {output_stl_path}.")
    return True, output_log