"""
Data processing module for the 3D-LSPO project.

This file contains the core logic for handling the initial dataset of 3D models.
Its primary responsibility is to load raw 3D model files (e.g., .step, .stl)
and convert them into a structured, sequential representation of Constructive
Solid Geometry (CSG) operations using the cadquery library. This sequential
representation is referred to as a "design trace."

These design traces serve as the fundamental input for the MotifEncoder model,
which learns to embed them into a latent space.
"""

import os
import logging
from pathlib import Path
from typing import List, Generator, Tuple

# Attempt to import cadquery, but handle cases where it might not be installed.
try:
    import cadquery as cq
except ImportError:
    # This allows the module to be imported and inspected even if cadquery is not available,
    # though the functions that require it will fail at runtime.
    cq = None

# Configure logging for clear output during the processing pipeline.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A type alias for clarity, representing a sequence of cadquery operations.
DesignTrace = List[str]


def _convert_model_to_design_trace(model_path: Path) -> DesignTrace:
    """
    Converts a single 3D model file into a design trace.

    This is the core conversion function. It loads a model from the given
    path and attempts to reverse-engineer it into a sequence of
    `cadquery` Python API calls.

    NOTE: The process of reverse-engineering a boundary representation (B-rep) or
    mesh file (like STEP or STL) into a procedural CSG script is an unsolved
    and extremely complex research problem. The current implementation is a
    significant simplification and serves as a placeholder. It imports the model
    and generates a design trace consisting of a single command to create a
    bounding box of the same dimensions. This provides a basic, consistent
    representation of the model's overall size but loses all geometric detail.

    Args:
        model_path: The absolute path to the 3D model file (e.g., .step).

    Returns:
        A list of strings representing the design trace. Returns an empty
        list if the conversion fails.
    """
    if not cq:
        logging.error("CadQuery library is not installed. Cannot process models.")
        return []

    try:
        # Load the shape from the file. STEP format is preferred for its accuracy.
        shape = cq.importers.importStep(str(model_path))
        if shape.val() is None:
             raise ValueError("Failed to import STEP file or the file is empty.")

        # Get the bounding box of the loaded shape.
        bb = shape.val().BoundingBox()

        # Create a simplified design trace based on the bounding box.
        # This is a placeholder for a much more complex feature decomposition logic.
        trace = [
            f'result = cq.Workplane("XY").box({bb.xlen:.4f}, {bb.ylen:.4f}, {bb.zlen:.4f})'
        ]
        return trace

    except Exception as e:
        # This broad exception catches errors from cq.importers and other issues.
        logging.warning(f"Could not process file {model_path.name}: {e}")
        return []


def _save_design_trace(trace: DesignTrace, output_path: Path) -> None:
    """
    Saves a generated design trace to a text file.

    Each operation in the trace is written as a new line in the output file.
    If the trace is empty, no file is created.

    Args:
        trace: A list of strings, where each string is a cadquery operation.
        output_path: The path to the file where the trace will be saved.
    """
    if not trace:
        return

    try:
        # Ensure the parent directory of the output file exists.
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the file in write mode and save the trace.
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(trace))
    except IOError as e:
        logging.error(f"Failed to save design trace to {output_path}: {e}")


def _find_model_files(input_dir: Path) -> Generator[Path, None, None]:
    """
    Scans a directory for supported 3D model files.

    Recursively searches the input directory and yields the path for each
    file with a supported extension. This implementation prioritizes STEP files
    as they are more suitable for B-rep analysis than mesh files like STL.

    Args:
        input_dir: The directory to search for model files.

    Yields:
        A Path object for each found model file.
    """
    supported_extensions = {'.step', '.stp'}
    
    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        return

    logging.info(f"Scanning for model files in '{input_dir}' with extensions: {supported_extensions}")
    found_count = 0
    for file_path in input_dir.rglob('*'):
        if file_path.suffix.lower() in supported_extensions:
            yield file_path
            found_count += 1
    
    if found_count == 0:
        logging.warning(f"No model files with extensions {supported_extensions} found in '{input_dir}'.")


def process_raw_models(input_dir: Path, output_dir: Path) -> None:
    """
    Processes an entire directory of raw 3D models into design traces.

    This function orchestrates the conversion process. It finds all supported
    model files in the `input_dir`, converts each one to a design trace using
    `_convert_model_to_design_trace`, and saves the resulting trace as a
    .txt file in the `output_dir`, preserving the original filename and
    subdirectory structure.

    For example, a model at `input_dir/subdir/my_model.step` will be processed
    and saved as `output_dir/subdir/my_model.txt`.

    Args:
        input_dir: The root directory containing raw 3D model files.
        output_dir: The root directory where processed design trace files (.txt)
                    will be saved. This directory will be created if it doesn't
                    exist.
    """
    # Create the main output directory if it doesn't exist.
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting processing of raw models from '{input_dir}'.")
    logging.info(f"Output will be saved to '{output_dir}'.")

    processed_count = 0
    failed_count = 0
    
    model_files = list(_find_model_files(input_dir))
    total_files = len(model_files)

    for i, model_path in enumerate(model_files):
        logging.info(f"[{i+1}/{total_files}] Processing: {model_path.relative_to(input_dir)}")

        # Convert the model to a design trace.
        trace = _convert_model_to_design_trace(model_path)

        if trace:
            # Construct the corresponding output path, preserving the subdirectory
            # structure from the input and changing the extension to .txt.
            relative_path = model_path.relative_to(input_dir)
            output_path = (output_dir / relative_path).with_suffix('.txt')

            # Save the design trace to the new path.
            _save_design_trace(trace, output_path)
            processed_count += 1
        else:
            logging.warning(f"-> Failed to convert {model_path.name}. Skipping.")
            failed_count += 1
            
    logging.info("--- Data Processing Complete ---")
    logging.info(f"Successfully processed: {processed_count} files.")
    logging.info(f"Failed to process: {failed_count} files.")