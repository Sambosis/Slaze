# 3d_lspo/data/processor.py

"""
Contains functions for processing raw 3D model files from the dataset.

This module is responsible for the critical first step of the project:
converting standard 3D model formats (like .stl or .step) into a
programmatic representation as a sequence of Constructive Solid Geometry (CSG)
operations. This "design trace" is a string of executable `cadquery` code
that serves as the input for training the motif encoder.
"""

import os
from pathlib import Path
from typing import List, Optional

import cadquery as cq


def convert_model_to_csg_trace(model_path: Path) -> Optional[str]:
    """Converts a single 3D model file into a `cadquery` CSG operation sequence.

    This function attempts the complex task of "reverse engineering" a 3D model
    file (like a STEP or STL) into a parametric script. It loads the geometry
    and applies feature recognition logic to generate a string of Python code
    using the `cadquery` library that can parametrically reconstruct the model.

    This is a highly challenging task, especially for mesh-based formats like STL.
    The success of this function is fundamental to creating the design traces
    for the motif discovery phase.

    Args:
        model_path: The file path to the 3D model (e.g., 'path/to/model.step').

    Returns:
        A string representing the executable `cadquery` script if conversion
        is successful, otherwise None. For example:
        'result = cq.Workplane("XY").box(10, 10, 5).faces(">Z").circle(2).cutThruAll()'
    """
    # --- Implementation Note ---
    # The task of reverse-engineering a boundary representation (B-rep) from a
    # STEP file or, even more difficult, a mesh from an STL file, into a
    # sequence of parametric CSG operations is a significant research problem in
    # computational geometry and CAD. A full implementation would require complex
    # feature recognition algorithms (e.g., for holes, pockets, bosses, fillets)
    # and is beyond the scope of this project's core focus.
    #
    # As a pragmatic and simplified placeholder, this function imports the shape
    # and then generates a CSG operation that creates a bounding box of the same
    # dimensions. This provides a very basic, dimensionally-aware "design trace"
    # that can be used for initial testing of downstream models. It does NOT
    # preserve the model's actual geometry but serves as a starting point.

    if not model_path.is_file():
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        # 1. Load the model using cq.importers.importShape().
        shape = cq.importers.importShape(str(model_path))

        # 2. Analyze the resulting shape's geometry to infer a primitive.
        #    Here, we simplify this to just getting the bounding box.
        if shape.isValid():
            bb = shape.BoundingBox()
            # 3. Build the string representation of a cadquery operation.
            # We create a simple box primitive as a representative trace.
            # The 'result =' part is important for the csg_executor.
            csg_trace = (
                f'import cadquery as cq\n\n'
                f'result = cq.Workplane("XY").box({bb.xlen:.4f}, {bb.ylen:.4f}, {bb.zlen:.4f})'
            )
            return csg_trace
        else:
            print(f"Warning: Imported shape from {model_path} is not valid.")
            return None

    except Exception as e:
        print(f"Error processing model file {model_path}: {e}")
        return None


def process_raw_models(
    input_dir: Path,
    output_dir: Path,
    file_extensions: Optional[List[str]] = None
) -> List[Path]:
    """Processes all 3D models in a directory, converting them to CSG traces.

    This function acts as a batch processor. It recursively scans the `input_dir`
    for files with the specified extensions (defaulting to .step and .stl),
    invokes `convert_model_to_csg_trace` on each, and saves the resulting script
    to the `output_dir` with a corresponding '.py' extension.

    Args:
        input_dir: The path to the directory containing raw 3D model files.
        output_dir: The path to the directory where the processed CSG trace
                    script files (*.py) will be saved.
        file_extensions: A list of file extensions to process (e.g.,
                         ['.step', '.stl']). Defaults to common CAD formats.

    Returns:
        A list of `pathlib.Path` objects pointing to the successfully created
        CSG trace files in the output directory.
    """
    # 1. Set default file extensions if None.
    if file_extensions is None:
        file_extensions = ['.step', '.stp', '.stl']
    # Ensure extensions are lowercase for case-insensitive matching
    file_extensions = [ext.lower() for ext in file_extensions]

    # 2. Ensure the output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir.resolve()}")

    processed_files: List[Path] = []
    print(f"Scanning for models in: {input_dir.resolve()} with extensions: {file_extensions}")

    # 3. Iterate through files in input_dir using Path.rglob for recursion.
    for model_path in input_dir.rglob('*'):
        if model_path.is_file() and model_path.suffix.lower() in file_extensions:
            print(f"Processing {model_path.name}...")

            # 4. For each file matching the extensions, call convert_model_to_csg_trace.
            trace_string = convert_model_to_csg_trace(model_path)

            # 5. If a valid trace string is returned, save it.
            if trace_string:
                # a. Construct the output path (e.g., 'model.step' -> 'model.py').
                output_filename = model_path.stem + ".py"
                output_path = output_dir / output_filename

                # b. Write the string to the output file.
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(trace_string)
                    # c. Add the new file path to a list of processed files.
                    processed_files.append(output_path)
                    print(f"  -> Successfully converted. Saved to {output_path}")
                except IOError as e:
                    print(f"  -> ERROR: Could not write to file {output_path}: {e}")
            else:
                print(f"  -> Failed to convert {model_path.name}.")

    print(f"\nProcessing complete. Successfully converted {len(processed_files)} files.")
    # 6. Return the list of created file paths.
    return processed_files