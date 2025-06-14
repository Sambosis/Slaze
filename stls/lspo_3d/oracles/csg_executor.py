# csg_executor.py
# /3d_lspo/oracles/csg_executor.py

"""
Executes a CadQuery script in a secure, sandboxed environment.

This module provides a utility function to take a string containing a CadQuery
Python script, execute it, and save the resulting 3D model to a specified
output file path as an STL. The execution is handled in a controlled scope
to minimize security risks from running generated code.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union

# External library imports
import cadquery as cq

# Set up a logger for this module
logger = logging.getLogger(__name__)


def _sanitize_script_string(script: str) -> str:
    """Return a version of ``script`` safe for ``exec``.

    The raw output from language models may contain curly quotes or other
    non-ASCII characters that cause ``exec`` to fail with a syntax error.
    This helper normalizes common curly quotes to standard ASCII quotes and
    strips characters outside the ASCII range.
    """
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for bad, good in replacements.items():
        script = script.replace(bad, good)
    # Remove any remaining non-ASCII characters
    script = script.encode("ascii", "ignore").decode("ascii")
    return script


def execute_cad_script(
    script_string: str, output_dir: Union[str, Path], output_filename: str
) -> Optional[str]:
    """
    Executes a CadQuery script string and exports the result to an STL file.

    This function creates a sandboxed environment to execute the provided
    Python code string. It expects the script to generate a CadQuery object
    (Workplane or Assembly) and assign it to a variable named 'result'.
    If the execution is successful and a valid CadQuery object is produced,
    it is exported to the specified file path.

    The function ensures the output directory exists before attempting to save.

    Args:
        script_string (str): A string containing the Python CadQuery script to execute.
            The script must assign the final CadQuery object to a variable
            named `result`.
        output_dir (Union[str, Path]): The path to the directory where the output STL file
            will be saved.
        output_filename (str): The base name for the output file (e.g.,
            "generated_model_1"). The `.stl` extension will be appended.

    Returns:
        Optional[str]: The full path to the created STL file if the script
                       execution and export are successful. Returns None if
                       an error occurs during execution (e.g., invalid syntax,
                       no 'result' object found, export failure).
    """
    # 1. Ensure the output directory exists.
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory '{output_dir}': {e}")
        return None

    # 2. Construct the full output file path.
    # The filename should not contain directory parts.
    base_filename = os.path.basename(output_filename)
    # Ensure it ends with .stl
    if not base_filename.lower().endswith('.stl'):
        base_filename += '.stl'
    full_output_path = os.path.join(output_dir, base_filename)

    # Sanitize the script before execution to avoid issues with non-ASCII
    # characters commonly produced by language models.
    script_string = _sanitize_script_string(script_string)


    # 3. Set up a controlled environment for `exec()` to act as a sandbox.
    #    This scope should include the `cadquery` module (`cq`).
    #    This is not a truly secure sandbox, but provides a controlled scope.
    sandbox_scope: Dict[str, Any] = {"cq": cq}

    # 4. Use a try/except block to safely execute the script_string.
    try:
        # Compile the script first so syntax errors are caught explicitly
        compiled_script = compile(script_string, "<string>", "exec")
        # Execute the script within the defined scope

        exec(compiled_script, sandbox_scope)


        # 5. After execution, retrieve the `result` object from the sandbox scope.
        result_object = sandbox_scope.get("result")

        if result_object is None:
            logger.warning(
                "Script executed successfully, but no 'result' variable was found in the scope."
            )
            return None

        # 6. Validate that 'result' is a cq.Workplane or cq.Assembly object.
        if not isinstance(result_object, (cq.Workplane, cq.Assembly)):
            logger.warning(
                f"Variable 'result' is of type {type(result_object)}, "
                "not a valid CadQuery Workplane or Assembly."
            )
            return None
        
        # Check if the result object is empty or invalid
        if not result_object.solids:
            logger.warning("The 'result' CadQuery object contains no solids to export.")
            return None

        # 7. If valid, use cq.exporters.export() to save the object as an STL file.
        cq.exporters.export(val=result_object, fname=full_output_path)
        logger.info(f"Successfully exported model to {full_output_path}")

        # 8. Return the full file path on success
        return full_output_path

    except Exception as e:
        # This will catch syntax errors from exec(), runtime errors within the
        # script, or errors from the cq.exporters.export() function.
        logger.error(
            f"An exception occurred during CadQuery script execution or export "
            f"for '{output_filename}': {e}",
            exc_info=False # Set to True for full traceback in logs
        )
        return None