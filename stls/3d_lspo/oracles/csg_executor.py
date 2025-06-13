"""
Executes a cadquery script in a sandboxed environment and saves the result.

This module provides a utility function to safely execute a string of Python code
that uses the cadquery library to generate a 3D model. The final model is then
exported to a specified file path in STL format.
"""

import os
import cadquery as cq
from cadquery.occ_impl.shapes import Shape
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union


class CSGExecutionError(Exception):
    """Custom exception for failures during CSG script execution or export."""
    pass


def execute_cad_script(
    script_content: str,
    output_stl_path: Union[str, Path],
    result_variable_name: str = "result"
) -> Tuple[bool, Optional[str]]:
    """
    Executes a string containing a cadquery script and saves the resulting model.

    This function programmatically executes the provided Python code string, which is
    expected to generate a cadquery object. It captures the final object from the
    script's local execution scope and exports it as an STL file. The execution is
    sandboxed to prevent it from affecting the global state. All potential
    exceptions during execution and export are caught and handled.

    Args:
        script_content (str):
            A string containing the multi-line cadquery script to be executed.
        output_stl_path (Union[str, Path]):
            The file path where the resulting .stl model should be saved.
            The parent directory will be created if it does not exist.
        result_variable_name (str):
            The name of the variable in the `script_content` that holds the final
            cadquery Workplane or Shape object to be exported. Defaults to "result".

    Returns:
        Tuple[bool, Optional[str]]:
            A tuple where the first element is a boolean indicating success (True) or
            failure (False), and the second element is an error message string if
            execution failed, otherwise None.
    """
    try:
        # 1. Convert output_stl_path to a Path object for robust handling.
        path_obj = Path(output_stl_path)

        # 2. Ensure the parent directory of the output path exists.
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # 3. Define a controlled environment (scope) for `exec`.
        # This provides the necessary imports to the script without polluting
        # the global namespace of the main application.
        execution_scope: Dict[str, Any] = {'cq': cq}

        # 4a. Execute the script within the controlled scope.
        exec(script_content, execution_scope)

        # 4b. Check if the expected result variable was created by the script.
        if result_variable_name not in execution_scope:
            raise CSGExecutionError(
                f"Script executed successfully but did not produce the expected "
                f"result variable named '{result_variable_name}'."
            )

        # 4c. Retrieve the result object from the execution scope.
        model = execution_scope[result_variable_name]

        # 4d. Validate that the retrieved object is a valid CadQuery object.
        # It could be a Workplane or a raw Shape.
        if not isinstance(model, (cq.Workplane, Shape)):
            raise CSGExecutionError(
                f"The result variable '{result_variable_name}' was of type "
                f"{type(model).__name__}, not a valid CadQuery Workplane or Shape."
            )

        # 4e. Export the model to the specified STL file path.
        # We convert the Path object to a string as `export` expects a string path.
        cq.exporters.export(model, str(path_obj))
        
        # 4f. If all steps succeed, return a success status.
        return (True, None)

    # 5. Catch any exception that occurs during the process.
    except Exception as e:
        # Formulate a descriptive error message and return a failure status.
        error_message = f"Failed to execute/export CAD script: {type(e).__name__}: {e}"
        return (False, error_message)