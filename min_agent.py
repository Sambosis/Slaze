#!/usr/bin/env python3
import asyncio
import os
from pathlib import Path
from tools.envsetup import ProjectSetupTool, ProjectCommand
from tools.write_code import WriteCodeTool
from tools import ToolResult
from rich import print as rr

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


async def create_python_venv(
    project_path: str, packages: list[str] = None
) -> ToolResult:
    """
    Create a Python virtual environment (venv) for the project in Docker.

    This helper function uses the ProjectSetupTool with the "setup_project" command to
    initialize the project environment (including directory creation, venv setup, and
    installation of any specified packages).

    Args:
        project_path (str): The absolute path of the project on the host, e.g. "/home/myuser/apps/nogame"
        packages (list[str], optional): A list of Python packages to install in the venv. Defaults to None.

    Returns:
        ToolResult: The result object returned by the ProjectSetupTool containing output, status, and messages.
    """
    tool = ProjectSetupTool()
    tool_input = {
        "command": "setup_project",
        "project_path": project_path,
        "packages": packages,
    }
    return await tool(**tool_input)


async def write_code_to_file(
    project_path: str, python_filename: str, code_description: str
) -> ToolResult:
    """
    Write a single code file to the project using the WriteCodeTool.

    This helper function directs the WriteCodeTool to generate and write a complete Python file
    based on the provided code description. The tool will use an intermediate code skeleton generated via LLM calls and will write the resulting file into the Docker container as well as log it locally.

    Args:
        project_path (str): The base project path (e.g., "/home/myuser/apps/nogame").
        python_filename (str): The relative file path to create within the project (e.g., "src/greeter.py").
        code_description (str): A detailed description of the required code, including functionality, class definitions,
                                  and docstrings.

    Returns:
        ToolResult: The result object from the WriteCodeTool call.
    """
    tool = WriteCodeTool()
    tool_input = {
        "command": "write_code_to_file",
        "project_path": project_path,
        "python_filename": python_filename,
        "code_description": code_description,
    }
    return await tool(**tool_input)


async def write_code_multiple_files(project_path: str, files: list[dict]) -> ToolResult:
    """
    Write multiple code files at once using the WriteCodeTool.

    The function accepts a list of file descriptions, where each entry is a dictionary containing:
      - "file_path": The relative path (within the project) for the file.
      - "code_description": A detailed description of the code to be generated for that file.

    The tool will generate and write each file using LLM calls that first generate a code skeleton and then the full implementation.

    Args:
        project_path (str): The base project path where files will be written (e.g., "/home/myuser/apps/nogame").
        files (list[dict]): A list of dictionaries with keys "file_path" and "code_description" for each file to create.

    Returns:
        ToolResult: A ToolResult object that includes a concatenated summary of all file outputs.
    """
    tool = WriteCodeTool()
    tool_input = {
        "command": "write_code_multiple_files",
        "project_path": project_path,
        "files": files,
    }
    return await tool(**tool_input)


async def run_python_app(project_path: str, entry_filename: str) -> ToolResult:
    """
    Run the main Python application in the Docker container.

    This helper function uses the ProjectSetupTool with the "run_app" command to start the application.
    It ensures that the correct environment is activated and directs Docker to run the Python script.

    Args:
        project_path (str): The base project path on the host (e.g., "/home/myuser/apps/nogame").
        entry_filename (str): The entry point Python file to run within the project (e.g., "main.py" or "src/app.py").

    Returns:
        ToolResult: The result object from the run_app command, including status, output, and any errors.
    """
    tool = ProjectSetupTool()
    tool_input = {
        "command": "run_app",
        "project_path": project_path,
        "entry_filename": entry_filename,
        "environment": "python",
    }
    return await tool(**tool_input)


# -----------------------------------------------------------------------------
# Main Functi\sult)

    # Example 2: Write a single code file using the write_code tool
    single_file_description = (
        "Create a Python class named 'Greeter' with an __init__ that accepts a 'name' (str) "
        "and a method called 'greet' that returns a greeting message. Include proper import statements and docstrings."
    )
    single_file_result = await write_code_to_file(
        project_path="/home/myuser/apps/nogame",
        python_filename="src/greeter.py",
        code_description=single_file_description,
    )
    rr(single_file_result)

    # Example 3: Write multiple code files using the write_code tool
    files_to_create = [
        {
            "file_path": "src/config.py",
            "code_description": "Define game configuration constants such as screen dimensions, colors, and basic game rules.",
        },

        {
            "file_path": "src/utils/logger.py",
            "code_description": "Create a logging utility that supports multiple log levels (debug, info, warning, error) and outputs logs to both a file and the console.",
        },
    ]
    multiple_files_result = await write_code_multiple_files(
        project_path="/home/myuser/apps/nogame",
        files=files_to_create,
    )
    rr(multiple_files_result)

    # Example 4: Run the main Python application using the run_app command
    run_result = await run_python_app(
        project_path="/home/myuser/apps/nogame",
        entry_filename="src/greeter.py",  
    )
    rr(run_result)


if __name__ == "__main__":
    asyncio.run(main())
