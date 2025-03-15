#!/usr/bin/env python3
import asyncio
import os
from pathlib import Path
from tools.envsetup import ProjectSetupTool, ProjectCommand
from tools.write_code import WriteCodeTool
from utils.docker_service import DockerService
from tools import ToolResult
from rich import print as rr
from openai import OpenAI, AsyncOpenAI
import agent
from config import PROMPTS_DIR, LOGS_DIR, get_project_dir, get_docker_project_dir, set_project_dir
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
# Main Function for Testing
# -----------------------------------------------------------------------------

import os
import re
import ast
import asyncio
from openai import AsyncOpenAI

# --- Helper Functions ---
# We only need extract_local_imports, since you handle the LLM call and writing.


# --- Helper Functions ---
import ast


import os
import ast
from pathlib import Path
from typing import List


async def lmin_agent(agent):
    """
    Main agent function that:
      - Retrieves the project task.
      - Gets a comma-separated list of relative file paths from the LLM.
      - Converts the relative paths (with forward slashes) to full Windows paths.
      - Iteratively generates files and checks each file for local imports.
    """
    project_dir = get_project_dir()  # Should return a Path (Windows-style)
    task = await agent.get_task()
    print(f"Project Dir = {project_dir}")

    client = AsyncOpenAI()
    model = "gpt-4o"

    # --- 1. Get the list of files ---
    initial_prompt = f"""
        You are an AI assistant tasked with creating a Python project.
        Given the following task, generate a list of the files needed for the project.
        Provide the list of files in a comma-separated format, including the file extension.
        Do not provide any additional explanation. Only provide file names.
        If the file should be in a subdirectory include that in the name.
        Task: {task}
        """

    messages = [{"role": "user", "content": initial_prompt}]

    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        )

    file_list_str = completion.choices[0].message.content
    rr(f"String from LLM    {file_list_str}")
    file_list = [f.strip() for f in file_list_str.split(",") if f.strip()]
    print(f"File list: {file_list}")

    written_files = set()
    files_to_write = [file_list[0]]  # Start with the first file

    while files_to_write:
        current_file = files_to_write.pop(0)
        if current_file in written_files:
            continue

        # Convert the Linux-style relative path to a Windows path.
        full_path = project_dir / Path(current_file.replace("/", os.sep))

        # Generate code for the file. Using a simple description here.
        code_description = f"Generate code for {current_file}. This is part of the project. Task: {task}"
        success = await write_code_to_file(
            project_path=project_dir,
            python_filename=current_file,
            code_description=code_description,
        )
        if not success:
            print(f"Error writing {current_file}")
            return

        written_files.add(current_file)
        # Read the current file to check for local imports.
        with open(full_path, "r") as f:
            code = f.read()
        # Check for local imports in the newly written file.
        local_imports = find_local_imports(code, file_list)
        print(f"For file {current_file}, found local imports: {local_imports}")
        for imp in local_imports:
            candidate = imp  # already a path like 'ui/stats_display.py'
            if candidate not in written_files and candidate in file_list:
                files_to_write.append(candidate)
                rr(f"Adding {candidate} to files_to_write")


def find_local_imports(code: str, file_list: set) -> set:
    imports = set(
        re.findall(
            r"^\s*import ([\w\.]+)|^\s*from ([\w\.]+) import", code, re.MULTILINE
        )
    )
    flat_imports = set()
    for imp in imports:
        imp_name = imp[0] or imp[1]
        imp_file_path = imp_name.replace(".", "/") + ".py"
        if imp_file_path in file_list:
            flat_imports.add(imp_file_path)
    return flat_imports


# async def lmin_agent(agent):
#     docker_tool = DockerService()
#     DOCKER_DIR = get_docker_project_dir()
#     PROJECT_DIR = get_project_dir()
#     task = await agent.get_task()
#     client = AsyncOpenAI()
#     model = "gpt-3.5-turbo-1106"
#     print(f"Docker Project Dir = {DOCKER_DIR}")
#     print(f"Project Dir = {PROJECT_DIR}")
#     tool_project = docker_tool.from_docker_path(DOCKER_DIR)
#     print(f"tool_project = {tool_project}")
#     test_file = "dog.py"
#     test_file_path = docker_tool.from_docker_path(test_file)
#     print(f"test_file_path = {test_file_path}")
#     docker_file = docker_tool.to_docker_path(test_file)
#     print(f"docker_file = {docker_file}")
#     # --- 1. Get the list of files ---
#     initial_prompt = f"""
#     You are an AI assistant tasked with creating a Python project.
#     Given the following task, generate a list of the files needed for the project.
#     Provide the list of files in a comma-separated format, including the file extension.
#     Do not provide any additional explanation. Only provide file names.
#     If the file should be in a subdirectory include that in the name.
#     Task: {task}
#     """
#     messages = [{"role": "user", "content": initial_prompt}]
#     try:
#         completion = await client.chat.completions.create(
#             model=model,
#             messages=messages,
#         )
#         file_list_str = completion.choices[0].message.content
#         # Normalize paths in file_list to use os.path.join
#         file_list = [
#             os.path.normpath(os.path.join(DOCKER_DIR, f.strip()))
#             for f in file_list_str.split(",")
#             if f.strip()
#         ]

#         print(f"File List: {file_list}")
#     except Exception as e:
#         print(f"Error generating file list: {e}")
#         return

#     # --- 2. Generate code iteratively ---

#     written_files = set()
#     files_to_write = [file_list[0]]  # Start with the first file
#     while files_to_write:
#         current_file = files_to_write.pop(0)
#         if current_file in written_files:
#             continue

#         # full_file_path = os.path.join(DOCKER_DIR, current_file) # Already have the full path

#         if current_file == file_list[0]:
#             code_description = f"Generate the code for {os.path.basename(current_file)}. This is the entry point of the application. Project Task: {task}"
#         else:
#             code_description = f"Generate the code for {os.path.basename(current_file)}. Project Task: {task}"

#         # Use YOUR write_code_to_file function
#         result = await write_code_to_file(
#             project_path=DOCKER_DIR,
#             python_filename=os.path.relpath(
#                 current_file, DOCKER_DIR
#             ),  # Pass relative path
#             code_description=code_description,
#         )
#         # Check for errors using the __bool__ method of ToolResult
#         if not result:
#             print(f"Error writing {current_file}: {result.error}")
#             return

#         written_files.add(current_file)

#         # Find local imports and add them to the queue.
#         imports = extract_local_imports(
#             current_file, file_list
#         )  # current_file already has full path
#         for imported_module in imports:
#             # Construct full paths for imported modules
#             imported_file_rel = imported_module.replace(".", "/") + ".py"
#             imported_file = os.path.normpath(
#                 os.path.join(DOCKER_DIR, imported_file_rel)
#             )

#             if imported_file not in written_files and imported_file in file_list:
#                 files_to_write.append(imported_file)  # Use full path
#             # Handle cases where import is just the module name (no subdirs)
#             elif imported_module + ".py" not in written_files:
#                 imported_file_simple = os.path.normpath(
#                     os.path.join(DOCKER_DIR, imported_module + ".py")
#                 )
#                 if imported_file_simple in file_list:
#                     converted_path = docker_tool.from_docker_path(imported_file_simple)
#                     files_to_write.append(converted_path)

#
async def main():
    import asyncio


async def main():
    # Project directory
    project_path = "/home/myuser/apps/graphingCalc"

    # Create a Python virtual environment in the project directory with needed packages.
    venv_result = await create_python_venv(
        project_path=project_path,
        packages=["matplotlib", "numpy"],  # Tkinter is included with Python
    )
    rr(venv_result)

    # List of files to create for the Graphing Calculator application.
    files_to_create = [
        {
            "file_path": "graphing_calculator.py",
            "code_description": (
                "Main entry point for the graphing calculator GUI application. "
                "Initializes the Tkinter GUI and imports the MainWindow from gui/main_window.py."
            ),
        },
        {
            "file_path": "gui/main_window.py",
            "code_description": (
                "Defines the main application window using Tkinter. Sets up the overall layout by integrating "
                "an input panel, toolbar, and graph canvas. This window frames the entire calculator GUI."
            ),
        },
        {
            "file_path": "gui/graph_canvas.py",
            "code_description": (
                "Defines a custom canvas widget that embeds a Matplotlib figure within a Tkinter interface. "
                "Supports plotting, zooming, panning, and graph updates based on user actions."
            ),
        },
        {
            "file_path": "gui/input_panel.py",
            "code_description": (
                "Creates a panel for user input of mathematical expressions. Facilitates function entry, "
                "color selection, and communicates with the function parser for validation."
            ),
        },
        {
            "file_path": "gui/toolbar.py",
            "code_description": (
                "Provides GUI controls such as zoom in/out, pan, and save. Contains buttons and logic to "
                "interact with the graph canvas and manipulate the graph view."
            ),
        },
        {
            "file_path": "core/function_parser.py",
            "code_description": (
                "Handles parsing, validation, and safe evaluation of mathematical expressions. Uses NumPy "
                "to process arithmetic, trigonometric, logarithmic, and exponential functions for graphing."
            ),
        },
        {
            "file_path": "core/graph_manager.py",
            "code_description": (
                "Manages the plotting and updating of multiple function graphs. Tracks functions, their "
                "colors, and visibility, and triggers redraws in the graph canvas when necessary."
            ),
        },
        {
            "file_path": "core/function.py",
            "code_description": (
                "Contains the Function class, which encapsulates a mathematical function. Stores properties "
                "such as the expression, color, and visibility, and defines methods for evaluating the function over a range."
            ),
        },
        {
            "file_path": "utils/math_helpers.py",
            "code_description": (
                "Provides helper functions for performing mathematical operations and coordinate transformations. "
                "Used by the graph manager and canvas for proper plotting ranges and scaling."
            ),
        },
        {
            "file_path": "utils/color_palette.py",
            "code_description": (
                "Defines color schemes and functions to generate distinct colors for plotting multiple functions. "
                "Ensures that graphs have visually distinct colors."
            ),
        },
    ]

    # Write all the necessary files to the project structure.
    multiple_files_result = await write_code_multiple_files(
        project_path=project_path,
        files=files_to_create,
    )
    rr(multiple_files_result)

    # Run the main application to test that the graphing calculator GUI starts up as expected.
    run_result = await run_python_app(
        project_path=project_path,
        entry_filename="graphing_calculator.py",
    )
    rr(run_result)


if __name__ == "__main__":
    asyncio.run(main())


# --- Main Agent Function (Modified) ---


# async def lmin_agent(agent):
#     docker_tool = DockerService()
#     DOCKER_DIR = agent.get_docker_project_dir()
#     task = await agent.get_task()
#     client = AsyncOpenAI()
#     model = "gpt-3.5-turbo-1106"
#     print(f"Docker Project Dir = {DOCKER_DIR}")

#     # --- 1. Get the list of files ---
#     initial_prompt = f"""
#     You are an AI assistant tasked with creating a Python project.
#     Given the following task, generate a list of the files needed for the project.
#     Provide the list of files in a comma-separated format, including the file extension.
#     Do not provide any additional explanation. Only provide file names.
#     If the file should be in a subdirectory include that in the name.
#     Task: {task}
#     """
#     messages = [{"role": "user", "content": initial_prompt}]
#     try:
#         completion = await client.chat.completions.create(
#             model=model,
#             messages=messages,
#         )
#         file_list_str = completion.choices[0].message.content
#         # Normalize paths in file_list to use os.path.join
#         file_list = [
#             os.path.normpath(os.path.join(DOCKER_DIR, f.strip()))
#             for f in file_list_str.split(",")
#             if f.strip()
#         ]
#         print(f"File List: {file_list}")
#     except Exception as e:
#         print(f"Error generating file list: {e}")
#         return

#     # --- 2. Generate code iteratively ---

#     written_files = set()
#     files_to_write = [file_list[0]]  # Start with the first file

#     while files_to_write:
#         current_file = files_to_write.pop(0)
#         if current_file in written_files:
#             continue

#         # full_file_path = os.path.join(DOCKER_DIR, current_file) # Already have the full path

#         if current_file == file_list[0]:
#             code_description = f"Generate the code for {os.path.basename(current_file)}. This is the entry point of the application. Project Task: {task}"
#         else:
#             code_description = f"Generate the code for {os.path.basename(current_file)}. Project Task: {task}"

#         # Use YOUR write_code_to_file function
#         result = await write_code_to_file(
#             project_path=DOCKER_DIR,
#             python_filename=os.path.relpath(
#                 current_file, DOCKER_DIR
#             ),  # Pass relative path
#             code_description=code_description,
#         )
#         # Check for errors using the __bool__ method of ToolResult
#         if not result:
#             print(f"Error writing {current_file}: {result.error}")
#             return

#         written_files.add(current_file)

#         # Find local imports and add them to the queue.
#         imports = extract_local_imports(
#             current_file, file_list
#         )  # current_file already has full path
#         for imported_module in imports:
#             # Construct full paths for imported modules
#             imported_file_rel = imported_module.replace(".", "/") + ".py"
#             imported_file = os.path.normpath(
#                 os.path.join(DOCKER_DIR, imported_file_rel)
#             )

#             if imported_file not in written_files and imported_file in file_list:
#                 files_to_write.append(imported_file)  # Use full path
#             # Handle cases where import is just the module name (no subdirs)
#             elif imported_module + ".py" not in written_files:
#                 imported_file_simple = os.path.normpath(
#                     os.path.join(DOCKER_DIR, imported_module + ".py")
#                 )
#                 if imported_file_simple in file_list:
#                     files_to_write.append(imported_file_simple)
#     await agent.set_status("completed")


# --- Main Agent Function (Modified) ---
