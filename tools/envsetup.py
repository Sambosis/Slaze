from enum import Enum
from typing import Literal, List
from pathlib import Path
import os # Added os import
# type: ignore[override]
from .base import ToolResult, BaseAnthropicTool
import subprocess
import logging
from config import get_constant

# from loguru import logger as ll # Removed loguru
from rich import print as rr # Removed rich print

# Configure logging to a file # Removed loguru configuration
# ll.add(
#     "my_log_file.log",
#     rotation="500 KB",
#     level="DEBUG",
#     format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} - {message}",
# )

logger = logging.getLogger(__name__)

class ProjectCommand(str, Enum):
    """Commands supported by the ProjectSetupTool."""

    SETUP_PROJECT = "setup_project"
    ADD_DEPENDENCIES = "add_additional_depends"
    RUN_APP = "run_app"
    RUN_PROJECT = "run_project"


class ProjectSetupTool(BaseAnthropicTool):
    """
    A tool that sets up Python project environments using uv and manages script execution on Windows or Linux.
    """

    name: Literal["project_setup"] = "project_setup"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "A tool for Python project management. "
        "setup_project: create a uv virtual environment and install packages. "
        "add_dependencies: install additional packages. "
        "run_app: execute a Python file using uv. "
        "run_project: run the project entry file."
    )

    def __init__(self, display=None):
        """Initialize the ProjectSetupTool instance."""
        super().__init__(display=display)
        self.display = display  # Explicitly set self.display

    def to_params(self) -> dict:
        """Convert the tool to a parameters dictionary for the API."""
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": [cmd.value for cmd in ProjectCommand],
                            "description": "Command to execute",
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project directory",
                        },
                        "environment": {
                            "type": "string",
                            "enum": ["python"],
                            "description": "Environment type (python)",
                            "default": "python",
                        },
                        "packages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of packages to install, This can be used during the setup_project command or the add_dependencies command. this should be a list of strings with each package in quotes and separated by commas, with the list enclosed in square brackets. Example: ['package1', 'package2', 'package3']",
                        },
                        "entry_filename": {
                            "type": "string",
                            "description": "Name of the entry point file to run",
                            "default": "app.py",
                        },
                    },
                    "required": ["command", "project_path"],
                },
            },
        }
        return params

    def format_output(self, data: dict) -> str:
        """Format the output data as a readable string."""
        output_lines = []
        output_lines.append(f"Command: {data['command']}")
        output_lines.append(f"Status: {data['status']}")
        # output_lines.append(f"Project Path: {data['project_path']}")

        if data["status"] == "error":
            output_lines.append("\nErrors:")
            output_lines.append(f"{data.get('error', 'Unknown error')}")

        if "packages_installed" in data:
            output_lines.append("\nPackages Installed:")
            for package in data["packages_installed"]:
                output_lines.append(f"- {package}")

        if "run_output" in data and data["run_output"]:
            output_lines.append("\nOutput:")
            output_lines.append(data["run_output"])

        return "\n".join(output_lines)

    def _get_venv_executable(self, venv_dir_path: Path, executable_name: str) -> Path:
        """Gets the path to an executable in the venv's scripts/bin directory."""
        if os.name == 'nt':  # Windows
            return venv_dir_path / "Scripts" / f"{executable_name}.exe"
        else:  # POSIX (Linux, macOS)
            return venv_dir_path / "bin" / executable_name

    async def setup_project(self, project_path: Path, packages: List[str]) -> ToolResult:
        """Set up a local Python project."""
        host_repo_dir = get_constant("REPO_DIR")

        host_repo_path = Path(host_repo_dir)
        installed_packages = []
        project_path_obj = Path(project_path)

        relative_subpath = project_path_obj
        project_path = (host_repo_path / relative_subpath).resolve()

        project_path.mkdir(parents=True, exist_ok=True)
        venv_dir = project_path / ".venv"

        try:
            if self.display is not None:
                self.display.add_message("user", f"Creating virtual environment in {project_path}")

            if not venv_dir.exists():
                try:
                    subprocess.run(["uv", "venv", str(venv_dir)], check=True, cwd=project_path, capture_output=True)
                except subprocess.CalledProcessError as e:
                    error_result = {
                        "command": "setup_project",
                        "status": "error",
                        "error": f"Failed to create virtual environment: {e.stderr if e.stderr else str(e)}",
                        "project_path": str(project_path),
                    }
                    return ToolResult(
                        error=f"Failed to create virtual environment: {e.stderr if e.stderr else str(e)}",
                        message=self.format_output(error_result),
                        command="setup_project",
                        tool_name=self.name
                    )

            # Run 'uv init' directly in the project directory; no venv activation is performed here
            try:
                subprocess.run(args=["uv", "init"], cwd=project_path, check=True)
                logger.info(f"Project initialized at {project_path}")
            except subprocess.CalledProcessError as e:
                error_result = {
                    "command": "setup_project", 
                    "status": "error",
                    "error": f"Failed to initialize project: {e.stderr if e.stderr else str(e)}",
                    "project_path": str(project_path),
                }
                return ToolResult(
                    error=f"Failed to initialize project: {e.stderr if e.stderr else str(e)}",
                    message=self.format_output(error_result),
                    command="setup_project",
                    tool_name=self.name
                )

            if packages:
                if isinstance(packages, str):
                    # Attempt to handle simple string list if accidentally passed
                    packages = [p.strip() for p in packages.split(',') if p.strip()]

                for package_group in packages: # packages could be a list of strings, where each string is a list of packages
                    actual_packages_to_install = []
                    if isinstance(package_group, str):
                        # Further split if a single string in the list contains multiple packages
                        # e.g. "packageA packageB" or "['packageA', 'packageB']"
                        if '[' in package_group and ']' in package_group: # Looks like a stringified list
                            try:
                                # This is a basic attempt, for more complex strings, json.loads might be better
                                # but pip install can often handle multiple package names in one command.
                                # For simplicity, we'll assume packages are space-separated if not proper list items.
                                cleaned_str = package_group.strip("[]'\" ")
                                actual_packages_to_install.extend([p.strip(" '\"") for p in cleaned_str.split(',') if p.strip(" '\"")])
                            except Exception:
                                # If parsing fails, treat the whole string as one package (or rely on pip to split)
                                actual_packages_to_install.append(package_group.strip())
                        else: # Space separated or single package
                            actual_packages_to_install.extend(package_group.split())
                    elif isinstance(package_group, list): # if it's already a list of packages
                        actual_packages_to_install.extend(package_group)
                    else: # Assuming it's a single package name
                        actual_packages_to_install.append(str(package_group))

                    for clean_pkg in actual_packages_to_install:
                        if not clean_pkg:
                            continue
                        logger.info(f"Attempting to install package: {clean_pkg} using uv")
                        try:
                            result = subprocess.run(["uv", "add", clean_pkg], capture_output=True, text=True, cwd=project_path, check=True)
                            installed_packages.append(clean_pkg)
                            logger.info(f"Successfully installed {clean_pkg}")
                        except subprocess.CalledProcessError as e:
                            logger.error(f"Failed to install {clean_pkg}: {e.stderr if e.stderr else str(e)}")
                            error_result = {
                                "command": "setup_project",
                                "status": "error",
                                "error": f"uv add {clean_pkg} failed: {e.stderr if e.stderr else str(e)}",
                                "project_path": str(project_path),
                            }
                            return ToolResult(
                                error=f"uv add {clean_pkg} failed: {e.stderr if e.stderr else str(e)}",
                                message=self.format_output(error_result),
                                command="setup_project",
                                tool_name=self.name
                            )
                if self.display is not None:
                    self.display.add_message("user", f"Project setup complete in {project_path}")
            rr(result)
            success_result = {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "packages_installed": installed_packages,
            }
            return ToolResult(
                output="Project setup completed successfully",
                message=self.format_output(success_result),
                command="setup_project",
                tool_name=self.name
            )
        except Exception as e:
            error_message = str(e)
            error_result = {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up project: {error_message}",
                "project_path": str(project_path),
            }
            return ToolResult(
                error=f"Failed to set up project: {error_message}",
                message=self.format_output(error_result),
                command="setup_project",
                tool_name=self.name
            )

    async def add_dependencies(self, project_path: Path, packages: List[str]) -> ToolResult:
        installed_packages = []

        try:
            for i, package_item in enumerate(packages, 1):
                pkgs_to_install_this_round = []
                if isinstance(package_item, str):
                    if '[' in package_item and ']' in package_item:
                        try:
                            cleaned_str = package_item.strip("[]'\" ")
                            pkgs_to_install_this_round.extend([p.strip(" '\"") for p in cleaned_str.split(',') if p.strip(" '\"")])
                        except Exception:
                            pkgs_to_install_this_round.append(package_item.strip())
                    else:
                        pkgs_to_install_this_round.extend(package_item.split())
                elif isinstance(package_item, list):
                    pkgs_to_install_this_round.extend(package_item)
                else:
                    pkgs_to_install_this_round.append(str(package_item))

                for package in pkgs_to_install_this_round:
                    if not package:
                        continue
                    if self.display is not None:
                        self.display.add_message(
                            "user", f"Installing package {i}/{len(packages)}: {package} using uv"
                        )
                    result = subprocess.run([
                        "uv",
                        "add",
                        package,
                    ], capture_output=True, text=True, cwd=project_path)
                    if result.returncode == 0:
                        installed_packages.append(package)
                        if self.display is not None:
                            self.display.add_message("user", f"Successfully installed {package}")
                    else:
                        error_result = {
                            "command": "add_additional_depends",
                            "status": "error",
                            "error": result.stderr,
                            "project_path": str(project_path),
                            "packages_installed": installed_packages,
                            "packages_failed": packages[i - 1 :],
                            "failed_at": package,
                        }
                        return ToolResult(
                            error=result.stderr,
                            message=self.format_output(error_result),
                            command="add_additional_depends",
                            tool_name=self.name
                        )

            success_result = {
                "command": "add_additional_depends",
                "status": "success",
                "project_path": str(project_path),
                "packages_installed": installed_packages,
            }
            return ToolResult(
                output="Dependencies added successfully",
                message=self.format_output(success_result),
                command="add_additional_depends", 
                tool_name=self.name
            )

        except Exception as e:
            error_result = {
                "command": "add_additional_depends",
                "status": "error",
                "error": str(e),
                "project_path": str(project_path),
                "packages_attempted": packages,
                "packages_installed": installed_packages,
            }
            return ToolResult(
                error=str(e),
                message=self.format_output(error_result),
                command="add_additional_depends",
                tool_name=self.name
            )

    async def run_app(self, project_path: Path, filename: str) -> ToolResult:
        """Runs a Python application locally."""
        try:
            print(f"Running app at {project_path} with filename {filename}")
            file_path = project_path / filename
            # get the REPO_DIR constant
            host_repo_dir = get_constant("REPO_DIR")
            cmd = ["uv", "run", str(file_path)]
            print(f"Running command: {' '.join(cmd)} in {host_repo_dir}")
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True, cwd=host_repo_dir
                )
                rr(result)
                run_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
                logger.info(f"Run app output for {file_path}:\n{run_output}")
                rr(f"Run app output for {file_path}:\n{run_output}")  # Using rich print for better formatting
                return ToolResult(
                    output=result.stdout,
                    error=result.stderr if result.stderr else None,
                    message=f"App executed successfully at {project_path}",
                    command="run_app",
                    tool_name=self.name
                )
            except subprocess.CalledProcessError as e:
                run_output = f"stdout: {e.stdout}\nstderr: {e.stderr}"
                logger.error(f"Run app failed for {file_path}:\n{run_output}")
                return ToolResult(
                    output=e.stdout,
                    error=e.stderr,
                    message=f"App execution failed at {project_path}",
                    command="run_app",
                    tool_name=self.name
                )
        except Exception as e:
            return ToolResult(
                error=str(e),
                message=f"Unexpected error running app at {project_path}",
                command="run_app",
                tool_name=self.name
            )

    async def run_project(
        self, project_path: Path, entry_filename: str = "app.py"
        ) -> ToolResult:
        """Runs a Python project locally."""
        try:
            print(f"Running project at {project_path} with entry file {entry_filename}")
            entry_file = project_path / entry_filename
            if not entry_file.exists():
                return ToolResult(
                    error=f"Entry file {entry_filename} does not exist",
                    message=f"Entry file {entry_filename} not found in {project_path}",
                    command="run_project",
                    tool_name=self.name
                )
            # get the REPO_DIR constant
            host_repo_dir = get_constant("REPO_DIR")
            cmd = ["uv", "run", str(entry_file)]
            print(f"Running command: {' '.join(cmd)} in {host_repo_dir}")
            # Use subprocess to run the command in the host_repo_dir
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=host_repo_dir)
                rr(result)
                return ToolResult(
                    output=result.stdout,
                    error=result.stderr if result.stderr else None,
                    message=f"Project executed successfully at {project_path}",
                    command="run_project",
                    tool_name=self.name
                )
            except subprocess.CalledProcessError as e:
                return ToolResult(
                    output=e.stdout,
                    error=e.stderr,
                    message=f"Project execution failed at {project_path}",
                    command="run_project",
                    tool_name=self.name
                )

        except Exception as e:
            if self.display is not None:
                self.display.add_message(
                    "assistant", f"ProjectSetupTool error: {str(e)}"
                )
            return ToolResult(
                error=f"Failed to execute run_project: {str(e)}",
                message=f"Unexpected error running project at {project_path}",
                command="run_project",
                tool_name=self.name
            )

    async def __call__(
        self,
        *,
        command: ProjectCommand,
        project_path: str,
        environment: str = "python",
        packages: List[str] | None = None,
        entry_filename: str = "app.py",
        **kwargs,
    ) -> ToolResult:  # type: ignore[override]
        """Executes the specified command for project management."""
        try:
            # Handle both string and Enum types for command
            if hasattr(command, "value"):
                command_value = command.value
            else:
                # If command is a string, try to convert it to an Enums
                try:
                    command = ProjectCommand(command)
                    command_value = command.value
                except ValueError:
                    # If conversion fails, return an error
                    return ToolResult(
                        error=f"Unknown command: {command}", tool_name=self.name
                    )

            if self.display is not None:
                self.display.add_message(
                    "user", f"ProjectSetupTool Command: {command_value}\nProject Path: {project_path}, Environment: {environment}, Packages: {packages}, Entry Filename: {entry_filename}"
                )

            # Set default packages if not provided
            if packages is None:
                packages = []

            # Convert string path to Path object
            project_path_obj = Path(project_path)

            if command == ProjectCommand.SETUP_PROJECT:
                result = await self.setup_project(project_path_obj, packages)

            elif command == ProjectCommand.ADD_DEPENDENCIES:
                result = await self.add_dependencies(project_path_obj, packages)

            elif command == ProjectCommand.RUN_APP:
                result = await self.run_app(project_path_obj, entry_filename)

            elif command == ProjectCommand.RUN_PROJECT:
                result = await self.run_project(project_path_obj, entry_filename)

            else:
                return ToolResult(error=f"Unknown command: {command_value}", tool_name=self.name)

            return result

        except Exception as e:
            logger.exception(f"Exception in ProjectSetupTool __call__ for command {command if 'command' in locals() else 'unknown'}")
            err_msg = f"Failed to execute {command.value if hasattr(command, 'value') else command}: {str(e)}"
            if self.display is not None:
                self.display.add_message(
                    "assistant", f"ProjectSetupTool error: {err_msg}"
                )
            return ToolResult(error=err_msg, tool_name=self.name)
