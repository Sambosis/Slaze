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

    async def setup_project(self, project_path: Path, packages: List[str]) -> dict:
        """Set up a local Python project."""
        host_repo_dir = get_constant("REPO_DIR")
        if not host_repo_dir:
            prompt_name = get_constant("PROMPT_NAME")
            if prompt_name:
                host_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo" / prompt_name
            else:
                host_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo"
        host_repo_path = Path(host_repo_dir)
        installed_packages = []
        project_path_obj = Path(project_path)
        if project_path_obj.is_absolute():
            if "repo" in project_path_obj.parts:
                repo_index = project_path_obj.parts.index("repo")
                relative_subpath = Path(*project_path_obj.parts[repo_index + 1 :])
            else:
                relative_subpath = Path(project_path_obj.name)
        else:
            relative_subpath = project_path_obj
        project_path = (host_repo_path / relative_subpath).resolve()

        project_path.mkdir(parents=True, exist_ok=True)
        venv_dir = project_path / ".venv"

        try:
            if self.display is not None:
                self.display.add_message("user", f"Creating virtual environment in {project_path}")

            if not venv_dir.exists():
                subprocess.run(["uv", "venv", str(venv_dir)], check=True, cwd=project_path, capture_output=True)

            # Run 'uv init' directly in the project directory; no venv activation is performed here
            subprocess.run(args=["uv", "init", "--no-workspace"], cwd=project_path, check=True)
            logger.info(f"Project initialized at {project_path}")

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
                        result = subprocess.run(["uv", "add", clean_pkg], capture_output=True, text=True, cwd=project_path, check=True)
                        if result.returncode == 0:
                            installed_packages.append(clean_pkg)
                            logger.info(f"Successfully installed {clean_pkg}")
                        else:
                            logger.error(f"Failed to install {clean_pkg}: {result.stderr}")
                            raise RuntimeError(f"uv add {clean_pkg} failed: {result.stderr}")
                self.display.add_message("user", f"Project setup complete in {project_path}")
            rr(result)
            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "packages_installed": installed_packages,
            }
        except Exception as e:
            error_message = str(e)
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up project: {error_message}",
                "project_path": str(project_path),
            }

    async def add_dependencies(self, project_path: Path, packages: List[str]) -> dict:
        installed_packages = []
        venv_dir = project_path / ".venv"
        python_executable = self._get_venv_executable(venv_dir, "python")

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
                        return {
                            "command": "add_additional_depends",
                            "status": "error",
                            "error": result.stderr,
                            "project_path": str(project_path),
                            "packages_installed": installed_packages,
                            "packages_failed": packages[i - 1 :],
                            "failed_at": package,
                        }

            return {
                "command": "add_additional_depends",
                "status": "success",
                "project_path": str(project_path),
                "packages_installed": installed_packages,
            }

        except Exception as e:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": str(e),
                "project_path": str(project_path),
                "packages_attempted": packages,
                "packages_installed": installed_packages,
            }

    async def run_app(self, project_path: Path, filename: str) -> dict:
        """Runs a Python application locally."""
        try:
            file_path = project_path / filename
            # get the REPO_DIR constant
            host_repo_dir = get_constant("REPO_DIR")
            cmd = ["uv", "run", str(file_path)]
            print(f"Running command: {' '.join(cmd)} in {host_repo_dir}")
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=host_repo_dir
            )
            rr(result)
            run_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
            logger.info(f"Run app output for {file_path}:\n{run_output}")
            rr(f"Run app output for {file_path}:\n{run_output}")  # Using rich print for better formatting
            return {
                "command": "run_app",
                "status": "success" if result.returncode == 0 else "error",
                "project_path": str(project_path),
                "run_output": run_output,
                "errors": result.stderr,
            }
        except Exception as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "errors": str(e),
            }

    async def run_project(
        self, project_path: Path, entry_filename: str = "app.py"
        ) -> dict:
        """Runs a Python project locally."""
        try:
            entry_file = project_path / entry_filename
            if not entry_file.exists():
                return {
                    "command": "run_project",
                    "status": "error",
                    "error": f"Entry file {entry_filename} does not exist",
                    "project_path": str(project_path),
                }
            # get the REPO_DIR constant
            host_repo_dir = get_constant("REPO_DIR")
            cmd = ["uv", "run", str(entry_file)]
            print(f"Running command: {' '.join(cmd)} in {host_repo_dir}")
            # Use subprocess to run the command in the host_repo_dir
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=host_repo_dir)
            run_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
            rr(result)
            return {
                "command": "run_project",
                "status": "success" if result.returncode == 0 else "error",
                "project_path": str(project_path),
                "run_output": run_output,
                "errors": result.stderr,
            }
        except Exception as e:
            if self.display is not None:
                self.display.add_message(
                    "assistant", f"ProjectSetupTool error: {str(e)}"
                )
            return ToolResult(error=f"Failed to execute run_project: {str(e)}")

    async def __call__(
        self,
        *,
        command: ProjectCommand,
        project_path: str,
        environment: str = "python",
        packages: List[str] = None,
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
            project_path = Path(project_path)

            if environment != "python":
                return ToolResult(error="Only python environment is supported", tool_name=self.name)

            if command == ProjectCommand.SETUP_PROJECT:
                result = await self.setup_project(project_path, packages)

            elif command == ProjectCommand.ADD_DEPENDENCIES:
                result = await self.add_dependencies(project_path, packages)

            elif command == ProjectCommand.RUN_APP:
                result = await self.run_project(project_path, entry_filename)

            elif command == ProjectCommand.RUN_PROJECT:
                result = await self.run_project(project_path, entry_filename)

            else:
                return ToolResult(error=f"Unknown command: {command_value}", tool_name=self.name)

            # Fix for 'ToolResult' object is not subscriptable error
            if isinstance(result, ToolResult):
                if result.error:
                    return result # Return error ToolResult directly
                # If it's a success ToolResult, extract output if possible, or create a default dict
                # This path is less likely given current internal methods return dicts for success.
                if result.output: # Assuming output is a string that format_output can handle or a dict
                    if isinstance(result.output, str): # If format_output expects dict, this needs adjustment
                        # This case means a ToolResult has string output. format_output might need to handle this.
                        # For now, let's assume if result.output is a string, it's pre-formatted.
                        return result
                    # If result.output is dict, it's fine for format_output
                    return ToolResult(output=self.format_output(result.output)) # format_output expects dict
                else: # ToolResult without error and without output, unlikely for this tool
                    return ToolResult(output=self.format_output({"status": "success", "message": "Operation completed.", "command": command_value}))

            # If result is a dictionary (expected case for successful operations from internal methods)
            return ToolResult(output=self.format_output(result))

        except Exception as e:
            logger.exception(f"Exception in ProjectSetupTool __call__ for command {command if 'command' in locals() else 'unknown'}")
            err_msg = f"Failed to execute {command.value if hasattr(command, 'value') else command}: {str(e)}"
            if self.display is not None:
                self.display.add_message(
                    "assistant", f"ProjectSetupTool error: {err_msg}"
                )
            return ToolResult(error=err_msg, tool_name=self.name)
