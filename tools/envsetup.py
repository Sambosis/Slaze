from enum import Enum
from typing import Literal, List
from pathlib import Path
import os # Added os import

from config import PROJECT_DIR
from .base import ToolResult, BaseAnthropicTool
import subprocess
from loguru import logger as ll
from rich import print as rr

# Configure logging to a file
ll.add(
    "my_log_file.log",
    rotation="500 KB",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} - {message}",
)


class ProjectCommand(str, Enum):
    """Commands supported by the ProjectSetupTool."""

    SETUP_PROJECT = "setup_project"
    ADD_DEPENDENCIES = "add_additional_depends"
    RUN_APP = "run_app"
    RUN_PROJECT = "run_project"


class ProjectSetupTool(BaseAnthropicTool):
    """
    A tool that sets up project environments and manages script execution locally.
    Provides utilities for both Python and Node.js projects without Docker.
    """

    name: Literal["project_setup"] = "project_setup"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "A tool for project management: setup projects, add dependencies, and run applications. "
        ": Sets up a Python or Node.js project with virtual environment. Optionally, you can provide a list of dependencies to be installed. "
        "add_dependencies: Adds additional dependencies to an existing project. "
        "run_app: This is the only command that needs to be used any time you want to run a python file."
        "run_project: Runs the entire project, including setup and dependencies. "
        "Supports Python and Node.js environments locally."
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
                            "enum": ["python", "node"],
                            "description": "Environment type (python or node)",
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
        project_path.mkdir(parents=True, exist_ok=True)
        venv_dir = project_path / ".venv"

        try:
            if self.display is not None:
                self.display.add_message("user", f"Creating virtual environment in {project_path}")

            python_executable = "python3" # Default system python
            if not venv_dir.exists():
                # For creating venv, we might use system python or a specific one if defined
                subprocess.run([python_executable, "-m", "venv", str(venv_dir)], check=True)

            pip_executable = self._get_venv_executable(venv_dir, "pip")
            installed_packages = []

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
                        if not clean_pkg: continue # Skip empty strings
                        rr(f"Attempting to install package: {clean_pkg} using {pip_executable}")
                        result = subprocess.run(
                            [str(pip_executable), "install", clean_pkg],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            installed_packages.append(clean_pkg)
                            rr(f"Successfully installed {clean_pkg}")
                        else:
                            rr(f"Failed to install {clean_pkg}: {result.stderr}")
                            raise RuntimeError(f"pip install {clean_pkg} failed: {result.stderr}")


            if self.display is not None:
                self.display.add_message("user", f"Project setup complete in {project_path}")

            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "packages_installed": installed_packages,
            }
        except Exception as e:
            error_message = str(e)
            if self.display is not None:
                self.display.add_message("user", f"Error setting up project: {error_message}")
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up project: {error_message}",
                "project_path": str(project_path),
            }

    async def add_dependencies(self, project_path: Path, packages: List[str]) -> dict:
        """Adds additional Python dependencies to an existing project."""
        venv_dir = project_path / ".venv"
        pip_executable = self._get_venv_executable(venv_dir, "pip")
        installed_packages = []

        try:
            for i, package_item in enumerate(packages, 1):
                # Handle if package_item is a string that needs splitting (e.g. "pkg1 pkg2" or "['pkg1','pkg2']")
                # or if it's already a clean package name.
                pkgs_to_install_this_round = []
                if isinstance(package_item, str):
                    if '[' in package_item and ']' in package_item: # Looks like a stringified list
                        try:
                            cleaned_str = package_item.strip("[]'\" ")
                            pkgs_to_install_this_round.extend([p.strip(" '\"") for p in cleaned_str.split(',') if p.strip(" '\"")])
                        except Exception:
                             pkgs_to_install_this_round.append(package_item.strip()) # Treat as one
                    else: # Space separated or single package
                        pkgs_to_install_this_round.extend(package_item.split())
                elif isinstance(package_item, list):
                     pkgs_to_install_this_round.extend(package_item) # Should not happen based on schema but good practice
                else: # Assuming single package name
                     pkgs_to_install_this_round.append(str(package_item))

                for package in pkgs_to_install_this_round:
                    if not package: continue # Skip empty strings
                    if self.display is not None:
                        self.display.add_message(
                            "user", f"Installing package {i}/{len(packages)}: {package} using {pip_executable}"
                        )

                    result = subprocess.run(
                        [str(pip_executable), "install", package],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed_packages.append(package)
                        if self.display is not None:
                            self.display.add_message(
                                "user", f"Successfully installed {package}"
                            )
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
            venv_dir = project_path / ".venv"
            python_executable = self._get_venv_executable(venv_dir, "python3") if venv_dir.exists() else "python3"

            if venv_dir.exists():
                cmd = [str(python_executable), str(file_path)]
            else: # Fallback to system python if no venv
                cmd = ["python3", str(file_path)]
                rr(f"Warning: venv not found at {venv_dir}, attempting to run {filename} with system python3.")

            result = subprocess.run(cmd, capture_output=True, text=True)

            run_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
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

    # === Node.js Environment Methods ===
    async def setup_project_node(self, project_path: Path, packages: List[str]) -> dict:
        """Sets up a local Node.js project."""
        project_path.mkdir(parents=True, exist_ok=True)

        try:
            if self.display is not None:
                self.display.add_message("user", f"Initializing Node.js project in {project_path}")

            subprocess.run(["npm", "init", "-y"], cwd=project_path, check=True)

            installed_packages = []
            base_packages = ["jest", "eslint"]

            for package in base_packages:
                try:
                    subprocess.run(["npm", "install", "--save-dev", package], cwd=project_path, check=True)
                    installed_packages.append(package)
                except Exception:
                    if self.display is not None:
                        self.display.add_message("user", f"Warning: Failed to install {package}, continuing anyway")

            if packages:
                for package in packages:
                    try:
                        subprocess.run(["npm", "install", package], cwd=project_path, check=True)
                        installed_packages.append(package)
                    except Exception as e:
                        return {
                            "command": "setup_project",
                            "status": "error",
                            "error": f"Failed to install {package}: {str(e)}",
                            "project_path": str(project_path),
                            "packages_installed": installed_packages,
                        }

            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "packages_installed": installed_packages,
            }

        except Exception as e:
            return {
                "command": "setup_project",
                "status": "error",
                "error": str(e),
                "project_path": str(project_path),
            }

    async def add_dependencies_node(
        self, project_path: Path, packages: List[str]
        ) -> dict:
        """Adds additional Node.js dependencies to an existing project."""
        installed_packages = []

        try:
            package_json = project_path / "package.json"
            if not package_json.exists():
                return {
                    "command": "add_additional_depends",
                    "status": "error",
                    "error": "package.json not found. Please run setup_project first.",
                    "project_path": str(project_path),
                }

            for i, package in enumerate(packages, 1):
                if self.display is not None:
                    self.display.add_message(
                        "user", f"Installing package {i}/{len(packages)}: {package}"
                    )

                is_dev = any(
                    kw in package.lower() for kw in [
                        "test",
                        "jest",
                        "mocha",
                        "chai",
                        "eslint",
                        "prettier",
                        "babel",
                        "webpack",
                        "typescript",
                    ]
                )

                cmd = ["npm", "install"]
                if is_dev:
                    cmd.append("--save-dev")
                cmd.append(package)

                result = subprocess.run(cmd, cwd=project_path, capture_output=True, text=True)
                if result.returncode == 0:
                    installed_packages.append(package)
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

    async def run_app_node(self, project_path: Path, filename: str) -> dict:
        """Runs a Node.js application locally."""
        try:
            package_json = project_path / "package.json"
            if not package_json.exists():
                return {
                    "command": "run_app",
                    "status": "error",
                    "error": "package.json not found. Please run setup_project first.",
                    "project_path": str(project_path),
                }

            result = subprocess.run(
                ["node", filename],
                cwd=project_path,
                capture_output=True,
                text=True,
            )

            return {
                "command": "run_app",
                "status": "success" if result.returncode == 0 else "error",
                "project_path": str(project_path),
                "run_output": result.stdout,
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

            venv_dir = project_path / ".venv"
            python_executable = self._get_venv_executable(venv_dir, "python3") if venv_dir.exists() else "python3"

            if venv_dir.exists():
                cmd = [str(python_executable), str(entry_file)]
            else: # Fallback to system python if no venv
                cmd = ["python3", str(entry_file)]
                rr(f"Warning: venv not found at {venv_dir}, attempting to run {entry_filename} with system python3.")

            result = subprocess.run(cmd, capture_output=True, text=True)
            run_output = f"stdout: {result.stdout}\nstderr: {result.stderr}"
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

    async def __call__(self,*, command: ProjectCommand,
        project_path: str,
        environment: str = "python",
        packages: List[str] = None,
        entry_filename: str = "app.py",
        **kwargs,
        ) -> ToolResult:
        """Executes the specified command for project management."""
        try:
            # Handle both string and Enum types for command
            if hasattr(command, "value"):
                command_value = command.value
            else:
                # If command is a string, try to convert it to an Enum
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
                    "user", f"ProjectSetupTool Command: {command_value}"
                )

            # Set default packages if not provided
            if packages is None:
                packages = []

            # Convert string path to Path object
            project_path = Path(project_path)

            if command == ProjectCommand.SETUP_PROJECT:
                if environment == "python":
                    result = await self.setup_project(project_path, packages)
                else:
                    result = await self.setup_project_node(project_path, packages)

            elif command == ProjectCommand.ADD_DEPENDENCIES:
                if environment == "python":
                    result = await self.add_dependencies(project_path, packages)
                else:
                    result = await self.add_dependencies_node(project_path, packages)

            elif command == ProjectCommand.RUN_APP:
                if environment == "python":
                    result = await self.run_project(project_path, entry_filename)
                else:
                    result = await self.run_app_node(project_path, entry_filename)

            elif command == ProjectCommand.RUN_PROJECT:
                result = await self.run_project(project_path, entry_filename)

            else:
                return ToolResult(
                    error=f"Unknown command: {command_value}", tool_name=self.name # Use command_value
                )

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
            ll.exception(f"Exception in ProjectSetupTool __call__ for command {command if 'command' in locals() else 'unknown'}")
            err_msg = f"Failed to execute {command.value if hasattr(command, 'value') else command}: {str(e)}"
            if self.display is not None:
                self.display.add_message(
                    "assistant", f"ProjectSetupTool error: {err_msg}"
                )
            return ToolResult(error=err_msg, tool_name=self.name)
