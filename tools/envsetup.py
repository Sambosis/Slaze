from enum import Enum
from typing import Literal, List
from pathlib import Path

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


    async def setup_project(self, project_path: Path, packages: List[str]) -> dict:
        """Set up a local Python project."""
        project_path.mkdir(parents=True, exist_ok=True)
        venv_dir = project_path / ".venv"

        try:
            if self.display is not None:
                self.display.add_message("user", f"Creating virtual environment in {project_path}")

            if not venv_dir.exists():
                subprocess.run(["python3", "-m", "venv", str(venv_dir)], check=True)

            installed_packages = []

            if packages:
                if isinstance(packages, str):
                    packages = packages.split()
                for package in packages:
                    clean_pkg = package.strip("[],'\" ")
                    result = subprocess.run(
                        [str(venv_dir / "bin" / "pip"), "install", clean_pkg],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        installed_packages.append(clean_pkg)
                    else:
                        raise RuntimeError(result.stderr)

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
        installed_packages = []

        try:
            for i, package in enumerate(packages, 1):
                if self.display is not None:
                    self.display.add_message(
                        "user", f"Installing package {i}/{len(packages)}: {package}"
                    )

                result = subprocess.run(
                    [str(venv_dir / "bin" / "pip"), "install", package],
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
            if venv_dir.exists():
                cmd = [str(venv_dir / "bin" / "python3"), str(file_path)]
            else:
                cmd = ["python3", str(file_path)]

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
            if venv_dir.exists():
                cmd = [str(venv_dir / "bin" / "python3"), str(entry_file)]
            else:
                cmd = ["python3", str(entry_file)]

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
                    error=f"Unknown command: {command}", tool_name=self.name
                )

            return ToolResult(output=self.format_output(result))

        except Exception as e:
            if self.display is not None:
                self.display.add_message(
                    "assistant", f"ProjectSetupTool error: {str(e)}"
                )
            return ToolResult(error=f"Failed to execute {command}: {str(e)}")
