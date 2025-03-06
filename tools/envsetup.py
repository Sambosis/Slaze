from enum import Enum
from typing import Literal, List
from pathlib import Path
from .base import ToolResult, BaseAnthropicTool
import os
from icecream import ic
from utils.docker_service import DockerService, DockerResult, DockerServiceError
from config import get_constant


class ProjectCommand(str, Enum):
    """Commands supported by the ProjectSetupTool."""

    SETUP_PROJECT = "setup_project"
    ADD_DEPENDENCIES = "add_additional_depends"
    RUN_APP = "run_app"


class ProjectSetupTool(BaseAnthropicTool):
    """
    A tool that sets up project environments and manages script execution using Docker.
    Provides consistent Linux environment for both Python and Node.js projects.
    """

    name: Literal["project_setup"] = "project_setup"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "A tool for project management: setup projects, add dependencies, and run applications. "
        "Supports Python and Node.js environments within Docker."
    )

    def __init__(self, display=None):
        """Initialize the ProjectSetupTool with display for UI feedback."""
        super().__init__(display)
        # Initialize Docker service
        self.docker = DockerService()
        self._docker_available = self.docker.is_available()

    def to_params(self) -> dict:
        """Define the input schema for the ProjectSetupTool."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [cmd.value for cmd in ProjectCommand],
                        "description": "Command to execute: The options are setup_project( which will create the project direct and set up the project virtual environment), add_additional_depends which will install a list of dependicies passed to it, or run_app which will run the application. These are the prefered methods to do these operations.",
                    },
                    "environment": {
                        "type": "string",
                        "enum": ["python", "node"],
                        "description": "Type of project environment to setup",
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of packages to install. This should be a list of strings contained inside of []",
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory",
                    },
                    "entry_filename": {
                        "type": "string",
                        "description": "Name of the file to run (e.g., app.py, app.js)",
                    },
                },
                "required": ["command", "project_path", "environment"],
            },
        }

    def get_docker_path(self, project_path: Path) -> str:
        """
        Get the correct Docker path for a project directory.

        Args:
            project_path: Path to the project directory on the host

        Returns:
            str: Docker path for the project
        """
        # Use Docker project directory from config if available
        docker_project_dir = get_constant("DOCKER_PROJECT_DIR")

        if docker_project_dir:
            return str(docker_project_dir)

        # Otherwise build the path with correct format
        project_name = project_path.name
        return f"/home/myuser/apps/{project_name}"

    def format_output(self, data: dict) -> str:
        """Format the output data as a readable string."""
        output_lines = []
        output_lines.append(f"Command: {data['command']}")
        output_lines.append(f"Status: {data['status']}")
        output_lines.append(f"Project Path: {data['project_path']}")

        if "docker_path" in data:
            output_lines.append(f"Docker Path: {data['docker_path']}")

        if "packages_installed" in data:
            output_lines.append("Packages Installed:")
            for package in data["packages_installed"]:
                output_lines.append(f"  - {package}")

        if "run_output" in data and data["run_output"]:
            run_output = data["run_output"]
            if len(run_output) > 200000:
                run_output = (
                    run_output[:100000] + " ... [TRUNCATED] ... " + run_output[-100000:]
                )
            output_lines.append("\nApplication Output:")
            output_lines.append(run_output)

        if "errors" in data and data["errors"]:
            errors = data["errors"]
            if len(errors) > 200000:
                errors = errors[:100000] + " ... [TRUNCATED] ... " + errors[-100000:]
            output_lines.append("\nErrors:")
            output_lines.append(errors)

        return "\n".join(output_lines)

    def validate_docker_path(self, docker_path: str) -> str:
        """
        Validate and fix Docker path if it's malformed.
        
        Args:
            docker_path: Docker path to validate
            
        Returns:
            str: Corrected Docker path
        """
        # Detect common issues in Docker paths
        if not docker_path:
            return "/home/myuser/apps"

        docker_path = str(docker_path)

        # Fix double path issues (homemyuserappsminecraft)
        if "homemyuser" in docker_path:
            # Extract project name (last part of the path)
            parts = docker_path.split('/')
            if len(parts) > 0:
                project_name = parts[-1]
                if "homemyuserapps" in project_name:
                    # Extract actual project name after homemyuserapps
                    actual_name = project_name.split('homemyuserapps')[-1]
                    # Reconstruct proper path
                    return f"/home/myuser/apps/{actual_name}"

        # Fix Windows-style backslashes
        docker_path = docker_path.replace("\\", "/")

        # Ensure path starts with /
        if not docker_path.startswith("/"):
            docker_path = "/" + docker_path

        return docker_path

    # === Python Environment Methods ===
    async def setup_project(self, project_path: Path, packages: List[str]) -> dict:
        """Sets up a Python project inside the Docker container."""
        if not self._docker_available:
            return {
                "command": "setup_project",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path),
            }

        # Make sure local directory exists
        project_path.mkdir(parents=True, exist_ok=True)

        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)

        # Validate and fix the Docker path if needed
        docker_path = self.validate_docker_path(docker_path)

        try:
            # Create the project directory in Docker
            ic(f"Creating directory in Docker: {docker_path}")
            mkdir_result = self.docker.execute_command(f"mkdir -p {docker_path}")

            # Create virtual environment in Docker
            ic("Creating Python virtual environment in Docker...")
            if self.display:  # Check if display exists before using it
                self.display.add_message(
                    "user", "Creating Python virtual environment in Docker..."
                )
            venv_result = self.docker.execute_command(
                f"cd {docker_path} && python3 -m venv .venv"
            )

            # Install packages in Docker
            installed_packages = []
            if packages:
                ic("Installing Python packages in Docker...")
                self.display.add_message(
                    "user", f"Installing {len(packages)} Python packages in Docker..."
                )

                # First upgrade pip
                self.docker.execute_command(
                    f"cd {docker_path} && .venv/bin/pip install --upgrade pip"
                )

                # Install each package
                for package in packages:
                    ic(f"Installing package: {package}")
                    try:
                        pkg_result = self.docker.execute_command(
                            f"cd {docker_path} && .venv/bin/pip install {package}"
                        )
                        installed_packages.append(package)
                        self.display.add_message("user", f"Installed {package}")
                    except Exception as e:
                        self.display.add_message(
                            "user", f"Failed to install {package}: {str(e)}"
                        )

            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": installed_packages,
            }
        except DockerServiceError as e:
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Docker service error: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path if "docker_path" in locals() else "unknown",
            }
        except Exception as e:
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up project in Docker: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path if "docker_path" in locals() else "unknown",
            }

    async def add_dependencies(self, project_path: Path, packages: List[str]) -> dict:
        """Adds Python dependencies to an existing project in the Docker container."""
        if not self._docker_available:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path),
            }

        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)

        try:
            ic(f"Installing {len(packages)} additional Python packages in Docker...")
            if self.display:
                self.display.add_message(
                    "user",
                    f"Installing {len(packages)} additional Python packages in Docker...",
                )

            # Check if virtual environment exists
            venv_check = self.docker.execute_command(
                f"[ -d {docker_path}/.venv ] && echo 'exists' || echo 'not exists'"
            )

            if "not exists" in venv_check.stdout:
                # Create virtual environment if it doesn't exist
                self.display.add_message(
                    "user", "Creating virtual environment first..."
                )
                self.docker.execute_command(
                    f"cd {docker_path} && python3 -m venv .venv"
                )
                self.docker.execute_command(
                    f"cd {docker_path} && .venv/bin/pip install --upgrade pip"
                )

            # Install packages
            installed_packages = []
            for i, package in enumerate(packages, 1):
                ic(f"Installing package {i}/{len(packages)}: {package}")
                if self.display:
                    self.display.add_message(
                        "user", f"Installing package {i}/{len(packages)}: {package}"
                    )

                try:
                    result = self.docker.execute_command(
                        f"cd {docker_path} && .venv/bin/pip install {package}"
                    )
                    installed_packages.append(package)
                    if self.display:
                        self.display.add_message(
                            "user", f"Successfully installed {package}"
                        )
                except Exception as e:
                    return {
                        "command": "add_additional_depends",
                        "status": "error",
                        "error": f"Failed to install {package}: {str(e)}",
                        "project_path": str(project_path),
                        "docker_path": docker_path,
                        "packages_installed": installed_packages,
                        "packages_failed": packages[i - 1 :],
                        "failed_at": package,
                    }

            if self.display:
                self.display.add_message(
                    "user", "All packages installed successfully in Docker"
                )

            return {
                "command": "add_additional_depends",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": installed_packages,
            }

        except Exception as e:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": f"Docker error: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_attempted": packages,
                "packages_installed": (
                    installed_packages if "installed_packages" in locals() else []
                ),
            }

    async def run_app(self, project_path: Path, filename: str) -> dict:
        """Runs a Python application inside the Docker container with X11 forwarding."""
        if not self._docker_available:
            return {
                "command": "run_app",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }

        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)

        # Validate and fix the Docker path if needed
        docker_path = self.validate_docker_path(docker_path)

        # Print paths for debugging
        ic(f"Windows project path: {project_path}")
        ic(f"Docker project path: {docker_path}")

        try:
            # Run Python script with X11 forwarding
            ic(f"Running {filename} in Docker container at {docker_path}")
            self.display.add_message("user", f"Running {filename} in Docker container")

            # Check if virtual environment exists
            venv_check = self.docker.execute_command(
                    f"[ -d {docker_path}/.venv ] && echo 'exists' || echo 'not exists'"
                )

            # Set up command with the appropriate Python interpreter
            if "exists" in venv_check.stdout:
                cmd = f"cd {docker_path} && .venv/bin/python3 {filename}"
            else:
                cmd = f"cd {docker_path} && python3 {filename}"

            # Execute with DISPLAY set for GUI applications
            result = self.docker.execute_command(
                cmd, env_vars={"DISPLAY": "host.docker.internal:0"}
            )

            return {
                "command": "run_app",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "run_output": result.stdout,
                "errors": result.stderr,
            }
        except Exception as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "errors": f"Failed to run app in Docker: {str(e)}",
            }

    # === Node.js Environment Methods ===
    async def setup_project_node(self, project_path: Path, packages: List[str]) -> dict:
        """Sets up a Node.js project inside the Docker container."""
        if not self._docker_available:
            return {
                "command": "setup_project",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path),
            }

        # Make sure local directory exists
        project_path.mkdir(parents=True, exist_ok=True)

        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)

        try:
            # Create the project directory in Docker
            ic(f"Creating directory in Docker: {docker_path}")
            self.docker.execute_command(f"mkdir -p {docker_path}")

            # Initialize Node.js project in Docker
            ic("Initializing Node.js project in Docker...")
            self.display.add_message(
                "user", "Initializing Node.js project in Docker..."
            )
            self.docker.execute_command(f"cd {docker_path} && npm init -y")

            # Install packages in Docker
            installed_packages = []
            if packages:
                ic("Installing Node.js packages in Docker...")
                self.display.add_message(
                    "user", f"Installing {len(packages)} Node.js packages in Docker..."
                )

                for package in packages:
                    ic(f"Installing package: {package}")
                    try:
                        result = self.docker.execute_command(
                            f"cd {docker_path} && npm install {package}"
                        )
                        installed_packages.append(package)
                        self.display.add_message("user", f"Installed {package}")
                    except Exception as e:
                        self.display.add_message(
                            "user", f"Failed to install {package}: {str(e)}"
                        )

            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": installed_packages,
            }
        except Exception as e:
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up Node.js project in Docker: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path if "docker_path" in locals() else "unknown",
            }

    async def add_dependencies_node(
        self, project_path: Path, packages: List[str]
        ) -> dict:
        """Adds Node.js dependencies to an existing project in the Docker container."""
        if not self._docker_available:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path),
            }

        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)

        try:
            ic(f"Installing {len(packages)} additional Node.js packages in Docker...")
            if self.display:
                self.display.add_message(
                    "user",
                    f"Installing {len(packages)} additional Node.js packages in Docker...",
                )

            # Check if package.json exists
            pkg_check = self.docker.execute_command(
                f"[ -f {docker_path}/package.json ] && echo 'exists' || echo 'not exists'"
            )

            if "not exists" in pkg_check.stdout:
                # Initialize Node.js project if package.json doesn't exist
                self.display.add_message(
                    "user", "Initializing Node.js project first..."
                )
                self.docker.execute_command(f"cd {docker_path} && npm init -y")

            # Install packages
            installed_packages = []
            for i, package in enumerate(packages, 1):
                ic(f"Installing Node.js package {i}/{len(packages)}: {package}")
                if self.display:
                    self.display.add_message(
                        "user",
                        f"Installing Node.js package {i}/{len(packages)}: {package}",
                    )

                try:
                    result = self.docker.execute_command(
                        f"cd {docker_path} && npm install {package}"
                    )
                    installed_packages.append(package)
                    if self.display:
                        self.display.add_message(
                            "user", f"Successfully installed {package}"
                        )
                except Exception as e:
                    return {
                        "command": "add_additional_depends",
                        "status": "error",
                        "error": f"Failed to install Node.js package {package}: {str(e)}",
                        "project_path": str(project_path),
                        "docker_path": docker_path,
                        "packages_installed": installed_packages,
                        "packages_failed": packages[i - 1 :],
                        "failed_at": package,
                    }

            if self.display:
                self.display.add_message(
                    "user", "All Node.js packages installed successfully in Docker"
                )

            return {
                "command": "add_additional_depends",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": installed_packages,
            }

        except Exception as e:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": f"Docker error: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_attempted": packages,
                "packages_installed": (
                    installed_packages if "installed_packages" in locals() else []
                ),
            }

    async def run_app_node(self, project_path: Path, filename: str) -> dict:
        """Runs a Node.js application inside the Docker container."""
        if not self._docker_available:
            return {
                "command": "run_app",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path),
            }

        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)

        try:
            # Run Node.js script
            ic(
                f"Running Node.js script {filename} in Docker container at {docker_path}"
            )
            self.display.add_message(
                "user", f"Running Node.js script {filename} in Docker container"
            )

            # Execute with DISPLAY set for GUI applications (in case of Electron apps)
            result = self.docker.execute_command(
                f"cd {docker_path} && node {filename}",
                env_vars={"DISPLAY": "host.docker.internal:0"},
            )

            return {
                "command": "run_app",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "run_output": result.stdout,
                "errors": result.stderr,
            }
        except Exception as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "errors": f"Failed to run Node.js app in Docker: {str(e)}",
            }

    async def __call__(
        self,
        *,
        command: ProjectCommand,
        project_path: str,
        environment: str = "python",
        packages: List[str] = None,
        entry_filename: str = "app.py",
        **kwargs,
    ) -> ToolResult:
        """Executes the specified command for project management."""
        if packages is None:
            packages = []

        try:
            if self.display:
                self.display.add_message(
                    "user",
                    f"ProjectSetupTool executing command: {command} in {environment} environment (Docker)",
                )

            # Convert project_path string to Path object
            project_dir = get_constant("PROJECT_DIR")
            if not project_dir:
                project_path_obj = Path(project_path)
            else:
                project_path_obj = project_dir

            # Dispatch based on environment and command
            command_str = command.value if hasattr(command, 'value') else str(command)
            
            if environment == "python":
                if command == ProjectCommand.SETUP_PROJECT:
                    result_data = await self.setup_project(project_path_obj, packages)
                elif command == ProjectCommand.ADD_DEPENDENCIES:
                    result_data = await self.add_dependencies(
                        project_path_obj, packages
                    )
                elif command == ProjectCommand.RUN_APP:
                    result_data = await self.run_app(project_path_obj, entry_filename)
                else:
                    return ToolResult(error=f"Unknown command: {command}", tool_name=self.name, command=command_str)
            elif environment == "node":
                if command == ProjectCommand.SETUP_PROJECT:
                    result_data = await self.setup_project_node(
                        project_path_obj, packages
                    )
                elif command == ProjectCommand.ADD_DEPENDENCIES:
                    result_data = await self.add_dependencies_node(
                        project_path_obj, packages
                    )
                elif command == ProjectCommand.RUN_APP:
                    result_data = await self.run_app_node(
                        project_path_obj, entry_filename
                    )
                else:
                    return ToolResult(error=f"Unknown command: {command}", tool_name=self.name, command=command_str)
            else:
                return ToolResult(error=f"Unsupported environment: {environment}", tool_name=self.name, command=command_str)

            formatted_output = self.format_output(result_data)
            if self.display:
                self.display.add_message(
                    "user", f"ProjectSetupTool completed: {formatted_output}"
                )
                
            # Create a new ToolResult instead of modifying an existing one
            return ToolResult(output=formatted_output, tool_name=self.name, command=command_str)

        except Exception as e:
            if self.display:
                self.display.add_message("user", f"ProjectSetupTool error: {str(e)}")
            error_msg = f"Failed to execute {command}: {str(e)}"
            command_str = command.value if hasattr(command, 'value') else str(command)
            return ToolResult(error=error_msg, tool_name=self.name, command=command_str)
