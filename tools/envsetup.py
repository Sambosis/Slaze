from enum import Enum
from typing import Literal, List
from pathlib import Path
from .base import ToolResult, BaseAnthropicTool
import os
import subprocess
from icecream import ic
from rich import print as rr
import json
from pydantic import BaseModel
from config import get_constant, set_constant
from utils.context_helpers import *

class ProjectCommand(str, Enum):
    SETUP_PROJECT = "setup_project"
    ADD_DEPENDENCIES = "add_additional_depends"
    RUN_APP = "run_app"

class ProjectSetupTool(BaseAnthropicTool):
    """
    A tool that sets up various project environments and manages script execution.
    Uses Docker for all project types to ensure consistent Linux environment.
    """

    name: Literal["project_setup"] = "project_setup"
    api_type: Literal["custom"] = "custom"
    description: str = ("A tool for project management: setup projects, add dependencies, and run applications. "
                        "Supports Python (in Docker) and Node.js environments.")
    
    # Docker container name
    container_name = "python-dev-container"

    def __init__(self, display=None):
        super().__init__(display)
        # Check Docker availability at initialization
        self._docker_available = self._check_docker()
    
    def _check_docker(self):
        """Check if Docker is available and the container exists"""
        try:
            # Check if Docker is running
            result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False

    def docker_exec(self, cmd: str, capture_output=True) -> subprocess.CompletedProcess:
        """Execute a command in the Docker container"""
        # Properly escape the command for passing to bash -c
        # escaped_cmd = cmd.replace('"', '\\"')
        docker_cmd = f'docker exec -e DISPLAY=host.docker.internal:0 {self.container_name} bash -c "{cmd}"'
        ic(f"Running Docker command: {docker_cmd}")
        try:
            result = subprocess.run(
                docker_cmd,
                shell=True,
                check=True,
                capture_output=capture_output,
                text=True
            )
            return result
        except subprocess.CalledProcessError as e:
            ic(f"Error executing Docker command: {docker_cmd}")
            ic(f"Error details: {e}")
            raise

    def to_docker_path(self, path) -> str:
        """Convert path to Docker container path with proper Linux formatting"""
        if isinstance(path, str):
            path = Path(path)
            
        # Get the project name from the path
        project_name = path.name
        
        # Use defined constant or fallback to standard Docker path
        docker_project_dir = get_constant("DOCKER_PROJECT_DIR") or f"/home/myuser/apps/{project_name}"
        
        # Make sure we're using Linux format
        docker_project_dir = str(docker_project_dir).replace("\\", "/")
        
        return docker_project_dir
    
    def to_params(self) -> dict:
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
                        "description": "Command to execute: setup_project, add_additional_depends, or run_app"
                    },
                    "environment": {
                        "type": "string",
                        "enum": ["python", "node"],
                        "description": "Type of project environment to setup"
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of packages/gems to install"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory"
                    },
                    "entry_filename": {
                        "type": "string",
                        "description": "Name of the file to run (e.g., app.py, app.js, or app.rb)"
                    }
                },
                "required": ["command", "project_path", "environment"]
            }
        }

    def format_output(self, data: dict) -> str:
        """Format the output data as a readable string"""
        output_lines = []
        output_lines.append(f"Command: {data['command']}")
        output_lines.append(f"Status: {data['status']}")
        output_lines.append(f"Project Path: {data['project_path']}")
        if 'docker_path' in data:
            output_lines.append(f"Docker Path: {data['docker_path']}")
        if 'packages_installed' in data:
            output_lines.append("Packages Installed:")
            for package in data['packages_installed']:
                output_lines.append(f"  - {package}")
        if 'run_output' in data and data['run_output']:
            run_output = data['run_output']
            if len(run_output) > 200000:
                run_output = run_output[:100000] + " ... [TRUNCATED] ... " + run_output[-100000:]
            output_lines.append("\nApplication Output:")
            output_lines.append(run_output)
        if 'errors' in data and data['errors']:
            errors = data['errors']
            if len(errors) > 200000:
                errors = errors[:100000] + " ... [TRUNCATED] ... " + errors[-100000:]
            output_lines.append("\nErrors:")
            output_lines.append(errors)
        return "\n".join(output_lines)

    # === Python Environment Methods (Docker-enabled) ===
    async def setup_project(self, project_path: Path, packages: List[str]) -> dict:
        """Sets up a Python project inside the Docker container"""
        if not self._docker_available:
            return {
                "command": "setup_project",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
            
        # Make sure local directory exists (will be mounted to Docker)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Use proper Docker paths for Linux
        docker_path = self.to_docker_path(project_path)
        
        try:
            # Create the project directory in Docker
            ic(f"Creating directory in Docker: {docker_path}")
            self.docker_exec(f"mkdir -p {docker_path}")
            
            # Create virtual environment in Docker
            ic("Creating Python virtual environment in Docker...")
            self.docker_exec(f"cd {docker_path} && python3 -m venv .venv")
            
            # Install packages in Docker
            ic("Installing Python packages in Docker...")
            for package in packages:
                ic(f"Installing package: {package}")
                self.docker_exec(f"cd {docker_path} && .venv/bin/pip install --upgrade pip && .venv/bin/pip install {package}")
            
            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": packages
            }
        except Exception as e:
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up project in Docker: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path
            }

    async def add_dependencies(self, project_path: Path, packages: List[str]) -> dict:
        """Adds Python dependencies inside the Docker container"""
        if not self._docker_available:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
        
        # Use proper Docker paths for Linux
        docker_path = self.to_docker_path(project_path)
        
        try:
            ic(f"Installing {len(packages)} additional Python packages in Docker...")
            if self.display:
                self.display.add_message("user", f"Installing {len(packages)} additional Python packages in Docker...")
            
            installed_packages = []
            
            # Install packages with progress updates
            for i, package in enumerate(packages, 1):
                ic(f"Installing package {i}/{len(packages)}: {package}")
                if self.display:
                    self.display.add_message("user", f"Installing package {i}/{len(packages)}: {package}")
                
                try:
                    result = self.docker_exec(
                        f"cd {docker_path} && .venv/bin/pip install {package}"
                    )
                    installed_packages.append(package)
                    if self.display:
                        self.display.add_message("user", f"Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    return {
                        "command": "add_additional_depends",
                        "status": "error",
                        "error": f"Failed to install {package}: {str(e)}",
                        "project_path": str(project_path),
                        "docker_path": docker_path,
                        "packages_installed": installed_packages,
                        "packages_failed": packages[i-1:],
                        "failed_at": package
                    }
            
            if self.display:
                self.display.add_message("user", "All packages installed successfully in Docker")
            
            return {
                "command": "add_additional_depends",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": installed_packages
            }
            
        except Exception as e:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": f"Docker error: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_attempted": packages,
                "packages_installed": installed_packages if 'installed_packages' in locals() else []
            }

    async def run_app(self, project_path: Path, filename: str) -> dict:
        """Runs a Python application inside the Docker container with X11 forwarding"""
        if not self._docker_available:
            return {
                "command": "run_app",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
        
        # Use proper Docker paths for Linux
        docker_path = self.to_docker_path(project_path)
        
        try:
            # Run Python script with DISPLAY environment variable set
            ic(f"Running {filename} in Docker container")
            result = self.docker_exec(
                f"cd {docker_path} && .venv/bin/python {filename}"
            )
            
            return {
                "command": "run_app",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "run_output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.CalledProcessError as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "errors": f"Failed to run app in Docker: {str(e)}\nStderr: {e.stderr if hasattr(e, 'stderr') else 'No error output'}"
            }
        except Exception as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "errors": f"Failed to run app: {str(e)}"
            }

    # === Node.js Environment Methods (now Docker-based) ===
    async def setup_project_node(self, project_path: Path, packages: List[str]) -> dict:
        """Sets up a Node.js project inside the Docker container"""
        if not self._docker_available:
            return {
                "command": "setup_project",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
            
        # Make sure local directory exists (will be mounted to Docker)
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Use proper Docker paths for Linux
        docker_path = self.to_docker_path(project_path)
        
        try:
            # Create the project directory in Docker
            ic(f"Creating directory in Docker: {docker_path}")
            self.docker_exec(f"mkdir -p {docker_path}")
            
            # Initialize Node.js project in Docker
            ic("Initializing Node.js project in Docker...")
            self.docker_exec(f"cd {docker_path} && npm init -y")
            
            # Install packages in Docker
            ic("Installing Node.js packages in Docker...")
            for package in packages:
                ic(f"Installing package: {package}")
                self.docker_exec(f"cd {docker_path} && npm install {package}")
            
            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": packages
            }
        except Exception as e:
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up Node.js project in Docker: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path
            }

    async def add_dependencies_node(self, project_path: Path, packages: List[str]) -> dict:
        """Adds Node.js dependencies inside the Docker container"""
        if not self._docker_available:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
        
        # Use proper Docker paths for Linux
        docker_path = self.to_docker_path(project_path)
        
        try:
            ic(f"Installing {len(packages)} additional Node.js packages in Docker...")
            if self.display:
                self.display.add_message("user", f"Installing {len(packages)} additional Node.js packages in Docker...")
            
            installed_packages = []
            
            # Install packages with progress updates
            for i, package in enumerate(packages, 1):
                ic(f"Installing package {i}/{len(packages)}: {package}")
                if self.display:
                    self.display.add_message("user", f"Installing package {i}/{len(packages)}: {package}")
                
                try:
                    result = self.docker_exec(
                        f"cd {docker_path} && npm install {package}"
                    )
                    installed_packages.append(package)
                    if self.display:
                        self.display.add_message("user", f"Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    return {
                        "command": "add_additional_depends",
                        "status": "error",
                        "error": f"Failed to install {package}: {str(e)}",
                        "project_path": str(project_path),
                        "docker_path": docker_path,
                        "packages_installed": installed_packages,
                        "packages_failed": packages[i-1:],
                        "failed_at": package
                    }
            
            if self.display:
                self.display.add_message("user", "All Node.js packages installed successfully in Docker")
            
            return {
                "command": "add_additional_depends",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": installed_packages
            }
            
        except Exception as e:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": f"Docker error: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_attempted": packages,
                "packages_installed": installed_packages if 'installed_packages' in locals() else []
            }

    async def run_app_node(self, project_path: Path, filename: str) -> dict:
        """Runs a Node.js application inside the Docker container"""
        if not self._docker_available:
            return {
                "command": "run_app",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
        
        # Use proper Docker paths for Linux
        docker_path = self.to_docker_path(project_path)
        
        try:
            # Run Node.js script
            ic(f"Running {filename} in Docker container")
            result = self.docker_exec(
                f"cd {docker_path} && node {filename}"
            )
            
            return {
                "command": "run_app",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "run_output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.CalledProcessError as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "errors": f"Failed to run app in Docker: {str(e)}\nStderr: {e.stderr if hasattr(e, 'stderr') else 'No error output'}"
            }
        except Exception as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "errors": f"Failed to run app: {str(e)}"
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
        """
        Executes the specified command for project management in the Docker container.
        Both Python and Node.js projects run in Docker for consistent Linux environment.
        """
        if packages is None:
            packages = []

        try:
            if self.display:
                self.display.add_message("user", f"ProjectSetupTool executing command: {command} in {environment} environment (Docker)")
            
            # Convert project_path string to Path object
            project_path_obj = Path(project_path)
            
            # Dispatch based on environment and command
            if environment == "python":
                if command == ProjectCommand.SETUP_PROJECT:
                    result_data = await self.setup_project(project_path_obj, packages)
                elif command == ProjectCommand.ADD_DEPENDENCIES:
                    result_data = await self.add_dependencies(project_path_obj, packages)
                elif command == ProjectCommand.RUN_APP:
                    result_data = await self.run_app(project_path_obj, entry_filename)
                else:
                    return ToolResult(error=f"Unknown command: {command}")
            elif environment == "node":
                if command == ProjectCommand.SETUP_PROJECT:
                    result_data = await self.setup_project_node(project_path_obj, packages)
                elif command == ProjectCommand.ADD_DEPENDENCIES:
                    result_data = await self.add_dependencies_node(project_path_obj, packages)
                elif command == ProjectCommand.RUN_APP:
                    result_data = await self.run_app_node(project_path_obj, entry_filename)
                else:
                    return ToolResult(error=f"Unknown command: {command}")
            else:
                return ToolResult(error=f"Unsupported environment: {environment}")

            formatted_output = self.format_output(result_data)
            if self.display:
                self.display.add_message("user", f"ProjectSetupTool completed: {formatted_output}")
            return ToolResult(output=formatted_output)

        except Exception as e:
            if self.display:
                self.display.add_message("user", f"ProjectSetupTool error: {str(e)}")
            error_msg = f"Failed to execute {command}: {str(e)}"
            return ToolResult(error=error_msg)