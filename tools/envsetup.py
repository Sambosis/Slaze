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
    RUN_PROJECT = "run_project"


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
        """Initialize the ProjectSetupTool instance."""
        super().__init__(display=display)
        self.display = display  # Explicitly set self.display
        self._docker_available = False
        self.docker = DockerService()
        self._docker_available = self.docker.is_available()

    def to_params(self) -> dict:
        """Convert the tool to a parameters dictionary for the API."""
        ic(f"ProjectSetupTool.to_params called with api_type: {self.api_type}")
        # Use the format that has worked in the past
        params = {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [cmd.value for cmd in ProjectCommand],
                        "description": "Command to execute"
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory"
                    },
                    "environment": {
                        "type": "string",
                        "enum": ["python", "node"],
                        "description": "Environment type (python or node)",
                        "default": "python"
                    },
                    "packages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of packages to install"
                    },
                    "entry_filename": {
                        "type": "string",
                        "description": "Name of the entry point file to run",
                        "default": "app.py"
                    }
                },
                "required": ["command", "project_path"]
            }
        }
        ic(f"ProjectSetupTool params: {params}")
        return params

    def get_docker_path(self, project_path: Path) -> str:
        """Convert a local Windows path to a Docker container path."""
        if not isinstance(project_path, Path):
            project_path = Path(project_path)
            
        # If the path starts with /home/myuser/apps/, assume it's already in Docker format
        if str(project_path).startswith("/home/myuser/apps/"):
            # Already in Docker format
            return str(project_path)
            
        # Get just the final directory name for Docker
        project_name = project_path.name
        
        # Construct the Docker path with proper slashes
        docker_path = f"/home/myuser/apps/{project_name}"
        
        # Ensure proper formatting
        docker_path = self.validate_docker_path(docker_path)
        
        return docker_path
        
    def format_output(self, data: dict) -> str:
        """Format the output data as a readable string."""
        output_lines = []
        output_lines.append(f"Command: {data['command']}")
        output_lines.append(f"Status: {data['status']}")
        output_lines.append(f"Project Path: {data['project_path']}")
        
        if 'docker_path' in data:
            output_lines.append(f"Docker Path: {data['docker_path']}")
        
        if data['status'] == 'error':
            output_lines.append(f"\nErrors:")
            output_lines.append(f"{data.get('error', 'Unknown error')}")
        
        if 'packages_installed' in data:
            output_lines.append(f"\nPackages Installed:")
            for package in data['packages_installed']:
                output_lines.append(f"- {package}")
        
        if 'run_output' in data and data['run_output']:
            output_lines.append(f"\nOutput:")
            output_lines.append(data['run_output'])
                
        return "\n".join(output_lines)
        
    def validate_docker_path(self, docker_path: str) -> str:
        """Ensure the Docker path is valid and in the correct format."""
        # Remove any Windows-style drive letters
        if ':' in docker_path:
            parts = docker_path.split(':')
            docker_path = parts[-1]
            
        # Replace backslashes with forward slashes
        docker_path = docker_path.replace('\\', '/')
        
        # Ensure it starts with /
        if not docker_path.startswith('/'):
            docker_path = '/' + docker_path
            
        # Basic validation - ensure the path starts with /home/myuser/apps
        if not docker_path.startswith('/home/myuser/apps'):
            # Extract the project name
            parts = docker_path.split('/')
            project_name = parts[-1] if parts[-1] else parts[-2]  # Handle trailing slash
            docker_path = f"/home/myuser/apps/{project_name}"
        
        # Ensure there are no double slashes or missing slashes
        while '//' in docker_path:
            docker_path = docker_path.replace('//', '/')
            
        # Ensure the path format is correct for Docker
        parts = docker_path.split('/')
        # Filter out empty parts that might come from consecutive slashes
        parts = [part for part in parts if part]
        
        # Reconstruct the path with proper slashes
        docker_path = '/' + '/'.join(parts)
            
        return docker_path
        
    async def setup_project(self, project_path: Path, packages: List[str]) -> dict:
        """Sets up a Python project inside the Docker container."""
        if not self._docker_available:
            return {
                "command": "setup_project",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
            
        # Make sure local directory exists
        project_path.mkdir(parents=True, exist_ok=True)
        
        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)
        
        try:
            # Create the project directory in Docker
            ic(f"Creating directory in Docker: {docker_path}")
            if self.display is not None:
                self.display.add_message(
                    "user", f"Creating project directory in Docker: {docker_path}"
                )
            
            # Log the exact command being executed
            mkdir_cmd = f"mkdir -p {docker_path}"
            ic(f"Docker mkdir command: {mkdir_cmd}")
            
            # Execute the command to create the directory
            mkdir_result = self.docker.execute_command(mkdir_cmd)
            if not mkdir_result.success:
                return {
                    "command": "setup_project",
                    "status": "error",
                    "error": f"Failed to create directory: {mkdir_result.stderr}",
                    "project_path": str(project_path),
                    "docker_path": docker_path
                }
            
            # Verify the directory was created
            verify_cmd = f"ls -la {docker_path}"
            verify_result = self.docker.execute_command(verify_cmd)
            ic(f"Directory verification: {verify_result.stdout}")
            
            # Create a virtual environment in Docker
            ic("Creating virtual environment in Docker...")
            if self.display is not None:
                self.display.add_message(
                    "user", "Creating Python virtual environment in Docker"
                )
                
            # Set the Docker project directory for the Docker service
            self.docker._docker_project_dir = docker_path
                
            # Ensure packages is a list of strings
            if isinstance(packages, str):
                packages = packages.split()
                
            venv_result = self.docker.create_virtual_env()
            
            if not venv_result.success:
                return {
                    "command": "setup_project",
                    "status": "error",
                    "error": f"Failed to create virtual environment: {venv_result.stderr}",
                    "project_path": str(project_path),
                    "docker_path": docker_path
                }
                
            # Initialize installed packages list
            installed_packages = []
            
            # Install base packages with pip in the virtual environment
            ic("Installing packages in Docker...")
            base_packages = ["pytest", "pytest-xvfb", "pytest-cov"]
            
            for package in base_packages:
                try:
                    result = self.docker.execute_command(
                        f"cd {docker_path} && .venv/bin/pip install {package}"
                    )
                    if result.success:
                        installed_packages.append(package)
                    else:
                        ic(f"Failed to install {package}: {result.stderr}")
                except Exception as e:
                    ic(f"Error installing {package}: {str(e)}")
                    
            # Install additional packages if provided
            if packages and len(packages) > 0:
                ic(f"Installing additional packages: {packages}")
                if self.display is not None:
                    self.display.add_message(
                        "user", f"Installing packages: {', '.join(packages)}"
                    )
                    
                # Ensure packages is a list of strings
                if isinstance(packages, str):
                    packages = packages.split()
                    
                try:
                    # Install packages one by one to better track failures
                    for package in packages:
                        result = self.docker.execute_command(
                            f"cd {docker_path} && .venv/bin/pip install {package}"
                        )
                        if result.success:
                            installed_packages.append(package)
                        else:
                            ic(f"Failed to install {package}: {result.stderr}")
                except Exception as e:
                    ic(f"Error installing packages: {str(e)}")
            
            if self.display is not None:
                self.display.add_message(
                    "user", f"Project setup complete in {docker_path}"
                )
                
            # Create a simple README.md file
            readme_content = f"""# Python Project

This project was set up with the following packages:
{', '.join(installed_packages)}

## Running the project

To run the project, use:
```
python app.py
```
"""
            try:
                readme_path = f"{docker_path}/README.md"
                self.docker.execute_command(f'echo "{readme_content}" > {readme_path}')
            except Exception as e:
                ic(f"Error creating README: {str(e)}")
                
            return {
                "command": "setup_project",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "packages_installed": installed_packages
            }
            
        except Exception as e:
            error_message = str(e)
            ic(f"Error in setup_project: {error_message}")
            if self.display is not None:
                self.display.add_message(
                    "user", f"Error setting up project: {error_message}"
                )
            return {
                "command": "setup_project",
                "status": "error",
                "error": f"Failed to set up project: {error_message}",
                "project_path": str(project_path),
                "docker_path": docker_path if 'docker_path' in locals() else None
            }
            
    async def add_dependencies(self, project_path: Path, packages: List[str]) -> dict:
        """Adds additional Python dependencies to an existing project."""
        if not self._docker_available:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
            
        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)
        
        # Validate and fix the Docker path if needed
        docker_path = self.validate_docker_path(docker_path)
        
        try:
            # Install packages
            ic(f"Installing additional packages in Docker: {packages}")
            if self.display is not None:
                self.display.add_message(
                    "user", f"Installing additional packages in Docker: {packages}"
                )
                
            # Initialize the list of installed packages
            installed_packages = []
            
            # Install each package
            for i, package in enumerate(packages, 1):
                ic(f"Installing package {i}/{len(packages)}: {package}")
                if self.display is not None:
                    self.display.add_message(
                        "user", f"Installing package {i}/{len(packages)}: {package}"
                    )

                try:
                    result = self.docker.execute_command(
                        f"cd {docker_path} && .venv/bin/pip install {package}"
                    )
                    installed_packages.append(package)
                    if self.display is not None:
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

            if self.display is not None:
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
            
            # Check if self.display is not None before calling add_message
            if self.display is not None:
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
            if self.display is not None:
                self.display.add_message(
                    "user", f"Creating project directory in Docker: {docker_path}"
                )
            self.docker.execute_command(f"mkdir -p {docker_path}")

            # Initialize Node.js project in Docker
            ic("Initializing Node.js project in Docker...")
            if self.display is not None:
                self.display.add_message(
                    "user", "Initializing Node.js project in Docker"
                )
            
            # Create package.json with npm init -y
            init_result = self.docker.execute_command(
                f"cd {docker_path} && npm init -y"
            )
            
            if not init_result.success:
                return {
                    "command": "setup_project",
                    "status": "error",
                    "error": f"Failed to initialize Node.js project: {init_result.stderr}",
                    "project_path": str(project_path),
                    "docker_path": docker_path
                }
                
            # Initialize installed packages list
            installed_packages = []
            
            # Install base development packages
            base_packages = ["jest", "eslint"]
            
            for package in base_packages:
                try:
                    result = self.docker.execute_command(
                        f"cd {docker_path} && npm install --save-dev {package}"
                    )
                    installed_packages.append(package)
                    if self.display is not None:
                        self.display.add_message("user", f"Installed {package}")
                except Exception:
                    if self.display is not None:
                        self.display.add_message(
                            "user", f"Warning: Failed to install {package}, continuing anyway"
                        )
                    
            # Install user-specified packages
            if packages:
                for package in packages:
                    try:
                        result = self.docker.execute_command(
                            f"cd {docker_path} && npm install {package}"
                        )
                        installed_packages.append(package)
                        if self.display is not None:
                            self.display.add_message(
                                "user", f"Installed {package}"
                            )
                    except Exception as e:
                        return {
                            "command": "setup_project",
                            "status": "error",
                            "error": f"Failed to install {package}: {str(e)}",
                            "project_path": str(project_path),
                            "docker_path": docker_path,
                            "packages_installed": installed_packages,
                        }
                        
            if self.display is not None:
                self.display.add_message(
                    "user", "Node.js project setup completed successfully in Docker"
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
                "error": f"Docker error: {str(e)}",
                "project_path": str(project_path),
                "docker_path": docker_path if 'docker_path' in locals() else "unknown",
            }
    
    async def add_dependencies_node(
        self, project_path: Path, packages: List[str]
        ) -> dict:
        """Adds additional Node.js dependencies to an existing project."""
        if not self._docker_available:
            return {
                "command": "add_additional_depends",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
            
        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)
        
        # Validate and fix the Docker path if needed
        docker_path = self.validate_docker_path(docker_path)
        
        try:
            # Install packages
            ic(f"Installing additional Node.js packages in Docker: {packages}")
            if self.display is not None:
                self.display.add_message(
                    "user", f"Installing additional Node.js packages in Docker: {packages}"
                )
                
            # Initialize the list of installed packages
            installed_packages = []
            
            # Check if package.json exists
            pkg_json_check = self.docker.execute_command(
                f"[ -f {docker_path}/package.json ] && echo 'exists' || echo 'not exists'"
            )
            
            if "not exists" in pkg_json_check.stdout:
                return {
                    "command": "add_additional_depends",
                    "status": "error",
                    "error": "package.json not found. Please run setup_project first.",
                    "project_path": str(project_path),
                    "docker_path": docker_path
                }
                
            # Install each package
            for i, package in enumerate(packages, 1):
                ic(f"Installing package {i}/{len(packages)}: {package}")
                if self.display is not None:
                    self.display.add_message(
                        "user", f"Installing package {i}/{len(packages)}: {package}"
                    )

                try:
                    # Check if it's a dev dependency
                    is_dev = any(dev_keyword in package.lower() for dev_keyword in 
                                ['test', 'jest', 'mocha', 'chai', 'eslint', 'prettier', 'babel', 'webpack', 'typescript'])
                    
                    if is_dev:
                        cmd = f"cd {docker_path} && npm install --save-dev {package}"
                    else:
                        cmd = f"cd {docker_path} && npm install {package}"
                        
                    result = self.docker.execute_command(cmd)
                    installed_packages.append(package)
                    if self.display is not None:
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

            if self.display is not None:
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
            # Run Node.js script
            ic(f"Running {filename} in Docker container at {docker_path}")
            if self.display is not None:
                self.display.add_message(
                    "user", f"Running Node.js app {filename} in Docker container"
                )

            # Check if package.json exists
            pkg_json_check = self.docker.execute_command(
                f"[ -f {docker_path}/package.json ] && echo 'exists' || echo 'not exists'"
            )
            
            if "not exists" in pkg_json_check.stdout:
                return {
                    "command": "run_app",
                    "status": "error",
                    "error": "package.json not found. Please run setup_project first.",
                    "project_path": str(project_path),
                    "docker_path": docker_path
                }

            # Execute the Node.js script
            cmd = f"cd {docker_path} && node {filename}"
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
                "errors": f"Failed to run Node.js app in Docker: {str(e)}",
            }
    
    async def run_project(self, project_path: Path, entry_filename: str = "app.py") -> dict:
        """Runs a Python project inside the Docker container."""
        if not self._docker_available:
            return {
                "command": "run_project",
                "status": "error",
                "error": "Docker is not available or not running",
                "project_path": str(project_path)
            }
            
        # Get Docker path with correct format
        docker_path = self.get_docker_path(project_path)
        
        # Ensure the entry file path is correctly formatted
        entry_file = f"{docker_path}/{entry_filename}"
        entry_file = entry_file.replace('//', '/')  # Remove any double slashes
        
        try:
            # Check if the entry file exists
            ic(f"Checking if entry file exists: {entry_file}")
            if self.display is not None:
                self.display.add_message(
                    "user", f"Checking if entry file exists: {entry_file}"
                )
                
            # Verify the file exists
            file_check_cmd = f"[ -f {entry_file} ] && echo 'exists' || echo 'not exists'"
            file_check = self.docker.execute_command(file_check_cmd)
            
            if "not exists" in file_check.stdout:
                return {
                    "command": "run_project",
                    "status": "error",
                    "error": f"Entry file {entry_filename} does not exist in {docker_path}",
                    "project_path": str(project_path),
                    "docker_path": docker_path
                }

            # Run Python script with X11 forwarding
            ic(f"Running {entry_filename} in Docker container at {docker_path}")
            
            # Check if self.display is not None before calling add_message
            if self.display is not None:
                self.display.add_message("user", f"Running {entry_filename} in Docker container")

            # Check if virtual environment exists
            venv_check = self.docker.execute_command(
                    f"[ -d {docker_path}/.venv ] && echo 'exists' || echo 'not exists'"
                )

            # Set up command with the appropriate Python interpreter
            if "exists" in venv_check.stdout:
                cmd = f"cd {docker_path} && .venv/bin/python3 {entry_file}"
            else:
                cmd = f"cd {docker_path} && python3 {entry_file}"

            # Execute with DISPLAY set for GUI applications
            result = self.docker.execute_command(
                cmd, env_vars={"DISPLAY": "host.docker.internal:0"}
            )

            return {
                "command": "run_project",
                "status": "success",
                "project_path": str(project_path),
                "docker_path": docker_path,
                "run_output": result.stdout,
                "errors": result.stderr,
            }
        except Exception as e:
            if self.display is not None:
                self.display.add_message("user", f"ProjectSetupTool error: {str(e)}")
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
    ) -> ToolResult:
        """Executes the specified command for project management."""
        try:
            # Handle both string and Enum types for command
            if hasattr(command, 'value'):
                command_value = command.value
            else:
                # If command is a string, try to convert it to an Enum
                try:
                    command = ProjectCommand(command)
                    command_value = command.value
                except ValueError:
                    # If conversion fails, return an error
                    return ToolResult(
                        error=f"Unknown command: {command}",
                        tool_name=self.name
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
                    error=f"Unknown command: {command}",
                    tool_name=self.name
                )
                
            return ToolResult(output=self.format_output(result))
            
        except Exception as e:
            if self.display is not None:
                self.display.add_message("user", f"ProjectSetupTool error: {str(e)}")
            return ToolResult(error=f"Failed to execute {command}: {str(e)}")
