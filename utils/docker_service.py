import os
import subprocess
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List
import logging
from dataclasses import dataclass

from config import get_constant, set_constant

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DockerService")
# set the logger level to DEBUG
# logger.setLevel(logging.DEBUG)
# set the logger level to INFO
logger.setLevel(logging.INFO)

@dataclass
class DockerResult:
    """Structured result from Docker command execution"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    command: str

class DockerServiceError(Exception):
    """Base exception for Docker service errors"""
    pass

class ContainerNotFoundError(DockerServiceError):
    """Exception raised when container is not found"""
    pass

class CommandExecutionError(DockerServiceError):
    """Exception raised when command execution fails"""
    pass

class DockerNotAvailableError(DockerServiceError):
    """Exception raised when Docker is not available"""
    pass

class DockerService:
    """Service class for Docker operations in the project"""

    def __init__(self):
        """Initialize the Docker service"""
        self._container_name = get_constant("DOCKER_CONTAINER_NAME") or "python-dev-container"
        self._docker_available = self._check_docker_available()
        self._path_cache = {}  # Cache for path translations

        # Get Docker project directory from config
        raw_docker_dir = get_constant("DOCKER_PROJECT_DIR")
        self._docker_project_dir = self._format_docker_path(raw_docker_dir) if raw_docker_dir else None
        self._project_dir = get_constant("PROJECT_DIR")

        # Default display settings
        self._default_display = "host.docker.internal:0"

        if self._docker_available:
            logger.info(f"Docker service initialized with container: {self._container_name}")
            logger.info(f"Docker project directory: {self._docker_project_dir}")
        else:
            logger.warning("Docker not available. Operations will fail.")

    def _check_docker_available(self) -> bool:
        """Check if Docker is available and the container is running"""
        try:
            result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={self._container_name}"],
                capture_output=True,
                check=False
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except Exception as e:
            logger.error(f"Error checking Docker availability: {str(e)}")
            return False

    def is_available(self) -> bool:
        """Check if Docker service is available"""
        # Refresh availability status
        self._docker_available = self._check_docker_available()
        return self._docker_available

    def _format_docker_path(self, path: Union[str, Path]) -> str:
        """Format a path for use in Docker commands to ensure proper slashes."""
        if path is None:
            return None
            
        # Convert to string if it's a Path
        docker_path = str(path)
        
        # Replace backslashes with forward slashes
        docker_path = docker_path.replace('\\', '/')
        
        # Ensure there are no double slashes
        while '//' in docker_path:
            docker_path = docker_path.replace('//', '/')
            
        # Ensure it starts with /
        if not docker_path.startswith('/'):
            docker_path = '/' + docker_path
            
        # Split the path and filter out empty parts
        parts = docker_path.split('/')
        parts = [part for part in parts if part]
        
        # Reconstruct the path with proper slashes
        docker_path = '/' + '/'.join(parts)
        
        return docker_path

    def to_docker_path(self, host_path: Union[str, Path]) -> Path:
        """Convert a host path to a Docker container path"""
        if isinstance(host_path, str):
            host_path = Path(host_path)

        # Check cache first
        cache_key = str(host_path)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # If no project directory is set, we can't do path translation
        project_dir = self._project_dir
        if not project_dir:
            raise DockerServiceError("Project directory not set in configuration")

        # Extract project name (directory name only, not full path)
        project_name = project_dir.name

        # Set the Docker base path
        docker_base = "/home/myuser/apps"

        try:
            # Determine if path is within project directory
            if host_path == project_dir or project_dir in host_path.parents:
                # Get the relative path from project directory
                rel_path = host_path.relative_to(project_dir)
                # Construct proper Docker path
                if str(rel_path) == ".":  # If it's the project directory itself
                    docker_path = f"{docker_base}/{project_name}"
                else:
                    docker_path = f"{docker_base}/{project_name}/{rel_path}"
            else:
                # Not within project directory, use default handling
                docker_path = f"{docker_base}/{host_path.name}"

            # Log the path translation for debugging
            logger.debug(f"Path translation: {host_path} -> {docker_path}")

            # Cache and return result
            docker_path = Path(docker_path)
            self._path_cache[cache_key] = docker_path
            return docker_path

        except ValueError:
            # Fallback if relative_to raises error
            docker_path = Path(f"{docker_base}/{host_path.name}")
            self._path_cache[cache_key] = docker_path
            return docker_path

    def execute_command(self, command: str, env_vars: Optional[Dict[str, str]] = None) -> DockerResult:
        """Execute a command in the Docker container"""
        if not self.is_available():
            raise DockerNotAvailableError("Docker is not available or container is not running")

        # Prepare environment variables string
        env_string = ""
        if env_vars:
            env_string = " ".join([f"export {k}={v} &&" for k, v in env_vars.items()])
            env_string += " "

        # Escape the command for passing to bash -c
        escaped_command = command.replace('"', '\\"')

        # Build the full Docker command
        docker_command = f'docker exec {self._container_name} bash -c "{env_string}{escaped_command}"'
        # ic(docker_command)
        logger.info(f"Executing Docker command: {docker_command}")

        try:
            result = subprocess.run(
                docker_command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False
            )

            return DockerResult(
                success=(result.returncode == 0),
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                command=command
            )
        except Exception as e:
            logger.error(f"Error executing Docker command: {str(e)}")
            raise CommandExecutionError(f"Docker command execution failed: {str(e)}")

    def run_python_script(self, script_path: Union[str, Path], 
                         gui: bool = False, 
                         cwd: Optional[Union[str, Path]] = None,
                         env_vars: Optional[Dict[str, str]] = None) -> DockerResult:
        """Run a Python script in the Docker container"""
        if isinstance(script_path, str):
            script_path = Path(script_path)

        # Convert to Docker paths
        docker_script_path = self.to_docker_path(script_path)

        # Set working directory
        working_dir = cwd if cwd else self._docker_project_dir
        if isinstance(working_dir, Path):
            working_dir = str(working_dir)

        # Set up default environment variables
        if env_vars is None:
            env_vars = {}

        # Add DISPLAY variable for GUI applications
        if gui and "DISPLAY" not in env_vars:
            env_vars["DISPLAY"] = self._default_display

        # Check if virtual environment exists
        venv_check = self.execute_command(
            f"[ -d {self._docker_project_dir}/.venv ] && echo 'exists' || echo 'not exists'"
        )

        # Determine Python command based on venv existence
        if venv_check.success and "exists" in venv_check.stdout:
            python_cmd = f".venv/bin/python {docker_script_path}"
        else:
            python_cmd = f"python3 {docker_script_path}"

        # Build the full command
        command = f"cd {working_dir} && {python_cmd}"

        # Execute the command
        return self.execute_command(command, env_vars)

    def run_bash_script(self, script_path: Union[str, Path],
                       cwd: Optional[Union[str, Path]] = None,
                       env_vars: Optional[Dict[str, str]] = None) -> DockerResult:
        """Run a bash script in the Docker container"""
        if isinstance(script_path, str):
            script_path = Path(script_path)

        # Convert to Docker paths
        docker_script_path = self.to_docker_path(script_path)

        # Set working directory
        working_dir = cwd if cwd else self._docker_project_dir
        if isinstance(working_dir, Path):
            working_dir = str(working_dir)

        # Build the full command
        command = f"cd {working_dir} && bash {docker_script_path}"

        # Execute the command
        return self.execute_command(command, env_vars)

    def check_file_exists(self, file_path: Union[str, Path]) -> bool:
        """Check if a file exists in the Docker container"""
        if isinstance(file_path, str):
            file_path = Path(file_path)

        docker_path = self.to_docker_path(file_path)

        result = self.execute_command(f"[ -f {docker_path} ] && echo 'exists' || echo 'not exists'")
        return result.success and "exists" in result.stdout

    def create_virtual_env(self, packages: Optional[List[str]] = None) -> DockerResult:
        """Create a Python virtual environment in the Docker container"""
        # Ensure docker_project_dir is properly formatted with slashes
        docker_dir = self._format_docker_path(self._docker_project_dir)
        
        # Log the directory we're using
        logger.info(f"Creating virtual environment in directory: {docker_dir}")
        
        commands = [
            f"cd {docker_dir}",
            "python3 -m venv .venv",
            ".venv/bin/pip install --upgrade pip"
        ]

        if packages and len(packages) > 0:
            pkg_list = " ".join(packages)
            commands.append(f".venv/bin/pip install {pkg_list}")

        command = " && ".join(commands)
        return self.execute_command(command)

    def install_packages(self, packages: List[str]) -> DockerResult:
        """Install packages in the Docker container's virtual environment"""
        if not packages:
            return DockerResult(True, "No packages to install", "", 0, "")

        # Ensure docker_project_dir is properly formatted with slashes
        docker_dir = self._format_docker_path(self._docker_project_dir)
        
        # Log the directory we're using
        logger.info(f"Installing packages in directory: {docker_dir}")

        venv_check = self.execute_command(
            f"[ -d {docker_dir}/.venv ] && echo 'exists' || echo 'not exists'"
        )

        if not (venv_check.success and "exists" in venv_check.stdout):
            # Create virtual environment if it doesn't exist
            venv_result = self.create_virtual_env()
            if not venv_result.success:
                return venv_result

        # Install packages
        pkg_list = " ".join(packages)
        command = f"cd {docker_dir} && .venv/bin/pip install {pkg_list}"
        return self.execute_command(command)

    def start_container(self) -> bool:
        """Start the Docker container if it exists but is not running"""
        try:
            # Check if container exists but is not running
            exists_result = subprocess.run(
                ["docker", "ps", "-a", "-q", "-f", f"name={self._container_name}"],
                capture_output=True,
                text=True,
                check=False
            )

            running_result = subprocess.run(
                ["docker", "ps", "-q", "-f", f"name={self._container_name}"],
                capture_output=True,
                text=True,
                check=False
            )

            if exists_result.stdout.strip() and not running_result.stdout.strip():
                # Container exists but is not running, so start it
                start_result = subprocess.run(
                    ["docker", "start", self._container_name],
                    capture_output=True,
                    check=False
                )

                if start_result.returncode == 0:
                    logger.info(f"Started Docker container: {self._container_name}")
                    self._docker_available = True
                    return True

            return False
        except Exception as e:
            logger.error(f"Error starting Docker container: {str(e)}")
            return False

    def get_logs(self, lines: int = 100) -> str:
        """Get container logs"""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), self._container_name],
                capture_output=True,
                text=True,
                check=False
            )
            return result.stdout
        except Exception as e:
            logger.error(f"Error getting Docker logs: {str(e)}")
            return f"Error: {str(e)}"
