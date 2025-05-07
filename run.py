# run.py - updated with Docker container management
import asyncio
import subprocess
import webbrowser
import time
import os
import sys
from pathlib import Path
from utils.agent_display_web_with_prompt import create_app


def check_docker_running():
    """Check if Docker is running on the system"""
    try:
        subprocess.run(["docker", "info"], check=True, capture_output=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def check_container_exists(container_name="python-dev-container"):
    """Check if the Docker container exists (running or stopped)"""
    result = subprocess.run(
        ["docker", "ps", "-a", "-q", "-f", f"name={container_name}"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def check_container_running(container_name="python-dev-container"):
    """Check if the Docker container is currently running"""
    result = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name={container_name}"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def build_docker_container():
    """Build the Docker container if it doesn't exist"""
    print("Building Docker container for Python development environment...")
    # Get the path to the Dockerfile
    docker_dir = Path(__file__).parent
    dockerfile_path = docker_dir / "Dockerfile"

    if not dockerfile_path.exists():
        print(f"Error: Dockerfile not found at {dockerfile_path}")
        print("Please make sure the Docker directory is set up correctly.")
        return False

    try:
        # Build the Docker image
        subprocess.run(
            [
                "docker",
                "build",
                "-t",
                "python-dev-image",
                "-f",
                str(dockerfile_path),
                str(docker_dir),
            ],
            check=True,
        )

        # Create and start the container
        app_dir = Path(__file__).parent / "repo"
        app_dir.mkdir(exist_ok=True)

        subprocess.run(
            [
                "docker",
                "run",
                "-d",  # Run in detached mode
                "-p",
                "5901:5901",  # VNC port
                "-v",
                f"{app_dir.absolute()}:/home/myuser/apps",  # Mount app directory
                "--name",
                "python-dev-container",
                "python-dev-image",
            ],
            check=True,
        )

        print("Docker container built and started successfully.")
        # Give the container time to initialize
        time.sleep(5)
        return True

    except subprocess.SubprocessError as e:
        print(f"Error building Docker container: {e}")
        return False


def start_docker_container(container_name="python-dev-container"):
    """Start the Docker container if it exists but is not running"""
    print(f"Starting Docker container {container_name}...")
    try:
        subprocess.run(["docker", "start", container_name], check=True)
        print("Docker container started successfully.")
        # Give the container time to initialize
        time.sleep(5)
        return True
    except subprocess.SubprocessError as e:
        print(f"Error starting Docker container: {e}")
        return False


def ensure_docker_ready():
    """Ensure that Docker is running and the container is available"""
    if not check_docker_running():
        print("Docker is not running. Please start Docker Desktop or Docker service.")
        print("After starting Docker, please run this application again.")
        return False

    container_name = "python-dev-container"

    if check_container_running(container_name):
        print(f"Docker container '{container_name}' is already running.")
        return True

    if check_container_exists(container_name):
        return start_docker_container(container_name)
    else:
        return build_docker_container()


def setup_x11_display():
    """Setup environment for X11 forwarding"""
    if os.name == "nt":  # Windows
        os.environ["DISPLAY"] = "host.docker.internal:0.0"
        print("X11 display set to 'host.docker.internal:0.0'")
    else:  # Unix-like systems
        os.environ["DISPLAY"] = ":0"
        print("X11 display set to ':0'")


# Main execution starts here
if __name__ == "__main__":
    # First ensure Docker is ready
    if not ensure_docker_ready():
        print("Docker setup failed. Exiting application.")
        sys.exit(1)

    # Setup X11 display for GUI applications
    setup_x11_display()

    # Create new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Create app with the loop
    app = create_app(loop)

    # Start the server
    server_process = subprocess.Popen(["./.venv/Scripts/python", "serve.py"])

    # Wait a moment for the server to start
    time.sleep(2)

    # Open the browser
    webbrowser.open("http://127.0.0.1:5001/select_prompt")

    try:
        # Keep the main thread running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        server_process.terminate()
        # Optionally stop the Docker container if you want
        # subprocess.run(["docker", "stop", "python-dev-container"], check=False)
