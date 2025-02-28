import subprocess
import sys
import os


def build_and_run():
    """Builds and runs the Docker image."""

    image_name = "python-dev-env"
    dockerfile_path = "Dockerfile"
    context_path = "."  # Current directory (repo)

    # Get the absolute path to the app directory
    app_dir = os.path.abspath("app")

    try:
        print("Building Docker image...")
        build_result = subprocess.run(
            ["docker", "build", "-t", image_name, "-f", dockerfile_path, context_path],
            capture_output=False,
            check=True,
        )
        print("Docker image built successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error building Docker image: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        print("Running Docker container...")
        run_result = subprocess.run(
            [
                "docker",
                "run",
                "-d",  # Run in detached mode (background)
                "-p",
                "5901:5901",  # Port mapping for VNC
                "-p",
                "8888:8888",  # Port mapping for Jupyter (if needed)
                "-v",
                f"{app_dir}:/home/myuser/apps",  # Mount app directory
                "--name",
                "python-dev-container",
                image_name,
            ],
            capture_output=True,
            check=True,
        )
        container_id = run_result.stdout.decode().strip()
        print(f"Docker container running. Container ID: {container_id}")
        print("\nAccess Information:")
        print("  VNC Connection:")
        print("  - Use a VNC client (like TightVNC Viewer, RealVNC, etc.)")
        print("  - Connect to: localhost:5901")
        print("  - Use the password: mypassword")
        print(
            "\n  Your local app/ directory is mounted in the container at /home/myuser/apps"
        )
        print(
            "  Any changes you make to files in either location will sync automatically."
        )

    except subprocess.CalledProcessError as e:
        print(f"Error running Docker container: {e}", file=sys.stderr)
        print(e.stderr.decode())
        sys.exit(1)


if __name__ == "__main__":
    build_and_run()
