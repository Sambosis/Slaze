# -*- coding: utf-8 -*-
"""
Python script adapted from a Colab notebook.
Original file was located at
    https://colab.research.google.com/drive/1wQ0uF1e6bMM51gPf2zetfWgbSJksxqf3
"""

import os
import subprocess
import sys

# Define the repository and directory names
repo_url = "http://github.com/sambosis/Slaze"
repo_dir_name = "Slaze"
# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
slaze_base_dir = (
    script_dir  # Base directory for cloning Slaze, can be changed if needed
)
slaze_repo_path = os.path.join(slaze_base_dir, repo_dir_name)


def run_command(command, working_dir=None, env_vars=None):
    """Helper function to run shell commands."""
    print(f"Running command: {' '.join(command)} in {working_dir or os.getcwd()}")
    current_env = os.environ.copy()
    if env_vars:
        current_env.update(env_vars)

    process = subprocess.Popen(
        command,
        cwd=working_dir,
        env=current_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=sys.platform == "win32",
    )  # Use shell=True on Windows for git, uv etc. if they are batch files or need shell interpretation

    # Stream stdout and stderr
    if process.stdout:
        for line in process.stdout:
            print(line, end="")
    if process.stderr:
        for line in process.stderr:
            print(f"Error: {line}", end="", file=sys.stderr)

    process.wait()  # Wait for the command to complete

    if process.returncode != 0:
        print(
            f"Command {' '.join(command)} failed with exit code {process.returncode}",
            file=sys.stderr,
        )
        # sys.exit(process.returncode) # Optionally exit if a command fails
    return process.returncode


# Clone the repository if it doesn't exist
if not os.path.exists(slaze_repo_path):
    print(f"Directory '{slaze_repo_path}' not found. Cloning repository...")
    run_command(["git", "clone", repo_url, slaze_repo_path], working_dir=slaze_base_dir)
else:
    print(f"Directory '{slaze_repo_path}' already exists. Skipping clone.")

# Change to the Slaze directory for subsequent commands
print(f"Changing working directory to: {slaze_repo_path}")
# Note: os.chdir changes the global CWD for the script.
# Subsequent run_command calls will use this CWD if not overridden.

# Run uv sync
run_command(["uv", "sync", "-q"], working_dir=slaze_repo_path)

# Run the runner script
runner_script_path = os.path.join("stls", "runner.py")  # Relative to slaze_repo_path
run_command(["uv", "run", runner_script_path], working_dir=slaze_repo_path)

# Set environment variable for MPLBACKEND
print("Setting MPLBACKEND=agg")
# This environment variable will be set for subsequent commands run via run_command if passed
# or can be set globally for the script's process:
os.environ["MPLBACKEND"] = "agg"

# Run agent training script
agent_training_script_path = os.path.join(
    ".", "stls", "run_agent_training.py"
)  # Relative to slaze_repo_path
run_command(["uv", "run", agent_training_script_path], working_dir=slaze_repo_path)

print("Script finished.")

if __name__ == "__main__":
    # The main logic is already at the top level,
    # but you could move it into a main() function if preferred.
    pass
