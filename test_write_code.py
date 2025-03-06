#!/usr/bin/env python3
import asyncio
import os
from pathlib import Path

# IMPORTANT:  Adjust these imports to match your project structure
from tools.write_code import WriteCodeTool  # Assuming WriteCodeTool is in tools/write_code.py
from utils.docker_service import DockerService # Import if you use the DockerService
from config import * # Import your config to get docker paths and project directory
from dotenv import load_dotenv
load_dotenv()
# Mock display for testing (optional, but good practice)
class MockDisplay:
    def add_message(self, role, content):
        print(f"[{role}]: {content}")

async def test_write_code():
    """
    Test function to call WriteCodeTool.write_code_to_file with hardcoded parameters.
    """
    # --- HARDCODE YOUR TEST PARAMETERS HERE ---
    code_description = """
Create a simple Python class called 'Greeter' with:
- An __init__ method that takes a 'name' (str) as an argument.
- A method called 'greet' that returns a greeting string like "Hello, {name}!".
- Proper docstrings.
"""
    current_dir = Path(os.getcwd())
    project_name = "my_test_project" # Project name
    set_project_dir(project_name)  # Set the project directory to the current directory
    project_path_str = str(current_dir / project_name)
    project_path = Path(project_path_str)
    filename = "src/greeter.py"    # File to create within the project

    # Create Project Directory on host:
    project_path.mkdir(parents=True, exist_ok=True)

    # --- END OF HARDCODED PARAMETERS ---

    # Initialize the tool (use a mock display for testing)
    tool = WriteCodeTool()#display=MockDisplay())

    # Call write_code_to_file with the hardcoded parameters
    result = await tool.write_code_to_file(
        code_description=code_description,
        project_path=project_path,
        filename=filename
    )

    print("\nResult:")
    print(result)
    # You can check the output

    # Example of how to check for success and get the Docker path:
    if result['status'] == 'success':
        docker_path = result['docker_path']
        print(f"\nFile written to Docker at: {docker_path}")

        # Verify using DockerService (if available):
        if hasattr(tool, 'docker') and tool.docker.is_available():
            file_exists = tool.docker.check_file_exists(docker_path)
            print(f"File exists in Docker: {file_exists}")
            if file_exists:
                # cat the file.
                docker_result = tool.docker.execute_command(f"cat {docker_path}")
                if docker_result.success:
                    print(f"File contents:\n{docker_result.stdout}")

    else:
        print(f"File writing failed: {result['error']}")

if __name__ == "__main__":
    asyncio.run(test_write_code())
