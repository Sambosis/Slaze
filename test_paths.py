"""
Test script to verify proper path handling and Docker path translation.
"""

from pathlib import Path
from utils.file_logger import convert_to_docker_path
from config import get_constant, set_project_dir


def test_path_translations():
    """Test various path translation scenarios"""
    print("=== Testing Path Translations ===\n")

    # Get the current system paths
    repo_dir = get_constant("REPO_DIR")
    if isinstance(repo_dir, str):
        repo_dir = Path(repo_dir)

    # Test project path setup
    test_project_name = "test_paths_project"
    project_dir = set_project_dir(test_project_name)

    # Print out the important paths
    print(f"Project name: {test_project_name}")
    print(f"Repo directory: {repo_dir}")
    print(f"Project directory: {project_dir}")
    print(f"Docker project directory: {get_constant('DOCKER_PROJECT_DIR')}")

    print("\n=== Path Conversion Tests ===\n")

    # Test paths for conversion
    test_paths = [
        project_dir,
        project_dir / "test_file.py",
        project_dir / "subdir" / "test_file.py",
        f"C:\\Users\\Machine81\\Slazy\\repo\\{test_project_name}\\test_file.py",
        f"/c/Users/Machine81/Slazy/repo/{test_project_name}/test_file.py",
        f"{project_dir}\\test_file.py",
        f" {get_constant('DOCKER_PROJECT_DIR')}\\test_file.py",
        "test_file.py",  # Relative path
    ]

    for path in test_paths:
        docker_path = convert_to_docker_path(path)
        print(f"Original: {path}")
        print(f"Docker:   {docker_path}")
        print("-" * 50)

    print("\n=== Path Writing Tests ===\n")

    # Create a test directory structure
    test_dir = project_dir / "test_subdir"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create a test file
    test_file = test_dir / "test_file.txt"
    test_file.write_text("This is a test file", encoding="utf-8")

    print(f"Created test file at: {test_file}")
    print(f"Docker path would be: {convert_to_docker_path(test_file)}")

    # Read the file back to verify
    content = test_file.read_text(encoding="utf-8")
    print(f"File content: {content}")

    print("\nTests completed!")


if __name__ == "__main__":
    test_path_translations()
