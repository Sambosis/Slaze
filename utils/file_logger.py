import json
import datetime
from pathlib import Path
from config import get_constant
from icecream import ic

import ast
import inspect
import re
from typing import Union, Optional
import os
from typing import Dict, List, Optional, Set, Tuple
import mimetypes
import base64

try:
    from config import get_constant
    # Import the function but don't redefine it
    try:
        from config import convert_to_docker_path
    except ImportError:
        # Define our own if not available in config
        def convert_to_docker_path(path: Union[str, Path]) -> str:
            """
            Convert a local Windows path to a Docker container path.
            
            Args:
                path: The local path to convert
                
            Returns:
                The Docker container path
            """
            if path is None:
                return ""
                
            # Ensure path is a string
            if isinstance(path, Path):
                path = str(path)
                
            # Replace Windows-style paths with Linux-style for comparison
            norm_path = path.replace('\\', '/')
            
            try:
                # Get project directory constants
                project_dir = get_constant('PROJECT_DIR')
                repo_dir = get_constant('REPO_DIR')
                
                # Skip conversion if already in Docker format
                if norm_path.startswith('/home/myuser/apps/'):
                    return norm_path
                    
                # Ensure project_dir is normalized
                if isinstance(project_dir, Path):
                    project_dir = str(project_dir).replace('\\', '/')
                
                # First check if path is directly in the repo directory
                if isinstance(repo_dir, Path):
                    repo_dir = str(repo_dir).replace('\\', '/')
                
                # The correct conversion logic for mapping to Docker paths
                if project_dir and norm_path.startswith(project_dir):
                    # Extract just the prompt name (project name) from the project directory
                    project_name = Path(project_dir).name
                    
                    # Get the path relative to the project directory
                    rel_path = os.path.relpath(norm_path, project_dir)
                    rel_path = rel_path.replace('\\', '/')
                    
                    # Build the Docker path
                    if rel_path == '.':
                        return f'/home/myuser/apps/{project_name}'
                    else:
                        return f'/home/myuser/apps/{project_name}/{rel_path}'
                
                # For files directly in the repo directory
                elif repo_dir and norm_path.startswith(repo_dir):
                    # Get path relative to repo dir
                    rel_path = os.path.relpath(norm_path, repo_dir)
                    rel_path = rel_path.replace('\\', '/')
                    
                    # Extract the prompt name if it's directly under repo
                    parts = rel_path.split('/', 1)
                    if len(parts) > 1:
                        project_name = parts[0]
                        sub_path = parts[1]
                        return f'/home/myuser/apps/{project_name}/{sub_path}'
                    else:
                        return f'/home/myuser/apps/{rel_path}'
            except Exception as e:
                # Log the error but continue
                print(f"Error in convert_to_docker_path: {e}")
            
            # Fallback to simple conversion if the above fails
            norm_path = path.replace('\\', '/')
            
            # Remove Windows drive letter if present
            if ':' in norm_path:
                norm_path = norm_path.split(':', 1)[1]
                
            # Ensure the path starts with a forward slash
            if not norm_path.startswith('/'):
                norm_path = '/' + norm_path
                
            # Default docker path prefix - if all else fails
            if not norm_path.startswith('/home/myuser/apps/'):
                # Extract the last directory as project name
                path_parts = norm_path.strip('/').split('/')
                if len(path_parts) >= 2:
                    project_name = path_parts[-2]  # Use second-to-last part as project name
                    file_name = path_parts[-1]     # Use last part as file name
                    return f'/home/myuser/apps/{project_name}/{file_name}'
                else:
                    return f'/home/myuser/apps/{norm_path.strip("/")}'
            
            return norm_path
except ImportError:
    # Fallback if config module is not available
    def get_constant(name):
        # Default values for essential constants
        defaults = {
            'PROJECT_DIR': os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'DOCKER_PROJECT_DIR': '/home/myuser/apps',
            'LOG_FILE': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "file_log.json")
        }
        return defaults.get(name)
        
    def convert_to_docker_path(path: Union[str, Path]) -> str:
        """
        Convert a local Windows path to a Docker container path.
        
        Args:
            path: The local path to convert
            
        Returns:
            The Docker container path
        """
        if path is None:
            return ""
            
        # Ensure path is a string
        if isinstance(path, Path):
            path = str(path)
            
        # Replace Windows-style paths with Linux-style for comparison
        norm_path = path.replace('\\', '/')
        
        try:
            # Get project directory constants
            project_dir = get_constant('PROJECT_DIR')
            repo_dir = get_constant('REPO_DIR')
            
            # Skip conversion if already in Docker format
            if norm_path.startswith('/home/myuser/apps/'):
                return norm_path
                
            # Ensure project_dir is normalized
            if isinstance(project_dir, Path):
                project_dir = str(project_dir).replace('\\', '/')
            
            # First check if path is directly in the repo directory
            if isinstance(repo_dir, Path):
                repo_dir = str(repo_dir).replace('\\', '/')
            
            # The correct conversion logic for mapping to Docker paths
            if project_dir and norm_path.startswith(project_dir):
                # Extract just the prompt name (project name) from the project directory
                project_name = Path(project_dir).name
                
                # Get the path relative to the project directory
                rel_path = os.path.relpath(norm_path, project_dir)
                rel_path = rel_path.replace('\\', '/')
                
                # Build the Docker path
                if rel_path == '.':
                    return f'/home/myuser/apps/{project_name}'
                else:
                    return f'/home/myuser/apps/{project_name}/{rel_path}'
            
            # For files directly in the repo directory
            elif repo_dir and norm_path.startswith(repo_dir):
                # Get path relative to repo dir
                rel_path = os.path.relpath(norm_path, repo_dir)
                rel_path = rel_path.replace('\\', '/')
                
                # Extract the prompt name if it's directly under repo
                parts = rel_path.split('/', 1)
                if len(parts) > 1:
                    project_name = parts[0]
                    sub_path = parts[1]
                    return f'/home/myuser/apps/{project_name}/{sub_path}'
                else:
                    return f'/home/myuser/apps/{rel_path}'
        except Exception as e:
            # Log the error but continue
            print(f"Error in convert_to_docker_path: {e}")
        
        # Fallback to simple conversion if the above fails
        norm_path = path.replace('\\', '/')
        
        # Remove Windows drive letter if present
        if ':' in norm_path:
            norm_path = norm_path.split(':', 1)[1]
            
        # Ensure the path starts with a forward slash
        if not norm_path.startswith('/'):
            norm_path = '/' + norm_path
            
        # Default docker path prefix - if all else fails
        if not norm_path.startswith('/home/myuser/apps/'):
            # Extract the last directory as project name
            path_parts = norm_path.strip('/').split('/')
            if len(path_parts) >= 2:
                project_name = path_parts[-2]  # Use second-to-last part as project name
                file_name = path_parts[-1]     # Use last part as file name
                return f'/home/myuser/apps/{project_name}/{file_name}'
            else:
                return f'/home/myuser/apps/{norm_path.strip("/")}'
        
        return norm_path

# File for logging operations
try:
    LOG_FILE = get_constant('LOG_FILE')
except:
    LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "file_log.json")

# In-memory tracking of file operations
FILE_OPERATIONS = {}

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# If log file doesn't exist, create an empty one
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w') as f:
        json.dump({"files": {}}, f)

# Track file operations
file_operations = []
tracked_files = set()
file_contents = {}

def log_file_operation(file_path: Path, operation: str, content: str = None, metadata: dict = None):
    """
    Log a file operation (create, update, delete) with enhanced metadata handling.
    
    Args:
        file_path: Path to the file
        operation: Type of operation ('create', 'update', 'delete')
        content: Optional content for the file
        metadata: Optional dictionary containing additional metadata (e.g., image generation prompt)
    """
    # Defensively initialize metadata to prevent NoneType errors
    if metadata is None:
        metadata = {}
    
    # Ensure file_path is a Path object
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
        
    # Create a string representation of the file path for consistent logging
    file_path_str = str(file_path)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extension = file_path.suffix.lower() if file_path.suffix else ""
    
    # Determine if the file is an image
    is_image = extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp']
    mime_type = mimetypes.guess_type(file_path_str)[0]
    if mime_type and mime_type.startswith('image/'):
        is_image = True
    
    # Track file operations in memory
    if file_path_str not in FILE_OPERATIONS:
        FILE_OPERATIONS[file_path_str] = {
            "operations": [],
            "last_updated": timestamp,
            "extension": extension,
            "is_image": is_image,
            "mime_type": mime_type
        }
    
    # Update the in-memory tracking
    FILE_OPERATIONS[file_path_str]["operations"].append({
        "timestamp": timestamp,
        "operation": operation
    })
    FILE_OPERATIONS[file_path_str]["last_updated"] = timestamp
    
    # Load existing log data or create a new one
    log_data = {
        "files": {}
    }
    
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            # If the log file is corrupted, start fresh
            log_data = {"files": {}}
    
    # Create or update the file entry in the log
    if file_path_str not in log_data["files"]:
        log_data["files"][file_path_str] = {
            "operations": [],
            "metadata": {},
            "content": None,
            "extension": extension,
            "is_image": is_image,
            "mime_type": mime_type,
            "last_updated": timestamp
        }
    
    # Add the operation to the log
    log_data["files"][file_path_str]["operations"].append({
        "timestamp": timestamp,
        "operation": operation
    })
    
    # Update the metadata if provided
    if metadata:
        # Ensure we have a metadata dictionary
        if "metadata" not in log_data["files"][file_path_str]:
            log_data["files"][file_path_str]["metadata"] = {}
            
        # Update with new metadata
        log_data["files"][file_path_str]["metadata"].update(metadata)
    
    # Store the content if provided, otherwise try to read it from the file
    file_content = content
    
    try:
        # Only try to read the file if it exists and content wasn't provided
        if file_content is None and file_path.exists() and file_path.is_file():
            try:
                # Handle different file types appropriately
                if is_image:
                    # For images, store base64 encoded content
                    with open(file_path, 'rb') as f:
                        img_content = f.read()
                        log_data["files"][file_path_str]["content"] = base64.b64encode(img_content).decode('utf-8')
                        # Add file size to metadata
                        if "metadata" not in log_data["files"][file_path_str]:
                            log_data["files"][file_path_str]["metadata"] = {}
                        log_data["files"][file_path_str]["metadata"]["size"] = len(img_content)
                
                elif extension in ['.py', '.js', '.html', '.css', '.json', '.md']:
                    # For code and text files, store as text
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                        log_data["files"][file_path_str]["content"] = text_content
                
                else:
                    # For other binary files, store base64 encoded
                    with open(file_path, 'rb') as f:
                        bin_content = f.read()
                        log_data["files"][file_path_str]["content"] = base64.b64encode(bin_content).decode('utf-8')
                        # Add file size to metadata
                        if "metadata" not in log_data["files"][file_path_str]:
                            log_data["files"][file_path_str]["metadata"] = {}
                        log_data["files"][file_path_str]["metadata"]["size"] = len(bin_content)
            
            except Exception as read_error:
                print(f"Error reading file content: {read_error}")
                # Don't fail the entire operation, just log the error
                if "metadata" not in log_data["files"][file_path_str]:
                    log_data["files"][file_path_str]["metadata"] = {}
                log_data["files"][file_path_str]["metadata"]["read_error"] = str(read_error)
        
        elif file_content is not None:
            # Use the provided content
            log_data["files"][file_path_str]["content"] = file_content
    
    except Exception as e:
        print(f"Error processing file content: {e}")
        # Don't fail the entire operation, just log the error
        if "metadata" not in log_data["files"][file_path_str]:
            log_data["files"][file_path_str]["metadata"] = {}
        log_data["files"][file_path_str]["metadata"]["processing_error"] = str(e)
    
    # Update last_updated timestamp
    log_data["files"][file_path_str]["last_updated"] = timestamp
    
    # Write the updated log data back to the log file
    try:
        with open(LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as write_error:
        print(f"Error writing to log file: {write_error}")

def aggregate_file_states() -> str:
    """
    Collect information about all tracked files and their current state.
    
    Returns:
        A formatted string with information about all files.
    """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    if not LOG_FILE.exists():
        return "No files have been tracked yet."
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            log_data = json.loads(f.read())
    except (json.JSONDecodeError, FileNotFoundError):
        return "Error reading log file."
    
    if not log_data:
        return "No files have been tracked yet."
    
    # Group files by type
    image_files = []
    code_files = []
    text_files = []
    other_files = []
    
    for file_path, file_info in log_data.items():
        file_type = file_info.get("file_type", "other")
        
        # Get the Docker path for display
        docker_path = file_info.get("docker_path", convert_to_docker_path(file_path))
        
        # Sort operations by timestamp to get the latest state
        operations = sorted(file_info.get("operations", []), 
                           key=lambda x: x.get("timestamp", ""), 
                           reverse=True)
        
        latest_operation = operations[0] if operations else {"operation": "unknown"}
        
        if file_type == "image":
            image_metadata = file_info.get("image_metadata", {})
            image_files.append({
                "path": docker_path,
                "operation": latest_operation.get("operation"),
                "prompt": image_metadata.get("prompt", "No prompt available"),
                "dimensions": image_metadata.get("dimensions", "Unknown"),
                "created_at": image_metadata.get("created_at", file_info.get("created_at", "Unknown"))
            })
        elif file_type == "code":
            code_files.append({
                "path": docker_path,
                "operation": latest_operation.get("operation"),
                "content": file_info.get("content", ""),
                "skeleton": file_info.get("skeleton", "No skeleton available")
            })
        elif file_type == "text":
            text_files.append({
                "path": docker_path,
                "operation": latest_operation.get("operation"),
                "content": file_info.get("content", "")
            })
        else:
            basic_info = file_info.get("basic_info", {})
            other_files.append({
                "path": docker_path,
                "operation": latest_operation.get("operation"),
                "mime_type": basic_info.get("mime_type", "Unknown"),
                "size": basic_info.get("size", 0)
            })
    
    # Format the output
    output = []
    
    if image_files:
        output.append("## Image Files")
        for img in image_files:
            output.append(f"### {img['path']}")
            output.append(f"- **Operation**: {img['operation']}")
            output.append(f"- **Created**: {img['created_at']}")
            output.append(f"- **Prompt**: {img['prompt']}")
            output.append(f"- **Dimensions**: {img['dimensions']}")
            output.append("")
    
    if code_files:
        output.append("## Code Files")
        for code in code_files:
            output.append(f"### {code['path']}")
            output.append(f"- **Operation**: {code['operation']}")
            
            # Add syntax highlighting based on file extension
            extension = Path(code['path']).suffix.lower()
            lang = get_language_from_extension(extension)
            
            output.append(f"- **Structure**:")
            output.append(f"```{lang}")
            output.append(code['skeleton'])
            output.append("```")
            
            output.append(f"- **Content**:")
            output.append(f"```{lang}")
            output.append(code['content'])
            output.append("```")
            output.append("")
    
    if text_files:
        output.append("## Text Files")
        for text in text_files:
            output.append(f"### {text['path']}")
            output.append(f"- **Operation**: {text['operation']}")
            
            # Add syntax highlighting based on file extension
            extension = Path(text['path']).suffix.lower()
            lang = get_language_from_extension(extension)
            
            output.append(f"- **Content**:")
            output.append(f"```{lang}")
            output.append(text['content'])
            output.append("```")
            output.append("")
    
    if other_files:
        output.append("## Other Files")
        for other in other_files:
            output.append(f"### {other['path']}")
            output.append(f"- **Operation**: {other['operation']}")
            output.append(f"- **MIME Type**: {other['mime_type']}")
            output.append(f"- **Size**: {other['size']} bytes")
            output.append("")
    
    return "\n".join(output)

def list_images_created() -> str:
    """
    List all images that have been created.
    
    Returns:
        A formatted string with information about all images.
    """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    if not LOG_FILE.exists():
        return "No images have been created yet."
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            log_data = json.loads(f.read())
    except (json.JSONDecodeError, FileNotFoundError):
        return "Error reading log file."
    
    image_files = []
    
    for file_path, file_info in log_data.items():
        if file_info.get("file_type") == "image":
            # Get the Docker path for display
            docker_path = file_info.get("docker_path", convert_to_docker_path(file_path))
            
            image_metadata = file_info.get("image_metadata", {})
            image_files.append({
                "path": docker_path,
                "prompt": image_metadata.get("prompt", "No prompt available"),
                "dimensions": image_metadata.get("dimensions", "Unknown"),
                "created_at": image_metadata.get("created_at", file_info.get("created_at", "Unknown"))
            })
    
    if not image_files:
        return "No images have been created yet."
    
    output = ["## Generated Images"]
    
    for img in image_files:
        output.append(f"### {img['path']}")
        output.append(f"- **Created**: {img['created_at']}")
        output.append(f"- **Prompt**: {img['prompt']}")
        output.append(f"- **Dimensions**: {img['dimensions']}")
        output.append("")
    
    return "\n".join(output)

def extract_files_content() -> str:
    """
    Extract structured file contents for use in prompts.
    This function reads the log file and extracts the file contents in a structured way.
    
    Returns:
        A markdown-formatted string of file contents
    """
    output = []
    try:
        # Ensure log file exists and has valid structure
        if not os.path.exists(LOG_FILE):
            print(f"Log file not found: {LOG_FILE}")
            # Initialize a new log file so future operations work
            with open(LOG_FILE, 'w') as f:
                json.dump({"files": {}}, f)
            return "No files have been created yet."
            
        # Read log file with robust error handling
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Log file exists but contains invalid JSON: {LOG_FILE}")
            # Reset the log file with valid JSON
            with open(LOG_FILE, 'w') as f:
                json.dump({"files": {}}, f)
            return "Error reading log file. Log file has been reset."
        except Exception as e:
            print(f"Unexpected error reading log file: {e}")
            return f"Error reading log file: {str(e)}"
            
        # Validate log structure
        if not isinstance(log_data, dict) or "files" not in log_data:
            print("Log file has invalid format (missing 'files' key)")
            # Fix the format
            log_data = {"files": {}}
            with open(LOG_FILE, 'w') as f:
                json.dump(log_data, f)
            return "Error reading log file. Log file has been reset with correct structure."
            
        # If there are no files, return early
        if not log_data.get("files"):
            return "No files have been created yet."
            
        # Group files by type for better organization
        code_files = []
        text_files = []
        other_files = []
        
        # Process each file in the log
        for file_path, file_data in log_data.get("files", {}).items():
            try:
                # Skip files that have been deleted (last operation is 'delete')
                operations = file_data.get("operations", [])
                if operations and operations[-1].get("operation") == "delete":
                    continue
                    
                # Get content and metadata
                content = file_data.get("content")
                if content is None:
                    continue
                    
                extension = file_data.get("extension", "").lower()
                is_image = file_data.get("is_image", False)
                mime_type = file_data.get("mime_type", "")
                
                # Skip images for code context
                if is_image or (mime_type and mime_type.startswith("image/")):
                    continue
                    
                # Categorize file by type
                if extension in ['.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs']:
                    code_files.append({
                        "path": file_path,
                        "content": content,
                        "operation": operations[-1].get("operation") if operations else "unknown",
                        "extension": extension
                    })
                elif extension in ['.json', '.md', '.yaml', '.yml', '.toml', '.ini', '.config', '.txt', '.csv']:
                    text_files.append({
                        "path": file_path,
                        "content": content,
                        "operation": operations[-1].get("operation") if operations else "unknown",
                        "extension": extension
                    })
                else:
                    # Skip binary files for code context
                    file_size = len(content) if isinstance(content, str) else 0
                    other_files.append({
                        "path": file_path,
                        "operation": operations[-1].get("operation") if operations else "unknown",
                        "extension": extension,
                        "mime_type": mime_type,
                        "size": file_size
                    })
            except Exception as file_error:
                print(f"Error processing file {file_path}: {file_error}")
                # Continue to next file rather than failing entirely
                
        # Format the output
        output.append("# All Code Files")
        
        if code_files:
            for code in code_files:
                try:
                    output.append(f"## {code['path']}")
                    output.append(f"```{get_language_from_extension(code['extension'])}")
                    output.append(code['content'])
                    output.append("```")
                    output.append("")  # Empty line for readability
                except Exception as format_error:
                    print(f"Error formatting code file {code.get('path')}: {format_error}")
                    # Add a simpler version without failing
                    output.append(f"## {code.get('path', 'Unknown file')}")
                    output.append("```")
                    output.append("Error displaying file content")
                    output.append("```")
                    output.append("")
        else:
            output.append("No code files have been created yet.")
            
        if text_files:
            output.append("# Text Files")
            for text in text_files:
                try:
                    output.append(f"## {text['path']}")
                    output.append(f"```{get_language_from_extension(text['extension'])}")
                    output.append(text['content'])
                    output.append("```")
                    output.append("")
                except Exception as format_error:
                    print(f"Error formatting text file {text.get('path')}: {format_error}")
                    output.append(f"## {text.get('path', 'Unknown file')}")
                    output.append("```")
                    output.append("Error displaying file content")
                    output.append("```")
                    output.append("")
                    
        return "\n".join(output)
        
    except Exception as e:
        print(f"Critical error in extract_files_content: {e}")
        return "Error reading log file. Please check application logs for details."

def extract_code_skeleton(source_code: Union[str, Path]) -> str:
    """
    Extract a code skeleton from existing Python code.
    
    This function takes Python code and returns just the structure: imports,
    class definitions, method/function signatures, and docstrings, with
    implementations replaced by 'pass' statements.
    
    Args:
        source_code: Either a path to a Python file or a string containing Python code
        
    Returns:
        str: The extracted code skeleton
    """
    # Load the source code
    if isinstance(source_code, (str, Path)) and Path(source_code).exists():
        with open(source_code, 'r', encoding='utf-8') as file:
            code_str = file.read()
    else:
        code_str = str(source_code)
    
    # Parse the code into an AST
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"# Error parsing code: {e}\n{code_str}"
    
    # Extract imports
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports.append(f"import {name.name}" + 
                               (f" as {name.asname}" if name.asname else ""))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = ", ".join(name.name + (f" as {name.asname}" if name.asname else "") 
                             for name in node.names)
            imports.append(f"from {module} import {names}")
    
    # Helper function to handle complex attributes
    def format_attribute(node):
        """Helper function to recursively format attribute expressions"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{format_attribute(node.value)}.{node.attr}"
        # Add support for ast.Subscript nodes (like List[int])
        elif isinstance(node, ast.Subscript):
            # Use ast.unparse for Python 3.9+ or manual handling for earlier versions
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            else:
                # Simplified handling for common cases
                if isinstance(node.value, ast.Name):
                    base = node.value.id
                else:
                    base = format_attribute(node.value)
                # Simple handling for slice
                if isinstance(node.slice, ast.Index) and hasattr(node.slice, 'value'):
                    if isinstance(node.slice.value, ast.Name):
                        return f"{base}[{node.slice.value.id}]"
                    else:
                        return f"{base}[...]"  # Fallback for complex slices
                return f"{base}[...]"  # Fallback for complex cases
        else:
            # Fallback for other node types - use ast.unparse if available
            if hasattr(ast, 'unparse'):
                return ast.unparse(node)
            return str(node)
    
    # Get docstrings and function/class signatures
    class CodeSkeletonVisitor(ast.NodeVisitor):
        def __init__(self):
            self.skeleton = []
            self.indent_level = 0
            self.imports = []
        
        def visit_Import(self, node):
            # Already handled above
            pass
            
        def visit_ImportFrom(self, node):
            # Already handled above
            pass
            
        def visit_ClassDef(self, node):
            # Extract class definition with inheritance
            bases = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    bases.append(base.id)
                elif isinstance(base, ast.Attribute):
                    # Use the helper function to handle nested attributes
                    bases.append(format_attribute(base))
                else:
                    # Fallback for other complex cases
                    if hasattr(ast, 'unparse'):
                        bases.append(ast.unparse(base))
                    else:
                        bases.append("...")
            
            class_def = f"class {node.name}"
            if bases:
                class_def += f"({', '.join(bases)})"
            class_def += ":"
            
            # Add class definition
            self.skeleton.append("\n" + "    " * self.indent_level + class_def)
            
            # Add docstring if it exists
            docstring = ast.get_docstring(node)
            if docstring:
                doc_lines = docstring.split('\n')
                if len(doc_lines) == 1:
                    self.skeleton.append("    " * (self.indent_level + 1) + 
                                        f'"""{docstring}"""')
                else:
                    self.skeleton.append("    " * (self.indent_level + 1) + '"""')
                    for line in doc_lines:
                        self.skeleton.append("    " * (self.indent_level + 1) + line)
                    self.skeleton.append("    " * (self.indent_level + 1) + '"""')
            
            # Increment indent for class members
            self.indent_level += 1
            
            # Visit all class members
            for item in node.body:
                if not isinstance(item, ast.Expr) or not isinstance(item.value, ast.Str):
                    self.visit(item)
            
            # If no members were added, add a pass statement
            if len(self.skeleton) > 0 and not self.skeleton[-1].strip().startswith("def "):
                if "pass" not in self.skeleton[-1]:
                    self.skeleton.append("    " * self.indent_level + "pass")
            
            # Restore indent
            self.indent_level -= 1
            
        def visit_FunctionDef(self, node):
            # Extract function signature
            args = []
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
            
            # Process regular arguments
            for i, arg in enumerate(node.args.args):
                arg_str = arg.arg
                # Add type annotation if available
                if arg.annotation:
                    # Use the helper function to handle complex types
                    if hasattr(ast, 'unparse'):
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    else:
                        if isinstance(arg.annotation, ast.Name):
                            arg_str += f": {arg.annotation.id}"
                        elif isinstance(arg.annotation, ast.Attribute):
                            arg_str += f": {format_attribute(arg.annotation)}"
                        elif isinstance(arg.annotation, ast.Subscript):
                            arg_str += f": {format_attribute(arg.annotation)}"
                        else:
                            arg_str += f": ..."  # Fallback for complex annotations
                
                # Add default value if available
                if defaults[i] is not None:
                    if hasattr(ast, 'unparse'):
                        arg_str += f" = {ast.unparse(defaults[i])}"
                    else:
                        # Simplified handling for common default values
                        if isinstance(defaults[i], (ast.Str, ast.Num, ast.NameConstant)):
                            arg_str += f" = {ast.literal_eval(defaults[i])}"
                        elif isinstance(defaults[i], ast.Name):
                            arg_str += f" = {defaults[i].id}"
                        elif isinstance(defaults[i], ast.Attribute):
                            arg_str += f" = {format_attribute(defaults[i])}"
                        else:
                            arg_str += f" = ..."  # Fallback for complex defaults
                
                args.append(arg_str)
            
            # Handle *args
            if node.args.vararg:
                args.append(f"*{node.args.vararg.arg}")
            
            # Handle keyword-only args
            if node.args.kwonlyargs:
                if not node.args.vararg:
                    args.append("*")
                for i, kwarg in enumerate(node.args.kwonlyargs):
                    kw_str = kwarg.arg
                    if kwarg.annotation:
                        if hasattr(ast, 'unparse'):
                            kw_str += f": {ast.unparse(kwarg.annotation)}"
                        else:
                            kw_str += f": {format_attribute(kwarg.annotation)}"
                    if i < len(node.args.kw_defaults) and node.args.kw_defaults[i] is not None:
                        if hasattr(ast, 'unparse'):
                            kw_str += f" = {ast.unparse(node.args.kw_defaults[i])}"
                        else:
                            kw_str += f" = ..."  # Fallback for complex defaults
                    args.append(kw_str)
            
            # Handle **kwargs
            if node.args.kwarg:
                args.append(f"**{node.args.kwarg.arg}")
            
            # Build function signature
            func_def = f"def {node.name}({', '.join(args)})"
            
            # Add return type if specified
            if node.returns:
                if hasattr(ast, 'unparse'):
                    func_def += f" -> {ast.unparse(node.returns)}"
                else:
                    func_def += f" -> {format_attribute(node.returns)}"
            
            func_def += ":"
            
            # Add function definition
            self.skeleton.append("\n" + "    " * self.indent_level + func_def)
            
            # Add docstring if it exists
            docstring = ast.get_docstring(node)
            if docstring:
                doc_lines = docstring.split('\n')
                if len(doc_lines) == 1:
                    self.skeleton.append("    " * (self.indent_level + 1) + 
                                       f'"""{docstring}"""')
                else:
                    self.skeleton.append("    " * (self.indent_level + 1) + '"""')
                    for line in doc_lines:
                        self.skeleton.append("    " * (self.indent_level + 1) + line)
                    self.skeleton.append("    " * (self.indent_level + 1) + '"""')
            
            # Add pass statement in place of the function body
            self.skeleton.append("    " * (self.indent_level + 1) + "pass")
    
    # Run the visitor on the AST
    visitor = CodeSkeletonVisitor()
    visitor.visit(tree)
    
    # Combine imports and code skeleton
    result = []
    
    # Add all imports first
    if imports:
        result.extend(imports)
        result.append("")  # Add a blank line after imports
    
    # Add the rest of the code skeleton
    result.extend(visitor.skeleton)
    
    return "\n".join(result)

def get_all_current_code() -> str:
    """
    Returns all the current code in the project as a string.
    This function is used to provide context about the existing code to the LLM.
    
    Returns:
        A string with all the current code.
    """
    try:
        # Ensure log file exists
        if not os.path.exists(LOG_FILE):
            print(f"Log file not found: {LOG_FILE}")
            # Initialize a new log file so future operations work
            with open(LOG_FILE, 'w') as f:
                json.dump({"files": {}}, f)
            return "No code has been written yet."
            
        # Load log data with robust error handling
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Log file exists but contains invalid JSON: {LOG_FILE}")
            # Reset the log file with valid JSON
            with open(LOG_FILE, 'w') as f:
                json.dump({"files": {}}, f)
            return "Error reading log file. Log file has been reset."
        except Exception as e:
            print(f"Unexpected error reading log file: {e}")
            return f"Error reading log file: {str(e)}"
            
        # Validate log structure
        if not isinstance(log_data, dict) or "files" not in log_data:
            print("Log file has invalid format (missing 'files' key)")
            # Fix the format
            log_data = {"files": {}}
            with open(LOG_FILE, 'w') as f:
                json.dump(log_data, f)
            return "Error reading log file. Log file has been reset with correct structure."
            
        output = []
        code_files = []

        # Process each file in the log
        for file_path, file_data in log_data.get("files", {}).items():
            try:
                # Skip files that have been deleted (last operation is 'delete')
                operations = file_data.get("operations", [])
                if operations and operations[-1].get("operation") == "delete":
                    continue
                
                # Get content and metadata
                content = file_data.get("content")
                if content is None:
                    continue
                
                # Skip images and binary files
                is_image = file_data.get("is_image", False)
                mime_type = file_data.get("mime_type", "")
                extension = file_data.get("extension", "").lower()
                
                if is_image or (mime_type and mime_type.startswith("image/")):
                    continue
                    
                # Only include code files
                if extension in ['.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs']:
                    code_files.append({
                        "path": file_path,
                        "content": content,
                        "extension": extension
                    })
            except Exception as file_error:
                print(f"Error processing file {file_path}: {file_error}")
                # Continue processing other files
        
        # Sort files by path for consistent output
        code_files.sort(key=lambda x: x["path"])
        
        # Format the output
        output.append("# Code\n")
        
        if code_files:
            for code in code_files:
                try:
                    output.append(f"## {code['path']}")
                    lang = get_language_from_extension(code['extension'])
                    output.append(f"```{lang}")
                    output.append(code['content'])
                    output.append("```\n")
                except Exception as format_error:
                    print(f"Error formatting code file {code.get('path')}: {format_error}")
                    # Add a simpler version without failing
                    output.append(f"## {code.get('path', 'Unknown file')}")
                    output.append("```")
                    output.append("Error displaying file content")
                    output.append("```\n")
        else:
            output.append("No code files have been created yet.\n")
        
        # Return the formatted output (This was missing)
        return "\n".join(output)

    except Exception as e:
        print(f"Critical error in get_all_current_code: {e}")
        return "Error reading code files. Please check application logs for details."

def get_all_current_skeleton() -> str:
    """
    Get the skeleton of all Python code files.
    
    Returns:
        A formatted string with the skeleton of all Python code files.
    """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    if not LOG_FILE.exists():
        return "No Python files have been tracked yet."
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return "Error reading log file."
    
    if not log_data or not log_data.get("files"):
        return "No Python files have been tracked yet."
    
    output = ["# All Python File Skeletons"]
    
    for file_path, file_info in log_data.get("files", {}).items():
        # Skip files that have been deleted (last operation is 'delete')
        operations = file_info.get("operations", [])
        if operations and operations[-1].get("operation") == "delete":
            continue
            
        # Only process Python files
        extension = file_info.get("extension", "").lower()
        if extension != '.py':
            continue
            
        # Get Docker path for display
        docker_path = convert_to_docker_path(file_path)
        
        # Look for skeleton in metadata
        metadata = file_info.get("metadata", {})
        skeleton = metadata.get("skeleton", "")
        
        # If no skeleton in metadata, try to extract it from the content
        if not skeleton and file_info.get("content"):
            try:
                skeleton = extract_code_skeleton(file_info.get("content", ""))
            except Exception as e:
                print(f"Error extracting skeleton from {file_path}: {e}")
                skeleton = "# Failed to extract skeleton"
        
        if skeleton:
            # Add file header
            output.append(f"## {docker_path}")
            
            # Add skeleton with syntax highlighting
            output.append("```python")
            output.append(skeleton)
            output.append("```")
            output.append("")
    
    return "\n".join(output)

def get_language_from_extension(extension: str) -> str:
    """
    Map file extensions to programming languages for syntax highlighting.

    Args:
        extension: The file extension (e.g., '.py', '.js')

    Returns:
        The corresponding language name for syntax highlighting.
    """
    extension = extension.lower()
    mapping = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'javascript',
        '.ts': 'typescript',
        '.tsx': 'typescript',
        '.html': 'html',
        '.css': 'css',
        '.json': 'json',
        '.md': 'markdown',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.sh': 'bash',
        '.rb': 'ruby',
        '.go': 'go',
        '.php': 'php',
        '.rs': 'rust',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.sql': 'sql'
    }
    return mapping.get(extension, '')

def test_file_logging():
    """Test function to verify file logging functionality."""
    print("Testing file logging functionality...")
    
    # Create a test file
    test_path = Path(__file__).parent.parent / "test_logging.txt"
    
    try:
        # Create the file directly
        with open(test_path, 'w') as f:
            f.write("Test content")
        print(f"Successfully created test file at {test_path}")
        
        # Log the file operation
        try:
            log_file_operation(
                file_path=test_path,
                operation="create",
                content="Test content",
                metadata={"test": "metadata"}
            )
            print("Successfully logged file operation")
        except Exception as e:
            print(f"Error logging file operation: {e}")
            
        # Verify log file was updated
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                log_data = json.load(f)
                print(f"Log file content: {json.dumps(log_data, indent=2)}")
        else:
            print(f"Log file not found at {LOG_FILE}")
            
    except Exception as e:
        print(f"Error in test: {e}")
    
    return "Test completed"

# Uncomment to run test when module is imported
# print(test_file_logging())

def should_skip_for_zip(path):
    """
    Determine if a file or directory should be skipped when creating a ZIP file.
    
    Args:
        path: Path to check
        
    Returns:
        bool: True if the path should be skipped, False otherwise
    """
    path_str = str(path).lower()
    
    # Skip virtual environment files and directories
    if '.venv' in path_str:
        # On Windows, particularly skip Linux-style virtual env paths
        if os.name == 'nt' and ('bin/' in path_str or 'lib/' in path_str):
            return True
    
    # Skip common directories not needed in the ZIP
    dirs_to_skip = [
        '__pycache__', '.git', '.idea', '.vscode', 
        'node_modules', '.pytest_cache'
    ]
    
    return any(skip_dir in path_str for skip_dir in dirs_to_skip)
