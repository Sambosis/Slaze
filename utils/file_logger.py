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

# Track file operations
file_operations = []
tracked_files = set()
file_contents = {}

def convert_to_docker_path(path: Union[str, Path]) -> str:
    """
    Convert a host system path to a Docker container path.
    
    Args:
        path: The host system path
        
    Returns:
        The equivalent path in the Docker container
    """
    # Convert path to string if it's a Path object
    if isinstance(path, Path):
        path = str(path)
        
    # Get project directory constants
    try:
        project_dir = get_constant('PROJECT_DIR')
        # Ensure project_dir is a string
        if isinstance(project_dir, Path):
            project_dir = str(project_dir)
            
        # Try to get docker project dir, but if it doesn't exist, use a default
        try:
            docker_project_dir = get_constant('DOCKER_PROJECT_DIR')
        except:
            docker_project_dir = '/app'
            
        # Ensure docker_project_dir is a string
        if isinstance(docker_project_dir, Path):
            docker_project_dir = str(docker_project_dir)
    except:
        # If constants aren't available, return the original path
        return path
        
    # Convert the path if it starts with the project directory
    if path.startswith(project_dir):
        return path.replace(project_dir, docker_project_dir)
    return path

def log_file_operation(file_path: Path, operation: str, content: str = None, metadata: dict = None):
    """
    Log a file operation (create, update, delete) with enhanced metadata handling.
    
    Args:
        file_path: Path to the file
        operation: Type of operation ('create', 'update', 'delete')
        content: Optional content for the file
        metadata: Optional dictionary containing additional metadata (e.g., image generation prompt)
    """
    # Ensure file_path is a Path object
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
        
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extension = file_path.suffix.lower() if file_path.suffix else ""
    
    # Determine if the file is an image
    is_image = extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp']
    mime_type = mimetypes.guess_type(str(file_path))[0]
    if mime_type and mime_type.startswith('image/'):
        is_image = True
    
    # Convert file_path to string for storage
    file_path_str = str(file_path)
    
    # Get Docker path
    docker_path = convert_to_docker_path(file_path)
    
    # Create operation data for in-memory tracking
    operation_data = {
        'path': file_path_str,
        'docker_path': docker_path,
        'operation': operation,
        'timestamp': timestamp,
        'is_image': is_image,
        'metadata': metadata or {}
    }
    
    # Update in-memory tracking
    if operation in ['create', 'update'] and content is not None:
        file_contents[file_path_str] = content
    elif operation == 'delete' and file_path_str in file_contents:
        del file_contents[file_path_str]
    
    file_operations.append(operation_data)
    tracked_files.add(file_path_str)
    
    # Now update the log file
    LOG_FILE = Path(get_constant('LOG_FILE'))
    log_data = {}
    
    # Create directory if it doesn't exist
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing log if it exists
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                log_content = f.read()
                if log_content:
                    log_data = json.loads(log_content)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading log file: {e}")
            log_data = {}
    
    # Create or update file entry in log
    if file_path_str not in log_data:
        log_data[file_path_str] = {
            "created_at": timestamp,
            "operations": [],
            "docker_path": docker_path,
            "file_type": "image" if is_image else "code" if extension in ['.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs'] else "text" if extension in ['.json', '.md', '.yaml', '.yml', '.toml', '.ini', '.config', '.txt', '.csv'] else "other"
        }
    
    # Add this operation
    log_data[file_path_str]["operations"].append({
        "operation": operation,
        "timestamp": timestamp
    })
    
    # Handle different file types
    if operation in ['create', 'update']:
        try:
            # If content is not provided but the file exists, read it directly from the file
            # Don't use the log file content as the file content
            file_content = content
            if file_content is None and file_path.exists() and file_path.is_file():
                try:
                    if file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                        with open(file_path, 'rb') as f:
                            content = base64.b64encode(f.read()).decode('utf-8')
                            metadata['content'] = content
                    elif file_path.suffix in ['.py', '.js', '.html', '.css', '.json', '.md']:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            metadata['content'] = content
                    else:
                        with open(file_path, 'rb') as f:
                            content = base64.b64encode(f.read()).decode('utf-8')
                            metadata['content'] = content
                except Exception as read_error:
                    print(f"Error reading file content: {read_error}")
                    file_content = None
            
            if file_content is not None:
                # Store content based on file type
                if is_image:
                    # For images, store metadata and path
                    log_data[file_path_str]["image_metadata"] = {
                        "path": docker_path,
                        "prompt": metadata.get("prompt", "No prompt available") if metadata else "No prompt available",
                        "dimensions": metadata.get("dimensions", "Unknown") if metadata else "Unknown",
                        "created_at": timestamp
                    }
                elif extension in ['.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs']:
                    # For code files, store both content and skeleton
                    log_data[file_path_str]["content"] = file_content
                    if extension == '.py':
                        try:
                            skeleton = extract_code_skeleton(file_content)
                            log_data[file_path_str]["skeleton"] = skeleton
                        except Exception as e:
                            log_data[file_path_str]["skeleton_error"] = str(e)
                elif extension in ['.json', '.md', '.yaml', '.yml', '.toml', '.ini', '.config', '.txt', '.csv']:
                    # For text files, store full content
                    log_data[file_path_str]["content"] = file_content
                else:
                    # For other files, just store basic info
                    log_data[file_path_str]["basic_info"] = {
                        "path": docker_path,
                        "extension": extension,
                        "mime_type": mime_type,
                        "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    }
        except Exception as e:
            log_data[file_path_str]["read_error"] = str(e)
    
    # Write updated log back to file - make sure we write valid JSON
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2)
    except Exception as write_error:
        print(f"Error writing to log file: {write_error}")
        # Try writing a simplified version if the full version fails
        try:
            # Create a minimal log entry to avoid circular references
            minimal_log = {file_path_str: {"operation": operation, "timestamp": timestamp}}
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(minimal_log, f)
        except:
            # If all else fails, try to reset the log file
            with open(LOG_FILE, 'w', encoding='utf-8') as f:
                f.write("{}")

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
            lang = ""
            if extension == '.py':
                lang = "python"
            elif extension in ['.js', '.jsx']:
                lang = "javascript"
            elif extension in ['.ts', '.tsx']:
                lang = "typescript"
            elif extension == '.html':
                lang = "html"
            elif extension == '.css':
                lang = "css"
            elif extension in ['.java', '.cpp', '.c', '.cs']:
                lang = "java" if extension == '.java' else "cpp" if extension in ['.cpp', '.c'] else "csharp"
            
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
            lang = ""
            if extension == '.json':
                lang = "json"
            elif extension == '.md':
                lang = "markdown"
            elif extension in ['.yaml', '.yml']:
                lang = "yaml"
            elif extension == '.toml':
                lang = "toml"
            
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
    Extract the content of all tracked files.
    
    Returns:
        A formatted string with the content of all files.
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
    
    output = ["# File Contents"]
    
    for file_path, file_info in log_data.items():
        # Skip image files
        if file_info.get("file_type") == "image":
            continue
            
        # Get the Docker path for display
        docker_path = file_info.get("docker_path", convert_to_docker_path(file_path))
        
        # Get content if available
        content = file_info.get("content", "")
        if not content:
            continue
            
        # Add file header
        output.append(f"## {docker_path}")
        
        # Add syntax highlighting based on file extension
        extension = Path(docker_path).suffix.lower()
        lang = ""
        if extension == '.py':
            lang = "python"
        elif extension in ['.js', '.jsx']:
            lang = "javascript"
        elif extension in ['.ts', '.tsx']:
            lang = "typescript"
        elif extension == '.html':
            lang = "html"
        elif extension == '.css':
            lang = "css"
        elif extension == '.json':
            lang = "json"
        elif extension == '.md':
            lang = "markdown"
        elif extension in ['.yaml', '.yml']:
            lang = "yaml"
        elif extension == '.toml':
            lang = "toml"
        elif extension in ['.java', '.cpp', '.c', '.cs']:
            lang = "java" if extension == '.java' else "cpp" if extension in ['.cpp', '.c'] else "csharp"
        
        # Add content with syntax highlighting
        output.append(f"```{lang}")
        output.append(content)
        output.append("```")
        output.append("")
    
    return "\n".join(output)

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
    Get the content of all code files.
    
    Returns:
        A formatted string with the content of all code files.
    """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    if not LOG_FILE.exists():
        return "No code files have been tracked yet."
    
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            log_data = json.loads(f.read())
    except (json.JSONDecodeError, FileNotFoundError):
        return "Error reading log file."
    
    if not log_data:
        return "No code files have been tracked yet."
    
    output = ["# All Code Files"]
    
    for file_path, file_info in log_data.items():
        # Only include code files
        if file_info.get("file_type") != "code":
            continue
            
        # Get the Docker path for display
        docker_path = file_info.get("docker_path", convert_to_docker_path(file_path))
        
        # Get content if available
        content = file_info.get("content", "")
        if not content:
            continue
            
        # Add file header
        output.append(f"## {docker_path}")
        
        # Add syntax highlighting based on file extension
        extension = Path(docker_path).suffix.lower()
        lang = ""
        if extension == '.py':
            lang = "python"
        elif extension in ['.js', '.jsx']:
            lang = "javascript"
        elif extension in ['.ts', '.tsx']:
            lang = "typescript"
        elif extension == '.html':
            lang = "html"
        elif extension == '.css':
            lang = "css"
        elif extension in ['.java', '.cpp', '.c', '.cs']:
            lang = "java" if extension == '.java' else "cpp" if extension in ['.cpp', '.c'] else "csharp"
        
        # Add content with syntax highlighting
        output.append(f"```{lang}")
        output.append(content)
        output.append("```")
        output.append("")
    
    return "\n".join(output)

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
            log_data = json.loads(f.read())
    except (json.JSONDecodeError, FileNotFoundError):
        return "Error reading log file."
    
    if not log_data:
        return "No Python files have been tracked yet."
    
    output = ["# All Python File Skeletons"]
    
    for file_path, file_info in log_data.items():
        # Only include Python files
        if file_info.get("file_type") != "code" or not file_path.endswith('.py'):
            continue
            
        # Get the Docker path for display
        docker_path = file_info.get("docker_path", convert_to_docker_path(file_path))
        
        # Get skeleton if available
        skeleton = file_info.get("skeleton", "")
        if not skeleton:
            continue
            
        # Add file header
        output.append(f"## {docker_path}")
        
        # Add skeleton with syntax highlighting
        output.append("```python")
        output.append(skeleton)
        output.append("```")
        output.append("")
    
    return "\n".join(output)
