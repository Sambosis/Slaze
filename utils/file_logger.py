import json
import datetime
from pathlib import Path
from config import get_constant
from icecream import ic

import ast
import inspect
import re
from typing import Union, Optional

def aggregate_file_states() -> str:
    """
    Aggregate file states. Here we reuse extract_files_content.
    """
    code_contents = get_all_current_skeleton()
    images_created = list_images_created()
    output = f"Here is the Skelatons of code created (Use Bash to see full code):\n{code_contents}\n\nImages Created:\n{images_created}"
    return output

def list_images_created() -> str:
    """ List all image files from the log """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    if not LOG_FILE.exists():
        return "No images created yet"
    
    # Initialize output
    image_paths = []
    image_suffixes = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    
    # Read log file and check each line for image paths
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if any(suffix in line.lower() for suffix in image_suffixes):
                # Extract the path between quotes if present
                if '"' in line:
                    path = line.split('"')[1]  # Get text between first pair of quotes
                    image_paths.append(path)
    
    return "\n".join(image_paths) if image_paths else "No images found"

def extract_files_content() -> str:
    """ Returns the complete contents of all files with special handling for images """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    # Create the file if it does not exist
    if not LOG_FILE.exists():
        LOG_FILE.touch()
    # Read the log file
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        # Handle case of empty file
        if f.read() == "":
            return "No code yet"
        f.seek(0)  # Reset file pointer to the beginning
        logs = json.loads(f.read())
    
    # Initialize output string
    output = []
    
    # Process each file in the logs
    for filepath in logs.keys():
        try:
            # Convert Windows path to Path object
            path = Path(filepath)
            
            # Skip if file doesn't exist
            if not path.exists():
                continue
            
            # Add file header
            output.append(f"# filepath: {filepath}")
            
            # Check if it's an image file
            if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                output.append("[Image File]")
                output.append(f"Created: {logs[filepath]['created_at']}")
                output.append(f"Last Operation: {logs[filepath]['operations'][-1]['operation']}")
            else:
                # For non-image files, include the content
                content = path.read_text(encoding='utf-8')
                output.append(content)
            
            output.append("\n" + "=" * 80 + "\n")  # Separator between files
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue
    
    # Combine all content
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
    Returns the complete contents of all files from the log file.
    Reads content directly from the log instead of opening each file again.

    Returns:
        str: Combined content of all files
    """
    try:
        LOG_FILE = Path(get_constant("LOG_FILE"))
        # Create the file if it does not exist
        if not LOG_FILE.exists():
            LOG_FILE.touch()
        # Read the log file
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            # Handle case of empty file
            if f.read() == "":
                return "No code yet"
            f.seek(0)  # Reset file pointer to the beginning
            logs = json.loads(f.read())

        # Initialize output string
        output = []

        # Process each file in the logs
        for filepath in logs.keys():
            try:
                # Add file header with full path
                output.append(f"# filepath: {filepath}")

                # Check if it's an image file
                path = Path(filepath)
                if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".gif", ".bmp"]:
                    output.append("[Image File]")
                else:
                    # For non-image files, include the content from our log
                    if "content" in logs[filepath]:
                        content = logs[filepath]["content"]
                        output.append(content)
                    else:
                        output.append("# Content not available in log")

                output.append("\n" + "=" * 80 + "\n")  # Separator between files

            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                continue

        # Combine all content
        return "\n".join(output)

    except Exception as e:
        return f"Error reading log file: {str(e)}"


def get_all_current_skeleton() -> str:
    """
    Returns the code skeleton of all Python files that have been logged.
    Uses extract_code_skeleton() to generate a structure-only view of each file.

    Returns:
        str: Combined code skeletons of all Python files
    """
    try:
        LOG_FILE = Path(get_constant("LOG_FILE"))
        # Create the file if it does not exist
        if not LOG_FILE.exists():
            LOG_FILE.touch()
        # Read the log file
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            # Handle case of empty file
            if f.read() == "":
                return "No code yet"
            f.seek(0)  # Reset file pointer to the beginning
            logs = json.loads(f.read())

        # Initialize output string
        output = []

        # Process each file in the logs
        for filepath in logs.keys():
            try:
                # Convert path string to Path object
                path = Path(filepath)

                # Skip if file doesn't exist
                if not path.exists():
                    continue

                # Add file header with full path
                output.append(f"# filepath: {filepath}")

                # Only extract skeletons for Python files
                if path.suffix.lower() == ".py":
                    # Extract and add skeleton
                    skeleton = extract_code_skeleton(path)
                    output.append(skeleton)
                else:
                    # For non-Python files, just note the file type
                    output.append(
                        f"[{path.suffix.upper()} File - No skeleton available]"
                    )

                output.append("\n" + "=" * 80 + "\n")  # Separator between files

            except Exception as e:
                print(f"Error processing skeleton for {filepath}: {str(e)}")
                continue

        # Combine all content
        return "\n".join(output)

    except Exception as e:
        return f"Error reading log file: {str(e)}"


def log_file_operation(path: Path, operation: str) -> None:
    """
    Log file information with simplified structure - preserving full path, content, and skeleton.
    Handles files with the same name in different directories correctly.

    Args:
        path (Path): Path to the file being operated on
        operation (str): Type of operation (e.g., 'create', 'modify', 'delete')
    """
    try:
        LOG_FILE = Path(get_constant("LOG_FILE"))

        # Initialize default log structure
        logs = {}

        # Read existing logs if file exists and has content
        if LOG_FILE.exists():
            content = LOG_FILE.read_text(encoding="utf-8").strip()
            if content:
                try:
                    logs = json.loads(content)
                except json.JSONDecodeError:
                    logs = {}

        path_str = str(path)

        # Skip if file doesn't exist (except for delete operations)
        if operation != "delete" and not path.exists():
            ic(f"Skipping log operation for non-existent file: {path_str}")
            return

        # Update the log entry with new structure
        if path.exists():
            # Read file content
            file_content = (
                path.read_text(encoding="utf-8") if path.is_file() else "[Not a file]"
            )

            # Get skeleton for Python files
            skeleton = ""
            if path.suffix.lower() == ".py":
                try:
                    skeleton = extract_code_skeleton(path)
                except Exception as e:
                    skeleton = f"# Error extracting skeleton: {str(e)}"

            # Update log with the new structure - store full path instead of just filename
            logs[path_str] = {
                "path": path_str,  # Store the full path
                "content": file_content,
                "skeleton": skeleton if path.suffix.lower() == ".py" else "",
            }
        elif operation == "delete" and path_str in logs:
            # Remove the file from logs if it was deleted
            del logs[path_str]

        # Write updated logs
        LOG_FILE.write_text(json.dumps(logs, indent=2), encoding="utf-8")

    except Exception as e:
        ic(f"Failed to log file operation: {str(e)}")
