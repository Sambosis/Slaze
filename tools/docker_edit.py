## edit.py
import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Literal, get_args, Dict
from anthropic.types.beta import BetaToolTextEditor20241022Param
from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import maybe_truncate
from typing import List, Optional
from icecream import ic
import sys
import ftfy
from rich import print as rr
import datetime
import json
from load_constants import write_to_file, ICECREAM_OUTPUT_FILE
from config import get_constant, set_constant, REPO_DIR, PROJECT_DIR, LOGS_DIR
from utils.file_logger import log_file_operation
from utils.docker_service import DockerService
from loguru import logger as ll

# # Configure logging to a file
# #ll.add(
#     "edit.log",
#     rotation="500 KB",
#     level="DEBUG",
#     format="{time: MM-DD HH:mm} | {level: <8} | {module}.{function}:{line} - {message}",
# )

# Configure icecream for debugging
ic.configureOutput(includeContext=True, outputFunction=write_to_file)

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]
SNIPPET_LINES: int = 4


class DockerEditTool(BaseAnthropicTool):
    description = """
    A cross-platform filesystem editor tool that works inside a Docker container.
    This tool mirrors the functionality of EditTool but uses DockerService for file operations.
    """
    api_type: Literal["custom"] = "custom"
    name: Literal["docker_edit"] = "docker_edit"
    LOG_FILE = Path(get_constant("LOG_FILE"))
    _file_history: dict[Path, list[str]]

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display  # Explicitly set self.display
        self._file_history = defaultdict(list)
        # Initialize Docker service
        ##ll.info(f"Initializing DockerEditTool")
        self.docker = DockerService()
        self._docker_available = self.docker.is_available()
        ##ll.info(f"Docker available: {self._docker_available}")

    def to_params(self) -> dict:
        #ll.debug(f"DockerEditTool.to_params called with api_type: {self.api_type}")
        # For custom tools, provide a detailed input schema
        params = {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [cmd for cmd in get_args(Command)],
                        "description": "Command to execute. Options: view, create, str_replace, insert, undo_edit"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file to operate on"
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Content to write to the file (required for create command)"
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Range of lines to view [start_line, end_line] (optional for view command)"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "String to be replaced (required for str_replace command)"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement string (required for str_replace and insert commands)"
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Line number where to insert text (required for insert command)"
                    }
                },
                "required": ["command", "path"]
            }
        }
        ic(f"DockerEditTool params: {params}")
        return params

    def format_output(self, data: Dict) -> str:
        """Format the output data similar to ProjectSetupTool style"""
        #ll.debug(f"Formatting output for data: {data}")
        output_lines = []

        # Add command type
        output_lines.append(f"Command: {data['command']}")
        if self.display is not None:
            self.display.add_message(
                "assistant", f"DockerEditTool Command: {data['command']}"
            )
        # Add status
        output_lines.append(f"Status: {data['status']}")

        # Add file path if present
        if "file_path" in data:
            output_lines.append(f"File Path: {data['file_path']}")

        # Add operation details if present
        if "operation_details" in data:
            output_lines.append(f"Operation: {data['operation_details']}")

        # Join all lines with newlines
        output = "\n".join(output_lines)
        if len(output) > 12000:
            output = f"{output[:5000]} ... (truncated) ... {output[-5000:]}"
        return output

    async def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs,
        ) -> ToolResult:
        """Execute the specified command with proper error handling and formatted output."""
        ##ll.info(f"DockerEditTool executing command: {command} on path: {path}")
        try:
            # Add display messages
            if self.display is not None:
                self.display.add_message(
                    "assistant",
                    f"DockerEditTool Executing Command: {command} on path: {path}",
                )

            # Normalize the path first - keep as string until we pass to specific methods
            _path = path
            #ll.debug(f"Normalized path: {_path}")
            
            if command == "create":
                ##ll.info(f"Creating new file at: {_path}")
                # if not file_text:
                #     #ll.error(f"Missing file_text parameter for create command")
                #     raise ToolError(
                #         "Parameter `file_text` is required for command: create"
                #     )
                #ll.debug(f"File content length: {len(file_text)} characters")
                self.write_file(_path, file_text)
                self._file_history[_path].append(file_text)
                log_file_operation(_path, "create")
                # ll.info(f"File created successfully: {_path}")
                if self.display is not None:
                    self.display.add_message(
                        "assistant",
                        f"DockerEditTool Command: {command} successfully created file {_path} !",
                    )
                    self.display.add_message("tool", file_text)
                output_data = {
                    "command": "create",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": "File created successfully",
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="create",
                )

            elif command == "view":
                #ll.info(f"Viewing file at: {_path}")
                result = await self.view(_path, view_range)
                #ll.debug(f"View result obtained, output length: {len(result.output) if result.output else 0}")
                if self.display is not None:
                    self.display.add_message(
                        "assistant",
                        f"DockerEditTool Command: {command} successfully viewed file\n {str(result.output[:100])} !",
                    )
                output_data = {
                    "command": "view",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": (
                        result.output if result.output else "No content to display"
                    ),
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="view",
                )

            elif command == "str_replace":
                #ll.info(f"Replacing string in file: {_path}")
                if not old_str:
                    ll.error(f"Missing old_str parameter for str_replace command")
                    raise ToolError(
                        "Parameter `old_str` is required for command: str_replace"
                    )
                ll.debug(f"Old string length: {len(old_str)}, New string length: {len(new_str) if new_str else 0}")
                result = self.str_replace(_path, old_str, new_str)
                #ll.info(f"String replacement completed in file: {_path}")
                if self.display is not None:
                    self.display.add_message(
                        "assistant",
                        f"DockerEditTool Command: {command} successfully replaced text in file {str(_path)} !",
                    )
                    self.display.add_message(
                        "assistant",
                        f"End of Old Text: {old_str[-200:] if len(old_str) > 200 else old_str}",
                    )
                    self.display.add_message(
                        "assistant",
                        f"End of New Text: {new_str[-200:] if new_str and len(new_str) > 200 else new_str}",
                    )
                output_data = {
                    "command": "str_replace",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": f"Replaced text in file",
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="str_replace",
                )

            elif command == "insert":
                #ll.info(f"Inserting text at line {insert_line} in file: {_path}")
                if insert_line is None:
                    ll.error(f"Missing insert_line parameter for insert command")
                    raise ToolError(
                        "Parameter `insert_line` is required for command: insert"
                    )
                if not new_str:
                    ll.error(f"Missing new_str parameter for insert command")
                    raise ToolError(
                        "Parameter `new_str` is required for command: insert"
                    )
                ll.debug(f"New string length: {len(new_str)}")
                result = self.insert(_path, insert_line, new_str)
                ll.info(f"Text inserted successfully at line {insert_line}")
                output_data = {
                    "command": "insert",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": f"Inserted text at line {insert_line}",
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="insert",
                )

            elif command == "undo_edit":
                #ll.info(f"Undoing last edit to file: {_path}")
                result = self.undo_edit(_path)
                #ll.info(f"Undo operation completed for file: {_path}")
                output_data = {
                    "command": "undo_edit",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": "Last edit undone successfully",
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="undo_edit",
                )

            else:
                ll.error(f"Unrecognized command: {command}")
                raise ToolError(
                    f'Unrecognized command {command}. The allowed commands are: {", ".join(get_args(Command))}'
                )

        except Exception as e:
            ll.error(f"Error executing {command} on {path}: {str(e)}")
            if self.display is not None:
                self.display.add_message("assistant", f"DockerEditTool error: {str(e)}")
            error_data = {
                "command": command,
                "status": "error",
                "file_path": str(_path) if "_path" in locals() else path,
                "operation_details": f"Error: {str(e)}",
            }
            return ToolResult(
                output=self.format_output(error_data),
                error=str(e),
                tool_name=self.name,
                command=str(command),
            )

    async def view(
        self, path: Path, view_range: Optional[List[int]] = None
        ) -> ToolResult:
        """Implement the view command using cross-platform methods."""
        #ll.info(f"View operation on path: {path}, range: {view_range}")
        ic(path)

        try:
            # Make sure we're working with a string path for Docker commands
            path_str = str(path)
            
            # Format the path for Docker command execution
            if not path_str.startswith('/'):
                path_str = '/' + path_str
            
            # Replace backslashes with forward slashes
            path_str = path_str.replace('\\', '/')
            
            # Ensure there are no double slashes
            while '//' in path_str:
                path_str = path_str.replace('//', '/')
                
            #ll.debug(f"Formatted path for Docker: {path_str}")
                
            # First check if path is a directory or file
            is_dir_cmd = f"[ -d \"{path_str}\" ] && echo 'dir' || echo 'not dir'"
            is_file_cmd = f"[ -f \"{path_str}\" ] && echo 'file' || echo 'not file'"
            
            is_dir_result = self.docker.execute_command(is_dir_cmd)
            is_file_result = self.docker.execute_command(is_file_cmd)
            
            is_dir = "dir" in is_dir_result.stdout
            is_file = "file" in is_file_result.stdout
            
            #ll.debug(f"Path check: is_dir={is_dir}, is_file={is_file}")
            
            # Prioritize file over directory when both are true
            # This handles cases where a path may be both a file and directory in Docker
            if is_file:
                # Handle file viewing using cat command
                #ll.debug(f"Reading file content from: {path_str}")
                cat_cmd = f"cat {path_str}"
                cat_result = self.docker.execute_command(cat_cmd)
                
                if not cat_result.success:
                    ll.error(f"Failed to read file: {cat_result.stderr}")
                    return ToolResult(output="", error=f"Failed to read file: {cat_result.stderr}")
                
                file_content = cat_result.stdout
                
                if not file_content:
                    ll.warning(f"File seems to be empty: {path_str}")
                    return ToolResult(output=f"File {path_str} appears to be empty.")
                
                # Handle view range if specified
                init_line = 1
                if view_range:
                    ll.debug(f"Processing view range: {view_range}")
                    if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                        ll.error(f"Invalid view range format: {view_range}")
                        raise ToolError(
                            "Invalid `view_range`. It should be a list of two integers."
                        )
                        
                    file_lines = file_content.split("\n")
                    n_lines_file = len(file_lines)
                    init_line, final_line = view_range
                    
                    # Validate view range
                    if init_line < 1 or init_line > n_lines_file:
                        raise ToolError(
                            f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be within the range of lines of the file: {[1, n_lines_file]}"
                        )
                    if final_line > n_lines_file:
                        raise ToolError(
                            f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be smaller than the number of lines in the file: `{n_lines_file}`"
                        )
                    if final_line != -1 and final_line < init_line:
                        raise ToolError(
                            f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be larger or equal than its first `{init_line}`"
                        )

                    # Apply the view range
                    if final_line == -1:
                        file_content = "\n".join(file_lines[init_line - 1:])
                    else:
                        file_content = "\n".join(file_lines[init_line - 1:final_line])
                        
                    #ll.debug(f"View range processed successfully")
                
                # Format the output for display
                #ll.info(f"Returning file content from line {init_line}")
                return ToolResult(
                    output=self._make_output(file_content, path_str, init_line=init_line)
                )
            elif is_dir:
                # Handle directory listing using ls command
                #ll.debug(f"Listing directory contents for: {path_str}")
                ls_cmd = f"ls -la {path_str}"
                ls_result = self.docker.execute_command(ls_cmd)
                
                if not ls_result.success:
                    #ll.error(f"Failed to list directory: {ls_result.stderr}")
                    return ToolResult(output="", error=f"Failed to list directory: {ls_result.stderr}")
                
                output = f"Directory listing for {path_str}:\n{ls_result.stdout}"
                #ll.debug(f"Directory listing completed successfully")
                return ToolResult(output=output)
            else:
                # Path doesn't exist
                ll.error(f"Path doesn't exist in Docker container: {path_str}")
                return ToolResult(
                    output="", 
                    error=f"Path {path_str} doesn't exist in Docker container."
                )
                
        except Exception as e:
            ll.error(f"Error in view operation: {str(e)}")
            return ToolResult(
                output="",
                error=f"Error viewing path {path}: {str(e)}"
            )

    def str_replace(
        self, path: Path, old_str: str, new_str: Optional[str]
        ) -> ToolResult:
        """Implement the str_replace command, which replaces old_str with new_str in the file content."""
        try:
            # Read the file content - ensure we're working with consistent line endings
            ll.info(f"Starting string replacement in file: {path}")
            
            file_content = self.read_file(path)
            # Normalize line endings and expand tabs for both strings
            file_content = file_content.expandtabs().replace('\r\n', '\n')
            old_str = old_str.expandtabs().replace('\r\n', '\n')
            new_str = new_str.expandtabs().replace('\r\n', '\n') if new_str is not None else ""
            
            # Check if the string exists in the file at all
            if old_str not in file_content:
                ll.warning(f"String not found in file: {path}")
                raise ToolError(
                    f"No replacement was performed, old_str did not appear verbatim in {path}."
                )
            
            # Count occurrences more reliably
            # Use a non-overlapping search to find all occurrences
            count = 0
            positions = []
            start_idx = 0
            
            # Find all occurrences and their positions
            while True:
                idx = file_content.find(old_str, start_idx)
                if idx == -1:
                    break
                count += 1
                positions.append(idx)
                start_idx = idx + len(old_str)
            
            if count == 0:
                ll.warning(f"String not found in file (after counting): {path}")
                raise ToolError(
                    f"No replacement was performed, old_str did not appear verbatim in {path}."
                )
            elif count > 1:
                # For multiple occurrences, find the line numbers
                lines = []
                for pos in positions:
                    line_num = file_content[:pos].count('\n') + 1
                    lines.append(line_num)
                
                ll.warning(f"Multiple ({count}) occurrences of old_str found in file at lines {lines}")
                raise ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str found at lines {lines}. Please ensure it is unique."
                )
            
            # Perform the replacement
            new_file_content = file_content.replace(old_str, new_str, 1)  # Replace just once to be safe
            
            # Verify the replacement was made
            if new_file_content == file_content:
                ll.warning("Replacement had no effect on file content")
                raise ToolError(
                    f"No changes were made to the file content. The replacement had no effect."
                )
            
            # Save original content to history before writing new content
            self._file_history[path].append(file_content)
            rr(path)
            # Write the new content to the file
            self.write_file(path, new_file_content)
            
            # Create a snippet showing the edited section
            pos = positions[0]  # We know there's exactly one occurrence
            lines_before = file_content[:pos].count('\n')
            replacement_line = lines_before + 1
            
            start_line = max(0, replacement_line - SNIPPET_LINES)
            end_line = replacement_line + SNIPPET_LINES + new_str.count('\n')
            
            snippet = "\n".join(new_file_content.split("\n")[start_line:end_line + 1])
            
            # Return success
            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(
                snippet, f"a snippet of {path}", start_line + 1
            )
            success_msg += "Review the changes and make sure they are as expected."
            return ToolResult(output=success_msg, error=None, base64_image=None)
        except Exception as e:
            ll.error(f"Error in string replacement: {str(e)}")
            return ToolResult(output=None, error=str(e), base64_image=None)

    def insert(self, path: Path, insert_line: int, new_str: str) -> ToolResult:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        #ll.info(f"Starting insert operation at line {insert_line} in file: {path}")
        ll.debug(f"New string length: {len(new_str)}")
        file_text = self.read_file(path).expandtabs()
        #ll.debug(f"File content read, length: {len(file_text)}")
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)
        #ll.debug(f"File has {n_lines_file} lines")
        if insert_line < 0 or insert_line > n_lines_file:
            #ll.error(f"Invalid insert line: {insert_line}, valid range is 0 to {n_lines_file}")
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )
        new_str_lines = new_str.split("\n")
        #ll.debug(f"New content has {len(new_str_lines)} lines")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )
        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)
        ll.debug(f"Writing updated file with {len(new_file_text_lines)} lines")
        self.write_file(path, new_file_text)
        ll.debug(f"Saving original content to history for: {path}")
        self._file_history[path].append(file_text)
        #ll.info(f"Insert operation completed successfully at line {insert_line} in file: {path}")
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return ToolResult(output=success_msg)

    def undo_edit(self, path: Path) -> ToolResult:
        """Implement the undo_edit command."""
        #ll.info(f"Starting undo operation for file: {path}")
        if not self._file_history[path]:
            ll.warning(f"No edit history found for file: {path}")
            raise ToolError(f"No edit history found for {path}.")
        ll.debug(f"Retrieving previous version from history")
        old_text = self._file_history[path].pop()
        ll.debug(f"Previous version retrieved, length: {len(old_text)}")
        ll.debug(f"Writing previous version back to file")
        self.write_file(path, old_text)

        #ll.info(f"Undo operation completed successfully for file: {path}")
        return ToolResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path) -> str:
        """Read file content from Docker container."""
        # Convert host path to Docker container path for a consistent read
        docker_path = self.docker.to_docker_path(path).as_posix()  # CHANGED
        # ...existing code...
        result = self.docker.execute_command(f"cat {docker_path}")
        if result.success:
            return result.stdout
        else:
            ll.error(f"Failed to read file from Docker: {result.stderr}")
            raise ToolError(f"Error reading file from Docker: {result.stderr}")

    def write_file(self, path: Path, file: str):
        """Write file content to Docker container."""
        #ll.info(f"Writing file to Docker: {path}, content length: {len(file)}")
        # Use DockerService to write the file by copying it into the container
        docker_file = self.docker.to_docker_path(path).as_posix()  # CHANGED
        #ll.debug(f"Converted to Docker path: {docker_file}")
        
        # Create parent directory in Docker container
        # #ll.debug(f"Creating parent directory in Docker: {parent_dir}")
        # mkdir_result = self.docker.execute_command(f"mkdir -p {parent_dir}")
        # if not mkdir_result.success:
        #     #ll.error(f"Failed to create parent directory: {mkdir_result.stderr}")
        
        # Write the file content to a temporary file and copy it to the Docker
        import tempfile, os, subprocess
        fd, temp_path = tempfile.mkstemp(text=True)
        #ll.debug(f"Created temporary file: {temp_path}")
        try:
            #ll.debug("Writing content to temporary file")
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(file)
            
            #ll.debug(f"Copying temporary file to Docker container: {docker_file}")
            docker_cmd = f'docker cp "{temp_path}" {self.docker._container_name}:"{docker_file}"'  # CHANGED
            ll.debug(f"Docker command: {docker_cmd}")
            subprocess.run(
                docker_cmd,
                shell=True,
                check=True,
            )
            
        except Exception as docker_e:
            #ll.error(f"Docker write operation failed: {str(docker_e)}")
            raise ToolError(f"Docker write failed: {str(docker_e)}")
        finally:
            ll.debug(f"Removing temporary file: {temp_path}")
            os.unlink(temp_path)

        try:
            #ll.debug(f"Logging file operation: {path}")
            log_file_operation(path, "modify")
        except Exception as log_e:
            ll.warning(f"Failed to log file operation: {str(log_e)}")

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
        ) -> str:
        """Generate output for the CLI based on the content of a file."""
        #ll.debug(f"Formatting output for {file_descriptor}, starting at line {init_line}")
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        #ll.debug(f"Output formatting completed, length: {len(file_content)}")
        return (
            f"Here's the result of running ` -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )
