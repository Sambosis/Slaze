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

# Configure logging to a file
ll.add(
    "edit.log",
    rotation="500 KB",
    level="DEBUG",
    format="{time: MM-DD HH:mm} | {level: <8} | {module}.{function}:{line} - {message}",
)

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
        ll.info(f"Initializing DockerEditTool")
        self.docker = DockerService()
        self._docker_available = self.docker.is_available()
        ll.info(f"Docker available: {self._docker_available}")

    def to_params(self) -> dict:
        ll.debug(f"DockerEditTool.to_params called with api_type: {self.api_type}")
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
        ll.debug(f"Formatting output for data: {data}")
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
        ll.info(f"DockerEditTool executing command: {command} on path: {path}")
        try:
            # Add display messages
            if self.display is not None:
                self.display.add_message(
                    "assistant",
                    f"DockerEditTool Executing Command: {command} on path: {path}",
                )

            # Normalize the path first
            _path = path
            ll.debug(f"Normalized path: {_path}")
            
            if command == "create":
                ll.info(f"Creating new file at: {_path}")
                if not file_text:
                    ll.error(f"Missing file_text parameter for create command")
                    raise ToolError(
                        "Parameter `file_text` is required for command: create"
                    )
                ll.debug(f"File content length: {len(file_text)} characters")
                self.write_file(_path, file_text)
                self._file_history[_path].append(file_text)
                log_file_operation(_path, "create")
                ll.info(f"File created successfully: {_path}")
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
                ll.info(f"Viewing file at: {_path}")
                result = await self.view(_path, view_range)
                ll.debug(f"View result obtained, output length: {len(result.output) if result.output else 0}")
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
                ll.info(f"Replacing string in file: {_path}")
                if not old_str:
                    ll.error(f"Missing old_str parameter for str_replace command")
                    raise ToolError(
                        "Parameter `old_str` is required for command: str_replace"
                    )
                ll.debug(f"Old string length: {len(old_str)}, New string length: {len(new_str) if new_str else 0}")
                result = self.str_replace(_path, old_str, new_str)
                ll.info(f"String replacement completed in file: {_path}")
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
                ll.info(f"Inserting text at line {insert_line} in file: {_path}")
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
                ll.info(f"Text inserted successfully at line {insert_line} in file: {_path}")
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
                ll.info(f"Undoing last edit to file: {_path}")
                result = self.undo_edit(_path)
                ll.info(f"Undo operation completed for file: {_path}")
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
        ll.info(f"View operation on path: {path}, range: {view_range}")
        ic(path)

        try:
            # Cross-platform directory listing using pathlib
            ll.debug(f"Checking if {path} is a directory")
            files = []
            for level in range(3):  # 0-2 levels deep
                if level == 0:
                    pattern = "*"
                else:
                    pattern = os.path.join(*["*"] * (level + 1))
                ll.debug(f"Glob pattern: {pattern} for level {level}")
                for item in path.glob(pattern):
                    # Skip hidden files and directories
                    if not any(part.startswith(".") for part in item.parts):
                        files.append(str(item.resolve()))  # Ensure absolute paths
                        ll.trace(f"Found file/directory: {item}")

            stdout = "\n".join(sorted(files))
            ll.debug(f"Found {len(files)} files/directories in {path}")
            stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
            ll.info(f"Directory listing completed for {path}")
            return ToolResult(output=stdout, error=None, base64_image=None)
        except Exception as e:
            ll.error(f"Error listing directory {path}: {str(e)}")
            return ToolResult(output="", error=str(e), base64_image=None)
        
        ll.info(f"Viewing file {path}")
        # If it's a file, read its content
        ll.info(f"Reading file content: {path}")
        file_content = self.read_file(path)
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

            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])
            ll.debug(f"View range processed successfully")

        ll.info(f"Returning file content from line {init_line}")
        return ToolResult(
            output=self._make_output(file_content, str(path), init_line=init_line),
            error=None,
            base64_image=None,
        )

    def str_replace(
        self, path: Path, old_str: str, new_str: Optional[str]
    ) -> ToolResult:
        """Implement the str_replace command, which replaces old_str with new_str in the file content."""
        try:
            # Read the file content
            ll.info(f"Starting string replacement in file: {path}")
            ll.debug(f"Old string length: {len(old_str)}")
            ll.debug(f"New string length: {len(new_str) if new_str else 0}")
            
            file_content = self.read_file(path).expandtabs()
            ll.debug(f"File content read, length: {len(file_content)}")
            
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs() if new_str is not None else ""

            # Check if old_str is unique in the file
            occurrences = file_content.count(old_str)
            ll.debug(f"Found {occurrences} occurrences of old_str in file")
            
            if occurrences == 0:
                ll.warning(f"String not found in file: {path}")
                raise ToolError(
                    f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
                )
            elif occurrences > 1:
                ll.warning(f"Multiple ({occurrences}) occurrences of old_str found in file")
                file_content_lines = file_content.split("\n")
                lines = [
                    idx + 1
                    for idx, line in enumerate(file_content_lines)
                    if old_str in line
                ]
                ll.debug(f"Occurrences found on lines: {lines}")
                raise ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
                )

            # Replace old_str with new_str
            ll.debug("Performing string replacement")
            new_file_content = file_content.replace(old_str, new_str)
            
            # Add validation to ensure the replacement actually changed the content
            if new_file_content == file_content:
                ll.warning("Replacement had no effect on file content")
                raise ToolError(
                    f"No changes were made to the file content. The replacement had no effect."
                )

            # Write the new content to the file
            ll.debug(f"Writing updated content to file: {path}")
            self.write_file(path, new_file_content)

            # Save the content to history
            ll.debug(f"Saving original content to history for: {path}")
            self._file_history[path].append(file_content)

            # Create a snippet of the edited section
            ll.debug("Creating snippet of edited section")
            replacement_line = file_content.split(old_str)[0].count("\n")
            start_line = max(0, replacement_line - SNIPPET_LINES)
            end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
            snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])
            ll.debug(f"Snippet created from lines {start_line+1} to {end_line+1}")

            # Prepare the success message
            ll.info(f"String replacement completed successfully in file: {path}")
            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(
                snippet, f"a snippet of {path}", start_line + 1
            )
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

            return ToolResult(output=success_msg, error=None, base64_image=None)

        except Exception as e:
            ll.error(f"Error in string replacement: {str(e)}")
            return ToolResult(output=None, error=str(e), base64_image=None)

    def insert(self, path: Path, insert_line: int, new_str: str) -> ToolResult:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        ll.info(f"Starting insert operation at line {insert_line} in file: {path}")
        ll.debug(f"New string length: {len(new_str)}")
        
        file_text = self.read_file(path).expandtabs()
        ll.debug(f"File content read, length: {len(file_text)}")
        
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)
        ll.debug(f"File has {n_lines_file} lines")

        if insert_line < 0 or insert_line > n_lines_file:
            ll.error(f"Invalid insert line: {insert_line}, valid range is 0 to {n_lines_file}")
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        ll.debug(f"New content has {len(new_str_lines)} lines")
        
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

        ll.info(f"Insert operation completed successfully at line {insert_line} in file: {path}")
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
        ll.info(f"Starting undo operation for file: {path}")
        
        if not self._file_history[path]:
            ll.warning(f"No edit history found for file: {path}")
            raise ToolError(f"No edit history found for {path}.")

        ll.debug(f"Retrieving previous version from history")
        old_text = self._file_history[path].pop()
        ll.debug(f"Previous version retrieved, length: {len(old_text)}")
        
        ll.debug(f"Writing previous version back to file")
        self.write_file(path, old_text)

        ll.info(f"Undo operation completed successfully for file: {path}")
        return ToolResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path) -> str:
        """Read file content from Docker container."""
        ll.info(f"Reading file content from Docker: {path}")
        # Use DockerService to read file content via 'cat'
        docker_path = self.docker.to_docker_path(path)
        ll.debug(f"Converted to Docker path: {docker_path}")
        
        result = self.docker.execute_command(f"cat {docker_path}")
        
        if result.success:
            ll.info(f"File read successfully, content length: {len(result.stdout)}")
            ll.debug(f"First 100 chars: {result.stdout[:100]}...")
            return result.stdout
        else:
            ll.error(f"Failed to read file from Docker: {result.stderr}")
            raise ToolError(f"Error reading file from Docker: {result.stderr}")
    
    def write_file(self, path: Path, file: str):
        """Write file content to Docker container."""
        ll.info(f"Writing file to Docker: {path}, content length: {len(file)}")
        # Use DockerService to write the file by copying it into the container
        docker_file = self.docker.to_docker_path(path)
        ll.debug(f"Converted to Docker path: {docker_file}")
        
        # Create parent directory in Docker container
        parent_dir = str(Path(docker_file).parent).replace("\\", "/")
        ll.debug(f"Creating parent directory in Docker: {parent_dir}")
        mkdir_result = self.docker.execute_command(f"mkdir -p {parent_dir}")
        
        if not mkdir_result.success:
            ll.error(f"Failed to create parent directory: {mkdir_result.stderr}")
        
        # Write the file content to a temporary file and copy it to the Docker
        import tempfile, os, subprocess
        fd, temp_path = tempfile.mkstemp(text=True)
        ll.debug(f"Created temporary file: {temp_path}")
        
        try:
            ll.debug("Writing content to temporary file")
            with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                temp_file.write(file)
                
            ll.debug(f"Copying temporary file to Docker container: {docker_file}")
            docker_cmd = f'docker cp "{temp_path}" {self.docker._container_name}:"{str(docker_file)}"'
            ll.debug(f"Docker command: {docker_cmd}")
            
            subprocess.run(
                docker_cmd,
                shell=True,
                check=True,
            )
            ll.info(f"File successfully written to Docker container")
            
        except Exception as docker_e:
            ll.error(f"Docker write operation failed: {str(docker_e)}")
            raise ToolError(f"Docker write failed: {str(docker_e)}")
        finally:
            ll.debug(f"Removing temporary file: {temp_path}")
            os.unlink(temp_path)
            
        try:
            ll.debug(f"Logging file operation: {path}")
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
        ll.debug(f"Formatting output for {file_descriptor}, starting at line {init_line}")
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        ll.debug(f"Output formatting completed, length: {len(file_content)}")
        return (
            f"Here's the result of running ` -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )
