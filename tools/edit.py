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
from load_constants import  write_to_file, ICECREAM_OUTPUT_FILE
from config import get_constant, set_constant, REPO_DIR, PROJECT_DIR, LOGS_DIR  # Updated import
from utils.file_logger import log_file_operation

# Reconfigure stdout to use UTF-8 encoding
# sys.stdout.reconfigure(encoding='utf-8')
# include the context for the icecream debugger
# ic.configureOutput(includeContext=True)
ic.configureOutput(includeContext=True, outputFunction=write_to_file)

# Reconfigure stdout to use UTF-8 encoding
Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]
SNIPPET_LINES: int = 4
PROJECT_DIR = Path(get_constant('PROJECT_DIR'))
PROJECT_DIR = Path.cwd() / PROJECT_DIR


# set_constant('LOG_FILE', str(LOG_FILE))  # Ensure LOG_FILE is a string path
class EditTool(BaseAnthropicTool):
    description="""
    A cross-platform filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters are defined by Anthropic and are not editable.
    """

    api_type: Literal["text_editor_20250124"] = "text_editor_20250124"
    name: Literal["str_replace_editor"] = "str_replace_editor"
    LOG_FILE = Path(get_constant('LOG_FILE'))
    _file_history: dict[Path, list[str]]

    def __init__(self, display=None):
        super().__init__(display)
        self._file_history = defaultdict(list)
        

    def to_params(self) -> BetaToolTextEditor20241022Param:
        return {
            "name": self.name,
            "type": self.api_type,
        }

    def format_output(self, data: Dict) -> str:
        """Format the output data similar to ProjectSetupTool style"""
        output_lines = []
        
        # Add command type
        output_lines.append(f"Command: {data['command']}")
        self.display.add_message("user", f"EditTool Command: {data['command']}")
        # Add status
        output_lines.append(f"Status: {data['status']}")
        
        # Add file path if present
        if 'file_path' in data:
            output_lines.append(f"File Path: {data['file_path']}")
            
        # Add operation details if present
        if 'operation_details' in data:
            output_lines.append(f"Operation: {data['operation_details']}")
            
        # Join all lines with newlines
        output =  "\n".join(output_lines)
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
        try:
            # Add display messages
            if self.display:
                self.display.add_message("user", f"EditTool Executing Command: {command} on path: {path}")

            # Normalize the path first
            _path = self.normalize_path(path)
            
            if command == "create":
                if not file_text:
                    raise ToolError("Parameter `file_text` is required for command: create")
                self.write_file(_path, file_text)
                self._file_history[_path].append(file_text)
                log_file_operation(_path, "create")  
                self.display.add_message("user", f"EditTool Command: {command}  successfully created file {_path} !")
                self.display.add_message("tool", file_text)
                output_data = {
                    "command": "create",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": "File created successfully"
                }
                return ToolResult(output=self.format_output(output_data))

            elif command == "view":
                result = await self.view(_path, view_range)
                self.display.add_message("user", f"EditTool Command: {command}  successfully viewed file\n {str(result.output[:100])} !")
                output_data = {
                    "command": "view",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": result.output if result.output else "No content to display"
                }
                return ToolResult(output=self.format_output(output_data))

            elif command == "str_replace":
                if not old_str:
                    raise ToolError("Parameter `old_str` is required for command: str_replace")
                result = self.str_replace(_path, old_str, new_str)
                self.display.add_message("user", f"EditTool Command: {command}  successfully replaced text in file{str(_path)} !")
                self.display.add_message("user", f"End of Old Text: {old_str[-200:]}")
                self.display.add_message("user", f"End of New Text: {new_str[-200:]}")
                output_data = {
                    "command": "str_replace",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": f"Replaced '{old_str}' with '{new_str}'"
                }
                return ToolResult(output=self.format_output(output_data))

            elif command == "insert":
                if insert_line is None:
                    raise ToolError("Parameter `insert_line` is required for command: insert")
                if not new_str:
                    raise ToolError("Parameter `new_str` is required for command: insert")
                result = self.insert(_path, insert_line, new_str)
                output_data = {
                    "command": "insert",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": f"Inserted text at line {insert_line}"
                }
                return ToolResult(output=self.format_output(output_data))

            elif command == "undo_edit":
                result = self.undo_edit(_path)
                output_data = {
                    "command": "undo_edit",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": "Last edit undone successfully"
                }
                return ToolResult(output=self.format_output(output_data))

            else:
                raise ToolError(
                    f'Unrecognized command {command}. The allowed commands are: {", ".join(get_args(Command))}'
                )

            if self.display:
                self.display.add_message("user", f"EditTool completed {command}")
            return ToolResult(output=self.format_output(output_data))

        except Exception as e:
            if self.display:
                self.display.add_message("user", f"EditTool error: {str(e)}")
            error_data = {
                "command": command,
                "status": "error",
                "file_path": str(_path) if '_path' in locals() else path,
                "operation_details": f"Error: {str(e)}"
            }
            return ToolResult(output=self.format_output(error_data))

    async def view(self, path: Path, view_range: Optional[List[int]] = None) -> ToolResult:
        """Implement the view command using cross-platform methods."""
        ic(path)
        if path.is_dir():
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            try:
                # Cross-platform directory listing using pathlib
                files = []
                for level in range(3):  # 0-2 levels deep
                    if level == 0:
                        pattern = "*"
                    else:
                        pattern = os.path.join(*["*"] * (level + 1))

                    for item in path.glob(pattern):
                        # Skip hidden files and directories
                        if not any(part.startswith('.') for part in item.parts):
                            files.append(str(item.resolve()))  # Ensure absolute paths

                stdout = "\n".join(sorted(files))
                stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
                return ToolResult(output=stdout, error=None, base64_image=None)
            except Exception as e:
                return ToolResult(output="", error=str(e), base64_image=None)

        # If it's a file, read its content
        file_content = self.read_file(path)
        init_line = 1
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError("Invalid `view_range`. It should be a list of two integers.")
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
                file_content = "\n".join(file_lines[init_line - 1:])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])
        return ToolResult(output=self._make_output(file_content, str(path), init_line=init_line), error=None, base64_image=None)
    def str_replace(self, path: Path, old_str: str, new_str: Optional[str]) -> ToolResult:
        """Implement the str_replace command, which replaces old_str with new_str in the file content."""
        try:
            # Read the file content
            ic(path)
            path = self.normalize_path(path)
            file_content = self.read_file(path).expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs() if new_str is not None else ""

            # Check if old_str is unique in the file
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                raise ToolError(f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}.")
            elif occurrences > 1:
                file_content_lines = file_content.split("\n")
                lines = [
                    idx + 1
                    for idx, line in enumerate(file_content_lines)
                    if old_str in line
                ]
                raise ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
                )

            # Replace old_str with new_str
            new_file_content = file_content.replace(old_str, new_str)

            # Write the new content to the file
            self.write_file(path, new_file_content)

            # Save the content to history
            self._file_history[path].append(file_content)

            # Create a snippet of the edited section
            replacement_line = file_content.split(old_str)[0].count("\n")
            start_line = max(0, replacement_line - SNIPPET_LINES)
            end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
            snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

            # Prepare the success message
            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(snippet, f"a snippet of {path}", start_line + 1)
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

            return ToolResult(output=success_msg, error=None, base64_image=None)

        except Exception as e:
            return ToolResult(output=None, error=str(e), base64_image=None)
    def insert(self, path: Path, insert_line: int, new_str: str) -> ToolResult:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        path = self.normalize_path(path)
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
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

        self.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."
        return ToolResult(output=success_msg)
    # def ensure_valid_repo_path(filename: str) -> str:
    #     ### Need to Try this out ###
    #     base_path = PROJECT_DIR
        
    #     # Normalize path separators for cross-platform compatibility
    #     filename = filename.replace("\\", "/")
        
    #     # Check if the filename already starts with the base path
    #     if not filename.startswith(base_path):
    #         # Prepend the base path if it's not present
    #         filename = os.path.join(base_path, filename.lstrip("/"))

        return os.path.normpath(filename)
    def undo_edit(self, path: Path) -> ToolResult:
        """Implement the undo_edit command."""
        path = self.normalize_path(path)
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)

        return ToolResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").encode('ascii', errors='replace').decode('ascii')
        except Exception as e:
            ic(f"Error reading file {path}: {e}")
            raise ToolError(f"Ran into {e} while trying to read {path}") from None

    def normalize_path(self, path: str | Path) -> Path:
        """
        Normalize a path to be within the project directory.
        Args:
            path: Path to normalize (can be string or Path object)
        Returns:
            Normalized Path object within project directory
        """
        return path
        # try:
        #     # Get project directory from config
        #     project_dir = Path(get_constant('PROJECT_DIR'))
        #     if not project_dir.exists():
        #         project_dir.mkdir(parents=True, exist_ok=True)
            
        #     # Convert input to Path object
        #     input_path = Path(path)
            
        #     # If absolute path, make relative to project dir
        #     if input_path.is_absolute():
        #         try:
        #             # Try to make relative to project dir
        #             relative_path = input_path.relative_to(project_dir)
        #             result_path = project_dir / relative_path
        #         except ValueError:
        #             # If not under project dir, take the name only
        #             result_path = project_dir / input_path.name
        #     else:
        #         # For relative paths, prepend project dir
        #         result_path = project_dir / input_path
            
        #     # Ensure result is within project directory
        #     try:
        #         result_path.relative_to(project_dir)
        #     except ValueError:
        #         raise ValueError(f"Path {result_path} must be within project directory {project_dir}")
                
        #     return result_path

        # except Exception as e:
        #     ic(f"Path normalization error: {e}")
        #     raise ValueError(f"Failed to normalize path '{path}': {str(e)}")

    def write_file(self, path: Path, file: str):
        """Write file content ensuring correct project directory"""
        try:
            # Normalize path to be within project directory
            ic(path)
            full_path = self.normalize_path(path)
            ic(full_path)
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            ic(file)
            # Write the file
            full_path.write_text(file, encoding="utf-8")
            # Log the file operation
            log_file_operation(full_path, "modify")
        except Exception as e:
            raise ToolError(f"Error writing to {path}: {str(e)}")

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        """Generate output for the CLI based on the content of a file."""
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )
        return (
            f"Here's the result of running ` -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )
