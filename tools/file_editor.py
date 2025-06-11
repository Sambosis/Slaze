from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Literal, Optional, get_args
import logging

# from loguru import logger as ll # Removed loguru import
# from icecream import ic # Removed
from lmnr import observe

from config import get_constant
from .base import BaseAnthropicTool, ToolError, ToolResult
from .run import maybe_truncate
from utils.file_logger import log_file_operation

Command = Literal["view", "create", "str_replace", "insert", "undo_edit"]
SNIPPET_LINES = 4

logger = logging.getLogger(__name__)

class FileEditorTool(BaseAnthropicTool):
    """A cross-platform filesystem editor tool. All operations run locally."""

    api_type: Literal["custom"] = "custom"
    name: Literal["file_editor"] = "file_editor"
    description = (
        "A cross-platform filesystem editor tool. All operations run locally."
    )
    LOG_FILE = Path(get_constant("LOG_FILE"))

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display
        self._file_history: Dict[Path, List[str]] = defaultdict(list)

    def to_params(self) -> dict:
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "enum": [cmd for cmd in get_args(Command)],
                            "description": (
                                "Command to execute. Options: view, create, "
                                "str_replace, insert, undo_edit"
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": "Path to the file to operate on",
                        },
                        "file_text": {
                            "type": "string",
                            "description": (
                                "Content to write to the file (required for "
                                "create command)"
                            ),
                        },
                        "view_range": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Range of lines to view [start_line, end_line] "
                                "(optional for view command)"
                            ),
                        },
                        "old_str": {
                            "type": "string",
                            "description": (
                                "String to be replaced (required for str_replace "
                                "command)"
                            ),
                        },
                        "new_str": {
                            "type": "string",
                            "description": (
                                "Replacement string (required for str_replace "
                                "and insert commands)"
                            ),
                        },
                        "insert_line": {
                            "type": "integer",
                            "description": (
                                "Line number where to insert text (required for "
                                "insert command)"
                            ),
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        }
        logger.debug(f"FileEditorTool params: {params}")
        return params

    def format_output(self, data: Dict) -> str:
        output_lines = [f"Command: {data['command']}"]
        if self.display is not None:
            self.display.add_message(
                "assistant", f"FileEditorTool Command: {data['command']}"
            )
        output_lines.append(f"Status: {data['status']}")
        if "file_path" in data:
            output_lines.append(f"File Path: {data['file_path']}")
        if "operation_details" in data:
            output_lines.append(f"Operation: {data['operation_details']}")
        output = "\n".join(output_lines)
        if len(output) > 12000:
            output = f"{output[:5000]} ... (truncated) ... {output[-5000:]}"
        return output

    @observe(name="docker_edit")
    async def __call__(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: List[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs,
    ) -> ToolResult:
        _path: Optional[Path] = None # Ensure _path is defined in outer scope for error reporting
        try:
            if not path or not isinstance(path, str):
                raise ToolError("Invalid or missing 'path' parameter. It must be a non-empty string.")
            try:
                _path = Path(path)
            except Exception as e:
                raise ToolError(f"Failed to convert path string '{path}' to Path object: {e}")

            if self.display is not None:
                self.display.add_message(
                    "user", f"FileEditorTool Executing Command: {command} on path: {str(_path)}"
                )

            # _path is now guaranteed to be a Path object if we proceed

            if command == "create":
                if file_text is None:
                    raise ToolError("Parameter `file_text` is required for command: create")
                # _path is already a Path object here
                self.write_file(_path, file_text)
                self._file_history[_path].append(file_text)
                log_file_operation(_path, "create")
                if self.display is not None:
                    self.display.add_message(
                        "assistant",
                        f"FileEditorTool Command: {command} successfully created file {_path} !",
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
                result = await self.view(_path, view_range)
                if self.display is not None:
                    self.display.add_message(
                        "assistant",
                        f"FileEditorTool Command: {command} successfully viewed file\n {str(result.output[:100])} !",
                    )
                output_data = {
                    "command": "view",
                    "status": "success",
                    "file_path": str(_path),
                    "operation_details": result.output if result.output else "No content to display",
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="view",
                )
            elif command == "str_replace":
                if not old_str:
                    raise ToolError("Parameter `old_str` is required for command: str_replace")
                result = self.str_replace(_path, old_str, new_str)
                if self.display is not None:
                    self.display.add_message(
                        "assistant",
                        f"FileEditorTool Command: {command} successfully replaced text in file {str(_path)} !",
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
                    "operation_details": "Replaced text in file",
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="str_replace",
                )
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
                    "operation_details": f"Inserted text at line {insert_line}",
                }
                return ToolResult(
                    output=self.format_output(output_data),
                    tool_name=self.name,
                    command="insert",
                )
            elif command == "undo_edit":
                result = self.undo_edit(_path)
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
                raise ToolError(
                    f"Unrecognized command {command}. The allowed commands are: {', '.join(get_args(Command))}"
                )
        except Exception as e:
            if self.display is not None:
                self.display.add_message("assistant", f"FileEditorTool error: {str(e)}")
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

    async def view(self, path: Path, view_range: Optional[List[int]] = None) -> ToolResult:
        logger.debug(f"Viewing path: {path}")
        try:
            files = []
            for level in range(3):
                pattern = "*" if level == 0 else os.path.join(*["*"] * (level + 1))
                for item in path.glob(pattern):
                    if not any(part.startswith(".") for part in item.parts):
                        files.append(str(item.resolve()))
            stdout = "\n".join(sorted(files))
            stdout = (
                f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
            )
            return ToolResult(output=stdout, error=None, base64_image=None)
        except Exception as e:
            return ToolResult(output="", error=str(e), base64_image=None)

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
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])
        return ToolResult(
            output=self._make_output(file_content, str(path), init_line=init_line),
            error=None,
            base64_image=None,
        )

    def str_replace(self, path: Path, old_str: str, new_str: Optional[str]) -> ToolResult:
        try:
            logger.debug(f"Replacing string in path: {path}")
            file_content = self.read_file(path).expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs() if new_str is not None else ""

            occurrences = file_content.count(old_str)
            if occurrences == 0:
                raise ToolError(
                    f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
                )
            elif occurrences > 1:
                file_content_lines = file_content.split("\n")
                lines = [idx + 1 for idx, line in enumerate(file_content_lines) if old_str in line]
                raise ToolError(
                    f"No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"
                )

            new_file_content = file_content.replace(old_str, new_str)
            self.write_file(path, new_file_content)
            self._file_history[path].append(file_content)
            replacement_line = file_content.split(old_str)[0].count("\n")
            start_line = max(0, replacement_line - SNIPPET_LINES)
            end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
            snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])
            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(snippet, f"a snippet of {path}", start_line + 1)
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."
            return ToolResult(output=success_msg, error=None, base64_image=None)
        except Exception as e:
            logger.error(f"Error in string replacement: {str(e)}", exc_info=True) # Replaced ll.error with logger.error and added exc_info
            return ToolResult(output=None, error=str(e), base64_image=None)

    def insert(self, path: Path, insert_line: int, new_str: str) -> ToolResult:
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)
        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )
        new_str_lines = new_str.split("\n")
        new_file_text_lines = file_text_lines[:insert_line] + new_str_lines + file_text_lines[insert_line:]
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

    def undo_edit(self, path: Path) -> ToolResult:
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")
        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)
        return ToolResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").encode("ascii", errors="replace").decode("ascii")
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}", exc_info=True)
            raise ToolError(f"Ran into {e} while trying to read {path}") from None

    def write_file(self, path: Path, file: str):
        try:
            if not isinstance(path, Path):
                logger.warning(f"FileEditorTool.write_file received a non-Path object for path: {type(path)}. Converting.")
                try:
                    path = Path(path)
                except Exception as e:
                    raise ToolError(f"Failed to convert path '{str(path)}' to Path object in write_file: {e}")

            # path is now guaranteed to be a Path object
            logger.debug(f"Writing to path: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"File content to write (first 100 chars): {file[:100]}")
            path.write_text(file, encoding="utf-8")
            log_file_operation(path, "modify") # path is Path object
        except Exception as e:
            # Use str(path) in error message as path could be None if conversion failed early
            raise ToolError(f"Error writing to {str(path)}: {str(e)}")

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join(
            [f"{i + init_line:6}\t{line}" for i, line in enumerate(file_content.split("\n"))]
        )
        return (
            f"Here's the result of running ` -n` on {file_descriptor}:\n" + file_content + "\n"
        )
