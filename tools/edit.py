## edit.py
import os
from pathlib import Path
from collections import defaultdict
from typing import Literal, get_args, Dict
from .base import BaseTool, ToolError, ToolResult
# from .run import maybe_truncate
from typing import List, Optional
import logging
from config import get_constant
from utils.file_logger import log_file_operation

logger = logging.getLogger(__name__)

Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]
SNIPPET_LINES: int = 4


class EditTool(BaseTool):
    description = """
    A cross-platform filesystem editor tool that allows the agent to view, create, and edit files.
    The tool parameters follow the OpenAI function calling format.
    """

    api_type: Literal["text_editor_20250124"] = "text_editor_20250124"
    name: Literal["str_replace_editor"] = "str_replace_editor"
    LOG_FILE = Path(get_constant("LOG_FILE"))
    _file_history: dict[Path, list[str]]

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display  # Explicitly set self.display
        self._file_history = defaultdict(list)

    def _format_terminal_output(self, 
                               command: str, 
                               path: str, 
                               result: str = None, 
                               error: str = None,
                               additional_info: str = None) -> str:
        """Format edit operations to look like terminal output."""
        output_lines = ["```console"]
        
        # Format the command with a pseudo-shell prompt
        output_lines.append(f"$ edit {command} {path}")
        
        # Add the result/output if provided
        if result:
            output_lines.extend(result.rstrip().split('\n'))
        
        # Add error if provided
        if error:
            output_lines.append(f"Error: {error}")
        
        # Add additional info if provided
        if additional_info:
            output_lines.extend(additional_info.rstrip().split('\n'))
        
        # End console formatting
        output_lines.append("```")
        
        return "\n".join(output_lines)

    def _resolve_path(self, path: str | Path) -> Path:
        """Resolve a given path relative to REPO_DIR if not absolute, and normalize it."""
        p = Path(path)
        if not p.is_absolute():
            repo_dir = get_constant("REPO_DIR")
            if repo_dir:
                p = Path(repo_dir) / p
        return p.resolve()

    def to_params(self) -> dict:
        logger.debug(
            f"EditTool.to_params called with api_type: {self.api_type}")
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
                            "enum": list(get_args(Command)),
                            "description": "The command to execute",
                        },
                        "path": {
                            "type": "string",
                            "description": "The path to file or directory",
                        },
                        "file_text": {
                            "type": "string"
                        },
                        "view_range": {
                            "type": "array",
                            "items": {
                                "type": "integer"
                            },
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "old_str": {
                            "type": "string",
                            "description": "The old string to replace",
                        },
                        "new_str": {
                            "type":
                            "string",
                            "description":
                            "The new string to replace the old one or the new string to add on an insert command",
                        },
                        "insert_line": {
                            "type":
                            "integer",
                            "description":
                            "The line number to insert the new string"
                        },
                    },
                    "required": ["command", "path"],
                },
            },
        }
        logger.debug(f"EditTool params: {params}")
        return params



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
            # Normalize the path first relative to REPO_DIR
            _path = self._resolve_path(path)
            if command == "create":
                if not file_text:
                    raise ToolError(
                        "Parameter `file_text` is required for command: create"
                    )
                self.write_file(_path, file_text)
                self._file_history[_path].append(file_text)
                log_file_operation(_path, "create")
                if self.display is not None:
                    console_output = self._format_terminal_output(
                        command="create", 
                        path=str(_path), 
                        result=f"File created successfully\nContent length: {len(file_text)} characters",
                        additional_info=f"Preview (first 200 chars):\n{file_text[:200]}{'...' if len(file_text) > 200 else ''}"
                    )
                    self.display.add_message("assistant", console_output)
                # Return detailed result with file creation confirmation
                success_msg = f"File {_path} has been created successfully.\n"
                success_msg += f"Content length: {len(file_text)} characters\n"
                success_msg += f"File path: {str(_path)}"

                return ToolResult(
                    output=success_msg,
                    tool_name=self.name,
                    command="create",
                )

            elif command == "view":
                result = await self.view(_path, view_range)
                if self.display is not None:
                    console_output = self._format_terminal_output(
                        command="view",
                        path=str(_path),
                        result=f"File viewed successfully",
                        additional_info=result.output[:500] + ("..." if len(result.output) > 500 else "")
                    )
                    self.display.add_message("assistant", console_output)
                # Return the detailed result from view method with proper metadata
                return ToolResult(
                    output=result.output,
                    error=result.error,
                    tool_name=self.name,
                    command="view",
                )

            elif command == "str_replace":
                if not old_str:
                    raise ToolError(
                        "Parameter `old_str` is required for command: str_replace"
                    )
                result = self.str_replace(_path, old_str, new_str)
                if self.display is not None:
                    old_preview = old_str[-200:] if len(old_str) > 200 else old_str
                    new_preview = new_str[-200:] if new_str and len(new_str) > 200 else new_str
                    console_output = self._format_terminal_output(
                        command="str_replace",
                        path=str(_path),
                        result=f"Text replacement completed successfully",
                        additional_info=f"Old text: {old_preview}\nNew text: {new_preview}"
                    )
                    self.display.add_message("assistant", console_output)

                # Return the detailed result from str_replace method with proper metadata
                return ToolResult(
                    output=result.output,
                    error=result.error,
                    tool_name=self.name,
                    command="str_replace",
                )

            elif command == "insert":
                if insert_line is None:
                    raise ToolError(
                        "Parameter `insert_line` is required for command: insert"
                    )
                if not new_str:
                    raise ToolError(
                        "Parameter `new_str` is required for command: insert")
                result = self.insert(_path, insert_line, new_str)
                if self.display is not None:
                    console_output = self._format_terminal_output(
                        command="insert",
                        path=str(_path),
                        result=f"Text inserted successfully at line {insert_line}",
                        additional_info=f"Inserted text: {new_str[:200]}{'...' if len(new_str) > 200 else ''}"
                    )
                    self.display.add_message("assistant", console_output)

                # Return the detailed result from insert method with proper metadata
                return ToolResult(
                    output=result.output,
                    error=result.error,
                    tool_name=self.name,
                    command="insert",
                )

            elif command == "undo_edit":
                result = self.undo_edit(_path)
                if self.display is not None:
                    console_output = self._format_terminal_output(
                        command="undo_edit",
                        path=str(_path),
                        result=f"Last edit undone successfully",
                        additional_info="Previous file state has been restored"
                    )
                    self.display.add_message("assistant", console_output)

                # Return the detailed result from undo_edit method with proper metadata
                return ToolResult(
                    output=result.output,
                    error=result.error,
                    tool_name=self.name,
                    command="undo_edit",
                )

            else:
                raise ToolError(
                    f"Unrecognized command {command}. The allowed commands are: {', '.join(get_args(Command))}"
                )

        except Exception as e:
            if self.display is not None:
                console_output = self._format_terminal_output(
                    command=str(command),
                    path=str(_path) if '_path' in locals() else path,
                    error=str(e)
                )
                self.display.add_message("assistant", console_output)

            # Return detailed error information
            error_msg = f"EditTool {command} command failed.\n"
            error_msg += f"File path: {str(_path) if '_path' in locals() else path}\n"
            error_msg += f"Error: {str(e)}"

            return ToolResult(
                output=error_msg,
                error=str(e),
                tool_name=self.name,
                command=str(command),
            )

    async def view(self,
                   path: Path,
                   view_range: Optional[List[int]] = None) -> ToolResult:
        """Implement the view command using cross-platform methods."""
        path = self._resolve_path(path)
        logger.debug(f"Viewing path: {path}")
        try:
            if path.is_dir():
                # Cross-platform directory listing using pathlib
                files = []
                for level in range(3):  # 0-2 levels deep
                    if level == 0:
                        pattern = "*"
                    else:
                        pattern = os.path.join(*["*"] * (level + 1))

                    for item in path.glob(pattern):
                        # Skip hidden files and directories
                        if not any(
                                part.startswith(".") for part in item.parts):
                            files.append(str(
                                item.resolve()))  # Ensure absolute paths

                stdout = "\n".join(sorted(files))
                stdout = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n{stdout}\n"
                return ToolResult(output=stdout, error=None, base64_image=None)
            elif path.is_file():
                # If it's a file, read its content
                file_content = self.read_file(path)
                init_line = 1
                if view_range:
                    if len(view_range) != 2 or not all(
                            isinstance(i, int) for i in view_range):
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
                        file_content = "\n".join(file_lines[init_line - 1:])
                    else:
                        file_content = "\n".join(file_lines[init_line -
                                                            1:final_line])
                return ToolResult(
                    output=self._make_output(file_content,
                                             str(path),
                                             init_line=init_line),
                    error=None,
                    base64_image=None,
                )
            else:
                return ToolResult(
                    output="",
                    error=
                    f"Path not found or is not a file or directory: {path}",
                    base64_image=None)
        except Exception as e:
            return ToolResult(output="", error=str(e), base64_image=None)

    def str_replace(self, path: Path, old_str: str,
                    new_str: Optional[str]) -> ToolResult:
        """Implement the str_replace command, which replaces old_str with new_str in the file content."""
        try:
            path = self._resolve_path(path)
            # Read the file content
            logger.debug(f"Replacing string in path: {path}")
            file_content = self.read_file(path).expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs() if new_str is not None else ""

            # Check if old_str is unique in the file
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                raise ToolError(
                    f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
                )
            elif occurrences > 1:
                file_content_lines = file_content.split("\n")
                lines = [
                    idx + 1 for idx, line in enumerate(file_content_lines)
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
            snippet = "\n".join(
                new_file_content.split("\n")[start_line:end_line + 1])

            # Prepare the success message
            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(snippet, f"a snippet of {path}",
                                             start_line + 1)
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

            return ToolResult(output=success_msg,
                              error=None,
                              base64_image=None)

        except Exception as e:
            return ToolResult(output=None, error=str(e), base64_image=None)

    def insert(self, path: Path, insert_line: int, new_str: str) -> ToolResult:
        """Implement the insert command, which inserts new_str at the specified line in the file content."""
        path = self._resolve_path(path)
        file_text = self.read_file(path).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within the range of lines of the file: {[0, n_lines_file]}"
            )

        new_str_lines = new_str.split("\n")
        new_file_text_lines = (file_text_lines[:insert_line] + new_str_lines +
                               file_text_lines[insert_line:])
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES):insert_line] +
            new_str_lines +
            file_text_lines[insert_line:insert_line + SNIPPET_LINES])

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
        """Implement the undo_edit command."""
        path = self._resolve_path(path)
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        self.write_file(path, old_text)

        return ToolResult(
            output=
            f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def read_file(self, path: Path) -> str:
        try:
            path = self._resolve_path(path)
            return (path.read_text(encoding="utf-8").encode(
                "ascii", errors="replace").decode("ascii"))
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}", exc_info=True)
            raise ToolError(
                f"Ran into {e} while trying to read {path}") from None

    def write_file(self, path: Path, file: str):
        """Write file content ensuring correct project directory"""
        try:
            # Normalize path to be within project directory
            full_path = self._resolve_path(path)
            logger.debug(f"Writing to path: {full_path}")
            logger.debug(f"Full path for writing: {full_path}")
            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(
                f"File content to write (first 100 chars): {file[:100]}")
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
        # file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()
        file_content = "\n".join([
            f"{i + init_line:6}\t{line}"
            for i, line in enumerate(file_content.split("\n"))
        ])
        return (f"Here's the result of running ` -n` on {file_descriptor}:\n" +
                file_content + "\n")
