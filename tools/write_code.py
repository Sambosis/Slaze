"""Utility tool for generating codebases via LLM calls."""

import asyncio
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, List, Dict

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel
import logging
# from rich import print as rr # Removed
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from tools.base import BaseAnthropicTool, ToolResult
from config import CODE_MODEL, get_constant
# from icecream import ic  # type: ignore # Removed
from pygments import highlight  # type: ignore
from pygments.formatters import HtmlFormatter  # type: ignore
from pygments.lexers import get_lexer_by_name, guess_lexer  # type: ignore
from utils.file_logger import (
    convert_to_docker_path,
    log_file_operation,
    get_language_from_extension,
)
from system_prompt.code_prompts import code_prompt_generate, code_skeleton_prompt
import ftfy




MODEL_STRING = CODE_MODEL  # Default model string, can be overridden in config

logger = logging.getLogger(__name__)

# --- Retry Predicate Function ---
def should_retry_llm_call(exception: Exception) -> bool:
    """Return True if the exception warrants a retry."""

    # Always retry on our custom LLMResponseError
    if isinstance(exception, LLMResponseError):
        logger.warning(
            f"Retry triggered by LLMResponseError: {str(exception)[:200]}"
        )
        return True

    # Retry on specific, transient OpenAI/network errors
    if isinstance(
        exception,
        (
            APIConnectionError,  # Network issues
            RateLimitError,  # Rate limits hit
            InternalServerError,  # Server-side errors from OpenAI (500 class)
        ),
    ):
        logger.warning(
            f"Retry triggered by OpenAI API Error ({type(exception).__name__}): {str(exception)[:200]}"
        )
        return True

    if isinstance(exception, APIStatusError):
        # Retry on general server errors (5xx) and specific client errors like Gateway Timeout (504)
        # or Request Timeout (408). 429 should ideally be RateLimitError.
        if exception.status_code >= 500 or exception.status_code in [
            408,
            429,
            502,
            503,
            504,
        ]:
            logger.warning(
                f"Retry triggered by OpenAI APIStatusError (status {exception.status_code}): {str(exception)[:200]}"
            )
            return True

    # For any other Exception, you might want to log it but not retry,
    # or add it here if known to be transient.
    # logger.error(f"Non-retryable exception encountered: {type(exception).__name__}: {exception}")
    return False


# --- LLMResponseError (already provided by you) ---
class LLMResponseError(Exception):
    """Custom exception for invalid or unusable responses from the LLM."""

    pass


class CodeCommand(str, Enum):
    WRITE_CODEBASE = "write_codebase"


class FileDetail(BaseModel):
    filename: str
    code_description: str
    external_imports: Optional[List[str]] = None
    internal_imports: Optional[List[str]] = None


class WriteCodeTool(BaseAnthropicTool):
    name: Literal["write_codebase_tool"] = "write_codebase_tool"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "Generates a codebase consisting of multiple files based on descriptions, skeletons, and import lists. "
        "This is the tool to use to generate a code files for a codebase, can create any amount of files. "
        "Use this tool to generate init files."
        "Creates skeletons first, then generates full code asynchronously, writing to the host filesystem."
        )

    def __init__(self, display=None):
        super().__init__(
            input_schema=None, display=display
        )  # Assuming BaseAnthropicTool takes these
        self.display = display
        logger.debug("Initializing WriteCodeTool")

    def to_params(self) -> dict:
        logger.debug(f"WriteCodeTool.to_params called with api_type: {self.api_type}")
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
                            "enum": [CodeCommand.WRITE_CODEBASE.value],
                            "description": "Command to perform. Only 'write_codebase' is supported.",
                        },
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "filename": {
                                        "type": "string",
                                        "description": "Name/path of the file relative to the project path.",
                                    },
                                    "code_description": {
                                        "type": "string",
                                        "description": "Detailed description of the code for this file.  This should be a comprehensive overview of the file's purpose, functionality, and any important details. It should include a general overview of the files implementation as well as how it interacts with the rest of the codebase.",
                                    },
                                    "external_imports": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of external libraries/packages required specifically for this file.",
                                        "default": [],
                                    },
                                    "internal_imports": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of internal modules/files within the codebase imported specifically by this file.",
                                        "default": [],
                                    },
                                },
                                "required": ["filename", "code_description"],
                            },
                            "description": "List of files to generate, each with a filename, description, and optional specific imports.",
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project directory (can be Docker-style or just project name). The actual write path will be resolved relative to the configured REPO_DIR on the host.",
                        },
                    },
                    "required": ["command", "files", "project_path"],
                },
            },
        }
        logger.debug(f"WriteCodeTool params: {params}")
        return params

    def _get_file_creation_log_content(self) -> str:
        """Reads the file creation log and returns its content."""
        from config import get_constant
        from pathlib import Path
        import logging

        logger = logging.getLogger(__name__)

        try:
            log_file_env_var = get_constant("LOG_FILE")
            if log_file_env_var:
                LOG_FILE_PATH = Path(log_file_env_var)
            else:
                # Default path relative to project root if constant is not set or None
                LOG_FILE_PATH = Path("logs") / "file_log.json"
        except KeyError:
            # Constant not found, use default
            LOG_FILE_PATH = Path("logs") / "file_log.json"
            logger.warning(
                "LOG_FILE constant not found in config, defaulting to %s",
                LOG_FILE_PATH,
            )
        except Exception as e:
            # Other potential errors from get_constant or Path()
            LOG_FILE_PATH = Path("logs") / "file_log.json"
            logger.error(
                "Error determining LOG_FILE_PATH, defaulting to %s: %s",
                LOG_FILE_PATH,
                e,
                exc_info=True,
            )

        try:
            if LOG_FILE_PATH.exists() and LOG_FILE_PATH.is_file():
                content = LOG_FILE_PATH.read_text(encoding="utf-8")
                if not content.strip():
                    logger.warning("File creation log %s is empty.", LOG_FILE_PATH)
                    return "File creation log is empty."
                return content
            else:
                logger.warning(
                    "File creation log not found or is not a file: %s", LOG_FILE_PATH
                )
                return "File creation log not found or is not a file."
        except IOError as e:
            logger.error(
                "IOError reading file creation log %s: %s",
                LOG_FILE_PATH,
                e,
                exc_info=True,
            )
            return f"Error reading file creation log: {e}"
        except Exception as e:
            logger.error(
                "Unexpected error reading file creation log %s: %s",
                LOG_FILE_PATH,
                e,
                exc_info=True,
            )
            return f"Unexpected error reading file creation log: {e}"

    # --- Logging Callback for Retries ---
    def _log_llm_retry_attempt(self, retry_state: RetryCallState):
        """Logs information about the current retry attempt."""
        fn_name = retry_state.fn.__name__ if retry_state.fn else "LLM_call"

        file_path_for_log = "unknown_file"
        # Try to get file_path or target_file_path from kwargs for contextual logging
        if retry_state.kwargs:
            fp_arg = retry_state.kwargs.get("file_path") or retry_state.kwargs.get(
                "target_file_path"
            )
            if isinstance(fp_arg, Path):
                file_path_for_log = fp_arg.name

        log_prefix = f"[bold magenta]Retry Log ({file_path_for_log})[/bold magenta] | "

        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            max_attempts_str = "N/A"
            stop_condition = retry_state.retry_object.stop
            if hasattr(stop_condition, "max_attempt_number"):
                max_attempts_str = str(stop_condition.max_attempt_number)

            log_msg = (
                f"{log_prefix}Retrying {fn_name} due to {type(exc).__name__}: {str(exc)[:150]}. "
                f"Attempt {retry_state.attempt_number} of {max_attempts_str}. "
                f"Waiting {retry_state.next_action.sleep:.2f}s..."
            )
        else:
            log_msg = (
                f"{log_prefix}Retrying {fn_name} (no direct exception, or outcome not yet available). "
                f"Attempt {retry_state.attempt_number}. Waiting {retry_state.next_action.sleep:.2f}s..."
            )
        logger.info(log_msg) # Rich text formatting removed

    async def __call__(
        self,
        *,
        command: CodeCommand,
        files: List[Dict[str, Any]],
        project_path: str,  # This might be a docker-style path or just a project name
        **kwargs,
        ) -> ToolResult:
        """
        Execute the write_codebase command.

        Args:
            command: The command to execute (should always be WRITE_CODEBASE).
            files: List of file details (filename, code_description, optional external_imports, optional internal_imports).
            project_path: Path/name identifier for the project.
            **kwargs: Additional parameters (ignored).

        Returns:
            A ToolResult object with the result of the operation.
        """
        if command != CodeCommand.WRITE_CODEBASE:
            return ToolResult(
                error=f"Unsupported command: {command}. Only 'write_codebase' is supported.",
                tool_name=self.name,
                command=command,
            )

        host_project_path_obj = None  # Initialize to None

        try:
            # --- START: Path Correction Logic ---
            # Ensure REPO_DIR is imported/available, e.g., from config import REPO_DIR
            host_repo_dir = get_constant("REPO_DIR")
            if not host_repo_dir:
                raise ValueError(
                    "REPO_DIR is not configured in config.py. Cannot determine host write path."
                )
            host_repo_path = Path(host_repo_dir)
            if not host_repo_path.is_dir():
                logger.warning(
                    f"Configured REPO_DIR '{host_repo_dir}' does not exist or is not a directory."
                )
                # Decide if you want to raise an error or attempt to create it
                # For now, let's proceed and let mkdir handle creation/errors later

            # project_path may include additional subdirectories. Preserve any
            # path components inside the repository directory so generated files
            # end up in e.g. repo/<prompt>/<app_name>/ rather than being
            # flattened.
            project_path_obj = Path(project_path)
            if project_path_obj.is_absolute():
                # If the absolute path contains the repo directory, strip that
                # portion so we keep the relative path under repo.
                if "repo" in project_path_obj.parts:
                    repo_index = project_path_obj.parts.index("repo")
                    relative_subpath = Path(*project_path_obj.parts[repo_index + 1 :])
                else:
                    # Fallback to just using the last component
                    relative_subpath = Path(project_path_obj.name)
            else:
                relative_subpath = project_path_obj

            # Construct the ACTUAL host path where files should be written
            host_project_path_obj = (host_repo_path / relative_subpath).resolve()
            logger.info(f"Resolved HOST project path for writing: {host_project_path_obj}")

            # Ensure the HOST directory exists
            host_project_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured host project directory exists: {host_project_path_obj}")
            # --- END: Path Correction Logic ---

            # Validate files input
            try:
                file_details = [FileDetail(**f) for f in files]
            except Exception as pydantic_error:
                logger.error(f"Pydantic validation error for 'files': {pydantic_error}", exc_info=True)
                return ToolResult(
                    error=f"Invalid format for 'files' parameter: {pydantic_error}",
                    tool_name=self.name,
                    command=command,
                )

            if not file_details:
                return ToolResult(
                    error="No files specified for write_codebase command",
                    tool_name=self.name,
                    command=command,
                )

            # --- Step 1: Generate Skeletons Asynchronously ---
            if self.display:
                self.display.add_message(
                    "assistant",
                    f"Generating skeletons for {len(file_details)} files...",
                )

            # CHANGE: Update the call to pass file_detail and all file_details
            skeleton_tasks = [
                self._call_llm_for_code_skeleton(
                    file,  # Pass the FileDetail object for the current file
                    host_project_path_obj
                    / file.filename,  # Pass the intended final host path
                    file_details,  # Pass the list of ALL FileDetail objects
                )
                for file in file_details  # Iterate through the validated FileDetail objects
            ]
            skeleton_results = await asyncio.gather(
                *skeleton_tasks, return_exceptions=True
            )

            skeletons: Dict[str, str] = {}
            errors_skeleton = []
            for i, result in enumerate(skeleton_results):
                filename_key = file_details[i].filename  # Use the relative filename
                if isinstance(result, Exception):
                    error_msg = (
                        f"Error generating skeleton for {filename_key}: {result}"
                    )
                    logger.error(error_msg, exc_info=True)
                    errors_skeleton.append(error_msg)
                    skeletons[filename_key] = (
                        f"# Error generating skeleton: {result}"  # Placeholder
                    )
                else:
                    skeletons[filename_key] = result
                    if self.display:
                        self.display.add_message(
                            "assistant", f"Skeleton generated for {filename_key}"
                        )

            # --- Step 2: Generate Full Code Asynchronously ---
            if self.display:
                self.display.add_message(
                    "assistant",
                    f"Generating full code for {len(file_details)} files using skeletons and specific imports...",
                )

            code_gen_tasks = [
                self._call_llm_to_generate_code(
                    file.code_description,
                    skeletons,
                    file.external_imports or [],
                    file.internal_imports or [],
                    # Pass the intended final host path to the code generator for context
                    host_project_path_obj / file.filename,
                )
                for file in file_details
            ]
            code_results = await asyncio.gather(*code_gen_tasks, return_exceptions=True)

            # --- Step 3: Write Files ---
            write_results = []
            errors_code_gen = []
            errors_write = []
            success_count = 0

            logger.info(
                f"Starting file writing phase for {len(code_results)} results to HOST path: {host_project_path_obj}"
            )

            for i, result in enumerate(code_results):
                file_detail = file_details[i]
                filename = file_detail.filename  # Relative filename
                # >>> USE THE CORRECTED HOST PATH FOR WRITING <<<
                absolute_path = (
                    host_project_path_obj / filename
                ).resolve()  # Ensure absolute path
                logger.info(f"Processing result for: {filename} (Host Path: {absolute_path})")

                if isinstance(result, Exception):
                    error_msg = f"Error generating code for {filename}: {result}"
                    logger.error(error_msg, exc_info=True)
                    errors_code_gen.append(error_msg)
                    write_results.append(
                        {"filename": filename, "status": "error", "message": error_msg}
                    )
                    # Attempt to write error file to the resolved host path
                    try:
                        logger.info(
                            f"Attempting to write error file for {filename} to {absolute_path}"
                        )
                        absolute_path.parent.mkdir(parents=True, exist_ok=True)
                        error_content = f"# Code generation failed: {result}\n\n# Skeleton:\n{skeletons.get(filename, '# Skeleton not available')}"
                        absolute_path.write_text(
                            error_content, encoding="utf-8", errors="replace"
                        )
                        logger.info(
                            f"Successfully wrote error file for {filename} to {absolute_path}"
                        )
                    except Exception as write_err:
                        logger.error(
                            f"Failed to write error file for {filename} to {absolute_path}: {write_err}", exc_info=True
                        )

                else:  # Code generation successful
                    code_content = result
                    logger.info(
                        f"Code generation successful for {filename}. Attempting to write to absolute host path: {absolute_path}"
                    )

                    if not code_content or not code_content.strip():
                        logger.warning(
                            f"Generated code content for {filename} is empty or whitespace only. Skipping write."
                        )
                        write_results.append(
                            {
                                "filename": filename,
                                "status": "error",
                                "message": "Generated code was empty",
                            }
                        )
                        continue  # Skip to next file

                    try:
                        logger.debug(f"Ensuring directory exists: {absolute_path.parent}")
                        absolute_path.parent.mkdir(parents=True, exist_ok=True)
                        operation = "modify" if absolute_path.exists() else "create"
                        logger.debug(f"Operation type for {filename}: {operation}")

                        fixed_code = ftfy.fix_text(code_content)
                        logger.debug(
                            f"Code content length for {filename} (after ftfy): {len(fixed_code)}"
                        )

                        # >>> THE WRITE CALL to the HOST path <<<
                        logger.debug(f"Executing write_text for: {absolute_path}")
                        absolute_path.write_text(
                            fixed_code, encoding="utf-8", errors="replace"
                        )
                        logger.info(f"Successfully executed write_text for: {absolute_path}")

                        # File existence and size check
                        if absolute_path.exists():
                            logger.info(
                                f"CONFIRMED: File exists at {absolute_path} after write."
                            )
                            try:
                                size = absolute_path.stat().st_size
                                logger.info(f"CONFIRMED: File size is {size} bytes.")
                                if size == 0 and len(fixed_code) > 0:
                                    logger.warning(
                                        "File size is 0 despite non-empty content being written!"
                                    )
                            except Exception as stat_err:
                                logger.warning(
                                    f"Could not get file stats for {absolute_path}: {stat_err}"
                                )
                        else:
                            logger.error(
                                f"FAILED: File DOES NOT exist at {absolute_path} immediately after write_text call!"
                            )

                        # Convert to Docker path FOR DISPLAY/LOGGING PURPOSES ONLY
                        docker_path_display = str(
                            absolute_path
                        )  # Default to host path if conversion fails
                        try:
                            # Ensure convert_to_docker_path can handle the absolute host path
                            docker_path_display = convert_to_docker_path(absolute_path)
                            logger.debug(
                                f"Converted host path {absolute_path} to display path {docker_path_display}"
                            )
                        except Exception as conv_err:
                            logger.warning(
                                f"Could not convert host path {absolute_path} to docker path for display: {conv_err}. Using host path for display."
                            )

                        # Log operation (using absolute_path for logging context)
                        try:
                            log_file_operation(
                                file_path=absolute_path,  # Log using the actual host path written to
                                operation=operation,
                                content=fixed_code,
                                metadata={
                                    "code_description": file_detail.code_description,
                                    "skeleton": skeletons.get(
                                        filename
                                    ),  # Use relative filename key
                                },
                            )
                            logger.debug(f"Logged file operation for {absolute_path}")
                        except Exception as log_error:
                            logger.error(
                                f"Failed to log code writing for {filename} ({absolute_path}): {log_error}", exc_info=True
                            )

                        # Use docker_path_display in the results if that's what the UI expects
                        write_results.append(
                            {
                                "filename": str(docker_path_display),
                                "status": "success",
                                "operation": operation,
                                # Add the generated code here
                                "code": fixed_code,
                            }
                        )
                        success_count += 1
                        logger.info(
                            f"Successfully processed and wrote {filename} to {absolute_path}"
                        )

                        # Display generated code (use docker_path_display if needed by UI)
                        if self.display:
                            self.display.add_message(
                                "user",
                                f"Code for {docker_path_display} generated successfully:",
                            )  # Use display path
                            language = get_language_from_extension(absolute_path.suffix)
                            formatted_code = html_format_code(fixed_code, language)  # noqa: F841
                            # Ensure display can handle html format correctly
                            # self.display.add_message("tool", {"html": formatted_code})
                            self.display.add_message("tool", fixed_code)

                    except Exception as write_error:
                        logger.error(
                            f"Caught exception during write operation for {filename} at path {absolute_path}", exc_info=True
                        )
                        errors_write.append(
                            f"Error writing file {filename}: {write_error}"
                        )
                        write_results.append(
                            {
                                "filename": filename,
                                "status": "error",
                                "message": f"Error writing file {filename}: {write_error}",
                            }
                        )

            # --- Step 4: Format and Return Result ---
            final_status = "success"
            if errors_skeleton or errors_code_gen or errors_write:
                final_status = "partial_success" if success_count > 0 else "error"

            # Use the resolved host path in the final message
            output_message = f"Codebase generation finished. Status: {final_status}. {success_count}/{len(file_details)} files written successfully to HOST path '{host_project_path_obj}'."
            if errors_skeleton:
                output_message += f"\nSkeleton Errors: {len(errors_skeleton)}"
            if errors_code_gen:
                output_message += f"\nCode Generation Errors: {len(errors_code_gen)}"
            if errors_write:
                output_message += f"\nFile Write Errors: {len(errors_write)}"

            result_data = {
                "status": final_status,
                "message": output_message,
                "files_processed": len(file_details),
                "files_successful": success_count,
                "project_path": str(
                    host_project_path_obj
                ),  # Report the actual host path used
                "results": write_results,
                "errors": errors_skeleton + errors_code_gen + errors_write,
            }

            return ToolResult(
                output=self.format_output(result_data),
                tool_name=self.name,
                command=command,
            )

        except ValueError as ve:  # Catch specific config/path errors
            error_message = f"Configuration Error in WriteCodeTool __call__: {str(ve)}"
            logger.critical(error_message, exc_info=True)
            return ToolResult(error=error_message, tool_name=self.name, command=command)
        except Exception as e:
            error_message = f"Critical Error in WriteCodeTool __call__: {str(e)}"
            logger.critical("Critical error during codebase generation", exc_info=True)
            # Optionally include host_project_path_obj if it was set
            if host_project_path_obj:
                error_message += f"\nAttempted Host Path: {host_project_path_obj}"
            # print(error_message) # Replaced by logger
            return ToolResult(error=error_message, tool_name=self.name, command=command)

    # --- Helper for logging final output ---
    def _log_generated_output(self, content: str, file_path: Path, output_type: str):
        """Helper to log the final generated content (code or skeleton) to CODE_FILE."""
        try:
            code_log_file_path_str = get_constant("CODE_FILE")
            if code_log_file_path_str:
                CODE_FILE = Path(code_log_file_path_str)
                CODE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CODE_FILE, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n--- Generated {output_type} for: {str(file_path)} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
                    )
                    f.write(f"{content}\n")
                    f.write(f"--- End {output_type} for: {str(file_path)} ---\n")
        except Exception as file_error:
            logger.error(
                f"Failed to log generated {output_type} for {file_path.name} to {get_constant('CODE_FILE')}: {file_error}", exc_info=True
            )

    def format_output(self, data: dict) -> str:
        """
        Format the output of the tool for display.

        Args:
            data: The data returned by the tool

        Returns:
            A formatted string for display
        """
        if "error" in data:
            return f"Error: {data['error']}"

        if "output" in data:
            return data["output"]

        # Handle different commands
        command = data.get("command", "")

        if command == "write_code_to_file":
            if data.get("status") == "success":
                return f"Successfully wrote code to {data.get('file_path', 'unknown file')}"
            else:
                return f"Failed to write code: {data.get('error', 'Unknown error')}"

        elif command == "write_code_multiple_files":
            if data.get("status") in ["success", "partial_success"]:
                return f"Wrote {data.get('files_processed', 0)} files to {data.get('project_path', 'unknown path')}\n{data.get('files_results', '')}"
            else:
                return f"Failed to write files: {data.get('errors', 'Unknown error')}"

        elif command == "get_all_current_skeleton":
            from utils.file_logger import get_all_current_skeleton

            return get_all_current_skeleton()

        elif command == "get_revised_version":
            if data.get("status") == "success":
                return f"Successfully revised {data.get('file_path', 'unknown file')}"
            else:
                return f"Failed to revise file: {data.get('error', 'Unknown error')}"

        # Default case
        return str(data)

    # --- Helper function to get task description ---
    def _get_task_description(self) -> str:
        """
        Tries to read the task description from various locations.
        Returns the task description or a default message if not found.
        """
        repo_dir = get_constant("REPO_DIR")
        possible_paths = []
        if repo_dir and isinstance(repo_dir, str): # Ensure repo_dir is a string
            possible_paths.append(os.path.join(repo_dir, "logs", "task.txt"))
        possible_paths.extend([
            os.path.join("logs", "task.txt"),
            "task.txt"
        ])

        for path in possible_paths:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as task_file:
                        task_desc = task_file.read().strip()
                        if task_desc:
                            logger.info(f"Read task description from: {path}")
                            return task_desc
            except Exception as e:
                logger.warning(f"Error reading task file {path}: {e}", exc_info=True)

        logger.warning("task.txt not found in any specified location. Trying 'TASK' constant as fallback.")
        # Try to get from "TASK" constant as a last resort if file is not found
        task_from_constant = get_constant("TASK")
        if task_from_constant and task_from_constant != "NOT YET CREATED" and task_from_constant.strip(): # Check if not default or empty
            logger.info("Using task description from 'TASK' constant as fallback.")
            return task_from_constant

        logger.error("No overall task description provided (task.txt not found and TASK constant not set, default, or empty).")
        return "No overall task description provided (task.txt not found and TASK constant not set, default, or empty)."

    # --- Refactored Code Generation Method ---
    async def _call_llm_to_generate_code(
        self,
        code_description: str,
        all_skeletons: Dict[str, str],
        external_imports: List[str],
        internal_imports: List[str],
        file_path: Path,
        ) -> str:
        """Generate full file content using provided skeletons."""
        # if self.display is not None:
            # self.display.add_message(
            #     "assistant", f"Generating code for: {file_path.name}"
            # )

        skeleton_context = "\n\n---\n\n".join(
            f"### Skeleton for {fname}:\n```\n{skel}\n```"
            for fname, skel in all_skeletons.items()
        )
        agent_task = self._get_task_description()
        log_content = self._get_file_creation_log_content()

        prepared_messages = code_prompt_generate(
            current_code_base="", # Assuming current_code_base is handled or intentionally empty
            code_description=code_description,
            research_string="", # Assuming research_string is handled or intentionally empty
            agent_task=agent_task,
            skeletons=skeleton_context,
            external_imports=external_imports,
            internal_imports=internal_imports,
            target_file=str(file_path.name),
            file_creation_log_content=log_content,
        )

        model_to_use = get_constant("CODE_GEN_MODEL") or MODEL_STRING
        final_code_string = (
            f"# Error: Code generation failed for {file_path.name} after all retries."
        )

        try:
            final_code_string = await self._llm_generate_code_core_with_retry(
                prepared_messages=prepared_messages,
                file_path=file_path,
                model_to_use=model_to_use,
            )
        except LLMResponseError as e:
            logger.error(f"LLMResponseError for {file_path.name} after all retries: {e}", exc_info=True)
            # rr( # Replaced by logger
            #     f"[bold red]LLM generated invalid content for {file_path.name} after retries: {e}[/bold red]"
            # )
            final_code_string = f"# Error generating code for {file_path.name}: LLMResponseError - {str(e)}"
        except APIError as e:  # Catch specific OpenAI errors
            logger.error(
                f"OpenAI APIError for {file_path.name} after all retries: {type(e).__name__} - {e}", exc_info=True
            )
            # rr( # Replaced by logger
            #     f"[bold red]LLM call failed due to APIError for {file_path.name} after retries: {e}[/bold red]"
            # )
            final_code_string = (
                f"# Error generating code for {file_path.name}: API Error - {str(e)}"
            )
        except Exception as e:
            logger.critical(
                f"Unexpected error during code generation for {file_path.name} after retries: {type(e).__name__} - {e}", exc_info=True
            )
            # rr( # Replaced by logger
            #     f"[bold red]LLM call ultimately failed for {file_path.name} due to unexpected error: {e}[/bold red]"
            # )
            final_code_string = (
                f"# Error generating code for {file_path.name} (final): {str(e)}"
            )

        self._log_generated_output(final_code_string, file_path, "Code")
        return final_code_string

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(should_retry_llm_call),
        reraise=True,
        before_sleep=_log_llm_retry_attempt,
        )
    async def _llm_generate_code_core_with_retry(
        self,
        prepared_messages: List[Dict[str, str]],
        file_path: Path,
        model_to_use: str,
        ) -> str:
        """Call the LLM to produce final code with retry logic."""
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set."
            )  # Should fail fast if not retryable

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY
        )

        current_attempt = getattr(
            self._llm_generate_code_core_with_retry.retry.statistics,
            "attempt_number",
            1,
        )
        logger.info(
            f"LLM Code Gen for {file_path.name}: Model {model_to_use}, Attempt {current_attempt}"
        )

        completion = await client.chat.completions.create(
            model=model_to_use, messages=prepared_messages
        )

        if not (
            completion
            and completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            logger.error(f"No valid completion content received for {file_path.name}")
            # rr( # Replaced by logger
            #     f"[bold red]Invalid or empty completion content from LLM for {file_path.name}[/bold red]"
            # )
            raise LLMResponseError(
                f"Invalid or empty completion content from LLM for {file_path.name}"
            )

        raw_code_string = completion.choices[0].message.content
        # Assuming self.extract_code_block is defined in your class
        code_string, detected_language = self.extract_code_block(
            raw_code_string, file_path
        )

        logger.info(
            f"Extracted code for {file_path.name}. Lang: {detected_language}. Raw len: {len(raw_code_string)}, Extracted len: {len(code_string or '')}"
        )

        if code_string == "No Code Found":  # Critical check
            if raw_code_string.strip():
                logger.error(
                    f"Could not extract code for {file_path.name}, raw response was not empty. LLM might have misunderstood."
                )
                raise LLMResponseError(
                    f"Extracted 'No Code Found' for {file_path.name}. Raw: '{raw_code_string[:100]}...'"
                )
            else:
                logger.error(
                    f"LLM response for {file_path.name} was effectively empty (raw string)."
                )
                raise LLMResponseError(
                    f"LLM response for {file_path.name} was effectively empty (raw string)."
                )

        if code_string.startswith(
            f"# Error: Code generation failed for {file_path.name}"
        ) or code_string.startswith(f"# Failed to generate code for {file_path.name}"):
            logger.error(
                f"LLM returned a placeholder error message for {file_path.name}: {code_string[:100]}"
            )
            raise LLMResponseError(
                f"LLM returned placeholder error for {file_path.name}: {code_string[:100]}"
            )

        return code_string

    # --- Refactored Skeleton Generation Method ---
    async def _call_llm_for_code_skeleton(
        self,
        file_detail: FileDetail,
        file_path: Path,
        all_file_details: List[FileDetail], # This parameter is no longer used directly by code_skeleton_prompt but might be kept for other reasons or future use.
        ) -> str:
        """Request a skeleton from the LLM for the target file."""
        target_file_name = file_path.name
        # if self.display:
        #     self.display.add_message(
        #         "assistant", f"Generating skeleton for {target_file_name}"
        #     )

        log_content = self._get_file_creation_log_content()
        agent_task = self._get_task_description()

        external_imports = file_detail.external_imports
        internal_imports = file_detail.internal_imports
        # all_files_dict_list = [f.model_dump() for f in all_file_details] # Removed as per requirement

        prepared_messages = code_skeleton_prompt(
            code_description=file_detail.code_description,
            target_file=target_file_name,
            agent_task=agent_task,
            external_imports=external_imports,
            internal_imports=internal_imports,
            file_creation_log_content=log_content,
        )

        model_to_use = get_constant("SKELETON_GEN_MODEL") or MODEL_STRING
        final_skeleton_string = f"# Error: Skeleton generation failed for {target_file_name} after all retries."

        try:
            final_skeleton_string = await self._llm_generate_skeleton_core_with_retry(
                prepared_messages=prepared_messages,
                target_file_path=file_path,
                model_to_use=model_to_use,
            )
        except LLMResponseError as e:
            logger.error(
                f"LLMResponseError for skeleton {target_file_name} after all retries: {e}", exc_info=True
            )
            # rr( # Replaced by logger
            #     f"[bold red]LLM generated invalid skeleton for {target_file_name} after retries: {e}[/bold red]"
            # )
            final_skeleton_string = f"# Error generating skeleton for {target_file_name}: LLMResponseError - {str(e)}"
        except APIError as e:  # Catch specific OpenAI errors
            logger.error(
                f"OpenAI APIError for skeleton {target_file_name} after all retries: {type(e).__name__} - {e}", exc_info=True
            )
            # rr( # Replaced by logger
            #     f"[bold red]LLM skeleton call failed due to APIError for {target_file_name} after retries: {e}[/bold red]"
            # )
            final_skeleton_string = f"# Error generating skeleton for {target_file_name}: API Error - {str(e)}"
        except Exception as e:
            logger.critical(
                f"Unexpected error during skeleton generation for {target_file_name} after retries: {type(e).__name__} - {e}", exc_info=True
            )
            # rr( # Replaced by logger
            #     f"[bold red]LLM skeleton call ultimately failed for {target_file_name} due to unexpected error: {e}[/bold red]"
            # )
            final_skeleton_string = (
                f"# Error generating skeleton for {target_file_name} (final): {str(e)}"
            )

        self._log_generated_output(final_skeleton_string, file_path, "Skeleton")
        logger.debug(
            f"Final Skeleton for {target_file_name}:\n{final_skeleton_string[:300]}..."
        )  # Log snippet
        return final_skeleton_string

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(should_retry_llm_call),
        reraise=True,
        before_sleep=_log_llm_retry_attempt,
        )
    async def _llm_generate_skeleton_core_with_retry(
        self,
        prepared_messages: List[Dict[str, str]],
        target_file_path: Path,
        model_to_use: str,
        ) -> str:
        """Generate a file skeleton with retry logic."""
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY
        )

        current_attempt = getattr(
            self._llm_generate_skeleton_core_with_retry.retry.statistics,
            "attempt_number",
            1,
        )
        logger.info(
            f"LLM Skeleton Gen for {target_file_path.name}: Model {model_to_use}, Attempt {current_attempt}"
        )

        completion = await client.chat.completions.create(
            model=model_to_use, messages=prepared_messages
        )

        if not (
            completion
            and completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            logger.error(
                f"No valid skeleton completion content received for {target_file_path.name}"
            )
            # rr( # Replaced by logger
            #     f"[bold red]Invalid or empty skeleton completion content from LLM for {target_file_path.name}[/bold red]"
            # )
            raise LLMResponseError(
                f"Invalid or empty skeleton completion from LLM for {target_file_path.name}"
            )

        raw_skeleton = completion.choices[0].message.content
        # Assuming self.extract_code_block is defined
        skeleton_string, detected_language = self.extract_code_block(
            raw_skeleton, target_file_path
        )

        logger.info(
            f"Extracted skeleton for {target_file_path.name}. Lang: {detected_language}. Raw len: {len(raw_skeleton)}, Extracted len: {len(skeleton_string or '')}"
        )

        if skeleton_string == "No Code Found":  # Critical check
            if raw_skeleton.strip():
                logger.error(
                    f"Could not extract skeleton for {target_file_path.name}, raw response was not empty."
                )
                raise LLMResponseError(
                    f"Extracted 'No Code Found' for skeleton {target_file_path.name}. Raw: '{raw_skeleton[:100]}...'"
                )
            else:
                logger.error(
                    f"LLM response for skeleton {target_file_path.name} was effectively empty (raw string)."
                )
                raise LLMResponseError(
                    f"LLM response for skeleton {target_file_path.name} was effectively empty (raw string)."
                )

        if skeleton_string.startswith(
            f"# Error: Skeleton generation failed for {target_file_path.name}"
        ) or skeleton_string.startswith(
            f"# Failed to generate skeleton for {target_file_path.name}"
        ):
            logger.error(
                f"LLM returned a placeholder error for skeleton {target_file_path.name}: {skeleton_string[:100]}"
            )
            raise LLMResponseError(
                f"LLM returned placeholder error for skeleton {target_file_path.name}: {skeleton_string[:100]}"
            )

        skeleton_string = ftfy.fix_text(skeleton_string)  # Apply ftfy only on success
        return skeleton_string


    def extract_code_block(
        self, text: str, file_path: Optional[Path] = None
        ) -> tuple[str, str]:
        """
        Extracts code based on file type. Special handling for Markdown files.
        Improved language guessing.
        Returns tuple of (content, language).
        """
        if file_path is not None and str(file_path).lower().endswith(
            (".md", ".markdown")
        ):
            return text, "markdown"

        if not text or not text.strip():
            return "No Code Found", "unknown"  # Consistent 'unknown'

        start_marker = text.find("```")
        if start_marker == -1:
            # No backticks, try guessing language from content
            try:
                language = guess_lexer(text).aliases[0]
                # Return the whole text as the code block
                return text.strip(), language
            except Exception:  # pygments.util.ClassNotFound or others
                logger.warning("Could not guess language for code without backticks.")
                return text.strip(), "unknown"  # Return unknown if guess fails

        # Found opening backticks ```
        language_line_end = text.find("\n", start_marker)
        if language_line_end == -1:  # Handle case where ``` is at the very end
            language_line_end = len(text)

        language = text[start_marker + 3 : language_line_end].strip().lower()

        code_start = language_line_end + 1
        end_marker = text.find("```", code_start)

        if end_marker == -1:
            # No closing backticks found, assume rest of text is code
            code_block = text[code_start:].strip()
        else:
            code_block = text[code_start:end_marker].strip()

        # If language wasn't specified after ```, try guessing from the extracted block
        if not language and code_block:
            try:
                language = guess_lexer(code_block).aliases[0]
            except Exception:
                logger.warning(
                    f"Could not guess language for extracted code block (File: {file_path})."
                )
                language = "unknown"  # Fallback if guess fails

        # If language is still empty, default to 'unknown'
        if not language:
            language = "unknown"

        return code_block if code_block else "No Code Found", language


def html_format_code(code, extension):
    """Format code with syntax highlighting for HTML display."""
    try:
        # Try to get a lexer based on the file extension
        try:
            lexer = get_lexer_by_name(extension.lower().lstrip("."))
        except Exception:
            # If that fails, try to guess the lexer from the code content
            lexer = guess_lexer(code)

        # Use a nice style for the highlighting
        formatter = HtmlFormatter(style="monokai", linenos=True, cssclass="source")

        # Highlight the code
        highlighted = highlight(code, lexer, formatter)

        # Add some CSS for better display
        css = formatter.get_style_defs(".source")
        html = f"""
            <style>
            {css}
            .source {{ background-color: #272822; padding: 10px; border-radius: 5px; }}
            </style>
            {highlighted}
            """
        return html
    except Exception:
        return f"<pre>{code}</pre>"
