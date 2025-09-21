# ignore: type
"""Utility tool for generating codebases via LLM calls."""
import asyncio
import json
import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import ftfy

# pyright: ignore[reportMissingImports]
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict

# from icecream import ic  # type: ignore # Removed
from pygments import highlight  # type: ignore
from pygments.formatters import HtmlFormatter  # type: ignore
from pygments.lexers import get_lexer_by_name, guess_lexer  # type: ignore

# from rich import print as rr # Removed
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import CODE_LIST, CODE_MODEL, get_constant
from system_prompt.code_prompts import code_prompt_generate
from tools.base import BaseAnthropicTool, ToolResult

# Import PictureGenerationTool for handling image files
from tools.create_picture import PictureCommand, PictureGenerationTool
from utils.file_logger import (
    convert_to_docker_path,
    get_language_from_extension,
    log_file_operation,
)

MODEL_STRING = CODE_MODEL  # Default model string, can be overridden in config

logger = logging.getLogger(__name__)

# Common image file extensions
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg',
    '.ico', '.psd', '.raw', '.heic', '.heif'
}


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS


# --- Retry Predicate Function ---
def should_retry_llm_call(exception: Exception) -> bool:
    """Return True if the exception warrants a retry."""

    # Always retry on our custom LLMResponseError
    if isinstance(exception, LLMResponseError):
        logger.warning(
            f"Retry triggered by LLMResponseError: {str(exception)[:200]}")
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
    model_config = ConfigDict(extra="forbid")
    filename: str
    code_description: str
    external_imports: Optional[List[str]] = None
    internal_imports: Optional[List[str]] = None


class WriteCodeTool(BaseAnthropicTool):
    name: Literal["write_codebase_tool"] = "write_codebase_tool"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "Generates a full or partial codebase consisting of up to 5 files based on descriptions and import lists. "
        "This is the tool to use to generate code files for a codebase, can create up to 5 files at a time. "
        "Use this tool to generate init files."
        "Generates full code asynchronously, writing to the host filesystem."
        "Generates a codebase from structured descriptions and import specifications."
    )

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        logger.debug("Initializing WriteCodeTool")
        # Initialize PictureGenerationTool for handling image files
        self.picture_tool = PictureGenerationTool(display=display)

    def _format_terminal_output(self, 
                               command: str, 
                               files: List[str] = None, 
                               result: str = None, 
                               error: str = None,
                               additional_info: str = None) -> str:
        """Format write code operations to look like terminal output."""
        output_lines = ["```console"]
        
        # Format the command with a pseudo-shell prompt
        if files:
            files_str = " ".join(files)
            output_lines.append(f"$ write_code {command} {files_str}")
        else:
            output_lines.append(f"$ write_code {command}")
        
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

    def to_params(self) -> dict:
        logger.debug(
            f"WriteCodeTool.to_params called with api_type: {self.api_type}")
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type":
                            "string",
                            "enum": [CodeCommand.WRITE_CODEBASE.value],
                            "description":
                            "Command to perform. Only 'write_codebase' is supported.",
                        },
                        "files": {
                            "type":
                            "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "filename": {
                                        "type":
                                        "string",
                                        "description":
                                        " The relative path for the file. The main entry point to the code should NOT have a directory structure, e.g., just `main.py`. Any other files that you would like to be in a directory structure should be specified with their relative paths, e.g., `/utils/helpers.py`.",
                                    },
                                    "code_description": {
                                        "type":
                                        "string",
                                        "description":
                                        "Detailed description of the code for this file.  This should be a comprehensive overview of the file's purpose, functionality, and any important details. It should include a general overview of the files implementation as well as how it interacts with the rest of the codebase.",
                                    },
                                    "external_imports": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "description":
                                        "List of external libraries/packages required specifically for this file.",
                                        "default": [],
                                    },
                                    "internal_imports": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                        "description":
                                        "List of internal modules/files within the codebase imported specifically by this file.",
                                        "default": [],
                                    },
                                },
                                "required": ["filename", "code_description"],
                            },
                            "description":
                            "List of files to generate, each with a filename, description, and optional specific imports.",
                        },
                        # Deprecated project_path removed. Files are now written
                        # directly relative to REPO_DIR.
                    },
                    "required": ["command", "files"],
                },
            },
        }
        logger.debug(f"WriteCodeTool params: {params}")
        return params

    def _get_file_creation_log_content(self) -> str:
        """Reads the file creation log and returns its content."""
        import logging
        from pathlib import Path

        from config import get_constant

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
                    logger.warning("File creation log %s is empty.",
                                   LOG_FILE_PATH)
                    return "File creation log is empty."
                return content
            else:
                logger.warning(
                    "File creation log not found or is not a file: %s",
                    LOG_FILE_PATH)
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
            fp_arg = retry_state.kwargs.get(
                "file_path") or retry_state.kwargs.get("target_file_path")
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
                f"Waiting {retry_state.next_action.sleep:.2f}s...")
        else:
            log_msg = (
                f"{log_prefix}Retrying {fn_name} (no direct exception, or outcome not yet available). "
                f"Attempt {retry_state.attempt_number}. Waiting {retry_state.next_action.sleep:.2f}s..."
            )
        logger.info(log_msg)  # Rich text formatting removed

    async def __call__(
        self,
        *,
        command: CodeCommand,
        files: List[Dict[str, Any]],
        **kwargs,
    ) -> ToolResult:
        """
        Execute the write_codebase command. All files are created relative to
        the configured REPO_DIR.

        Args:
            command: The command to execute (should always be WRITE_CODEBASE).
            files: List of file details (filename, code_description, optional external_imports, optional internal_imports).
            **kwargs: Additional parameters (ignored).

        Returns:
            A ToolResult object with the result of the operation.
        """
        if command != CodeCommand.WRITE_CODEBASE:
            return ToolResult(
                error=
                f"Unsupported command: {command}. Only 'write_codebase' is supported.",
                tool_name=self.name,
                command=command,
            )

        repo_path_obj = None  # Initialize to None

        try:
            # Determine the root path where files should be written
            host_repo_dir = get_constant("REPO_DIR")
            if not host_repo_dir:
                raise ValueError(
                    "REPO_DIR is not configured in config.py. Cannot determine host write path."
                )

            repo_path_obj = Path(host_repo_dir).resolve()
            if not repo_path_obj.exists():
                repo_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Resolved repository path for writing: {repo_path_obj}")

            # Validate files input
            try:
                file_details = [FileDetail(**f) for f in files]
            except Exception as pydantic_error:
                logger.error(
                    f"Pydantic validation error for 'files': {pydantic_error}",
                    exc_info=True)
                return ToolResult(
                    error=
                    f"Invalid format for 'files' parameter: {pydantic_error}",
                    tool_name=self.name,
                    command=command,
                )

            if not file_details:
                return ToolResult(
                    error="No files specified for write_codebase command",
                    tool_name=self.name,
                    command=command,
                )

            # --- Check for image files and handle them with PictureGenerationTool ---
            image_files = []
            code_files = []
            image_results = []

            for file_detail in file_details:
                if is_image_file(file_detail.filename):
                    image_files.append(file_detail)
                else:
                    code_files.append(file_detail)

            # Handle image files with PictureGenerationTool
            if image_files:
                if self.display:
                    image_filenames = [file.filename for file in image_files]
                    console_output = self._format_terminal_output(
                        command="generate_images",
                        files=image_filenames,
                        result="Detected image files, using PictureGenerationTool",
                        additional_info=f"Files to generate:\n" + "\n".join(f"  - {filename}" for filename in image_filenames)
                    )
                    self.display.add_message("assistant", console_output)

                for image_file in image_files:
                    try:
                        # Use the code_description as the prompt for image generation
                        # Set default dimensions if not specified
                        width = 1024
                        height = 1024

                        result = await self.picture_tool(
                            command=PictureCommand.CREATE,
                            prompt=image_file.code_description,
                            output_path=image_file.filename,
                            width=width,
                            height=height)
                        image_results.append(result)

                        if self.display:
                            console_output = self._format_terminal_output(
                                command="generate_image",
                                files=[image_file.filename],
                                result=f"Image generated successfully",
                                additional_info=f"Generated: {image_file.filename}"
                            )
                            self.display.add_message("assistant", console_output)
                    except Exception as e:
                        error_msg = f"Error generating image {image_file.filename}: {str(e)}"
                        logger.error(error_msg)
                        image_results.append(
                            ToolResult(
                                error=error_msg,
                                tool_name=self.name,
                                command=command,
                            ))

            # If only image files were requested, return the image results
            if not code_files:
                if len(image_results) == 1:
                    return image_results[0]
                else:
                    # Combine multiple image results
                    success_count = sum(1 for r in image_results
                                        if not r.error)
                    error_count = len(image_results) - success_count

                    output_messages = []
                    for i, result in enumerate(image_results):
                        if result.error:
                            output_messages.append(
                                f"Error with {image_files[i].filename}: {result.error}"
                            )
                        else:
                            output_messages.append(
                                f"Successfully generated {image_files[i].filename}"
                            )

                    return ToolResult(
                        output=
                        f"Generated {success_count} images successfully, {error_count} errors.\n"
                        + "\n".join(output_messages),
                        tool_name=self.name,
                        command=command,
                    )

            # Update file_details to only include code files for the rest of the process
            file_details = code_files

            # --- Generate Code Asynchronously ---
            if file_details:  # Only proceed if there are code files to process
                if self.display:
                    code_filenames = [file.filename for file in file_details]
                    console_output = self._format_terminal_output(
                        command="generate_code",
                        files=code_filenames,
                        result="Starting code generation process",
                        additional_info=f"Files to generate:\n" + "\n".join(f"  - {filename}" for filename in code_filenames)
                    )
                    self.display.add_message("assistant", console_output)

                code_gen_tasks = [
                    self._call_llm_to_generate_code(
                        file.code_description,
                        file.external_imports or [],
                        file.internal_imports or [],
                        # Pass the intended final host path to the code generator for context
                        repo_path_obj / file.filename,
                    ) for file in file_details
                ]
                code_results = await asyncio.gather(*code_gen_tasks,
                                                    return_exceptions=True)

                # --- Write Files ---
                write_results = []
                errors_code_gen = []
                errors_write = []
                success_count = 0
            else:
                # Initialize variables when there are no code files to process
                write_results = []
                errors_code_gen = []
                errors_write = []
                success_count = 0

            if file_details:  # Only process code results if there are code files
                logger.info(
                    f"Starting file writing phase for {len(code_results)} results to HOST path: {repo_path_obj}"
                )

                for i, result in enumerate(code_results):
                    file_detail = file_details[i]
                    filename = file_detail.filename  # Relative filename
                    # >>> USE THE CORRECTED HOST PATH FOR WRITING <<<
                    absolute_path = (
                        repo_path_obj /
                        filename).resolve()  # Ensure absolute path
                    logger.info(
                        f"Processing result for: {filename} (Host Path: {absolute_path})"
                    )

                    if isinstance(result, Exception):
                        error_msg = f"Error generating code for {filename}: {result}"
                        logger.error(error_msg, exc_info=True)
                        errors_code_gen.append(error_msg)
                        write_results.append({
                            "filename": filename,
                            "status": "error",
                            "message": error_msg
                        })
                        # Attempt to write error file to the resolved host path
                        try:
                            logger.info(
                                f"Attempting to write error file for {filename} to {absolute_path}"
                            )
                            absolute_path.parent.mkdir(parents=True,
                                                       exist_ok=True)
                            error_content = (
                                f"# Code generation failed for {filename}: {result}\n\n"
                                "# Code description:\n"
                                f"{file_detail.code_description}"
                            )
                            absolute_path.write_text(error_content,
                                                     encoding="utf-8",
                                                     errors="replace")
                            logger.info(
                                f"Successfully wrote error file for {filename} to {absolute_path}"
                            )
                        except Exception as write_err:
                            logger.error(
                                f"Failed to write error file for {filename} to {absolute_path}: {write_err}",
                                exc_info=True)

                    else:  # Code generation successful
                        code_content = result
                        logger.info(
                            f"Code generation successful for {filename}. Attempting to write to absolute host path: {absolute_path}"
                        )

                        if not code_content or not code_content.strip():
                            logger.warning(
                                f"Generated code content for {filename} is empty or whitespace only. Skipping write."
                            )
                            write_results.append({
                                "filename":
                                filename,
                                "status":
                                "error",
                                "message":
                                "Generated code was empty",
                            })
                            continue  # Skip to next file

                        try:
                            logger.debug(
                                f"Ensuring directory exists: {absolute_path.parent}"
                            )
                            absolute_path.parent.mkdir(parents=True,
                                                       exist_ok=True)
                            operation = "modify" if absolute_path.exists(
                            ) else "create"
                            logger.debug(
                                f"Operation type for {filename}: {operation}")

                            fixed_code = ftfy.fix_text(code_content)
                            logger.debug(
                                f"Code content length for {filename} (after ftfy): {len(fixed_code)}"
                            )

                            # >>> THE WRITE CALL to the HOST path <<<
                            logger.debug(
                                f"Executing write_text for: {absolute_path}")
                            absolute_path.write_text(fixed_code,
                                                     encoding="utf-8",
                                                     errors="replace")
                            logger.info(
                                f"Successfully executed write_text for: {absolute_path}"
                            )

                            # File existence and size check
                            if absolute_path.exists():
                                logger.info(
                                    f"CONFIRMED: File exists at {absolute_path} after write."
                                )
                                try:
                                    size = absolute_path.stat().st_size
                                    logger.info(
                                        f"CONFIRMED: File size is {size} bytes."
                                    )
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
                                docker_path_display = convert_to_docker_path(
                                    absolute_path)
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
                                    file_path=
                                    absolute_path,  # Log using the actual host path written to
                                    operation=operation,
                                    content=fixed_code,
                                    metadata={
                                        "code_description":
                                        file_detail.code_description,
                                    },
                                )
                                logger.debug(
                                    f"Logged file operation for {absolute_path}"
                                )
                            except Exception as log_error:
                                logger.error(
                                    f"Failed to log code writing for {filename} ({absolute_path}): {log_error}",
                                    exc_info=True)

                            # Use docker_path_display in the results if that's what the UI expects
                            write_results.append({
                                "filename":
                                str(docker_path_display),
                                "status":
                                "success",
                                "operation":
                                operation,
                                # Add the generated code here
                                "code":
                                fixed_code,
                            })
                            success_count += 1
                            logger.info(
                                f"Successfully processed and wrote {filename} to {absolute_path}"
                            )

                            # Display generated code (use docker_path_display if needed by UI)
                            if self.display:
                                language = get_language_from_extension(
                                    absolute_path.suffix)
                                # Determine the language for highlighting.
                                # The 'language' variable from get_language_from_extension might be simple (e.g., 'py')
                                # or more specific if html_format_code needs it.
                                # For pygments, simple extensions usually work.
                                formatted_code = html_format_code(
                                    fixed_code, language
                                    or absolute_path.suffix.lstrip('.'))
                                
                                # Show console output for successful file creation
                                console_output = self._format_terminal_output(
                                    command="write_file",
                                    files=[filename],
                                    result=f"File written successfully",
                                    additional_info=f"Path: {docker_path_display}\nSize: {len(fixed_code)} characters\nLanguage: {language or 'unknown'}"
                                )
                                self.display.add_message("assistant", console_output)
                                
                                # Also show the formatted code
                                # self.display.add_message("tool", formatted_code)

                        except Exception as write_error:
                            logger.error(
                                f"Caught exception during write operation for {filename} at path {absolute_path}",
                                exc_info=True)
                            errors_write.append(
                                f"Error writing file {filename}: {write_error}"
                            )
                            write_results.append({
                                "filename":
                                filename,
                                "status":
                                "error",
                                "message":
                                f"Error writing file {filename}: {write_error}",
                            })

            # --- Step 4: Format and Return Result ---
            final_status = "success"
            if errors_code_gen or errors_write:
                final_status = "partial_success" if success_count > 0 else "error"

            # Calculate image generation results
            image_success_count = sum(1 for r in image_results if not r.error)
            image_error_count = len(image_results) - image_success_count
            total_files = len(file_details) + len(image_files)
            total_success = success_count + image_success_count

            # Use the resolved host path in the final message
            if image_files:
                output_message = f"File generation finished. Status: {final_status}. {total_success}/{total_files} files created successfully to HOST path '{repo_path_obj}'."
                output_message += f"\n  - Code files: {success_count}/{len(file_details)} successful"
                output_message += f"\n  - Image files: {image_success_count}/{len(image_files)} successful"
            else:
                output_message = f"Codebase generation finished. Status: {final_status}. {success_count}/{len(file_details)} files written successfully to HOST path '{repo_path_obj}'."

            if errors_code_gen:
                output_message += f"\nCode Generation Errors: {len(errors_code_gen)}"
            if errors_write:
                output_message += f"\nFile Write Errors: {len(errors_write)}"
            if image_error_count > 0:
                output_message += f"\nImage Generation Errors: {image_error_count}"

            # Add image results to write_results
            for i, image_result in enumerate(image_results):
                if image_result.error:
                    write_results.append({
                        "filename":
                        image_files[i].filename,
                        "status":
                        "error",
                        "message":
                        f"Error generating image: {image_result.error}",
                    })
                else:
                    write_results.append({
                        "filename":
                        image_files[i].filename,
                        "status":
                        "success",
                        "operation":
                        "image_generation",
                        "message":
                        f"Successfully generated image: {image_files[i].filename}",
                    })

            result_data = {
                "status": final_status,
                "message": output_message,
                "files_processed": total_files,
                "files_successful": total_success,
                "code_files_processed": len(file_details),
                "code_files_successful": success_count,
                "image_files_processed": len(image_files),
                "image_files_successful": image_success_count,
                "write_path": str(repo_path_obj),
                "results": write_results,
                "errors": errors_code_gen + errors_write,
            }

            # Display final completion message
            if self.display:
                status_text = "completed successfully" if final_status == "success" else f"completed with status: {final_status}"
                all_filenames = [f.filename for f in file_details] + [f.filename for f in image_files]
                console_output = self._format_terminal_output(
                    command="write_codebase",
                    files=all_filenames,
                    result=f"Codebase generation {status_text}",
                    additional_info=f"Total files: {total_files}\nSuccessful: {total_success}\nCode files: {success_count}/{len(file_details)}\nImage files: {image_success_count}/{len(image_files)}\nWrite path: {repo_path_obj}"
                )
                self.display.add_message("assistant", console_output)

            return ToolResult(
                output=self.format_output(result_data),
                tool_name=self.name,
                command=command,
            )

        except ValueError as ve:  # Catch specific config/path errors
            error_message = f"Configuration Error in WriteCodeTool __call__: {str(ve)}"
            logger.critical(error_message, exc_info=True)
            
            if self.display:
                console_output = self._format_terminal_output(
                    command="write_codebase",
                    error=f"Configuration Error: {str(ve)}"
                )
                self.display.add_message("assistant", console_output)
            
            return ToolResult(error=error_message,
                              tool_name=self.name,
                              command=command)
        except Exception as e:
            error_message = f"Critical Error in WriteCodeTool __call__: {str(e)}"
            logger.critical("Critical error during codebase generation",
                            exc_info=True)
            
            if self.display:
                console_output = self._format_terminal_output(
                    command="write_codebase",
                    error=f"Critical Error: {str(e)}"
                )
                self.display.add_message("assistant", console_output)
            # Optionally include repo_path_obj if it was set
            if repo_path_obj:
                error_message += f"\nAttempted Host Path: {repo_path_obj}"
            # print(error_message) # Replaced by logger
            return ToolResult(error=error_message,
                              tool_name=self.name,
                              command=command)

    # --- Helper for logging final output ---
    def _log_generated_output(self, content: str, file_path: Path,
                              output_type: str):
        """Helper to log the final generated content to CODE_FILE."""
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
                    f.write(
                        f"--- End {output_type} for: {str(file_path)} ---\n")
        except Exception as file_error:
            logger.error(
                f"Failed to log generated {output_type} for {file_path.name} to {get_constant('CODE_FILE')}: {file_error}",
                exc_info=True)

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
                return f"Wrote {data.get('files_processed', 0)} files to {data.get('write_path', 'unknown path')}\n{data.get('files_results', '')}"
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
        possible_paths = []

        # Primary location: Use LOGS_DIR constant
        logs_dir = get_constant("LOGS_DIR")
        if logs_dir:
            possible_paths.append(os.path.join(str(logs_dir), "task.txt"))

        # Fallback for backward compatibility: repo/logs/task.txt
        repo_dir = get_constant("REPO_DIR")
        if repo_dir:
            possible_paths.append(
                os.path.join(str(repo_dir), "logs", "task.txt"))

        # Additional fallbacks for backward compatibility
        possible_paths.extend([
            os.path.join("logs", "task.txt"),  # Relative logs directory
            "task.txt"  # Project root
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
                logger.warning(f"Error reading task file {path}: {e}",
                               exc_info=True)

        logger.warning(
            "task.txt not found in any specified location. Trying 'TASK' constant as fallback."
        )
        # Try to get from "TASK" constant as a last resort if file is not found
        task_from_constant = get_constant("TASK")
        if task_from_constant and task_from_constant != "NOT YET CREATED" and task_from_constant.strip(
        ):  # Check if not default or empty
            logger.info(
                "Using task description from 'TASK' constant as fallback.")
            return task_from_constant

        logger.error(
            "No overall task description provided (task.txt not found and TASK constant not set, default, or empty)."
        )
        return "No overall task description provided (task.txt not found and TASK constant not set, default, or empty)."

    # --- Refactored Code Generation Method ---
    async def _call_llm_to_generate_code(
        self,
        code_description: str,
        external_imports: List[str],
        internal_imports: List[str],
        file_path: Path,
    ) -> str:
        """Generate full file content for the target file."""

        agent_task = self._get_task_description()
        log_content = self._get_file_creation_log_content()

        existing_code = ""
        if file_path.exists():
            try:
                existing_code = file_path.read_text(encoding="utf-8")
            except Exception as read_error:
                logger.warning(
                    "Unable to read existing file %s for additional context: %s",
                    file_path,
                    read_error,
                )

        prepared_messages = code_prompt_generate(
            current_code_base=existing_code,
            code_description=code_description,
            research_string="",
            agent_task=agent_task,
            external_imports=external_imports,
            internal_imports=internal_imports,
            target_file=str(file_path.name),
            file_creation_log_content=log_content,
        )

        model_to_use = get_constant("CODE_GEN_MODEL") or MODEL_STRING
        models_to_run = self._get_code_generation_models(model_to_use)

        candidate_results = await self._generate_code_candidates(
            prepared_messages=prepared_messages,
            file_path=file_path,
            models=models_to_run,
        )

        successful_candidates = [
            candidate for candidate in candidate_results
            if candidate.get("code")
        ]

        if not successful_candidates:
            error_messages = [
                candidate.get("error") or "Unknown error"
                for candidate in candidate_results
            ]
            combined_error = "; ".join(error_messages) if error_messages else "Unknown"
            final_code_string = (
                f"# Error generating code for {file_path.name}: {combined_error}"
            )
            self._log_generated_output(final_code_string, file_path, "Code")
            return final_code_string

        selection_details = None
        try:
            selection_details = await self._select_best_code_version(
                file_path=file_path,
                code_description=code_description,
                agent_task=agent_task,
                external_imports=external_imports,
                internal_imports=internal_imports,
                log_content=log_content,
                existing_code=existing_code,
                candidates=candidate_results,
            )
        except Exception as selection_error:
            logger.error(
                "Code selection model failed for %s: %s",
                file_path,
                selection_error,
                exc_info=True,
            )

        if selection_details and selection_details.get("selected_code"):
            final_code_string = selection_details["selected_code"]
            selected_model = selection_details.get("selected_model")
            selection_reason = selection_details.get("reason")
        else:
            fallback_candidate = successful_candidates[0]
            final_code_string = fallback_candidate.get("code", "")
            selected_model = fallback_candidate.get("model")
            selection_reason = (
                "Falling back to first successful candidate due to selection "
                "model failure or empty response."
            )

        if not final_code_string:
            final_code_string = (
                f"# Error generating code for {file_path.name}: Empty content after selection"
            )

        logger.info(
            "Selected code for %s using model %s",
            file_path.name,
            selected_model or "unknown",
        )

        if selection_details and selection_details.get("raw_response"):
            logger.debug(
                "Selection model raw response for %s: %s",
                file_path.name,
                selection_details["raw_response"],
            )

        if selection_reason:
            logger.info(
                "Selection reason for %s: %s",
                file_path.name,
                selection_reason,
            )

        if self.display:
            additional_info_lines = []
            if selected_model:
                additional_info_lines.append(f"Model: {selected_model}")
            if selection_reason:
                additional_info_lines.append(
                    f"Reason: {self._truncate_text(selection_reason, limit=300)}"
                )
            console_output = self._format_terminal_output(
                command="select_best_code",
                files=[file_path.name],
                result="Selected best generated code candidate",
                additional_info="\n".join(additional_info_lines) if additional_info_lines else None,
            )
            self.display.add_message("assistant", console_output)

        self._log_generated_output(final_code_string, file_path, "Code")
        return final_code_string

    def _get_code_generation_models(self, default_model: str) -> List[str]:
        """Return the list of models to use for code generation."""

        configured_models = get_constant("CODE_LIST", CODE_LIST)
        raw_models: List[Any] = []

        if isinstance(configured_models, list):
            raw_models = configured_models
        elif isinstance(configured_models, str):
            stripped_value = configured_models.strip()
            if stripped_value.startswith("[") and stripped_value.endswith("]"):
                try:
                    parsed = json.loads(stripped_value)
                    if isinstance(parsed, list):
                        raw_models = parsed
                except json.JSONDecodeError:
                    raw_models = []
            if not raw_models:
                raw_models = [
                    item.strip() for item in stripped_value.split(",") if item.strip()
                ]
        elif configured_models is None:
            raw_models = []
        else:
            raw_models = [configured_models]

        models: List[str] = []
        for model in raw_models:
            if not isinstance(model, str):
                continue
            model_str = model.strip()
            if not model_str:
                continue
            try:
                potential_path = Path(model_str)
                if potential_path.exists():
                    # Ignore filesystem paths that may have been provided accidentally.
                    continue
            except OSError:
                pass
            if model_str not in models:
                models.append(model_str)

        if not models:
            for model in CODE_LIST:
                if isinstance(model, str) and model and model not in models:
                    models.append(model)

        if default_model and default_model not in models:
            models.append(default_model)

        return models

    async def _generate_code_candidates(
        self,
        *,
        prepared_messages: List[Dict[str, str]],
        file_path: Path,
        models: List[str],
    ) -> List[Dict[str, Any]]:
        """Run code generation for each model in parallel."""

        if not models:
            return []

        tasks = [
            self._generate_code_for_model(
                prepared_messages=prepared_messages,
                file_path=file_path,
                model=model,
            )
            for model in models
        ]

        return await asyncio.gather(*tasks)

    async def _generate_code_for_model(
        self,
        *,
        prepared_messages: List[Dict[str, str]],
        file_path: Path,
        model: str,
    ) -> Dict[str, Any]:
        """Generate code for a single model with retry handling."""

        start_time = time.time()
        try:
            code_string = await self._llm_generate_code_core_with_retry(
                prepared_messages=prepared_messages,
                file_path=file_path,
                model_to_use=model,
            )
            elapsed = time.time() - start_time
            logger.info(
                "Generated candidate code for %s using model %s in %.2fs",
                file_path.name,
                model,
                elapsed,
            )
            return {"model": model, "code": code_string, "error": None}
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(
                "Code generation error for %s using model %s after %.2fs: %s",
                file_path.name,
                model,
                elapsed,
                exc,
                exc_info=True,
            )
            return {"model": model, "code": "", "error": str(exc)}

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(should_retry_llm_call),
        reraise=True,
        before_sleep=_log_llm_retry_attempt,
    )
    async def _call_selection_model(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        file_path: Path,
    ) -> str:
        """Call the selection model to choose the best candidate."""

        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        logger.info(
            "Selection model %s evaluating candidates for %s",
            model,
            file_path.name,
        )

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
        )

        if not (
            completion
            and completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            logger.error(
                "Invalid or empty selection response for %s", file_path.name
            )
            raise LLMResponseError(
                f"Invalid or empty selection response for {file_path.name}"
            )

        return completion.choices[0].message.content

    def _parse_selection_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the selection model."""

        if not response or not response.strip():
            raise LLMResponseError("Empty response from selection model")

        stripped_response = response.strip()
        try:
            return json.loads(stripped_response)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", stripped_response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError as exc:
                    logger.error("Failed to parse selection JSON: %s", exc)
            raise

    async def _select_best_code_version(
        self,
        *,
        file_path: Path,
        code_description: str,
        agent_task: str,
        external_imports: List[str],
        internal_imports: List[str],
        log_content: str,
        existing_code: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Use CODE_MODEL to select the best candidate implementation."""

        selector_model = get_constant("CODE_MODEL") or CODE_MODEL
        valid_candidates = [candidate for candidate in candidates if candidate.get("code")]

        if not valid_candidates:
            raise ValueError("No valid code candidates were generated.")

        candidate_sections = []
        for index, candidate in enumerate(candidates, start=1):
            model_name = candidate.get("model", f"candidate_{index}")
            header = f"Candidate {index} - Model {model_name}"
            if candidate.get("code"):
                truncated_code = self._truncate_text(candidate["code"], limit=12000)
                candidate_sections.append(
                    f"{header}\n```\n{truncated_code}\n```"
                )
            else:
                candidate_sections.append(
                    f"{header} encountered an error: {candidate.get('error', 'Unknown error')}"
                )

        existing_excerpt = (
            self._truncate_text(existing_code, limit=4000)
            if existing_code
            else "(No existing file content)"
        )
        log_excerpt = (
            self._truncate_text(log_content, limit=4000)
            if log_content
            else "(No recent file log entries available)"
        )

        agent_task_excerpt = (
            self._truncate_text(agent_task, limit=2000)
            if agent_task
            else "(No agent task provided)"
        )

        context_lines = [
            f"File path: {file_path}",
            f"File description: {code_description}",
            f"Agent task: {agent_task_excerpt}",
            "External imports: "
            + (", ".join(external_imports) if external_imports else "None"),
            "Internal imports: "
            + (", ".join(internal_imports) if internal_imports else "None"),
        ]

        user_message = (
            "You must pick the best candidate implementation for the target file.\n"
            + "\n".join(context_lines)
            + "\n\nExisting file content (if present):\n```")
        user_message += existing_excerpt + "\n```\n\n"
        user_message += "Recent file creation log excerpt:\n```\n"
        user_message += log_excerpt + "\n```\n\n"
        user_message += "Candidate implementations:\n\n"
        user_message += "\n\n".join(candidate_sections)
        user_message += (
            "\n\nRespond with a JSON object containing the keys "
            "'selected_model', 'selected_code', and 'reason'. Use one of the provided "
            "model identifiers exactly and prefer candidates that match the project "
            "requirements."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert software engineer reviewing multiple code implementations. "
                    "Evaluate correctness, completeness, and integration with the project. "
                    "Return only valid JSON with fields 'selected_model', 'selected_code', and 'reason'."
                ),
            },
            {"role": "user", "content": user_message},
        ]

        raw_response = await self._call_selection_model(
            messages=messages,
            model=selector_model,
            file_path=file_path,
        )

        parsed_response = self._parse_selection_response(raw_response)
        selected_model = (
            parsed_response.get("selected_model")
            or parsed_response.get("winner_model")
            or parsed_response.get("model")
        )
        selected_code = parsed_response.get("selected_code") or parsed_response.get("code")
        selection_reason = (
            parsed_response.get("reason")
            or parsed_response.get("justification")
            or parsed_response.get("rationale")
            or ""
        )

        selected_index = (
            parsed_response.get("selected_index")
            or parsed_response.get("winner_index")
            or parsed_response.get("index")
        )

        if (not selected_model) and selected_index is not None:
            try:
                index_value = int(selected_index)
            except (ValueError, TypeError):
                index_value = None
            if index_value and 1 <= index_value <= len(valid_candidates):
                candidate = valid_candidates[index_value - 1]
                selected_model = candidate.get("model")
                if not selected_code:
                    selected_code = candidate.get("code")

        candidate_map = {
            candidate.get("model"): candidate for candidate in valid_candidates if candidate.get("model")
        }

        if selected_model in candidate_map:
            if not selected_code:
                selected_code = candidate_map[selected_model].get("code")
        else:
            fallback_candidate = valid_candidates[0]
            selected_model = fallback_candidate.get("model")
            if not selected_code:
                selected_code = fallback_candidate.get("code")

        return {
            "selected_model": selected_model,
            "selected_code": selected_code,
            "reason": selection_reason,
            "raw_response": raw_response,
        }

    @staticmethod
    def _truncate_text(text: str, *, limit: int = 4000) -> str:
        """Truncate long text to keep prompts manageable."""

        if text is None:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit] + (
            f"\n... (truncated, original length {len(text)} characters)"
        )

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
            raise ValueError("OPENROUTER_API_KEY environment variable not set."
                             )  # Should fail fast if not retryable

        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                             api_key=OPENROUTER_API_KEY)

        current_attempt = getattr(
            self._llm_generate_code_core_with_retry.retry.statistics,
            "attempt_number",
            1,
        )
        logger.info(
            f"LLM Code Gen for {file_path.name}: Model {model_to_use}, Attempt {current_attempt}"
        )

        completion = await client.chat.completions.create(
            model=model_to_use, messages=prepared_messages)

        if not (completion and completion.choices
                and completion.choices[0].message
                and completion.choices[0].message.content):
            logger.error(
                f"No valid completion content received for {file_path.name}")
            logger.error(
                f"[bold red]Invalid or empty completion content from LLM for {file_path.name}[/bold red]"
            )
            raise LLMResponseError(
                f"Invalid or empty completion content from LLM for {file_path.name}"
            )

        raw_code_string = completion.choices[0].message.content
        # Assuming self.extract_code_block is defined in your class
        code_string, detected_language = self.extract_code_block(
            raw_code_string, file_path)

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
        ) or code_string.startswith(
                f"# Failed to generate code for {file_path.name}"):
            logger.error(
                f"LLM returned a placeholder error message for {file_path.name}: {code_string[:100]}"
            )
            raise LLMResponseError(
                f"LLM returned placeholder error for {file_path.name}: {code_string[:100]}"
            )

        return code_string


    def extract_code_block(
            self,
            text: str,
            file_path: Optional[Path] = None) -> tuple[str, str]:
        """
        Extracts code based on file type. Special handling for Markdown files.
        Improved language guessing.
        Returns tuple of (content, language).
        """
        if file_path is not None and str(file_path).lower().endswith(
            (".md", ".markdown")):
            return text, "markdown"

        if not text or not text.strip():
            return "No Code Found", "unknown"  # Consistent 'unknown'

        start_marker = text.find("```")
        if start_marker == -1:
            # No backticks, try guessing language from content
            try:
                language = guess_lexer(text).aliases[0]
                if language in {"text", "text only"}:
                    language = "unknown"
                # Return the whole text as the code block
                return text.strip(), language
            except Exception:  # pygments.util.ClassNotFound or others
                logger.warning(
                    "Could not guess language for code without backticks.")
                return text.strip(), "unknown"  # Return unknown if guess fails

        # Found opening backticks ```
        language_line_end = text.find("\n", start_marker)
        if language_line_end == -1:  # Handle case where ``` is at the very end
            language_line_end = len(text)

        language = text[start_marker + 3:language_line_end].strip().lower()

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

        if language in {"text", "text only"}:
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
        formatter = HtmlFormatter(style="monokai",
                                  linenos=True,
                                  cssclass="source")

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
