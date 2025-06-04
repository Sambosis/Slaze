import asyncio
import os
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, List, Dict

from openai import (
    APIConnectionError, APIError, APIStatusError, AsyncOpenAI, 
    InternalServerError, RateLimitError
)
from pydantic import BaseModel
from rich import print as rr
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from tools.base import BaseAnthropicTool, ToolResult
from config import REPO_DIR, get_constant
from icecream import ic  # type: ignore
from pygments import highlight  # type: ignore
from pygments.formatters import HtmlFormatter  # type: ignore
from pygments.lexers import get_lexer_by_name, guess_lexer  # type: ignore
from utils.file_logger import convert_to_docker_path, log_file_operation, get_language_from_extension
from system_prompt.code_prompts import (
    code_prompt_generate,
    code_skeleton_prompt,
)
from utils.docker_service import (
    DockerService,
)
from pygments import highlight
from pygments.formatters import HtmlFormatter
import ftfy
from pygments.lexers import get_lexer_by_name, guess_lexer
import traceback
from lmnr import observe

googlepro = "google/gemini-2.5-pro-preview"
oa4omini = "openai/o4-mini-high"
gflash = "google/gemini-2.5-flash-preview"
MODEL_STRING = googlepro  # Default model string, can be overridden in config


# --- Retry Predicate Function ---
def should_retry_llm_call(retry_state: RetryCallState) -> bool:
    """
    Determines if a retry should occur based on the raised exception.
    Accesses the exception from retry_state.outcome.exception().
    """
    if not retry_state.outcome: # Should not happen if called after an attempt
        return False

    exception = retry_state.outcome.exception()
    if not exception: # No exception, successful outcome
        return False

    # Always retry on our custom LLMResponseError
    if isinstance(exception, LLMResponseError):
        rr(f"[bold yellow]Retry triggered by LLMResponseError: {str(exception)[:200]}[/bold yellow]")
        return True

    # Retry on specific, transient OpenAI/network errors
    if isinstance(exception, (
        APIConnectionError,  # Network issues
        RateLimitError,     # Rate limits hit
        InternalServerError # Server-side errors from OpenAI (500 class)
    )):
        rr(f"[bold yellow]Retry triggered by OpenAI API Error ({type(exception).__name__}): {str(exception)[:200]}[/bold yellow]")
        return True

    if isinstance(exception, APIStatusError):
        # Retry on general server errors (5xx) and specific client errors like Gateway Timeout (504)
        # or Request Timeout (408). 429 should ideally be RateLimitError.
        if exception.status_code >= 500 or exception.status_code in [408, 429, 502, 503, 504]:
            rr(f"[bold yellow]Retry triggered by OpenAI APIStatusError (status {exception.status_code}): {str(exception)[:200]}[/bold yellow]")
            return True

    # For any other Exception, you might want to log it but not retry,
    # or add it here if known to be transient.
    # rr(f"[bold red]Non-retryable exception encountered: {type(exception).__name__}: {exception}[/bold red]")
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
        "Creates skeletons first, then generates full code asynchronously, writing to the host filesystem."
        )

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display) # Assuming BaseAnthropicTool takes these
        self.display = display
        try:
            self.docker = DockerService()
            self._docker_available = self.docker.is_available()

        except Exception as docker_init_err:
            rr(f"[bold red]Failed to initialize DockerService: {docker_init_err}[/bold red]")
            self.docker = None
            self._docker_available = False
        ic("Initializing WriteCodeTool")
        ic(f"Docker available: {self._docker_available}")

    def to_params(self) -> dict:
        ic(f"WriteCodeTool.to_params called with api_type: {self.api_type}")
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
                                        "description": "Detailed description of the code for this file.",
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
        ic(f"WriteCodeTool params: {params}")
        return params

    # --- Logging Callback for Retries ---
    def _log_llm_retry_attempt(self, retry_state: RetryCallState):
        """Logs information about the current retry attempt."""
        fn_name = retry_state.fn.__name__ if retry_state.fn else "LLM_call"

        file_path_for_log = "unknown_file"
        # Try to get file_path or target_file_path from kwargs for contextual logging
        if retry_state.kwargs:
            fp_arg = retry_state.kwargs.get('file_path') or retry_state.kwargs.get('target_file_path')
            if isinstance(fp_arg, Path):
                file_path_for_log = fp_arg.name

        log_prefix = f"[bold magenta]Retry Log ({file_path_for_log})[/bold magenta] | "

        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            max_attempts_str = "N/A"
            stop_condition = retry_state.retry_object.stop
            if hasattr(stop_condition, 'max_attempt_number'):
                max_attempts_str = str(stop_condition.max_attempt_number)

            log_msg = (
                f"{log_prefix}Retrying [u]{fn_name}[/u] due to [bold red]{type(exc).__name__}[/bold red]: {str(exc)[:150]}. "
                f"Attempt [bold cyan]{retry_state.attempt_number}[/bold cyan] of {max_attempts_str}. "
                f"Waiting [bold green]{retry_state.next_action.sleep:.2f}s[/bold green]..."
            )
        else:
            log_msg = (
                 f"{log_prefix}Retrying [u]{fn_name}[/u] (no direct exception, or outcome not yet available). "
                 f"Attempt [bold cyan]{retry_state.attempt_number}[/bold cyan]. Waiting [bold green]{retry_state.next_action.sleep:.2f}s[/bold green]..."
            )
        rr(log_msg)

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
                rr(
                    f"Configured REPO_DIR '{host_repo_dir}' does not exist or is not a directory."
                )
                # Decide if you want to raise an error or attempt to create it
                # For now, let's proceed and let mkdir handle creation/errors later

            # Assume project_path might be a full docker path or just the project name.
            # We want the final component to use relative to the host REPO_DIR.
            project_name = Path(project_path).name
            if not project_name or project_name in [".", ".."]:
                raise ValueError(
                    f"Could not extract a valid project name from input project_path: {project_path}"
                )

            # Construct the ACTUAL host path where files should be written
            host_project_path_obj = (
                host_repo_path / project_name
            ).resolve()  # Resolve to absolute path early
            rr(f"Resolved HOST project path for writing: {host_project_path_obj}")

            # Ensure the HOST directory exists
            host_project_path_obj.mkdir(parents=True, exist_ok=True)
            rr(f"Ensured host project directory exists: {host_project_path_obj}")
            # --- END: Path Correction Logic ---

            # Validate files input
            try:
                file_details = [FileDetail(**f) for f in files]
            except Exception as pydantic_error:
                rr(f"Pydantic validation error for 'files': {pydantic_error}")
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
                    rr(error_msg)
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

            rr(
                f"Starting file writing phase for {len(code_results)} results to HOST path: {host_project_path_obj}"
            )

            for i, result in enumerate(code_results):
                file_detail = file_details[i]
                filename = file_detail.filename  # Relative filename
                # >>> USE THE CORRECTED HOST PATH FOR WRITING <<<
                absolute_path = (
                    host_project_path_obj / filename
                ).resolve()  # Ensure absolute path
                rr(f"Processing result for: {filename} (Host Path: {absolute_path})")

                if isinstance(result, Exception):
                    error_msg = f"Error generating code for {filename}: {result}"
                    rr(error_msg)
                    errors_code_gen.append(error_msg)
                    write_results.append(
                        {"filename": filename, "status": "error", "message": error_msg}
                    )
                    # Attempt to write error file to the resolved host path
                    try:
                        rr(
                            f"Attempting to write error file for {filename} to {absolute_path}"
                        )
                        absolute_path.parent.mkdir(parents=True, exist_ok=True)
                        error_content = f"# Code generation failed: {result}\n\n# Skeleton:\n{skeletons.get(filename, '# Skeleton not available')}"
                        absolute_path.write_text(
                            error_content, encoding="utf-8", errors="replace"
                        )
                        rr(
                            f"Successfully wrote error file for {filename} to {absolute_path}"
                        )
                    except Exception as write_err:
                        rr(
                            f"Failed to write error file for {filename} to {absolute_path}: {write_err}"
                        )

                else:  # Code generation successful
                    code_content = result
                    rr(
                        f"Code generation successful for {filename}. Attempting to write to absolute host path: {absolute_path}"
                    )

                    if not code_content or not code_content.strip():
                        rr(
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
                        rr(f"Ensuring directory exists: {absolute_path.parent}")
                        absolute_path.parent.mkdir(parents=True, exist_ok=True)
                        operation = "modify" if absolute_path.exists() else "create"
                        rr(f"Operation type for {filename}: {operation}")

                        fixed_code = ftfy.fix_text(code_content)
                        rr(
                            f"Code content length for {filename} (after ftfy): {len(fixed_code)}"
                        )

                        # >>> THE WRITE CALL to the HOST path <<<
                        rr(f"Executing write_text for: {absolute_path}")
                        absolute_path.write_text(
                            fixed_code, encoding="utf-8", errors="replace"
                        )
                        rr(f"Successfully executed write_text for: {absolute_path}")

                        # File existence and size check
                        if absolute_path.exists():
                            rr(
                                f"CONFIRMED: File exists at {absolute_path} after write."
                            )
                            try:
                                size = absolute_path.stat().st_size
                                rr(f"CONFIRMED: File size is {size} bytes.")
                                if size == 0 and len(fixed_code) > 0:
                                    rr(
                                        "File size is 0 despite non-empty content being written!"
                                    )
                            except Exception as stat_err:
                                rr(
                                    f"Could not get file stats for {absolute_path}: {stat_err}"
                                )
                        else:
                            rr(
                                f"FAILED: File DOES NOT exist at {absolute_path} immediately after write_text call!"
                            )

                        # Convert to Docker path FOR DISPLAY/LOGGING PURPOSES ONLY
                        docker_path_display = str(
                            absolute_path
                        )  # Default to host path if conversion fails
                        try:
                            # Ensure convert_to_docker_path can handle the absolute host path
                            docker_path_display = convert_to_docker_path(absolute_path)
                            rr(
                                f"Converted host path {absolute_path} to display path {docker_path_display}"
                            )
                        except Exception as conv_err:
                            rr(
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
                            rr(f"Logged file operation for {absolute_path}")
                        except Exception as log_error:
                            rr(
                                f"Failed to log code writing for {filename} ({absolute_path}): {log_error}"
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
                        rr(
                            f"Successfully processed and wrote {filename} to {absolute_path}"
                        )

                        # Display generated code (use docker_path_display if needed by UI)
                        if self.display:
                            self.display.add_message(
                                "user",
                                f"Code for {docker_path_display} generated successfully:",
                            )  # Use display path
                            language = get_language_from_extension(absolute_path.suffix)
                            formatted_code = html_format_code(fixed_code, language)
                            # Ensure display can handle html format correctly
                            self.display.add_message("tool", {"html": formatted_code})

                    except Exception as write_error:
                        rr(
                            f"Caught exception during write operation for {filename} at path {absolute_path}"
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
            error_message = f"Configuration Error in WriteCodeTool __call__: {str(ve)}\n{traceback.format_exc()}"
            rr(error_message)
            return ToolResult(error=error_message, tool_name=self.name, command=command)
        except Exception as e:
            error_message = f"Critical Error in WriteCodeTool __call__: {str(e)}\n{traceback.format_exc()}"
            rr("Critical error during codebase generation")
            # Optionally include host_project_path_obj if it was set
            if host_project_path_obj:
                error_message += f"\nAttempted Host Path: {host_project_path_obj}"
            print(error_message)
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
                    f.write(f"\n--- Generated {output_type} for: {str(file_path)} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n")
                    f.write(f"{content}\n")
                    f.write(f"--- End {output_type} for: {str(file_path)} ---\n")
        except Exception as file_error:
            rr(f"[bold red]Failed to log generated {output_type} for {file_path.name} to {get_constant('CODE_FILE')}: {file_error}[/bold red]")

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

    # --- Refactored Code Generation Method ---
    async def _call_llm_to_generate_code(
        self,
        code_description: str,
        all_skeletons: Dict[str, str],
        external_imports: List[str],
        internal_imports: List[str],
        file_path: Path,
    ) -> str:
        if self.display is not None:
            self.display.add_message("assistant", f"Generating code for: {file_path.name}")

        skeleton_context = "\n\n---\n\n".join(
            f"### Skeleton for {fname}:\n```\n{skel}\n```" for fname, skel in all_skeletons.items()
        )
        agent_task = get_constant("TASK") or "No overall task description provided."
        if os.path.exists("task.txt"):
            with open("task.txt", "r") as task_file:
                task_desc_from_file = task_file.read().strip()
            agent_task = task_desc_from_file or agent_task

        prepared_messages = code_prompt_generate(
            current_code_base="", code_description=code_description, research_string="",
            agent_task=agent_task, skeletons=skeleton_context,
            external_imports=external_imports, internal_imports=internal_imports,
            target_file=str(file_path.name),
        )

        model_to_use = get_constant("CODE_GEN_MODEL") or MODEL_STRING
        final_code_string = f"# Error: Code generation failed for {file_path.name} after all retries."

        try:
            final_code_string = await self._llm_generate_code_core_with_retry(
                prepared_messages=prepared_messages, file_path=file_path, model_to_use=model_to_use
            )
        except LLMResponseError as e:
            ic(f"LLMResponseError for {file_path.name} after all retries: {e}")
            rr(f"[bold red]LLM generated invalid content for {file_path.name} after retries: {e}[/bold red]")
            final_code_string = f"# Error generating code for {file_path.name}: LLMResponseError - {str(e)}"
        except APIError as e: # Catch specific OpenAI errors
            ic(f"OpenAI APIError for {file_path.name} after all retries: {type(e).__name__} - {e}")
            rr(f"[bold red]LLM call failed due to APIError for {file_path.name} after retries: {e}[/bold red]")
            final_code_string = f"# Error generating code for {file_path.name}: API Error - {str(e)}"
        except Exception as e:
            ic(f"Unexpected error during code generation for {file_path.name} after retries: {type(e).__name__} - {e}")
            rr(f"[bold red]LLM call ultimately failed for {file_path.name} due to unexpected error: {e}[/bold red]")
            final_code_string = f"# Error generating code for {file_path.name} (final): {str(e)}"

        self._log_generated_output(final_code_string, file_path, "Code")
        return final_code_string

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(should_retry_llm_call),
        reraise=True,
        before_sleep=_log_llm_retry_attempt 
    )
    async def _llm_generate_code_core_with_retry(
        self, prepared_messages: List[Dict[str, str]], file_path: Path, model_to_use: str
    ) -> str:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.") # Should fail fast if not retryable

        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

        current_attempt = getattr(self._llm_generate_code_core_with_retry.retry.statistics, 'attempt_number', 1)
        rr(f"LLM Code Gen for [cyan]{file_path.name}[/cyan]: Model [yellow]{model_to_use}[/yellow], Attempt [bold]{current_attempt}[/bold]")

        completion = await client.chat.completions.create(model=model_to_use, messages=prepared_messages)

        if not (completion and completion.choices and completion.choices[0].message and completion.choices[0].message.content):
            ic(f"No valid completion content received for {file_path.name}")
            rr(f"[bold red]Invalid or empty completion content from LLM for {file_path.name}[/bold red]")
            raise LLMResponseError(f"Invalid or empty completion content from LLM for {file_path.name}")

        raw_code_string = completion.choices[0].message.content
        # Assuming self.extract_code_block is defined in your class
        code_string, detected_language = self.extract_code_block(raw_code_string, file_path) 

        rr(f"Extracted code for [cyan]{file_path.name}[/cyan]. Lang: {detected_language}. Raw len: {len(raw_code_string)}, Extracted len: {len(code_string or '')}")

        if code_string == "No Code Found": # Critical check
            if raw_code_string.strip():
                rr(f"[bold orange_red1]Could not extract code for {file_path.name}, raw response was not empty. LLM might have misunderstood.[/bold orange_red1]")
                raise LLMResponseError(f"Extracted 'No Code Found' for {file_path.name}. Raw: '{raw_code_string[:100]}...'")
            else:
                rr(f"[bold red]LLM response for {file_path.name} was effectively empty (raw string).[/bold red]")
                raise LLMResponseError(f"LLM response for {file_path.name} was effectively empty (raw string).")

        if code_string.startswith(f"# Error: Code generation failed for {file_path.name}") or \
           code_string.startswith(f"# Failed to generate code for {file_path.name}"):
            rr(f"[bold red]LLM returned a placeholder error message for {file_path.name}: {code_string[:100]}[/bold red]")
            raise LLMResponseError(f"LLM returned placeholder error for {file_path.name}: {code_string[:100]}")

        return code_string

    # --- Refactored Skeleton Generation Method ---
    @observe(name="generate_code_skeleton")
    async def _call_llm_for_code_skeleton(
        self, file_detail: FileDetail, file_path: Path, all_file_details: List[FileDetail]
    ) -> str:
        target_file_name = file_path.name
        if self.display:
            self.display.add_message("assistant", f"Generating skeleton for {target_file_name}")

        try: # Add try-finally for task.txt to ensure it's handled if missing
            with open("task.txt", "r") as task_file:
                task_description_from_file = task_file.read().strip()
            agent_task = task_description_from_file or "No overall task description provided."
        except FileNotFoundError:
            agent_task = "No overall task description provided (task.txt not found)."
            rr("[yellow]Warning: task.txt not found. Using default task description.[/yellow]")

        external_imports = file_detail.external_imports
        internal_imports = file_detail.internal_imports
        all_files_dict_list = [f.model_dump() for f in all_file_details]

        prepared_messages = code_skeleton_prompt(
            code_description=file_detail.code_description, target_file=target_file_name,
            agent_task=agent_task, external_imports=external_imports,
            internal_imports=internal_imports, all_file_details=all_files_dict_list,
        )

        model_to_use = get_constant("SKELETON_GEN_MODEL") or MODEL_STRING
        final_skeleton_string = f"# Error: Skeleton generation failed for {target_file_name} after all retries."

        try:
            final_skeleton_string = await self._llm_generate_skeleton_core_with_retry(
                prepared_messages=prepared_messages, target_file_path=file_path, model_to_use=model_to_use
            )
        except LLMResponseError as e:
            ic(f"LLMResponseError for skeleton {target_file_name} after all retries: {e}")
            rr(f"[bold red]LLM generated invalid skeleton for {target_file_name} after retries: {e}[/bold red]")
            final_skeleton_string = f"# Error generating skeleton for {target_file_name}: LLMResponseError - {str(e)}"
        except APIError as e: # Catch specific OpenAI errors
            ic(f"OpenAI APIError for skeleton {target_file_name} after all retries: {type(e).__name__} - {e}")
            rr(f"[bold red]LLM skeleton call failed due to APIError for {target_file_name} after retries: {e}[/bold red]")
            final_skeleton_string = f"# Error generating skeleton for {target_file_name}: API Error - {str(e)}"
        except Exception as e:
            ic(f"Unexpected error during skeleton generation for {target_file_name} after retries: {type(e).__name__} - {e}")
            rr(f"[bold red]LLM skeleton call ultimately failed for {target_file_name} due to unexpected error: {e}[/bold red]")
            final_skeleton_string = f"# Error generating skeleton for {target_file_name} (final): {str(e)}"

        self._log_generated_output(final_skeleton_string, file_path, "Skeleton")
        ic(f"Final Skeleton for {target_file_name}:\n{final_skeleton_string[:300]}...") # Log snippet
        return final_skeleton_string

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(3), 
        retry=retry_if_exception(should_retry_llm_call),
        reraise=True,
        before_sleep=_log_llm_retry_attempt
        )
    async def _llm_generate_skeleton_core_with_retry(
        self, prepared_messages: List[Dict[str, str]], target_file_path: Path, model_to_use: str
        ) -> str:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set.")

        client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

        current_attempt = getattr(self._llm_generate_skeleton_core_with_retry.retry.statistics, 'attempt_number', 1)
        rr(f"LLM Skeleton Gen for [cyan]{target_file_path.name}[/cyan]: Model [yellow]{model_to_use}[/yellow], Attempt [bold]{current_attempt}[/bold]")

        completion = await client.chat.completions.create(model=model_to_use, messages=prepared_messages)

        if not (completion and completion.choices and completion.choices[0].message and completion.choices[0].message.content):
            ic(f"No valid skeleton completion content received for {target_file_path.name}")
            rr(f"[bold red]Invalid or empty skeleton completion content from LLM for {target_file_path.name}[/bold red]")
            raise LLMResponseError(f"Invalid or empty skeleton completion from LLM for {target_file_path.name}")

        raw_skeleton = completion.choices[0].message.content
        # Assuming self.extract_code_block is defined
        skeleton_string, detected_language = self.extract_code_block(raw_skeleton, target_file_path)

        rr(f"Extracted skeleton for [cyan]{target_file_path.name}[/cyan]. Lang: {detected_language}. Raw len: {len(raw_skeleton)}, Extracted len: {len(skeleton_string or '')}")

        if skeleton_string == "No Code Found": # Critical check
            if raw_skeleton.strip():
                rr(f"[bold orange_red1]Could not extract skeleton for {target_file_path.name}, raw response was not empty.[/bold orange_red1]")
                raise LLMResponseError(f"Extracted 'No Code Found' for skeleton {target_file_path.name}. Raw: '{raw_skeleton[:100]}...'")
            else:
                rr(f"[bold red]LLM response for skeleton {target_file_path.name} was effectively empty (raw string).[/bold red]")
                raise LLMResponseError(f"LLM response for skeleton {target_file_path.name} was effectively empty (raw string).")

        if skeleton_string.startswith(f"# Error: Skeleton generation failed for {target_file_path.name}") or \
           skeleton_string.startswith(f"# Failed to generate skeleton for {target_file_path.name}"):
            rr(f"[bold red]LLM returned a placeholder error for skeleton {target_file_path.name}: {skeleton_string[:100]}[/bold red]")
            raise LLMResponseError(f"LLM returned placeholder error for skeleton {target_file_path.name}: {skeleton_string[:100]}")

        skeleton_string = ftfy.fix_text(skeleton_string) # Apply ftfy only on success
        return skeleton_string

    # You need to define extract_code_block method within this class or ensure it's accessible
    def extract_code_block(self, raw_response: str, file_path: Path) -> tuple[str, Optional[str]]:
        """
        Extracts a code block from the LLM's raw response.
        Handles different markdown formats and language detection.
        Returns the extracted code string and the detected language.
        If no code block is found, returns "No Code Found" and None.
        """
        # Simplified example, replace with your actual robust implementation
        # Your implementation might involve regex to find ```python ... ``` or similar

        # Attempt to guess language if not specified in markdown
        language = get_language_from_extension(file_path.suffix) # from your context_helpers

        # Common pattern: ```[language]\ncode\n``` or ```\ncode\n```
        code_block_match = re.search(r"```(?:[a-zA-Z0-9_.-]*\n)?(.*?)```", raw_response, re.DOTALL)

        if code_block_match:
            extracted_code = code_block_match.group(1).strip()
            # Try to get language from markdown if present
            lang_in_markdown_match = re.search(r"```([a-zA-Z0-9_.-]+)\n", raw_response)
            if lang_in_markdown_match:
                language = lang_in_markdown_match.group(1).lower()

            if not extracted_code: # Empty code block
                # If the raw response itself is just the code without backticks, and is short.
                if not "```" in raw_response and len(raw_response.strip()) > 0 and len(raw_response.strip()) < 300: # Heuristic
                    return raw_response.strip(), language # Assume raw is the code
                return "No Code Found", language

            # Basic check for placeholder/refusal, although LLMResponseError should catch most
            if "sorry" in extracted_code.lower() and "cannot generate" in extracted_code.lower():
                return "No Code Found", language # Or raise specific error

            return extracted_code, language

        # If no triple backticks, check if the entire response is code (heuristic)
        # This is risky and depends on LLM behavior.
        # Only do this if the response is relatively clean and doesn't look like prose.
        lines = raw_response.strip().split('\n')
        if len(lines) > 0 and not lines[0].startswith("# Error") and not lines[0].startswith("I cannot"):
            # Heuristic: if it doesn't look like a refusal and lacks prose markers.
            # This part needs careful tuning or removal if it causes issues.
            # For now, let's be more conservative and rely on backticks.
            # If the LLM sometimes returns code without backticks, this logic needs to be robust.
            # A simple check: if raw_response does not contain "```" at all.
            if "```" not in raw_response and raw_response.strip(): # and not any(prose_indicator in raw_response for prose_indicator in ["sorry", "unfortunately", "understand"]):
                return raw_response.strip(), language # Assume the whole thing is code

        return "No Code Found", language

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
                rr("Could not guess language for code without backticks.")
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
                rr(
                    f"Could not guess language for extracted code block (File: {file_path})."
                )
                language = "unknown"  # Fallback if guess fails

        # If language is still empty, default to 'unknown'
        if not language:
            language = "unknown"

        return code_block if code_block else "No Code Found", language

    # def format_output(self, data: dict) -> str:
    #     """
    #     Format the output of the tool for display.

    #     Args:
    #         data: The data returned by the tool's operation.

    #     Returns:
    #         A formatted string for display.
    #     """
    #     status = data.get("status", "Unknown")
    #     message = data.get("message", "No message provided.")
    #     errors = data.get("errors", [])
    #     files_processed = data.get("files_processed", 0)
    #     files_successful = data.get("files_successful", 0)

    #     output = f"Operation Status: {status.upper()}\n"
    #     output += f"Message: {message}\n"
    #     output += (
    #         f"Files Processed: {files_processed}, Successful: {files_successful}\n"
    #     )

    #     if errors:
    #         output += f"\nErrors Encountered ({len(errors)}):\n"
    #         # Show first few errors
    #         for i, err in enumerate(errors[:5]):
    #             output += f"- {err}\n"
    #         if len(errors) > 5:
    #             output += f"... and {len(errors) - 5} more errors.\n"
    #         output += "(Check logs for full details)\n"

    #     # Optionally add details about successful files if needed
    #     # results = data.get("results", [])
    #     # successful_files = [r['filename'] for r in results if r.get('status') == 'success']
    #     # if successful_files:
    #     #    output += f"\nSuccessful Files:\n" + "\n".join([f"- {f}" for f in successful_files])

    #     return output.strip()


def html_format_code(code, extension):
    """Format code with syntax highlighting for HTML display."""
    try:
        # Try to get a lexer based on the file extension
        try:
            lexer = get_lexer_by_name(extension.lower().lstrip("."))
        except:
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
    finally:
        # Ensure any cleanup or finalization if needed
        pass
