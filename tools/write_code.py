from __future__ import annotations
import asyncio
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Literal

import re
if TYPE_CHECKING:
    from agent_display import AgentDisplay

import ftfy

import logging
from enum import Enum
# pyright: ignore[reportMissingImports]
from openai import (
    APIError,
    APIStatusError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel

# from icecream import ic  # type: ignore # Removed
from pygments import highlight  # type: ignore
from pygments.formatters import HtmlFormatter  # type: ignore
from pygments.lexers import get_lexer_by_name, guess_lexer  # type: ignore

# from rich import print as rr # Removed
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    RetryCallState,
)

from config import CODE_MODEL, CODE_LIST, get_constant
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


def _log_llm_retry_attempt(retry_state: RetryCallState):
    """Logs information about the current retry attempt."""
    # The function being retried
    fn_name = retry_state.fn.__name__ if retry_state.fn else "LLM_call"

    # Extract file path from kwargs for better logging context
    file_path_for_log = "unknown_file"
    if retry_state.kwargs:
        fp_arg = retry_state.kwargs.get("file_path") or retry_state.kwargs.get("target_file_path")
        if isinstance(fp_arg, Path):
            file_path_for_log = fp_arg.name

    log_prefix = f"[bold magenta]Retry Log ({file_path_for_log})[/bold magenta] | "

    # Check if the retry is happening after a failure
    if retry_state.outcome and retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        stop_condition = retry_state.retry_object.stop
        max_attempts_str = str(stop_condition.max_attempt_number) if hasattr(stop_condition, "max_attempt_number") else "N/A"

        # Safely check for the next action and sleep duration
        wait_time = f"Waiting {retry_state.next_action.sleep:.2f}s..." if retry_state.next_action else "No further retries."

        log_msg = (
            f"{log_prefix}Retrying {fn_name} due to {type(exc).__name__}: {str(exc)[:150]}. "
            f"Attempt {retry_state.attempt_number} of {max_attempts_str}. {wait_time}"
        )
    else:
        # Case for retries not directly caused by an exception in the outcome
        wait_time = f"Waiting {retry_state.next_action.sleep:.2f}s..." if retry_state.next_action else "No further retries."
        log_msg = (
            f"{log_prefix}Retrying {fn_name}. "
            f"Attempt {retry_state.attempt_number}. {wait_time}"
        )
    logger.info(log_msg)


class LLMResponseError(Exception):
    """Custom exception for invalid or unusable responses from the LLM."""

    pass


class CodeCommand(str, Enum):
    WRITE_CODEBASE = "write_codebase"


class FileDetail(BaseModel):
    model_config = {"extra": "forbid"}
    
    filename: str
    code_description: str
    external_imports: Optional[List[str]] = None
    internal_imports: Optional[List[str]] = None


class WriteCodeTool(BaseAnthropicTool):
    """
    A tool for writing code to files, including generating code from descriptions.
    """
    
    def __init__(self, display: "AgentDisplay" = None):
        super().__init__(input_schema=None, display=display)
        self.picture_tool = PictureGenerationTool(display=display)
    
    @property
    def name(self) -> str:
        return "write_codebase_tool"
    
    @property
    def description(self) -> str:
        return (
            "Generates a full or partial codebase consisting of up to 3 files based on descriptions and import lists. "
            "This is the tool to use to generate code files for a codebase, can create up to 3 files at a time. "
            "Use this tool to generate init files. "
            "Generates full code asynchronously, writing to the host filesystem."
        )

    def to_params(self) -> dict:
        """Return the parameters for OpenAI function calling."""
        return {
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
                            "description": "The command to execute. Must be 'write_codebase'."
                        },
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "filename": {
                                        "type": "string",
                                        "description": "The name of the file to create (with extension)."
                                    },
                                    "code_description": {
                                        "type": "string",
                                        "description": "Detailed description of the code for this file. This should be a comprehensive overview of the file's purpose, functionality, and any important details. It should include a general overview of the files implementation as well as how it interacts with the rest of the codebase."
                                    },
                                    "external_imports": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of external libraries/packages required specifically for this file.",
                                        "default": []
                                    },
                                    "internal_imports": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of internal modules/files within the codebase imported specifically by this file.",
                                        "default": []
                                    }
                                },
                                "required": ["filename", "code_description"]
                            },
                            "description": "List of files to generate. Each file should have a filename and code_description."
                        }
                    },
                    "required": ["command", "files"]
                }
            }
        }

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
        """
        if command != CodeCommand.WRITE_CODEBASE:
            return ToolResult(
                error=f"Unsupported command: {command}. Only 'write_codebase' is supported.",
                tool_name=self.name,
                command=command,
            )

        repo_path_obj = None

        try:
            # Determine the root path where files should be written
            host_repo_dir = get_constant("REPO_DIR")
            if not host_repo_dir:
                raise ValueError("REPO_DIR is not configured in config.py. Cannot determine host write path.")

            repo_path_obj = Path(host_repo_dir).resolve()
            if not repo_path_obj.exists():
                repo_path_obj.mkdir(parents=True, exist_ok=True)
            logger.info(f"Resolved repository path for writing: {repo_path_obj}")

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

            # Separate image and code files
            image_files = [f for f in file_details if is_image_file(f.filename)]
            code_files = [f for f in file_details if not is_image_file(f.filename)]

            write_results = []
            errors_code_gen = []
            errors_write = []
            success_count = 0
            image_success_count = 0
            image_error_count = 0

            # Handle image files
            if image_files:
                img_results, img_success, img_errors = await self._handle_image_generation(image_files)
                write_results.extend(img_results)
                image_success_count = img_success
                image_error_count = img_errors

            # Handle code files
            if code_files:
                code_results, code_gen_errors = await self._handle_code_generation(code_files, repo_path_obj)
                errors_code_gen.extend(code_gen_errors)
                
                # Write code files
                write_res, write_errs, write_success = self._write_files_to_disk(
                    code_files, code_results, repo_path_obj
                )
                write_results.extend(write_res)
                errors_write.extend(write_errs)
                success_count = write_success

            # Format result
            final_status = "success"
            if errors_code_gen or errors_write or image_error_count > 0:
                final_status = "partial_success" if (success_count > 0 or image_success_count > 0) else "error"

            total_files = len(file_details)
            total_success = success_count + image_success_count

            output_message = f"Codebase generation finished. Status: {final_status}. {total_success}/{total_files} files processed."
            if success_count > 0:
                output_message += f"\n  - Code files: {success_count}/{len(code_files)} successful"
            if image_success_count > 0:
                output_message += f"\n  - Image files: {image_success_count}/{len(image_files)} successful"
            
            if errors_code_gen:
                output_message += f"\nCode Generation Errors: {len(errors_code_gen)}"
            if errors_write:
                output_message += f"\nFile Write Errors: {len(errors_write)}"
            if image_error_count > 0:
                output_message += f"\nImage Generation Errors: {image_error_count}"

            result_data = {
                "status": final_status,
                "message": output_message,
                "files_processed": total_files,
                "files_successful": total_success,
                "write_path": str(repo_path_obj),
                "results": write_results,
                "errors": errors_code_gen + errors_write,
            }

            if self.display:
                status_text = "completed successfully" if final_status == "success" else f"completed with status: {final_status}"
                all_filenames = [f.filename for f in file_details]
                console_output = self._format_terminal_output(
                    command="write_codebase",
                    files=all_filenames,
                    result=f"Codebase generation {status_text}",
                    additional_info=f"Total files: {total_files}\nSuccessful: {total_success}\nWrite path: {repo_path_obj}"
                )
                self.display.add_message("assistant", console_output)

            return ToolResult(
                output=self.format_output(result_data),
                tool_name=self.name,
                command=command,
            )

        except ValueError as ve:
            error_message = f"Configuration Error in WriteCodeTool __call__: {str(ve)}"
            logger.critical(error_message, exc_info=True)
            if self.display:
                self.display.add_message("assistant", self._format_terminal_output(command="write_codebase", error=str(ve)))
            return ToolResult(error=error_message, tool_name=self.name, command=command)
        except Exception as e:
            error_message = f"Critical Error in WriteCodeTool __call__: {str(e)}"
            logger.critical("Critical error during codebase generation", exc_info=True)
            if self.display:
                self.display.add_message("assistant", self._format_terminal_output(command="write_codebase", error=str(e)))
            return ToolResult(error=error_message, tool_name=self.name, command=command)

    async def _handle_image_generation(self, image_files: List[FileDetail]) -> tuple[List[dict], int, int]:
        """Handle generation of image files."""
        image_results = []
        success_count = 0
        error_count = 0

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
                result = await self.picture_tool(
                    command=PictureCommand.CREATE,
                    prompt=image_file.code_description,
                    output_path=image_file.filename,
                    width=1024,
                    height=1024
                )
                
                if result.error:
                    error_count += 1
                    image_results.append({
                        "filename": image_file.filename,
                        "status": "error",
                        "message": f"Error generating image: {result.error}",
                    })
                else:
                    success_count += 1
                    image_results.append({
                        "filename": image_file.filename,
                        "status": "success",
                        "operation": "image_generation",
                        "message": f"Successfully generated image: {image_file.filename}",
                    })
                    if self.display:
                        self.display.add_message("assistant", self._format_terminal_output(
                            command="generate_image",
                            files=[image_file.filename],
                            result="Image generated successfully"
                        ))

            except Exception as e:
                error_count += 1
                error_msg = f"Error generating image {image_file.filename}: {str(e)}"
                logger.error(error_msg)
                image_results.append({
                    "filename": image_file.filename,
                    "status": "error",
                    "message": error_msg,
                })

        return image_results, success_count, error_count

    async def _handle_code_generation(self, code_files: List[FileDetail], repo_path_obj: Path) -> tuple[List[Any], List[str]]:
        """Handle generation of code files."""
        if self.display:
            code_filenames = [file.filename for file in code_files]
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
                repo_path_obj / file.filename,
            ) for file in code_files
        ]
        
        code_results = await asyncio.gather(*code_gen_tasks, return_exceptions=True)
        
        errors = []
        for i, result in enumerate(code_results):
            if isinstance(result, Exception):
                errors.append(f"Error generating code for {code_files[i].filename}: {result}")
        
        return code_results, errors

    def _write_files_to_disk(self, file_details: List[FileDetail], code_results: List[Any], repo_path_obj: Path) -> tuple[List[dict], List[str], int]:
        """Write generated code to disk."""
        write_results = []
        errors_write = []
        success_count = 0

        logger.info(f"Starting file writing phase for {len(code_results)} results to HOST path: {repo_path_obj}")

        for i, result in enumerate(code_results):
            file_detail = file_details[i]
            filename = file_detail.filename
            absolute_path = (repo_path_obj / filename).resolve()
            
            logger.info(f"Processing result for: {filename} (Host Path: {absolute_path})")

            if isinstance(result, Exception):
                # Handle generation error - try to write error file
                write_results.append({
                    "filename": filename,
                    "status": "error",
                    "message": f"Error generating code: {result}"
                })
                try:
                    absolute_path.parent.mkdir(parents=True, exist_ok=True)
                    error_content = (
                        f"# Code generation failed for {filename}: {result}\n\n"
                        "# Code description:\n"
                        f"{file_detail.code_description}"
                    )
                    absolute_path.write_text(error_content, encoding="utf-8", errors="replace")
                except Exception as write_err:
                    logger.error(f"Failed to write error file for {filename}: {write_err}", exc_info=True)
            else:
                # Success case
                code_content = result
                if not code_content or not code_content.strip():
                    logger.warning(f"Generated code content for {filename} is empty. Skipping write.")
                    write_results.append({
                        "filename": filename,
                        "status": "error",
                        "message": "Generated code was empty",
                    })
                    continue

                try:
                    absolute_path.parent.mkdir(parents=True, exist_ok=True)
                    operation = "modify" if absolute_path.exists() else "create"
                    
                    fixed_code = ftfy.fix_text(code_content)
                    absolute_path.write_text(fixed_code, encoding="utf-8", errors="replace")
                    
                    # Log operation
                    log_file_operation(
                        file_path=absolute_path,
                        operation=operation,
                        content=fixed_code,
                        metadata={"code_description": file_detail.code_description},
                    )

                    # Display logic
                    docker_path_display = str(absolute_path)
                    try:
                        docker_path_display = convert_to_docker_path(absolute_path)
                    except Exception:
                        pass

                    write_results.append({
                        "filename": str(docker_path_display),
                        "status": "success",
                        "operation": operation,
                        "code": fixed_code,
                    })
                    success_count += 1
                    
                    if self.display:
                        self.display.add_message("assistant", self._format_terminal_output(
                            command="write_file",
                            files=[filename],
                            result="File written successfully",
                            additional_info=f"Path: {docker_path_display}\nSize: {len(fixed_code)} chars"
                        ))

                except Exception as write_error:
                    logger.error(f"Error writing file {filename}: {write_error}", exc_info=True)
                    errors_write.append(f"Error writing file {filename}: {write_error}")
                    write_results.append({
                        "filename": filename,
                        "status": "error",
                        "message": f"Error writing file: {write_error}",
                    })

        return write_results, errors_write, success_count

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

        if command == "write_codebase":
            if data.get("status") == "success":
                return f"Successfully wrote codebase to {data.get('write_path', 'unknown path')}"
            else:
                return f"Failed to write codebase: {data.get('message', 'Unknown error')}"

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
        """Generate full file content for the target file using multiple models from CODE_LIST."""
        
        # Use the new multi-model approach
        return await self._generate_code_with_multiple_models(
            code_description=code_description,
            external_imports=external_imports,
            internal_imports=internal_imports,
            file_path=file_path
        )

    async def _generate_code_with_multiple_models(
        self,
        code_description: str,
        external_imports: List[str],
        internal_imports: List[str],
        file_path: Path,
    ) -> str:
        """Generate code using multiple models from CODE_LIST in parallel, then select the best version."""
        
        # Prepare messages once for all models
        agent_task = self._get_task_description()
        log_content = self._get_file_creation_log_content()
        
        prepared_messages = code_prompt_generate(
            current_code_base="",
            code_description=code_description,
            research_string="",
            agent_task=agent_task,
            external_imports=external_imports,
            internal_imports=internal_imports,
            target_file=str(file_path.name),
            file_creation_log_content=log_content,
        )
        
        # Validate prepared_messages
        if prepared_messages is None:
            logger.error(f"code_prompt_generate returned None for {file_path.name}")
            raise LLMResponseError("Failed to prepare messages for code generation: code_prompt_generate returned None")
        
        if not isinstance(prepared_messages, list):
            logger.error(f"code_prompt_generate returned invalid type {type(prepared_messages)} for {file_path.name}")
            raise LLMResponseError(f"Failed to prepare messages for code generation: expected list, got {type(prepared_messages)}")
        
        # Generate code with all models in CODE_LIST in parallel
        logger.info(f"Generating code for {file_path.name} using {len(CODE_LIST)} models in parallel: {CODE_LIST}")
        
        generation_tasks = []
        for model in CODE_LIST:
            task = self._generate_code_with_single_model(
                prepared_messages=prepared_messages,
                file_path=file_path,
                model=model
            )
            generation_tasks.append(task)
        
        # Execute all generation tasks in parallel
        generation_results = await asyncio.gather(*generation_tasks, return_exceptions=True)
        
        # Filter successful results
        successful_generations = []
        for i, result in enumerate(generation_results):
            if isinstance(result, Exception):
                logger.warning(f"Model {CODE_LIST[i]} failed for {file_path.name}: {result}")
            elif isinstance(result, str) and result and not result.startswith("# Error"):
                successful_generations.append({
                    "model": CODE_LIST[i],
                    "code": result
                })
                logger.info(f"Model {CODE_LIST[i]} successfully generated {len(result)} characters for {file_path.name}")
        
        # If no successful generations, fall back to original method
        if not successful_generations:
            logger.warning(f"All models failed for {file_path.name}, falling back to single model retry")
            # Fallback to single model generation with retry
            model_to_use = get_constant("CODE_GEN_MODEL") or MODEL_STRING
            try:
                return await self._llm_generate_code_core_with_retry(
                    prepared_messages=prepared_messages,
                    file_path=file_path,
                    model_to_use=model_to_use
                )
            except Exception as e:
                logger.error(f"Fallback generation failed for {file_path.name}: {e}")
                return f"# Error: Code generation failed for {file_path.name} after all retries."
        
        # If only one successful generation, use it directly
        if len(successful_generations) == 1:
            logger.info(f"Only one model succeeded for {file_path.name}, using result from {successful_generations[0]['model']}")
            return successful_generations[0]["code"]
        
        # Multiple successful generations - use CODE_MODEL to synthesize the best one
        logger.info(f"Synthesizing best code from {len(successful_generations)} successful generations for {file_path.name}")
        best_code = await self._synthesize_best_code_version(
            successful_generations, code_description, file_path, agent_task
        )
        
        return best_code

    async def _generate_code_with_single_model(
        self,
        prepared_messages: List[Dict[str, str]],
        file_path: Path,
        model: str,
    ) -> str:
        """Generate code using a single model without retry logic for parallel execution."""
        try:
            OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")

            client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                                 api_key=OPENROUTER_API_KEY)

            logger.debug(f"Calling model {model} for {file_path.name}")

            completion = await client.chat.completions.create(
                model=model, messages=prepared_messages)

            if not (completion and completion.choices
                    and completion.choices[0].message
                    and completion.choices[0].message.content):
                raise LLMResponseError(f"Invalid or empty completion content from {model}")

            raw_code_string = completion.choices[0].message.content
            code_string, detected_language = self.extract_code_block(raw_code_string, file_path)

            if code_string == "No Code Found":
                raise LLMResponseError(f"No code found in response from {model}")

            if code_string.startswith("# Error: Code generation failed"):
                raise LLMResponseError(f"Model {model} returned error placeholder")

            return code_string

        except Exception as e:
            logger.warning(f"Model {model} failed for {file_path.name}: {e}")
            raise e

    async def _synthesize_best_code_version(
        self,
        code_versions: List[Dict[str, str]],
        code_description: str,
        file_path: Path,
        agent_task: str,
    ) -> str:
        """Use CODE_MODEL to synthesize the best code version from multiple generated versions."""
        
        # Prepare the synthesis prompt
        versions_text = ""
        for i, version in enumerate(code_versions, 1):
            versions_text += f"\n=== OPTION {i} (from {version['model']}) ===\n"
            versions_text += version['code']
            versions_text += f"\n=== END OPTION {i} ===\n"
        
        current_codebase = self._get_current_codebase_context()
        
        synthesis_prompt = f"""As an expert software developer, your task is to create the best possible version of a file by synthesizing from multiple AI-generated options.

**File to Create:** `{file_path.name}`
**Project Goal:** {agent_task}
**File Requirements:** {code_description}

**Codebase Context:**
{current_codebase}

**AI-Generated Options:**
{versions_text}

**Your Task:**
1.  **Analyze:** Carefully review each provided option. Identify their strengths (e.g., good structure, correctness, efficiency) and weaknesses (e.g., bugs, poor style, incomplete implementation).
2.  **Synthesize:** Do **not** just pick one. Create a new, superior version of the code. Combine the best parts of the given options, add your own expert improvements, and ensure the final code is correct, complete, and adheres to best practices.
3.  **Fit:** Ensure your final version fits seamlessly into the existing codebase structure and meets all stated requirements.

**Output ONLY the complete, final code for the file. Do not include any explanations, comments, or markdown formatting.**"""
        
        synthesis_fallback_reason = "Unknown reason"
        try:
            if self.display:
                console_output = self._format_terminal_output(
                    command="synthesize_code",
                    files=[file_path.name],
                    result=f"Synthesizing best version for {file_path.name} using {CODE_MODEL}",
                    additional_info=f"Analyzing {len(code_versions)} generated options."
                )
                self.display.add_message("assistant", console_output)

            OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY not set")

            client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

            synthesis_messages = [{"role": "user", "content": synthesis_prompt}]
            
            completion = await client.chat.completions.create(
                model=CODE_MODEL, 
                messages=synthesis_messages,
                max_tokens=4000, # Allow for larger synthesized files
                temperature=0.2
            )
            
            if completion and completion.choices and completion.choices[0].message and completion.choices[0].message.content:
                synthesized_code = completion.choices[0].message.content.strip()
                
                # Extract code block just in case the model wraps it
                final_code, _ = self.extract_code_block(synthesized_code, file_path)

                if final_code != "No Code Found" and final_code.strip():
                    logger.info(f"Successfully synthesized a new version for {file_path.name} using {CODE_MODEL}")
                    if self.display:
                        console_output = self._format_terminal_output(
                            command="synthesize_code",
                            files=[file_path.name],
                            result=f"Synthesis successful for {file_path.name}",
                            additional_info=f"Generated new version of {len(final_code)} characters."
                        )
                        self.display.add_message("assistant", console_output)
                    return final_code
                else:
                    synthesis_fallback_reason = f"Synthesis from {CODE_MODEL} resulted in empty code."
            else:
                synthesis_fallback_reason = f"No content in response from {CODE_MODEL} during synthesis."

        except Exception as e:
            synthesis_fallback_reason = f"Code synthesis failed: {e}"
            logger.warning(f"{synthesis_fallback_reason}, defaulting to first version for {file_path.name}")

        # Fallback to first version if synthesis fails
        logger.info(f"Using first version from {code_versions[0]['model']} for {file_path.name} due to synthesis fallback.")
        if self.display:
            model_name = code_versions[0]['model']
            console_output = self._format_terminal_output(
                command="synthesize_code",
                files=[file_path.name],
                result=f"Synthesis failed for {file_path.name}, using fallback.",
                additional_info=f"Using version from: {model_name}\nReason: {synthesis_fallback_reason}"
            )
            self.display.add_message("assistant", console_output)
        return code_versions[0]["code"]

    def _get_current_codebase_context(self) -> str:
        """Get current codebase context for code selection. This is a simplified version."""
        try:
            # Try to get recent file creation log for context
            log_content = self._get_file_creation_log_content()
            if log_content:
                return f"Recent file operations:\n{log_content[:1000]}..."
            return "No recent codebase context available."
        except Exception:
            return "No codebase context available."



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


    def _get_file_creation_log_content(self) -> str:
        """
        Retrieve the content of the file creation log for providing context to LLM.
        Returns a string representation of recent file operations.
        """
        try:
            log_file_path = get_constant("LOG_FILE")
            if not log_file_path or not Path(log_file_path).exists():
                return ""
            
            with open(log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return content
        except Exception as e:
            logger.warning(f"Failed to read file creation log: {e}")
            return ""

    def _log_generated_output(self, content: str, file_path: Path, output_type: str):
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

    def _format_terminal_output(self, 
                               command: str, 
                               files: Optional[List[str]] = None, 
                               result: Optional[str] = None, 
                               error: Optional[str] = None,
                               additional_info: Optional[str] = None) -> str:
        """Format write code operations to look like terminal output."""
        lines = ["```console"]
                                                                              
        # Add command line
        if files:
            files_str = " ".join(files) if len(files) <= 3 else f"{files[0]} ... +{len(files)-1} more"
            lines.append(f"$ write_code {command} {files_str}")
        else:
            lines.append(f"$ write_code {command}")
        
        # Add result or error
        if error:
            lines.append(f"Error: {error}")
        elif result:
            lines.append(result)
        
        # Add additional info if provided
        if additional_info:
            lines.extend(additional_info.split('\n'))
        
        lines.append("```")
        return '\n'.join(lines)

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
            # No backticks, assume it's plain text and return unknown language
            return text.strip(), "unknown"

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
