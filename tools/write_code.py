import asyncio
from enum import Enum
from typing import Literal, Optional, List, Dict, Any
from pathlib import Path
from .base import ToolResult, BaseAnthropicTool
import os
from icecream import ic
from rich import print as rr
import json
from pydantic import BaseModel
from load_constants import (
    write_to_file,
    ICECREAM_OUTPUT_FILE,
)  # Assuming ICECREAM_OUTPUT_FILE is used elsewhere
from tenacity import retry, stop_after_attempt, wait_fixed
from config import *  # Ensure PROJECT_DIR and REPO_DIR are available via this import
from openai import AsyncOpenAI
from utils.file_logger import *  # Ensure convert_to_docker_path, log_file_operation are here
from utils.context_helpers import *  # Ensure get_language_from_extension is here
import time
from system_prompt.code_prompts import (
    code_prompt_generate,
    code_skeleton_prompt,
)
import logging
from utils.docker_service import (
    DockerService,
    DockerResult,
    DockerServiceError,
)  # Assuming DockerService might be used elsewhere
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name
from dotenv import load_dotenv
import ftfy
from pygments.lexers import get_lexer_by_name, guess_lexer
from loguru import logger as ll
import traceback
from rich import print as rr
from lmnr import observe
googlepro = "google/gemini-2.5-pro-preview-03-25"
oa4omini = "openai/o4-mini-high"
MODEL_STRING = googlepro  # Default model string, can be overridden in config


class CodeCommand(str, Enum):
    """
    An enumeration of possible commands for the WriteCodeTool.
    """

    WRITE_CODEBASE = "write_codebase"

class FileDetail(BaseModel):
    """Model for specifying file details for code generation."""

    filename: str
    code_description: str
    external_imports: Optional[List[str]] = None
    internal_imports: Optional[List[str]] = None

class WriteCodeTool(BaseAnthropicTool):
    """
    A tool that takes a description of a codebase, including files, external and internal imports,
    generates code skeletons, and then generates the full code for each file asynchronously,
    writing them to the correct host path.
    """

    name: Literal["write_codebase_tool"] = "write_codebase_tool"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "Generates a codebase consisting of multiple files based on descriptions, skeletons, and import lists. Creates skeletons first, then generates full code asynchronously, writing to the host filesystem."
    )

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display
        # DockerService might be needed for other potential features, keep initialization
        try:
            self.docker = DockerService()
            self._docker_available = self.docker.is_available()
        except Exception as docker_init_err:
            # rr(f"Failed to initialize DockerService: {docker_init_err}")
            self.docker = None
            self._docker_available = False
        ic("Initializing WriteCodeTool")
        ic(f"Docker available: {self._docker_available}")

    def to_params(self) -> dict:
        ic(f"WriteCodeTool.to_params called with api_type: {self.api_type}")
        params = {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
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
        }
        ic(f"WriteCodeTool params: {params}")
        return params

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
                    file, # Pass the FileDetail object for the current file
                    host_project_path_obj / file.filename, # Pass the intended final host path
                    file_details # Pass the list of ALL FileDetail objects
                )
                for file in file_details # Iterate through the validated FileDetail objects
            ]
            skeleton_results = await asyncio.gather(
                *skeleton_tasks, return_exceptions=True
            )

            skeletons: Dict[str, str] = {}
            errors_skeleton = []
            for i, result in enumerate(skeleton_results):
                filename_key = file_details[i].filename # Use the relative filename
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
                rr(
                    f"Processing result for: {filename} (Host Path: {absolute_path})"
                )

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
                        rr(
                            f"Successfully executed write_text for: {absolute_path}"
                        )

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
                                        f"File size is 0 despite non-empty content being written!"
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

    
    @observe()
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(8))
    async def _call_llm_to_generate_code(
        self,
        code_description: str,
        all_skeletons: Dict[str, str],
        external_imports: List[str],
        internal_imports: List[str],
        file_path: Path,
        ) -> str:  # file_path is the intended final host path
        """Call LLM to generate code based on the code description, skeletons, and specific imports for this file."""

        if self.display is not None:
            self.display.add_message(
                "assistant", f"Generating code for: {file_path.name}"
                )  # Display relative name

        code_string = f"# Error: Code generation failed for {file_path.name}"

        # Create context from skeletons, using relative filenames
        skeleton_context = "\n\n---\n\n".join(
            f"### Skeleton for {fname}:\n```\n{skel}\n```"
            for fname, skel in all_skeletons.items()  # fname is already relative key
            )

        agent_task = get_constant("TASK") or "No overall task description provided."

        messages = code_prompt_generate(
            current_code_base="",  # Consider if current codebase context is needed differently now
            code_description=code_description,
            research_string="",  # Research step removed in this version
            agent_task=agent_task,
            skeletons=skeleton_context,
            external_imports=external_imports,
            internal_imports=internal_imports,
            target_file=str(file_path.name),  # Pass relative filename to prompt
            )

        ic(
            f"Messages for Code Generation ({file_path.name}):\n +++++ \n{messages}\n +++++"
            )

        try:
            OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")

            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                )
            # Consider making model configurable
            model = get_constant("CODE_GEN_MODEL") or MODEL_STRING
            rr(f"Using model {model} for code generation.")

            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                # Consider adding temperature, max_tokens etc. if needed
            )

            if (
                completion
                and completion.choices
                and completion.choices[0].message
                and completion.choices[0].message.content
            ):
                raw_code_string = completion.choices[0].message.content
                # Pass the host file_path to extract_code_block for context (e.g., .md handling)
                code_string, detected_language = self.extract_code_block(
                    raw_code_string, file_path
                )
                rr(
                    f"Extracted code block for {file_path.name}. Detected language: {detected_language}"
                )
                if code_string == "No Code Found":
                    # If extraction fails but content exists, use the raw response
                    if raw_code_string.strip():
                        code_string = raw_code_string
                        rr(
                            f"Could not extract code block for {file_path.name}, using raw response."
                        )
                    else:
                        rr(
                            f"LLM response for {file_path.name} was effectively empty."
                        )
                        code_string = f"# Failed to generate code for {file_path.name}\n# LLM response was empty."

            else:
                ic(f"No valid completion received for {file_path.name}")
                rr(
                    f"Invalid or empty completion received from LLM for {file_path.name}"
                )
                code_string = f"# Failed to generate code for {file_path.name}\n# LLM response was empty or invalid."

        except Exception as e:
            ic(f"Error in _call_llm_to_generate_code for {file_path.name}: {e}")
            rr(f"LLM call failed for {file_path.name}")
            code_string = f"# Error generating code for {file_path.name}: {str(e)}"

        # Log the generated code (or error message) to a central file if configured
        try:
            code_log_file_path = get_constant("CODE_FILE")
            if code_log_file_path:
                CODE_FILE = Path(code_log_file_path)
                CODE_FILE.parent.mkdir(
                    parents=True, exist_ok=True
                )  # Ensure log dir exists
                with open(CODE_FILE, "a", encoding="utf-8") as f:
                    f.write(
                        f"\n--- Generated Code for: {str(file_path)} ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
                    )
                    f.write(f"{code_string}\n")
                    f.write(f"--- End Code for: {str(file_path)} ---\n")
        except Exception as file_error:
            rr(
                f"Failed to log generated code for {file_path.name} to {get_constant('CODE_FILE')}: {file_error}"
            )

        return code_string

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(8))
    async def _call_llm_for_code_skeleton(
        self,
        file_detail: FileDetail,  # Pass the whole detail object
        file_path: Path,
        all_file_details: List[FileDetail],  # Pass the list of all file details
    ) -> str:  # file_path is the intended final host path
        """Call LLM to generate code skeleton based on the code description"""
        target_file_name = file_path.name  # Get relative name for prompt

        if self.display:
            self.display.add_message(
                "assistant", f"Generating skeleton for {target_file_name}"
            )

        skeleton_string = f"# Error: Skeleton generation failed for {target_file_name}"

        try:
            OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY environment variable not set.")

            client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            model = get_constant("SKELETON_GEN_MODEL") or MODEL_STRING
            rr(f"Using model {model} for skeleton generation for {target_file_name}.")

            # --- Get additional context ---
            agent_task = get_constant("TASK") or "No overall task description provided."
            external_imports = file_detail.external_imports  # Get from FileDetail
            internal_imports = file_detail.internal_imports  # Get from FileDetail

            # Convert FileDetail objects to simple dicts for the prompt function if needed
            all_files_dict_list = [f.model_dump() for f in all_file_details]
            # --- End Get additional context ---

            # CHANGE: Call updated prompt function with more args
            messages = code_skeleton_prompt(
                code_description=file_detail.code_description,  # From FileDetail
                target_file=target_file_name,  # Pass relative filename
                agent_task=agent_task,
                external_imports=external_imports,
                internal_imports=internal_imports,
                all_file_details=all_files_dict_list,  # Pass context of all files
            )
            ic(
                f"Messages for Skeleton ({target_file_name}):\n +++++ \n{messages}\n +++++"
            )

            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
            )

            if (
                completion
                and completion.choices
                and completion.choices[0].message
                and completion.choices[0].message.content
            ):
                raw_skeleton = completion.choices[0].message.content
                # Pass the host file_path to extract_code_block for context
                skeleton_string, detected_language = self.extract_code_block(
                    raw_skeleton, file_path
                )
                rr(
                    f"Extracted skeleton block for {target_file_name}. Detected language: {detected_language}"
                )

                if skeleton_string == "No Code Found":
                    if raw_skeleton.strip():
                        skeleton_string = raw_skeleton  # Use raw if extraction fails but content exists
                        rr(
                            f"Could not extract skeleton block for {target_file_name}, using raw response."
                        )
                    else:
                        rr(
                            f"LLM response for skeleton {target_file_name} was effectively empty."
                        )
                        skeleton_string = f"# Failed to generate skeleton for {target_file_name}\n# LLM response was empty."

                skeleton_string = ftfy.fix_text(skeleton_string)
            else:
                ic(f"No valid skeleton completion received for {target_file_name}")
                rr(
                    f"Invalid or empty completion received from LLM for skeleton {target_file_name}"
                )
                skeleton_string = f"# Failed to generate skeleton for {target_file_name}\n# LLM response was empty or invalid."

        except Exception as e:
            ic(f"Error in _call_llm_for_code_skeleton for {target_file_name}: {e}")
            rr(f"LLM skeleton call failed for {target_file_name}")
            skeleton_string = (
                f"# Error generating skeleton for {target_file_name}: {str(e)}"
            )

        ic(f"Skeleton for {target_file_name}:\n{skeleton_string}")
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

    def format_output(self, data: dict) -> str:
        """
        Format the output of the tool for display.

        Args:
            data: The data returned by the tool's operation.

        Returns:
            A formatted string for display.
        """
        status = data.get("status", "Unknown")
        message = data.get("message", "No message provided.")
        errors = data.get("errors", [])
        files_processed = data.get("files_processed", 0)
        files_successful = data.get("files_successful", 0)

        output = f"Operation Status: {status.upper()}\n"
        output += f"Message: {message}\n"
        output += (
            f"Files Processed: {files_processed}, Successful: {files_successful}\n"
        )

        if errors:
            output += f"\nErrors Encountered ({len(errors)}):\n"
            # Show first few errors
            for i, err in enumerate(errors[:5]):
                output += f"- {err}\n"
            if len(errors) > 5:
                output += f"... and {len(errors) - 5} more errors.\n"
            output += "(Check logs for full details)\n"

        # Optionally add details about successful files if needed
        # results = data.get("results", [])
        # successful_files = [r['filename'] for r in results if r.get('status') == 'success']
        # if successful_files:
        #    output += f"\nSuccessful Files:\n" + "\n".join([f"- {f}" for f in successful_files])

        return output.strip()


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
    except Exception as e:

        return f"<pre>{code}</pre>"
    finally:
        # Ensure any cleanup or finalization if needed
        pass
