import asyncio
from enum import Enum
from typing import Literal, Optional, List
from pathlib import Path
from .base import ToolResult, BaseAnthropicTool
import os
import subprocess
from icecream import ic
from rich import print as rr
import json
from pydantic import BaseModel
import tempfile
from load_constants import write_to_file, ICECREAM_OUTPUT_FILE
from tenacity import retry, stop_after_attempt, wait_fixed
from config import *
from openai import OpenAI, AsyncOpenAI
from utils.file_logger import *
from utils.context_helpers import *
import time
from system_prompt.code_prompts import (
    code_prompt_research,
    code_prompt_generate,
    code_skeleton_prompt,
    )
import logging
from utils.docker_service import DockerService, DockerResult, DockerServiceError
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from pygments.styles import get_style_by_name
from dotenv import load_dotenv
import ftfy
from pygments.lexers import get_lexer_by_name, guess_lexer

def send_email_attachment_of_code(filename, code_string):
    import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition, ContentId
import base64
load_dotenv()

ic.configureOutput(includeContext=True, outputFunction=write_to_file)


def write_chat_completion_to_file(response, filepath):
    """Appends an OpenAI ChatCompletion object to a file in a human-readable format."""
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(response.get('created', 0)))}\n")
            f.write(f"Model: {response.get('model', 'N/A')}\n")
            f.write("\nUsage:\n")
            usage = response.get("usage", {})
            for key, value in usage.items():
                if isinstance(value, dict):  # Nested details
                    f.write(f"  {key}:\n")
                    for k, v in value.items():
                        f.write(f"    {k}: {v}\n")
                else:
                    f.write(f"  {key}: {value}\n")

            f.write("\nChoices:\n")
            for i, choice in enumerate(response.get("choices", [])):
                f.write(f"  Choice {i+1}:\n")
                f.write(f"    Index: {choice.get('index', 'N/A')}\n")
                f.write(f"    Finish Reason: {choice.get('finish_reason', 'N/A')}\n")

                message = choice.get('message', {})
                if message:
                    f.write("    Message:\n")
                    f.write(f"      Role: {message.get('role', 'N/A')}\n")
                    f.write(f"      Content: {message.get('content', 'None')}\n")
                    
                    if 'refusal' in message:
                        f.write(f"      Refusal: {message['refusal']}\n")


                f.write("\n")

    except Exception as e:
        print(f"Error writing to file: {e}")

def send_email_attachment_of_code(filename, code_string):
    
    message = Mail(
        from_email='sambosisai@outlook.com',
        to_emails='sambosis@gmail.com',
        subject=f"Code File: {filename}",
        html_content='<strong>Here is the file</strong>',
    )


    content = base64.b64encode(code_string.encode('utf-8'))

    message.attachment = Attachment(
                                    FileContent(content.decode('utf-8')),
                                    FileName(filename),
                                    FileType('application/text'),
                                    Disposition('attachment'),
                                    ContentId('Content ID 1')
                                    )
        
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
        print(response.status_code)
        print(response.body)
        print(response.headers)
    except Exception as e:
        print(str(e))


class CodeCommand(str, Enum):
    """
    An enumeration of possible commands for the WriteCodeTool.

    Attributes:
        WRITE_CODE_TO_FILE (str): Command to write code to a file.
        WRITE_AND_EXEC (str): Command to write and execute code.
        WRITE_CODE_MULTIPLE_FILES (str): Command to write multiple files.
        GET_ALL_CODE (str): Command to get all current code.
        GET_REVISED_VERSION (str): Command to get a revised version of an existing file.

    """
    WRITE_CODE_TO_FILE = "write_code_to_file"
    WRITE_AND_EXEC = "write_and_exec"
    WRITE_CODE_MULTIPLE_FILES = "write_code_multiple_files"  # Added new command
    GET_ALL_CODE = "get_all_current_skeleton"
    GET_REVISED_VERSION = "get_revised_version"  # Add new command

class WriteCodeTool(BaseAnthropicTool):
    """
    A tool that takes a description of code that needs to be written and provides the actual programming code in the specified language. It can either execute the code or write it to a file depending on the command.
    """

    name: Literal["write_code"] = "write_code"
    api_type: Literal["custom"] = "custom"
    description: str = "A tool that takes a description of code that needs to be written and provides the actual programming code in the specified language. It can either execute the code or write it to a file depending on the command. It is also able to return all of the code written so far so you can view the contents of all files."
    def __init__(self, display=None):
        super().__init__(display)
        # Initialize Docker service
        self.docker = DockerService()
        self._docker_available = self.docker.is_available()
        ic("Initializing WriteCodeTool")
        ic(f"Docker available: {self._docker_available}")

    def to_params(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": [cmd.value for cmd in CodeCommand],
                        "description": "Command to perform. Options: write_code_to_file, write_and_exec, write_code_multiple_files, get_all_current_code, get_revised_version"
                    },
                    "code_description": {
                        "type": "string",
                        "description": "Description for single file code generation. This should be a very detailed description of the code to be created. Include any assumption and specific details including necessary imports and how to interact with other aspects of the code."
                    },
                    "project_path": {
                        "type": "string",
                        "description": "Path to the project directory."
                    },
                    "python_filename": {
                        "type": "string",
                        "description": "Filename for write_code_to_file command."
                    },
                    "files": {  # Added for multiple file support
                        "type": "array",
                        "description": "List of objects with 'code_description' and 'file_path' for each file to write.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code_description": {"type": "string"},
                                "file_path": {"type": "string"}
                            },
                            "required": ["code_description", "file_path"]
                        }
                    },
                    "revision_description": {
                        "type": "string",
                        "description": "Description of the changes to be made to an existing file"
                    },
                    "target_file": {
                        "type": "string",
                        "description": "Path to the file that needs to be revised"
                    }
                },
                "required": ["command", "project_path"]
            }
        }

    async def __call__(
        self,
        *,
        command: CodeCommand,
        code_description: str = "",  # default empty for multi-file command
        project_path: str = PROJECT_DIR,
        python_filename: str = "you_need_to_name_me.py",
        **kwargs,
        ) -> ToolResult:
        """
        Executes the specified command for project management.
        """

        try:
            if self.display is not None:
                self.display.add_message("user", f"WriteCodeTool Instructions: {code_description}")

            # Convert path string to Path object
            project_path = Path(get_constant("PROJECT_DIR"))
            ic(f"Project path: {project_path}")
            
            # Get command value as string for the ToolResult
            command_str = command.value if hasattr(command, 'value') else str(command)
            
            # Execute the appropriate command
            if command == CodeCommand.WRITE_CODE_TO_FILE:
                result_data = await self.write_code_to_file(code_description, project_path, python_filename)
            elif command == CodeCommand.WRITE_AND_EXEC:
                result_data = await self.write_and_exec(code_description, project_path)
            elif command == CodeCommand.WRITE_CODE_MULTIPLE_FILES:
                files = kwargs.get("files", [])
                if not files:
                    return ToolResult(error="No files provided for multiple file creation.", tool_name=self.name, command=command_str)
                result_data = await self.write_multiple_files(project_path, files)
            elif command == CodeCommand.GET_ALL_CODE:
                result_data = {
                    "command": "GET_ALL_CODE",
                    "status": "success",
                    "project_path": str(project_path),
                    "files_results": get_all_current_code(),
                }
            elif command == CodeCommand.GET_REVISED_VERSION:
                target_file = kwargs.get("target_file")
                revision_description = kwargs.get("revision_description")
                if not target_file or not revision_description:
                    return ToolResult(error="Both target_file and revision_description are required for revision", 
                                     tool_name=self.name, command=command_str)
                result_data = await self.get_revised_version(project_path, target_file, revision_description)
            else:
                ic(f"Unknown command: {command}")
                return ToolResult(error=f"Unknown command: {command}", tool_name=self.name, command=command_str)

            # Add tool name and command to the result data
            result_data["tool_name"] = self.name
            if "command" not in result_data:
                result_data["command"] = command_str

            # Convert result_data to formatted string
            formatted_output = self.format_output(result_data)
            ic(f"formatted_output: {formatted_output}")

            return ToolResult(output=formatted_output, tool_name=self.name, command=command_str)

        except Exception as e:
            error_msg = f"Failed to execute {command}: {str(e)}"
            command_str = command.value if hasattr(command, 'value') else str(command)
            return ToolResult(error=error_msg, tool_name=self.name, command=command_str)

    def extract_code_block(self, text: str, file_path: Optional[Path] = None) -> tuple[str, str]:
        """
        Extracts code based on file type. Special handling for Markdown files.
        Returns tuple of (content, language).
        """
        # If a file_path is provided and it's a Markdown file, return text as-is.
        if file_path is not None and str(file_path).lower().endswith(('.md', '.markdown')):
            return text, "markdown"

        # Original code block extraction logic for other files
        if not text.strip():
            return "No Code Found", "Unknown"

        start_marker = text.find("```")
        if (start_marker == -1):
            return text, "code"

        # Determine language (text immediately after opening delimiter)
        language_line_end = text.find("\n", start_marker)
        if (language_line_end == -1):
            language_line_end = start_marker + 3
        language = text[start_marker + 3:language_line_end].strip()
        if not language:
            language = "code"

        end_marker = text.find("```", language_line_end)
        if (end_marker == -1):
            code_block = text[language_line_end:].strip()
        else:
            code_block = text[language_line_end:end_marker].strip()

        return code_block if code_block else "No Code Found", language

    async def _call_llm_to_generate_code(self, code_description: str, research_string: str, file_path) -> str:
        """Call LLM to generate code based on the code description"""

        if self.display is not None:
            self.display.add_message("assistant", f"Generating code for: {file_path}")

        code_string = "no code created"

        current_code_base = get_all_current_skeleton()
        
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            )
        model = "google/gemini-2.0-flash-001"
        # client = AsyncOpenAI()
        # model = "o3-mini"

        # Prepare messages
        messages = code_prompt_generate(current_code_base, code_description, research_string)
        ic(f"Here are the messages being sent to Generate the Code\n +++++ \n +++++ \n{messages}")
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
            )

            # Ensure we handle None completion or empty response
            if completion is None or not hasattr(completion, 'choices') or not completion.choices:
                ic("No valid completion received from LLM")
                return f"# Failed to generate code for {file_path}\n# Please try again with more detailed description."
                
            code_string = completion.choices[0].message.content
            if not code_string:
                ic("Empty content returned from LLM")
                return f"# Failed to generate code for {file_path}\n# Empty response received."

        except Exception as e:
            ic(f"error in _call_llm_to_generate_code: {e}")
            return f"# Error generating code: {str(e)}"

        # Extract code using the new function
        try:
            code_string, detected_language = self.extract_code_block(code_string, file_path)
        except Exception as parse_error:
            ic(f"Error parsing code block: {parse_error}")
            # Return a safer fallback
            return f"# Failed to parse code block: {str(parse_error)}\n\n{code_string}"

        # Log the extraction
        try:
            CODE_FILE = Path(get_constant("CODE_FILE"))
            with open(CODE_FILE, "a", encoding="utf-8") as f:
                f.write(f"File Path: {str(file_path)}\n")
                f.write(f"Language detected: {detected_language}\n")
                f.write(f"{code_string}\n")
        except Exception as file_error:
            # Log failure but don't stop execution
            if self.display is not None:
                try:
                    self.display.add_message("user", f"Failed to log code: {str(file_error)}")
                except Exception:
                    pass

        if detected_language == "html":
            # Send the HTML content directly to the display for rendering
            if self.display is not None:
                self.display.add_message("tool", {"html": code_string})
            code_display = "HTML rendered in display."
            css_styles = ""
        else:
            ### Highlight the code
            # Attempt to get a lexer based on the detected language, otherwise guess
            try:
                if detected_language and detected_language != 'code':
                    lexer = get_lexer_by_name(detected_language, stripall=True)
                else:
                    lexer = guess_lexer(code_string)
            except Exception:
                lexer = PythonLexer()  # Fallback to Python lexer

            # Create an HTML formatter with full=False
            formatter = HtmlFormatter(style="monokai", full=False, linenos=True, wrap=True)        

            code_temp=f"#{str(file_path)}\n{code_string}"

            # Highlight the code
            code_display = highlight(code_temp, lexer, formatter)

            # Get CSS styles
            css_styles = formatter.get_style_defs(".highlight")

        # Return BOTH the highlighted code and the CSS
        if self.display is not None:
            self.display.add_message("tool", {"code": code_display, "css": css_styles})
            
        # send_email_attachment_of_code(str(file_path), code_string)
        return code_string

    async def _call_llm_to_research_code(self, code_description: str, file_path) -> str:
        """Call LLM to generate code based on the code description"""
        if self.display is not None:
            self.display.add_message("assistant", f"Researching code for: {file_path}")
            
        code_string = "no code created"
        current_code_base = get_all_current_skeleton()
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            )
        model = "google/gemini-2.0-flash-001"

        # Prepare messages
        messages = code_prompt_research(current_code_base, code_description)
        ic(f"Here are the messages being sent to Research the Code\n +++++ \n +++++ \n{messages}")
        try:
            completion =  await client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as e:
            ic(completion)
            ic(f"error: {e}")
            return code_description   

        # Handle both OpenRouter and standard OpenAI response formats
        try:
            if hasattr(completion.choices[0].message, 'content'):
                research_string = completion.choices[0].message.content
            else:
                research_string = completion.choices[0].message['content']
        except (AttributeError, KeyError, IndexError) as e:
            ic(f"Error extracting content: {e}")
            return code_description
        research_string = ftfy.fix_text(research_string)

        return research_string

    async def _call_llm_for_code_skeleton(self, code_description: str, file_path) -> str:
        """Call LLM to generate code skeleton based on the code description"""
        # if self.display is not None:
            # self.display.add_message("assistant", f"Creating code skeleton for: {file_path}")
            
        skeleton_string = "no code created"
        current_code_base = get_all_current_skeleton()
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            )
        model = "google/gemini-2.0-flash-001"

        # Prepare messages
        messages = code_skeleton_prompt(code_description)
        ic(f"Here are the messages being sent to Create Code Skeleton\n +++++ \n +++++ \n{messages}")
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as e:
            ic(completion)
            ic(f"error: {e}")
            return code_description   

        # Handle both OpenRouter and standard OpenAI response formats
        try:
            if hasattr(completion.choices[0].message, 'content'):
                skeleton_string = completion.choices[0].message.content
            else:
                skeleton_string = completion.choices[0].message['content']
        except (AttributeError, KeyError, IndexError) as e:
            ic(f"Error extracting content: {e}")
            return code_description

        skeleton_string = ftfy.fix_text(skeleton_string)
        ic(skeleton_string)
        return skeleton_string

    def format_output(self, data: dict) -> str:
        """Format the output data as a readable string"""
        output_lines = []

        # Add tool name 
        output_lines.append(f"Tool: {data.get('tool_name', self.name)}")

        # Add command type
        output_lines.append(f"Command: {data['command']}")

        # Add status
        output_lines.append(f"Status: {data['status']}")

        # Add project path
        output_lines.append(f"Project Path: {data['project_path']}")

        # Add filename if present (for single file operations)
        if 'filename' in data:
            output_lines.append(f"Filename: {data['filename']}")
            code_string = data.get('code_string', '')
            if len(code_string) > 150000:
                code_string = code_string[:75000] + " ... [TRUNCATED] ... " + code_string[-75000:]
            output_lines.append(f"Code:\n{code_string}")

        # Add files_results (for multiple files, this is already formatted)
        if 'files_results' in data:
            files_results = data['files_results']
            if len(files_results) > 150000:
                files_results = files_results[:75000] + " ... [TRUNCATED] ... " + files_results[-75000:]
            output_lines.append(files_results)

        # Add packages if present
        if 'packages_installed' in data:
            output_lines.append("Packages Installed:")
            for package in data['packages_installed']:
                output_lines.append(f"  - {package}")

        # Add run output if present
        if 'run_output' in data and data['run_output']:
            run_output = data['run_output']
            if len(run_output) > 150000:
                run_output = run_output[:75000] + " ... [TRUNCATED] ... " + run_output[-75000:]
            output_lines.append("\nApplication Output:")
            output_lines.append(run_output)

        if 'errors' in data and data['errors']:
            errors = data['errors']
            if len(errors) > 150000:
                errors = errors[:75000] + " ... [TRUNCATED] ... " + errors[-75000:]
            output_lines.append("\nErrors:")
            output_lines.append(errors)

        # Join all lines with newlines
        return "\n".join(output_lines)

    async def write_code_to_file(self, code_description: str, project_path: Path, filename) -> dict:
        """
        Write code to a file in the Docker container.
        
        Generates code based on description and writes it directly to the Docker container
        rather than the local file system.
        """
        local_file_path = project_path / filename
        ic(f"Writing code to file: {local_file_path}")
        try:
            # Initialize Docker service
            docker = DockerService()
            if not docker.is_available():
                return {
                    "command": "write_code_to_file",
                    "status": "error",
                    "project_path": str(project_path),
                    "filename": filename,
                    "error": "Docker service not available"
                }

            # Get the Docker path with normalized Linux forward slashes
            docker_project_dir = docker.to_docker_path(project_path)
            docker_project_dir_str = str(docker_project_dir).replace('\\', '/')

            # Ensure we have a correct Linux-style path with forward slashes
            docker_file_path = f"{docker_project_dir_str}/{filename}".replace('\\', '/')
            ic(f"Docker file path: {docker_file_path}")

            # Generate code skeleton
            code_research_string = await self._call_llm_for_code_skeleton(
                code_description, local_file_path
            )

            # Generate full code implementation
            code_string = await self._call_llm_to_generate_code(
                code_description, code_research_string, local_file_path
            )

            # Debug log the generated code length
            ic(f"Generated code length: {len(code_string)} chars")

            # Create parent directories in Docker
            parent_dir = str(Path(docker_file_path).parent).replace('\\', '/')
            mkdir_result = docker.execute_command(f"mkdir -p {parent_dir}")

            if not mkdir_result.success:
                return {
                    "command": "write_code_to_file",
                    "status": "error",
                    "project_path": str(project_path),
                    "filename": filename,
                    "error": f"Failed to create directory in Docker: {mkdir_result.stderr}"
                }

            # Write file directly using a temporary file and docker cp
            # Create a temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.py', text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as temp_file:
                    temp_file.write(code_string)

                # Ensure temp file has content
                temp_size = os.path.getsize(temp_path)
                ic(f"Temporary file created at {temp_path} with size: {temp_size} bytes")

                if temp_size == 0:
                    return {
                        "command": "write_code_to_file",
                        "status": "error", 
                        "project_path": str(project_path),
                        "filename": filename,
                        "error": "Generated content is empty"
                    }

                # Copy the file to Docker using docker cp with escaped paths
                docker_cp_command = f'docker cp "{temp_path}" {docker._container_name}:"{docker_file_path}"'
                ic(f"Docker copy command: {docker_cp_command}")

                cp_result = subprocess.run(
                    docker_cp_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=False
                )

                if cp_result.returncode != 0:
                    ic(f"Docker cp failed: {cp_result.stderr}")
                    return {
                        "command": "write_code_to_file",
                        "status": "error",
                        "project_path": str(project_path),
                        "filename": filename,
                        "error": f"Failed to copy file to Docker: {cp_result.stderr}"
                    }

                # Verify the file was created successfully
                check_cmd = f"[ -f {docker_file_path} ] && wc -c {docker_file_path} || echo 'File not found'"
                verify_result = docker.execute_command(check_cmd)

                ic(f"File verification result: {verify_result.stdout}")

                # Also write to local filesystem for logging
                await asyncio.to_thread(os.makedirs, local_file_path.parent, exist_ok=True)
                await asyncio.to_thread(local_file_path.write_text, code_string, encoding='utf-8')

                # Log the file creation
                await asyncio.to_thread(log_file_operation, local_file_path, "create")

            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    ic(f"Failed to delete temp file: {e}")

            return {
                "command": "write_code_to_file",
                "status": "success",
                "project_path": str(project_path),
                "filename": filename,
                "docker_path": docker_file_path,
                "code_string": code_string
            }

        except Exception as e:
            import traceback
            ic(f"Error in write_code_to_file: {str(e)}")
            ic(traceback.format_exc())
            return {
                "command": "write_code_to_file",
                "status": "error",
                "project_path": str(project_path),
                "filename": filename,
                "error": f"Error writing file to Docker: {str(e)}"
            }

    async def write_multiple_files(self, project_path: Path, files: List[dict]) -> dict:
        """
        Write multiple files based on descriptions.
        
        Args:
            project_path: Base project path
            files: List of dictionaries with 'code_description' and 'file_path' keys
            
        Returns:
            Dictionary with command, status, project_path, and files_results
        """
        results = []
        status = "success"
        errors = []
        
        # Process each file in the list
        for file_entry in files:
            file_path = file_entry.get("file_path", "")
            code_description = file_entry.get("code_description", "")
            
            if not file_path or not code_description:
                errors.append(f"Invalid file entry: Missing required fields: {file_entry}")
                continue
            
            try:
                # Use the existing write_code_to_file method to handle each file
                result = await self.write_code_to_file(code_description, project_path, file_path)
                if result.get("status") == "error":
                    status = "partial_success"  # At least one file had an error
                    errors.append(f"Error writing {file_path}: {result.get('error', 'Unknown error')}")
                
                results.append({
                    "file_path": file_path,
                    "status": result.get("status", "unknown"),
                    "code_length": len(result.get("code_string", "")) if "code_string" in result else 0
                })
                
            except Exception as e:
                status = "partial_success"
                errors.append(f"Exception while processing {file_path}: {str(e)}")
        
        # If all files failed, mark status as error
        if status == "success" and not results:
            status = "error"
        elif errors and len(errors) == len(files):
            status = "error"
        
        return {
            "command": "write_code_multiple_files",
            "status": status,
            "project_path": str(project_path),
            "files_processed": len(results),
            "files_results": "\n".join([f"- {r['file_path']}: {r['status']} ({r['code_length']} bytes)" for r in results]),
            "errors": "\n".join(errors) if errors else None
        }
