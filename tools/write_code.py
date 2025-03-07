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

def html_format_code(code, extension):
    """Format code with syntax highlighting for HTML display."""
    try:
        # Try to get a lexer based on the file extension
        try:
            lexer = get_lexer_by_name(extension.lower().lstrip('.'))
        except:
            # If that fails, try to guess the lexer from the code content
            lexer = guess_lexer(code)
            
        # Use a nice style for the highlighting
        formatter = HtmlFormatter(style='monokai', linenos=True, cssclass="source")
        
        # Highlight the code
        highlighted = highlight(code, lexer, formatter)
        
        # Add some CSS for better display
        css = formatter.get_style_defs('.source')
        html = f"""
        <style>
        {css}
        .source {{ background-color: #272822; padding: 10px; border-radius: 5px; }}
        </style>
        {highlighted}
        """
        return html
    except Exception as e:
        # If highlighting fails, return plain code in a pre tag
        return f"<pre>{code}</pre>"

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
        super().__init__(input_schema=None, display=display)
        self.display = display  # Explicitly set self.display
        # Initialize Docker service
        self.docker = DockerService()
        self._docker_available = self.docker.is_available()
        ic("Initializing WriteCodeTool")
        ic(f"Docker available: {self._docker_available}")

    def to_params(self) -> dict:
        ic(f"WriteCodeTool.to_params called with api_type: {self.api_type}")
        # Use the format that has worked in the past
        params = {
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
                    }
                },
                "required": ["command"]
            }
        }
        ic(f"WriteCodeTool params: {params}")
        return params

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
        Execute the tool with the given command and parameters.
        
        Args:
            command: The command to execute
            code_description: Description of the code to write
            project_path: Path to the project directory
            python_filename: Name of the file to write
            **kwargs: Additional parameters
            
        Returns:
            A ToolResult object with the result of the operation
        """
        import traceback
        
        try:
            # Convert project_path to Path object if it's a string
            if isinstance(project_path, str):
                from pathlib import Path
                project_path = Path(project_path)
            
            # Ensure the project directory exists
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Handle different commands
            if command == CodeCommand.WRITE_CODE_TO_FILE:
                result = await self.write_code_to_file(
                    code_description, project_path, python_filename
                )
                
                # Check if we have an error in the result
                if "error" in result and result["error"]:
                    return ToolResult(
                        error=result["error"],
                        output=result.get("output", "Failed to write code"),
                        tool_name=self.name,
                        command=command
                    )
                
                return ToolResult(
                    output=self.format_output(result),
                    tool_name=self.name,
                    command=command
                )
                
            elif command == CodeCommand.WRITE_CODE_MULTIPLE_FILES:
                files = kwargs.get("files", [])
                if not files:
                    return ToolResult(
                        error="No files specified for write_code_multiple_files command",
                        tool_name=self.name,
                        command=command
                    )
                    
                result = await self.write_multiple_files(project_path, files)
                
                # Check if we have an error in the result
                if "error" in result and result["error"]:
                    return ToolResult(
                        error=result["error"],
                        output=result.get("output", "Failed to write multiple files"),
                        tool_name=self.name,
                        command=command
                    )
                
                return ToolResult(
                    output=self.format_output(result),
                    tool_name=self.name,
                    command=command
                )
                
            elif command == CodeCommand.GET_ALL_CODE:
                from utils.file_logger import get_all_current_code
                return ToolResult(
                    output=get_all_current_code(),
                    tool_name=self.name,
                    command=command
                )
                
            elif command == CodeCommand.GET_REVISED_VERSION:
                file_path = kwargs.get("file_path", "")
                if not file_path:
                    return ToolResult(
                        error="No file_path specified for get_revised_version command",
                        tool_name=self.name,
                        command=command
                    )
                    
                # TODO: Implement get_revised_version
                return ToolResult(
                    error="get_revised_version command not implemented yet",
                    tool_name=self.name,
                    command=command
                )
                
            else:
                return ToolResult(
                    error=f"Unknown command: {command}",
                    tool_name=self.name,
                    command=command
                )
                
        except Exception as e:
            error_message = f"Error in WriteCodeTool: {str(e)}\n{traceback.format_exc()}"
            print(error_message)  # Print to console for debugging
            return ToolResult(
                error=error_message,
                tool_name=self.name,
                command=command
            )

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

    async def write_code_to_file(self, code_description: str, project_path: Path, filename) -> dict:
        """Generate and write code to a file based on description."""
        try:
            if not isinstance(project_path, Path):
                project_path = Path(project_path)
                
            # Ensure project_path exists
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Determine file path
            if os.path.isabs(filename):
                file_path = Path(filename)
            else:
                file_path = project_path / filename
                
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Research and generate code
            if self.display is not None:
                self.display.add_message("assistant", f"Generating code for: {file_path}")
            code = await self._research_and_generate_code(code_description, file_path)
            
            # Skip if no code was generated
            if not code:
                return {
                    "status": "error", 
                    "message": f"No code was generated for {file_path}",
                    "file_path": str(file_path),
                    "code": ""
                }
                
            # Determine operation type
            operation = "update" if file_path.exists() else "create"
                
            # Write code to file
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(code)
                    
                # Log file operation
                try:
                    log_file_operation(
                        file_path=file_path,
                        operation_type=operation,
                        content=code,
                        code_description=code_description,
                        language=get_language_from_extension(file_path.suffix),
                    )
                except Exception as file_error:
                    if self.display is not None:
                        self.display.add_message("user", f"Failed to log code: {str(file_error)}")
                        
                # Generate a nice HTML output with highlighted code
                try:
                    code_string = html_format_code(code, str(file_path).split('.')[-1])
                    if self.display is not None:
                        self.display.add_message("tool", {"html": code_string})
                except Exception as format_error:
                    if self.display is not None:
                        self.display.add_message("user", f"Note: Code highlighting failed: {str(format_error)}")
                        # Still display the code in a simple format
                        self.display.add_message("tool", {"text": f"```\n{code}\n```"})
                    
                # Return the result with file path and code
                return {
                    "status": "success", 
                    "operation": operation,
                    "message": f"Successfully wrote code to {file_path}",
                    "file_path": str(file_path),
                    "code": code
                }
                    
            except PermissionError:
                return {
                    "status": "error", 
                    "message": f"Permission denied when writing to {file_path}",
                    "file_path": str(file_path),
                    "code": code
                }
                
            except OSError as os_error:
                return {
                    "status": "error", 
                    "message": f"OS error when writing to {file_path}: {str(os_error)}",
                    "file_path": str(file_path),
                    "code": code
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "message": f"Error generating code: {str(e)}",
                "file_path": str(filename),
                "code": ""
            }

    async def _research_and_generate_code(self, code_description: str, file_path: Path) -> str:
        """Research and generate code based on description."""
        try:
            if self.display is not None:
                self.display.add_message("assistant", f"Researching code for: {file_path}")
                
            # Generate code using the LLM
            code = await self._call_llm_to_generate_code(
                code_description,
                await self._call_llm_to_research_code(code_description, file_path),
                file_path
            )
            
            # Extract only the code block if the LLM returned extra text
            if code:
                code, _ = self.extract_code_block(code, file_path)
                
            return code
        except Exception as e:
            print(f"Error in code research: {str(e)}")
            return ""

    async def write_multiple_files(self, project_path: Path, files: list) -> dict:
        """Write multiple files concurrently.
        Expects files as a list of dicts with 'code_description' and 'file_path' keys.
        Returns a structured result with information about all created files.
        """
        try:
            # Convert project_path to Path if it's a string
            if not isinstance(project_path, Path):
                project_path = Path(project_path)
                
            # Ensure project directory exists
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Set up tasks for concurrent execution
            tasks = []
            for file_info in files:
                desc = file_info.get("code_description", "")
                file_relative = file_info.get("file_path")
                
                # Ensure file_relative is a Path object
                file_path = Path(file_relative)
                
                # If file_path is not absolute, combine it with project_path
                if not file_path.is_absolute():
                    file_path = project_path / file_path
                    
                tasks.append(self.write_code_to_file(desc, project_path, file_path))
                
            # Execute all file creation tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_files = []
            errors = []
            output_string = ""
            
            for result in results:
                if isinstance(result, Exception):
                    error_message = f"Error writing file: {str(result)}"
                    errors.append({"message": error_message})
                    output_string += f"{error_message}\n"
                elif result.get("status") == "success":
                    successful_files.append(result)
                    file_path = result.get("file_path", "Unknown")
                    code = result.get("code", "")
                    output_string += f"Filename: {file_path}\n"
                    output_string += f"Code:\n{code}\n\n"
                    
                    # Try to display code with syntax highlighting
                    if self.display is not None:
                        try:
                            extension = str(file_path).split('.')[-1]
                            code_string = html_format_code(code, extension)
                            self.display.add_message("tool", {
                                "html": f"<h3>{file_path}</h3>{code_string}"
                            })
                        except Exception as format_error:
                            self.display.add_message("user", f"Note: Code highlighting failed for {file_path}: {str(format_error)}")
                            # Still display the code in a simple format
                            self.display.add_message("tool", {"text": f"# {file_path}\n```\n{code}\n```"})
                else:
                    error_message = f"Filename: {result.get('file_path', 'Unknown')}\nError: {result.get('message', 'Unknown error')}"
                    errors.append(result)
                    output_string += f"{error_message}\n\n"
            
            # Create the response in the format expected by the caller
            return {
                "command": "write_code_multiple_files",
                "status": "completed" if not errors else "partial_success",
                "project_path": str(project_path),
                "files_results": output_string,
                "files": successful_files,
                "errors": errors,
                "message": f"Successfully wrote {len(successful_files)} files" + 
                          (f", with {len(errors)} errors" if errors else "")
            }
            
        except Exception as e:
            error_message = f"Error in write_multiple_files: {str(e)}"
            if self.display is not None:
                self.display.add_message("user", error_message)
                
            return {
                "command": "write_code_multiple_files",
                "status": "error",
                "message": error_message,
                "project_path": str(project_path),
                "files_results": error_message,
                "files": [],
                "errors": [{"message": str(e)}]
            }
