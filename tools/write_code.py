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
from loguru import logger as ll

# Configure logging to a file
ll.add(
    "my_log_file.log",
    rotation="500 KB",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}.{function}:{line} - {message}")

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
                    self.display.add_message("assistant", f"Failed to log code: {str(file_error)}")
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
            from config import get_constant, get_project_dir, get_docker_project_dir, REPO_DIR
            from utils.file_logger import convert_to_docker_path
            import os
            
            # Ensure project_path is a Path object
            if not isinstance(project_path, Path):
                project_path = Path(project_path)
                
            # Ensure we're using the correct project directory structure
            repo_dir = get_constant('REPO_DIR')
            if repo_dir and isinstance(repo_dir, str):
                repo_dir = Path(repo_dir)
                
            # Extract the project name (prompt name) from the project path
            project_name = project_path.name
            
            # Check if we need to adjust the project path to be within repo directory
            if repo_dir and not str(project_path).startswith(str(repo_dir)):
                # Create correct project path within repo
                project_path = repo_dir / project_name
                
            # Ensure project_path exists
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Get the Docker project directory for logging
            docker_project_dir = f"/home/myuser/apps/{project_name}"
            
            # Normalize paths for consistent handling
            project_path_str = str(project_path).replace('\\', '/')
            
            # Determine file path
            if os.path.isabs(filename):
                file_path = Path(filename)
            else:
                file_path = project_path / filename
                
            # Create parent directories if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to string for consistent logging
            file_path_str = str(file_path).replace('\\', '/')

            code = await self._research_and_generate_code(code_description, file_path)
            
            # Skip if no code was generated
            if not code:
                return {
                    "status": "error",
                    "message": "No code was generated."
                }
                
            # Determine the operation (create or modify)
            operation = "modify" if file_path.exists() else "create"
            
            # Write the code to the file
            file_path.write_text(code, encoding="utf-8")
            ll.info(f"Code written to {file_path}")
            # Convert to Docker path for display
            docker_path = convert_to_docker_path(file_path)
            ll.info(f"Converted to Docker path: {docker_path}")
            # Log the file operation
            try:
                from utils.file_logger import log_file_operation
                log_file_operation(
                    file_path=file_path,
                    operation=operation,
                    content=code,
                    metadata={"code_description": code_description}
                )
            except Exception as log_error:
                ic(f"Warning: Failed to log code writing: {str(log_error)}")
            
            # If we have a display, add a message showing what happened
            if self.display is not None:
                self.display.add_message(
                    "user", f"Code for {docker_path} generated successfully:"
                )
                
            # Format the code for HTML display
            language = get_language_from_extension(file_path.suffix)
            formatted_code = html_format_code(code, language)
            
            # Return result dictionary
            result = {
                "status": "success",
                "operation": operation,
                "message": f"Successfully wrote code to {docker_path}",
                "file_path": str(docker_path),
                "code": code,
                "html": formatted_code
            }
            return result

        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            ic(f"Error in write_code_to_file: {str(e)}\n{stack_trace}")
            return {
                "status": "error",
                "message": f"Error in write_code_to_file: {str(e)}"
            }

    async def _research_and_generate_code(self, code_description: str, file_path: Path) -> str:
        """Research and generate code based on description."""
        try:
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
            ic(f"Error in code research: {str(e)}")
            return ""

    async def write_multiple_files(self, project_path: Path, files: list) -> dict:
        """
        Write multiple files with content at once.
        
        Args:
            project_path: Path to the project directory
            files: List of file information. Each entry should have:
                  - file_path or filename: Path to file (relative to project_path)
                  - content: The content to write (optional if code_description is provided)
                  - code_description: Description of the code to generate (optional if content is provided)
            
        Returns:
            Dictionary with operation results
        """
        try:
            from config import get_constant, REPO_DIR
            from utils.file_logger import convert_to_docker_path
            import os
            
            # Ensure project_path is a Path object
            if not isinstance(project_path, Path):
                project_path = Path(project_path)
                
            # Ensure we're using the correct project directory structure
            repo_dir = get_constant('REPO_DIR')
            if repo_dir and isinstance(repo_dir, str):
                repo_dir = Path(repo_dir)
                
            # Extract the project name (prompt name) from the project path
            project_name = project_path.name
            
            # Check if we need to adjust the project path to be within repo directory
            if repo_dir and not str(project_path).startswith(str(repo_dir)):
                # Create correct project path within repo
                project_path = repo_dir / project_name
            
            # Ensure project_path exists
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Get the Docker project directory for logging
            docker_project_dir = f"/home/myuser/apps/{project_name}"
                
            # Tasks to execute (may be completed results or coroutines to await)
            tasks = []
            
            # For each file specified
            for file_info in files:
                try:
                    # Check which format we're dealing with (new or old)
                    if "content" in file_info and "filename" in file_info:
                        # New format with direct content
                        filename = file_info["filename"]
                        content = file_info["content"]
                        
                        # Determine file path
                        if os.path.isabs(filename):
                            file_path = Path(filename)
                        else:
                            file_path = project_path / filename
                            
                        # Create parent directories if needed
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Determine operation (create or modify)
                        operation = "modify" if file_path.exists() else "create"
                        
                        # Write the file
                        file_path.write_text(content, encoding="utf-8")
                        
                        # Convert to Docker path for display
                        docker_path = convert_to_docker_path(file_path)
                        
                        # Log the file operation
                        try:
                            from utils.file_logger import log_file_operation
                            log_file_operation(
                                file_path=file_path,
                                operation=operation,
                                content=content,
                                metadata={"code_description": file_info.get("description", "")}
                            )
                        except Exception as log_error:
                            ic(f"Warning: Failed to log code writing: {str(log_error)}")
                            
                        # Add a completed result instead of a task
                        result = {
                            "status": "success", 
                            "operation": operation,
                            "message": f"Successfully wrote code to {docker_path}",
                            "file_path": str(docker_path), 
                            "code": content
                        }
                        tasks.append(result)
                    elif "file_path" in file_info and "code_description" in file_info:
                        # Old format that requires code generation
                        file_path = file_info["file_path"]
                        code_description = file_info["code_description"]
                        
                        # Create a coroutine to generate and write the file
                        # We'll await it later
                        task = self.write_code_to_file(
                            code_description, 
                            project_path, 
                            file_path
                        )
                        tasks.append(task)
                    else:
                        # Invalid file format
                        tasks.append({
                            "status": "error",
                            "message": f"Invalid file info: Must have either 'content'+'filename' or 'file_path'+'code_description'",
                            "file_info": str(file_info),
                        })
                except Exception as file_error:
                    ic(f"Error preparing file: {str(file_error)}")
                    tasks.append({
                        "status": "error",
                        "message": f"Error preparing file: {str(file_error)}",
                        "file_path": file_info.get("file_path", file_info.get("filename", "unknown")),
                    })    

            # Execute all file creation tasks concurrently (for code_description tasks only)
            results = []
            for task in tasks:
                if isinstance(task, dict):  # Already completed task
                    results.append(task)
                else:  # Coroutine to await
                    try:
                        result = await task
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "status": "error",
                            "message": f"Error writing file: {str(e)}",
                            })
            
            # Check how many succeeded and failed
            success_count = sum(1 for r in results if r.get("status") == "success")
            error_count = sum(1 for r in results if r.get("status") == "error")
            
            # Create the summary message
            if error_count == 0:
                message = f"Successfully wrote {success_count} files to {docker_project_dir}"
            else:
                message = f"Wrote {success_count} files, {error_count} failed"
                
            # Return the overall result
            return {
                "status": "success" if error_count == 0 else "partial",
                "message": message,
                "results": results,
                "project_path": str(docker_project_dir),
            }
            
        except Exception as e:
            import traceback
            stack_trace = traceback.format_exc()
            ic(f"Error in write_multiple_files: {str(e)}\n{stack_trace}")
            return {
                "status": "error",
                "message": f"Error in write_multiple_files: {str(e)}"
        }
