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
from system_prompt.code_prompts import code_prompt_research, code_prompt_generate
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
    GET_ALL_CODE = "get_all_current_code"
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

                await asyncio.sleep(0.2)

            # Convert path string to Path object
            project_path = Path(get_constant("PROJECT_DIR"))
            # Execute the appropriate command
            if command == CodeCommand.WRITE_CODE_TO_FILE:
                result_data = await self.write_code_to_file(code_description, project_path, python_filename)
                ic(f"result_data: {result_data}")
            elif command == CodeCommand.WRITE_AND_EXEC:
                result_data = await self.write_and_exec(code_description, project_path)
                ic(f"result_data: {result_data}")
            elif command == CodeCommand.WRITE_CODE_MULTIPLE_FILES:
                files = kwargs.get("files", [])
                if not files:
                    return ToolResult(error="No files provided for multiple file creation.")
                result_data = await self.write_multiple_files(project_path, files)
            elif command == CodeCommand.GET_ALL_CODE:
                result_data = {
                "command": "GET_ALL_CODE",
                "status": "success",
                "files_results ": get_all_current_code(),

            }
            elif command == CodeCommand.GET_REVISED_VERSION:
                target_file = kwargs.get("target_file")
                revision_description = kwargs.get("revision_description")
                if not target_file or not revision_description:
                    return ToolResult(error="Both target_file and revision_description are required for revision")
                result_data = await self.get_revised_version(project_path, target_file, revision_description)
            else:

                ic(f"Unknown command: {command}")
                return ToolResult(error=f"Unknown command: {command}")


            # Convert result_data to formatted string   
            formatted_output = self.format_output(result_data)
            ic(f"formatted_output: {formatted_output}")

            return ToolResult(output=formatted_output)

        except Exception as e:

            error_msg = f"Failed to execute {command}: {str(e)}"
            
            return ToolResult(error=error_msg)

    def extract_code_block(self, text: str) -> tuple[str, str]:
        """
        Extracts the first code block in the text with its language.
        If no code block is found, returns the full text and language "code".
        If text is empty, returns "No Code Found" and language "Unknown".
        """
        if not text.strip():
            return "No Code Found", "Unknown"
        
        start_marker = text.find("```")
        if start_marker == -1:
            return text, "code"
        
        # Determine language (text immediately after opening delimiter)
        language_line_end = text.find("\n", start_marker)
        if language_line_end == -1:
            language_line_end = start_marker + 3
        language = text[start_marker+3:language_line_end].strip()
        if not language:
            language = "code"
        
        end_marker = text.find("```", language_line_end)
        if end_marker == -1:
            code_block = text[language_line_end:].strip()
        else:
            code_block = text[language_line_end:end_marker].strip()
        
        return code_block if code_block else "No Code Found", language

    async def _call_llm_to_generate_code(self, code_description: str, research_string: str, file_path) -> str:
        """Call LLM to generate code based on the code description"""
        
        self.display.add_message("assistant", f"Generating code for: {file_path}")

        code_string="no code created"

        current_code_base = get_all_current_code()
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        # client = AsyncOpenAI(
        #     base_url="https://openrouter.ai/api/v1",
        #     api_key=OPENROUTER_API_KEY,
        #     )
        # model = "microsoft/wizardlm-2-8x22b:nitro"
        client = AsyncOpenAI()
        model = "o3-mini"
        ic(model)
        # Prepare messages
        messages = code_prompt_generate(current_code_base, code_description, research_string)
        ic(f"Here are the messages being sent to Generate the Code\n +++++ \n +++++ \n{messages}")
        try:
            completion =  await client.chat.completions.create(
                model=model,
                messages=messages)

        except Exception as e:
            ic(completion)
            ic(f"error: {e}")
            return code_description
        code_string = completion.choices[0].message.content

        # Extract code using the new function
        try:
            code_string, detected_language = self.extract_code_block(code_string)
            # ic(f"Code String: {code_string}\nLanguage: {detected_language}")
        except Exception as parse_error:
            error_msg = f"Failed to parse code block: {str(parse_error)}"
            if self.display is not None:
                try:
                    self.display.add_message("user", error_msg)
                except Exception as display_error:
                    return ToolResult(error=f"{error_msg}\nFailed to display error: {str(display_error)}")

            raise ToolResult(code_string)
        
        # Log the extraction
        try:
            CODE_FILE = Path(get_constant("CODE_FILE"))
            # ic(CODE_FILE)
            with open(CODE_FILE, "a", encoding="utf-8") as f:
                f.write(f"File Path: {str(file_path)}\n")
                f.write(f"Language detected: {detected_language}\n")
                f.write(f"{code_string}\n")
        except Exception as file_error:
            # Log failure but don't stop execution
            if self.display is not None:
                try:
                    self.display.add_message("user  ", f"Failed to log code: {str(file_error)}")
                except Exception:
                    pass

        if detected_language == "html":
            # Send the HTML content directly to the display for rendering
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
        self.display.add_message("tool", {"code": code_display, "css": css_styles})
        # send_email_attachment_of_code(str(file_path), code_string)
        return code_string

    async def _call_llm_to_research_code(self, code_description: str, file_path) -> str:
        """Call LLM to generate code based on the code description"""
        self.display.add_message("assistant", f"Researching code for: {file_path}")
        code_string = "no code created"
        current_code_base = get_all_current_code()
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            )
        model = "google/gemini-2.0-flash-001:nitro"
        # client = AsyncOpenAI()
        # model = "o3-mini"

        # Prepare messages
        messages = code_prompt_research(current_code_base, code_description)
        ic(f"Here are the messages being sent to Research the Code\n +++++ \n +++++ \n{messages}")
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages)

        except Exception as e:
            ic(completion)
            ic(f"error: {e}")
            return code_description   
        # ic(completion)

        # Handle both OpenRouter and standard OpenAI response formats
        try:
            if hasattr(completion.choices[0].message, 'content'):
                research_string = completion.choices[0].message.content
            else:
                research_string = completion.choices[0].message['content']
        except (AttributeError, KeyError, IndexError) as e:
            ic(f"Error extracting content: {e}")
            return code_description

        # ic(research_string)
        research_string = ftfy.fix_text(research_string)
        # ic(research_string)
        return research_string


    def format_output(self, data: dict) -> str:
        """Format the output data as a readable string"""
        output_lines = []
        
        # Add command type
        output_lines.append(f"Command: {data['command']}")
        
        # Add status
        output_lines.append(f"Status: {data['status']}")
        
        # Add project path
        output_lines.append(f"Project Path: {data['project_path']}")

        # Add filename if present (for single file operations)
        if 'filename' in data:
            output_lines.append(f"Filename: {data['filename']}")
            output_lines.append(f"Code:\n{data.get('code_string', '')}")

        # Add files_results (for multiple files, this is already formatted)
        if 'files_results' in data:
            output_lines.append(data['files_results'])
        
        # Add packages if present
        if 'packages_installed' in data:
            output_lines.append("Packages Installed:")
            for package in data['packages_installed']:
                output_lines.append(f"  - {package}")
        
        # Add run output if present
        if 'run_output' in data and data['run_output']:
            output_lines.append("\nApplication Output:")
            output_lines.append(data['run_output'])
        
        if 'errors' in data and data['errors']:
            output_lines.append("\nErrors:")
            output_lines.append(data['errors'])
        
        # Join all lines with newlines
        return "\n".join(output_lines)

    async def write_code_to_file(self, code_description: str,  project_path: Path, filename) -> dict:
        """write code to a permanent file"""
        file_path = project_path / filename
        code_research_string = await self._call_llm_to_research_code(code_description, file_path)
        # ic(code_research_string)

        # Write research result (optional: can also be wrapped if needed)
        await asyncio.to_thread(lambda: open("codeResearch.txt", "a", encoding='utf-8').write(code_research_string))
        # ic(code_research_string)
        code_string = await self._call_llm_to_generate_code(code_description, code_research_string, file_path)
        ic(code_string)
        
        # Create the directory if it does not exist
        try:
            await asyncio.to_thread(os.makedirs, file_path.parent, exist_ok=True)
        except Exception as dir_error:
            return {
                "command": "write_code_to_file",
                "status": "error",
                "project_path": str(project_path),
                "filename": filename,
                "error": f"Failed to create directory: {str(dir_error)}"
            }
        
        # Write the file asynchronously
        try:
            await asyncio.to_thread(file_path.write_text, code_string, encoding='utf-8')
        except Exception as write_error:
            return {
                "command": "write_code_to_file",
                "status": "error",
                "project_path": str(project_path),
                "filename": filename,
                "error": f"Failed to write file: {str(write_error)}"
            }
        
        # Log the file creation
        try:
            await asyncio.to_thread(log_file_operation, file_path, "create")
        except Exception as log_error:
            if self.display is not None:
                try:
                    self.display.add_message("user", f"Failed to log file operation: {str(log_error)}")
                except Exception:
                    pass
        
        return {
            "command": "write_code_to_file",
            "status": "success",
            "project_path": str(project_path),
            "filename": filename,
            "code_string": code_string
        }
           
    async def write_and_exec(self, code_description: str,  project_path: Path) -> dict:
        """Write code to a temp file and execute it"""
        os.chdir(project_path)    

        code_string = await self._call_llm_to_generate_code(code_description)
        

        
        # Create temp file with .py extension
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code_string)
            temp_path = temp_file.name
        try:
            result = subprocess.run(
                ["python", temp_path],
                capture_output=True,
                text=True,
                check=True
            )
            return {
                "command": "write_and_exec",
                "status": "success",
                "project_path": str(project_path),
                "run_output": result.stdout,
                "errors": result.stderr
            }
        except subprocess.CalledProcessError as e:
            return {
                "command": "run_app",
                "status": "error",
                "project_path": str(project_path),
                "errors": f"Failed to run app: {str(e)}\nOutput: {e.stdout}\nError: {e.stderr}"
            }

    async def write_multiple_files(self, project_path: Path, files: list) -> dict:
        """Write multiple files concurrently.
        Expects files as a list of dicts with 'code_description' and 'file_path' keys.
        Returns a single string containing all filenames and file contents.
        """
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
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Concatenate successful file creation results into a single string
        output_string = ""
        for result in results:
            if isinstance(result, Exception):
                output_string += f"Error writing file: {str(result)}\n"
            elif result.get("status") == "success":
                output_string += f"Filename: {result.get('filename')}\n"
                output_string += f"Code:\n{result.get('code_string')}\n\n"
            else:
                output_string += f"Filename: {result.get('filename', 'Unknown')}\n"
                output_string += f"Error: {result.get('error', 'Unknown error')}\n\n"

        return {
            "command": "write_code_multiple_files",
            "status": "completed",
            "project_path": str(project_path),
            "files_results": output_string  # Return the concatenated string
        }

    async def get_revised_version(self, project_path: Path, target_file: str, revision_description: str) -> dict:
        """Get a revised version of an existing file with specified changes."""
        file_path = project_path / target_file
        display.add_message("assistant", f"Revising file USING NEW TOOL {file_path}")
        display.add_message("user", f"Revising file USING NEW TOOL {file_path}")
        try:
            # Read existing file content
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_code = f.read()
                
            # Get revised version
            research_string = await self._call_llm_to_research_code(
                f"Revision needed: {revision_description}\nExisting code:\n{existing_code}", 
                file_path
            )
            
            revised_code = await self._call_llm_to_generate_code(
                f"Revision needed: {revision_description}\nExisting code:\n{existing_code}",
                research_string,
                file_path
            )
            # write the opdated version back to the file
            file_path.write_text(revised_code, encoding='utf-8')
            # Log the file creation
            try:
                await asyncio.to_thread(log_file_operation, file_path, "update")
            except Exception as log_error:
                if self.display is not None:
                    try:
                        self.display.add_message("user", f"Failed to log file operation: {str(log_error)}")
                    except Exception:
                        pass
                    
            return {
                "command": "get_revised_version",
                "status": "success",
                "project_path": str(project_path),
                "filename": target_file,
                "original_code": existing_code,
                "revised_code": revised_code
            }
            
        except FileNotFoundError:
            return {
                "command": "get_revised_version",
                "status": "error",
                "project_path": str(project_path),
                "filename": target_file,
                "error": f"File not found: {target_file}"
            }
        except Exception as e:
            return {
                "command": "get_revised_version",
                "status": "error",
                "project_path": str(project_path),
                "filename": target_file,
                "error": f"Error during revision: {str(e)}"
            }

