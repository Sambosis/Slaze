import asyncio
from pathlib import Path
from typing import ClassVar, Literal
from anthropic.types.beta import BetaToolBash20241022Param
import re
import os
import subprocess
import sys # ADDED: Import sys for OS detection
import io
import traceback
from datetime import datetime
from anthropic import Anthropic
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()
from .base import BaseAnthropicTool, ToolError, ToolResult
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt  # Add this line
from load_constants import  write_to_file, ICECREAM_OUTPUT_FILE
from icecream import ic
from config import * # BASH_PROMPT_FILE, get_constant
from utils.context_helpers import *

ic.configureOutput(includeContext=True, outputFunction=write_to_file)



def read_prompt_from_file(file_path: str, bash_command: str) -> str:
    """Read the prompt template from a file and format it with the given bash command."""
    project_dir = Path(get_constant("PROJECT_DIR"))
    try:
        with open(file_path, "r", encoding='utf-8', errors='replace') as file:
            prompt_string = file.read()
        prompt_string += f"Your project directory is {project_dir}. You need to make sure that all files you create and work you do is done in that directory. \n"
        prompt_string += f"Your bash command is: {bash_command}\n"
        return prompt_string
    except Exception as e:
        return f"Error reading prompt file: {str(e)}"

def extract_code_block(text: str) -> tuple[str, str]:
    """
    Extract code from a markdown code block, detecting the language if specified.
    Returns tuple of (code, language).
    If no code block is found, returns the original text and empty string for language.
    """
    # Find all code blocks in the text
    code_blocks = []
    lines = text.split('\n')
    in_block = False
    current_block = []
    current_language = ''

    for line in lines:
        if line.startswith('```'):
            if in_block:
                # End of block
                in_block = False
                code_blocks.append((current_language, '\n'.join(current_block)))
                current_block = []
                current_language = ''
            else:
                # Start of block
                in_block = True
                # Extract language if specified
                current_language = line.strip('`').strip()
                if current_language == '':
                    current_language = 'unknown'
        elif in_block:
            current_block.append(line)

    # If we found code blocks, return the most relevant one
    # (currently taking the first non-empty block)
    # Modified to return ALL code blocks found
    extracted_scripts = []
    for language, code in code_blocks:
        if code.strip():
            extracted_scripts.append({'language': language.lower(), 'code': code.strip()})
    if extracted_scripts:
        return extracted_scripts
    else: # If no code blocks found or all empty, return original text
        return [{'language': '', 'code': text.strip()}]


async def generate_script_with_llm(prompt: str) -> str:
    """Send a prompt to the LLM and return its response."""

    try:
        ic(prompt)
        # 1. Create a client with custom endpoint
        # client = OpenAI()
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            )
        model = "mistralai/codestral-2501:nitro"
        # 2. Use it like normal OpenAI calls
        response = await client.chat.completions.create(
            model=model,  # Use model name your endpoint expects
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ic(response)
        response_message = response.choices[0].message.content
        ic(response_message)
        return response_message
    except Exception as e:
        raise ToolError(f"Error during LLM API call: {e}")


def parse_llm_response(response: str):
    """Parse the LLM response to extract the script."""
    script_blocks = extract_code_block(response)
    parsed_scripts = []
    for block in script_blocks:
        script_type = block['language']
        script_code = block['code']
        if script_type == 'python':
            script_type = 'Python Script'
        elif script_type == 'powershell':
            script_type = 'PowerShell Script'
        elif script_type == 'unknown': # If language is unknown, default to powershell for windows, bash for linux (or handle as needed)
            if sys.platform == 'win32':
                script_type = 'PowerShell Script' # Default to powershell on windows if language is not specified.
            elif sys.platform.startswith('linux'):
                script_type = 'Bash Script' # You can add 'Bash Script' type and execution if needed for linux and unknown type.
            else:
                script_type = 'Unknown Script' # Or handle as error.

        if script_type: # Only add if script_type is determined.
            parsed_scripts.append({'script_type': script_type, 'script_code': script_code})
    return parsed_scripts


def generate_temp_filename(script_type: str) -> Path:
    """Generate a unique temporary filename in the project's temp directory."""
    project_dir = Path(get_constant("PROJECT_DIR"))
    temp_dir = project_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = ".py" if script_type == "Python Script" else ".ps1" if script_type == "PowerShell Script" else ".sh" # Added .sh for bash if needed
    return temp_dir / f"temp_script_{timestamp}{extension}"

def execute_script(script_type: str, script_code: str, display: AgentDisplayWebWithPrompt = None):
    """Execute the extracted script synchronously and capture output and errors."""
    output = ""
    error = ""
    success = False

    project_dir = Path(get_constant("PROJECT_DIR"))
    if os.name == 'nt':  # Windows
        python_path = project_dir / ".venv" / "Scripts" / "python"
    else:  # Unix-like
        python_path = project_dir / ".venv" / "bin" / "python"

    if script_type == "Python Script":
        if display:
            display.add_message("user", "Executing Python script synchronously...")
        # Generate unique temp file path
        script_file = generate_temp_filename(script_type)
        try:
            with open(script_file, "w", encoding="utf-8", errors='replace') as f:
                f.write(script_code)
            # Run the script synchronously and capture output and error
            result = subprocess.run(
                [str(python_path), str(script_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_dir),
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False
            )
            output = result.stdout
            error = result.stderr
            success = (result.returncode == 0)
            if display:
                display.add_message("user", f"Python script completed with return code {result.returncode}")
        except Exception as e:
            error = f"Error: {str(e)}\n{traceback.format_exc()}"
            success = False
            if display:
                display.add_message("user", f"Python Execution Error:\n{error}")

    elif script_type == "PowerShell Script":
        if display:
            display.add_message("user", "Executing PowerShell script synchronously...")
        script_file = generate_temp_filename(script_type)
        with open(script_file, "w", encoding="utf-8", errors='replace') as f:
            f.write(script_code)
        try:
            result = subprocess.run(
                ["powershell.exe", "-File", str(script_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_dir),
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False
            )
            output = result.stdout
            error = result.stderr
            success = (result.returncode == 0)
            if display:
                display.add_message("user", f"PowerShell script completed with return code {result.returncode}")
        except Exception as e:
            error = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            success = False
            if display:
                display.add_message("user", f"PowerShell Execution Error:\n{error}")

    elif script_type == "Bash Script":
        if display:
            display.add_message("user", "Executing Bash script synchronously...")
        script_file = generate_temp_filename(script_type)
        with open(script_file, "w", encoding="utf-8", errors='replace') as f:
            f.write(script_code)
        try:
            result = subprocess.run(
                ["bash", str(script_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(project_dir),
                text=True,
                encoding='utf-8',
                errors='replace',
                check=False
            )
            output = result.stdout
            error = result.stderr
            success = (result.returncode == 0)
            if display:
                display.add_message("user", f"Bash script completed with return code {result.returncode}")
        except Exception as e:
            error = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            success = False
            if display:
                display.add_message("user", f"Bash Execution Error:\n{error}")
    else:
        error_msg = f"Unsupported script type: {script_type}"
        if display:
            display.add_message("user", f"Error: {error_msg}")
        raise ValueError(error_msg)

    return {
        "success": success,
        "output": output,
        "error": error
    }


class BashTool(BaseAnthropicTool):
    def __init__(self, display: AgentDisplayWebWithPrompt = None):
        self.display = display
        super().__init__()


    description = """
        A tool that allows the agent to run bash commands. On Windows it uses PowerShell, on Linux it runs bash commands directly.
        The tool parameters are defined by Anthropic and are not editable.
        """

    name: ClassVar[Literal["bash"]] = "bash"
    api_type: ClassVar[Literal["bash_20241022"]] = "bash_20241022"

    async def __call__(self, command: str | None = None, **kwargs):
        if command is not None:
            return await self._run_command(command)
        raise ToolError("no command provided.")

    async def _run_command(self, command: str):
        """Execute a command in the shell.
        On Linux, executes the command directly.
        On Windows, generates and executes Python and PowerShell scripts, with fallback.
        """
        BASH_PROMPT_FILE= get_constant("BASH_PROMPT_FILE")
        output = ""
        error = ""
        success = False
        powershell_result = None # Initialize for tracking powershell result
        python_result = None # Initialize for tracking python result

        try:
            if self.display:
                self.display.add_message("user", f"Processing command: {command}")
                await asyncio.sleep(0.2)

            if sys.platform.startswith('linux'):
                # Linux: Execute command directly
                if self.display:
                    self.display.add_message("user", f"Executing bash command directly (Linux)...")
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    check=False
                )
                output = result.stdout
                error = result.stderr
                success = (result.returncode == 0)

            elif sys.platform == 'win32':
                # Windows: Generate and execute scripts via LLM with fallback
                if self.display:
                    self.display.add_message("user", "Generating and executing scripts via LLM (Windows)...")

                prompt = read_prompt_from_file(BASH_PROMPT_FILE, command)
                response = await generate_script_with_llm(prompt)
                await asyncio.sleep(0.2)
                parsed_scripts = parse_llm_response(response)  # List of script blocks in LLM-provided order

                for script_info in parsed_scripts:
                    script_type = script_info['script_type']
                    script_code = script_info['script_code']

                    if script_type not in ["Python Script", "PowerShell Script", "Bash Script"]:
                        error += f"Unsupported script type from LLM: {script_type}\n"
                        if self.display:
                            self.display.add_message("user", f"Error: Unsupported script type: {script_type}")
                        continue

                    result = execute_script(script_type, script_code, self.display)
                    output += f"{script_type} output:\n{result['output']}\n{script_type} error:\n{result['error']}\n"
                    if result['success']:
                        success = True
                        break  # Stop after successful execution
                    else:
                        error += f"{script_type} script failed:\n{result['error']}\n"

 
            else: # OS not detected as Linux or Windows
                error = "Unsupported operating system. Bash tool only supports Linux and Windows."
                if self.display:
                    self.display.add_message("user", f"Error: {error}")


            if success:
                return ToolResult(output=f"command: {command}\nsuccess: true\noutput: {output}\nerror: {error}")
            else:
                return ToolResult(output=f"command: {command}\nsuccess: false\noutput: {output}\nerror: {error}")


        except Exception as e:
            error = str(e)
            if self.display:
                self.display.add_message("user", f"Error: {error}")
            return ToolError(error)

    def to_params(self) -> BetaToolBash20241022Param:
        return {
            "type": self.api_type,
            "name": self.name,
        }


def save_successful_code(script_code: str) -> str:
    """Save successfully executed Python code to a file."""
    # Create directory if it doesn't exist
    llm_gen_code_dir = Path(get_constant("LLM_GEN_CODE_DIR"))
    save_dir = llm_gen_code_dir
    save_dir.mkdir(exist_ok=True)
    ic(script_code)
    # Extract first line of code for filename (cleaned)
    first_line = script_code.split("\n")[0].strip()
    # Clean the first line to create a valid filename
    clean_name = re.sub(r"[^a-zA-Z0-9]", "_", first_line)[:30]

    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{clean_name}_{timestamp}.py"

    # Save the code with UTF-8 encoding
    file_path = save_dir / filename
    with open(file_path, "w", encoding='utf-8', errors='replace') as f:
        f.write(script_code)

    return str(file_path)
 
