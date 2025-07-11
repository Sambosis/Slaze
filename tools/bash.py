from typing import ClassVar, Literal, Union
import subprocess
import re
from dotenv import load_dotenv
from regex import T
from pathlib import Path
from config import get_constant # check_docker_available removed
from .base import BaseTool, ToolError, ToolResult
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole
from utils.command_converter import convert_command_for_system
import logging
import os
from rich import print as rr

load_dotenv()
# ic.configureOutput(includeContext=True, outputFunction=write_to_file) # Removed icecream configuration

logger = logging.getLogger(__name__)

class BashTool(BaseTool):
    def __init__(self, display: Union[WebUI, AgentDisplayConsole] = None):
        self.display = display
        super().__init__(input_schema=None, display=display)

    description = """
        A tool that allows the agent to run bash commands directly on the host system.
        All commands are executed relative to the current project directory if one
        has been set via the configuration. The tool parameters follow the OpenAI
        function calling format.
        """

    name: ClassVar[Literal["bash"]] = "bash"
    api_type: ClassVar[Literal["bash_20250124"]] = "bash_20250124"

    async def __call__(self, command: str | None = None, **kwargs):
        if command is not None:
            if self.display is not None:
                self.display.add_message("user", f"Executing command: {command}")

            # Convert command using LLM for current system
            modified_command = await self._convert_command_for_system(command)
            return await self._run_command(modified_command)
        raise ToolError("no command provided.")

    async def _convert_command_for_system(self, command: str) -> str:
        """
        Convert command using LLM to be appropriate for the current system.
        Falls back to legacy regex-based modification if LLM conversion fails.
        """
        try:
            # Try LLM-based conversion first
            converted = await convert_command_for_system(command)
            if converted == command:
                # Ensure standard patterns are applied
                converted = self._legacy_modify_command(command)
            return converted
        except Exception as e:
            logger.warning(f"LLM command conversion failed, using fallback: {e}")
            # Fallback to legacy regex-based modification
            return self._legacy_modify_command(command)
    
    def _legacy_modify_command(self, command: str) -> str:
        """Legacy fallback method for command modification using regex patterns."""
        import platform
        os_name = platform.system()
        
        if os_name == "Windows":
            # On Windows, keep Windows commands as-is
            if command.startswith("dir"):
                return command  # dir is correct for Windows
            
            # Convert Linux commands to Windows equivalents
            if command.startswith("ls"):
                ls_pattern = r"^ls\s*(-\w+)?\s*(.*)"
                ls_match = re.match(ls_pattern, command)
                if ls_match:
                    path = ls_match.group(2) if ls_match.group(2) else ""
                    return f"dir {path}".strip()
            
            # Convert find to Windows equivalent
            find_pattern = r"^find\s+(\S+)\s+-type\s+f"
            find_match = re.match(find_pattern, command)
            if find_match:
                path = find_match.group(1)
                # Convert Unix path to Windows style
                win_path = path.replace("/", "\\")
                return f'dir /s /b {win_path}\\*'
        
        else:
            # On Linux/Unix systems
            # Handle find command
            find_pattern = r"^find\s+(\S+)\s+-type\s+f"
            find_match = re.match(find_pattern, command)
            if find_match:
                path = find_match.group(1)
                # Extract the part after the path and -type f
                rest_of_command = command[find_match.end() :]
                # Add the exclusion for hidden files/paths
                return f'find {path} -type f -not -path "*/\\.*"{rest_of_command}'

            # Handle ls -la command
            ls_pattern = r"^ls\s+-la\s+(\S+)"
            ls_match = re.match(ls_pattern, command)
            if ls_match:
                path = ls_match.group(1)
                # Use ls with grep to filter out hidden entries - improved pattern
                return f'ls -la {path} | grep -v "^\\."'
            
            # Convert Windows dir to Linux ls
            if command.startswith("dir"):
                dir_pattern = r"^dir\s*(.*)"
                dir_match = re.match(dir_pattern, command)
                if dir_match:
                    path = dir_match.group(1).strip()
                    return f"ls -la {path}".strip() if path else "ls -la"

        # Return the original command if it doesn't match any patterns
        return command

    async def _run_command(self, command: str):
        """Execute a command in the Docker container."""
        output = ""
        error = ""
        success = False
        cwd = None
        try:
            # Execute the command relative to REPO_DIR if set
            repo_dir = get_constant("REPO_DIR")
            if repo_dir and Path(repo_dir).exists():
                cwd = str(repo_dir)
            else:
                cwd = None
            terminal_display = f"terminal {cwd}>  {command}"
            if self.display is not None:
                self.display.add_message("assistant", terminal_display)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                cwd=cwd,
                )
            output = result.stdout
            error = result.stderr
            success = result.returncode == 0
            terminal_display = f"{output}\n{error}"
            if self.display is not None:
                self.display.add_message("assistant", terminal_display)
                # Record the executed command as a user message last for testing
                self.display.add_message("user", f"Executing command: {command}")
            if len(output) > 200000:
                output = f"{output[:100000]} ... [TRUNCATED] ... {output[-100000:]}"
            if len(error) > 200000:
                error = f"{error[:100000]} ... [TRUNCATED] ... {error[-100000:]}"

            formatted_output = (
                f"command: {command}\n"
                f"working_directory: {cwd}\n"
                f"success: {str(success).lower()}\n"
                f"output: {output}\n"
                f"error: {error}"
            )
            rr(formatted_output)
            return ToolResult(
                output=formatted_output,
                error=error,
                tool_name=self.name,
                command=command,
            )

        except Exception as e:
            error = str(e)
            rr(error)
            if self.display is not None:
                try:
                    self.display.add_message("assistant", f"Error: {error}")
                except Exception as display_error:
                    logger.error(f"Error displaying message: {display_error}", exc_info=True)
            return ToolResult(error=error, tool_name=self.name, command=command)

    def to_params(self) -> dict:
        logger.debug(f"BashTool.to_params called with api_type: {self.api_type}")
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
                            "description": "The bash command to be executed.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
        logger.debug(f"BashTool params: {params}")
        return params
