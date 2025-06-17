from typing import ClassVar, Literal
import subprocess
import re
from dotenv import load_dotenv
from config import get_constant, write_to_file # check_docker_available removed
from .base import BaseTool, ToolError, ToolResult
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
import logging
import os
from rich import print as rr

load_dotenv()
# ic.configureOutput(includeContext=True, outputFunction=write_to_file) # Removed icecream configuration

logger = logging.getLogger(__name__)

class BashTool(BaseTool):
    def __init__(self, display: AgentDisplayWebWithPrompt = None):
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
            # Modify commands to exclude hidden files/paths
            modified_command = self._modify_command_if_needed(command)
            return await self._run_command(modified_command)
        raise ToolError("no command provided.")

    def _modify_command_if_needed(self, command: str) -> str:
        """Modify find and ls commands to exclude hidden files/paths."""
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
            # Use ls with grep to filter out hidden entries
            return f'ls -la {path} | grep -v "^d*\\."'

        # Return the original command if it doesn't match any patterns
        return command

    async def _run_command(self, command: str):
        """Execute a command in the Docker container."""
        output = ""
        error = ""
        success = False
        timeout = 42
        try:
            if self.display is not None:
                self.display.add_message("user", f"Executing command: {command}")

            try:
                # Execute the command locally relative to PROJECT_DIR if set
                project_dir = get_constant("PROJECT_DIR")
                cwd = str(project_dir) if project_dir else None
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
            except Exception as e:
                error = f"Command execution error: {str(e)}"
                success = False
            if len(output) > 200000:
                output = output[:100000] + " ... [TRUNCATED] ... " + output[-100000:]
            if len(error) > 200000:
                error = error[:100000] + " ... [TRUNCATED] ... " + error[-100000:]

            # Determine working directory **before** we attempt the subprocess call
            project_dir = get_constant("PROJECT_DIR")
            cwd: str | None = str(project_dir) if project_dir else None

            try:
                result = subprocess.run(
                    ["bash", "-c", command],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                success = result.returncode == 0
                output = result.stdout
                error = result.stderr
            except subprocess.TimeoutExpired as e:
                success = False
                output = e.stdout or ""
                error = e.stderr or ""
            except Exception as e:
                success = False
                output = ""
                error = str(e)

            if len(error) > 200000:
                error = error[:100000] + " ... [TRUNCATED] ... " + error[-100000:]

            working_dir = cwd or os.getcwd()
            formatted_output = (
                f"command: {command}\n"
                f"working_directory: {working_dir}\n"
                f"success: {str(success).lower()}\n"
                f"output: {output}\n"
                f"error: {error}"
            )
            rr(formatted_output)
            # Create a new ToolResult instead of modifying an existing one
            return ToolResult(
                output=formatted_output, tool_name=self.name, command=command
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
