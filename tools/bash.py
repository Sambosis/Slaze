import asyncio
from pathlib import Path
from typing import ClassVar, Literal
from anthropic.types.beta import BetaToolBash20241022Param
import os
import subprocess
from dotenv import load_dotenv
from config import get_constant, check_docker_available
from .base import BaseAnthropicTool, ToolError, ToolResult
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from load_constants import write_to_file
from icecream import ic

load_dotenv()
ic.configureOutput(includeContext=True, outputFunction=write_to_file)

# Docker container name
DOCKER_CONTAINER_NAME = get_constant("DOCKER_CONTAINER_NAME") or "python-dev-container"


def run_docker_command(command: str, container_name=DOCKER_CONTAINER_NAME):
    """Execute a command in a Docker container and return the result."""
    try:
        # Properly escape the command for passing to bash -c
        escaped_command = command.replace('"', '\\"')
        docker_command = f'docker exec -e DISPLAY=host.docker.internal:0.0 {container_name} bash -c "{escaped_command}"'

        result = subprocess.run(
            docker_command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Docker command execution failed: {str(e)}")


class BashTool(BaseAnthropicTool):
    def __init__(self, display: AgentDisplayWebWithPrompt = None):
        self.display = display
        self._docker_available = check_docker_available()
        super().__init__()

    description = """
        A tool that allows the agent to run bash commands in a Docker container.
        The tool parameters are defined by Anthropic and are not editable.
        """

    name: ClassVar[Literal["bash"]] = "bash"
    api_type: ClassVar[Literal["bash_20250124"]] = "bash_20250124"

    async def __call__(self, command: str | None = None, **kwargs):
        if command is not None:
            return await self._run_command(command)
        raise ToolError("no command provided.")

    async def _run_command(self, command: str):
        """Execute a command in the Docker container."""
        output = ""
        error = ""
        success = False

        try:
            if self.display:
                self.display.add_message("user", f"Executing command: {command}")
                await asyncio.sleep(0.2)

            if not self._docker_available:
                error = "Docker is not available or the container is not running."
                if self.display:
                    self.display.add_message("user", f"Error: {error}")
                return ToolError(error)

            docker_project_dir = get_constant("DOCKER_PROJECT_DIR") or "/home/myuser/apps"
            # Use proper Linux path format
            docker_project_dir = str(docker_project_dir).replace("\\", "/")
            # Add prefix to ensure we're in the correct directory in Docker
            docker_command = f"cd {docker_project_dir} && {command}"
            try:
                result = run_docker_command(docker_command)
                output = result.stdout
                error = result.stderr
                success = result.returncode == 0
            except Exception as e:
                error = f"Docker execution error: {str(e)}"
                success = False
            if len(output) > 200000:
                output = output[:100000] + " ... [TRUNCATED] ... " + output[-100000:]
            if len(error) > 200000:
                error = error[:100000] + " ... [TRUNCATED] ... " + error[-100000:]
                
            return ToolResult(
                output=f"command: {command}\nsuccess: {str(success).lower()}\noutput: {output}\nerror: {error}"
            )

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
