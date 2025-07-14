from typing import ClassVar, Literal, Union
import logging

try:
    from interpreter import interpreter
except Exception:  # pragma: no cover - optional dependency
    interpreter = None


import os
from pathlib import Path
from config import get_constant
from .base import BaseTool, ToolError, ToolResult
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole

logger = logging.getLogger(__name__)

class OpenInterpreterTool(BaseTool):
    """Execute commands using the open-interpreter package."""

    name: ClassVar[Literal["open_interpreter"]] = "open_interpreter"
    api_type: ClassVar[Literal["open_interpreter_20250124"]] = "open_interpreter_20250124"
    description: str = (
        "Runs instructions using open-interpreter's interpreter.chat function."
    )

    def __init__(self, display: Union[WebUI, AgentDisplayConsole] = None):
        super().__init__(input_schema=None, display=display)

    async def __call__(self, message: str | None = None, **kwargs):
        if message is None:
            raise ToolError("no message provided")

        if self.display is not None:
            try:
                self.display.add_message("user", f"Interpreter request: {message}")
            except Exception as e:
                return ToolResult(error=str(e), tool_name=self.name, command=message)

        try:
            global interpreter
            if interpreter is None:
                from interpreter import interpreter as _interp
                interpreter = _interp

            result = interpreter.chat(message, display=False, stream=False, blocking=True)
            return ToolResult(
                output=str(result),
                tool_name=self.name,
                command=message,
            )
        except Exception as e:
            logger.error("Interpreter execution failed", exc_info=True)
            return ToolResult(
                output="",
                error=str(e),
                tool_name=self.name,
                command=message,
            )
    def __init__(self, display: Union[WebUI, AgentDisplayConsole] = None):
        self.display = display
        super().__init__(input_schema=None, display=display)

    description = """
        A tool that uses open-interpreter's interpreter.chat() method to execute commands
        and tasks. This provides an alternative to direct bash execution with enhanced
        AI-powered command interpretation and execution.
        """

    name: ClassVar[Literal["open_interpreter"]] = "open_interpreter"
    api_type: ClassVar[Literal["open_interpreter_20250124"]] = "open_interpreter_20250124"

    async def __call__(self, task_description: str | None = None, **kwargs):
        if task_description is not None:
            if self.display is not None:
                try:
                    self.display.add_message("user", f"Executing task with open-interpreter: {task_description}")
                except Exception as e:
                    return ToolResult(error=str(e), tool_name=self.name, command=task_description)

            return await self._execute_with_interpreter(task_description)
        raise ToolError("no task description provided.")

    async def _execute_with_interpreter(self, task_description: str):
        """
        Execute a task using open-interpreter's interpreter.chat() method.
        """
        output = ""
        error = ""
        success = False
        cwd = None
        
        try:
            # Get the current working directory
            repo_dir = get_constant("REPO_DIR")
            cwd = str(repo_dir) if repo_dir and Path(repo_dir).exists() else None
            
            # Try to import and use open-interpreter
            try:
                import interpreter
                
                # Configure interpreter settings
                interpreter.offline = False  # Allow online operations
                interpreter.auto_run = True  # Auto-run commands
                interpreter.verbose = False  # Reduce verbosity for tool usage
                
                # Create system information context
                system_info = self._get_system_info()
                full_task = f"{task_description}\n\nSystem Information: {system_info}"
                
                # Execute the task using interpreter.chat()
                result = interpreter.chat(full_task)
                
                # Extract output from the result
                if hasattr(result, 'messages'):
                    # Get the last assistant message which should contain the execution result
                    for message in reversed(result.messages):
                        if message.get('role') == 'assistant':
                            output = message.get('content', '')
                            break
                else:
                    output = str(result)
                
                success = True
                
            except ImportError:
                error = "open-interpreter package is not installed. Please install it with: pip install open-interpreter"
                success = False
                
        except Exception as e:
            error = str(e)
            success = False

        formatted_output = (
            f"task_description: {task_description}\n"
            f"working_directory: {cwd}\n"
            f"success: {str(success).lower()}\n"
            f"output: {output}\n"
            f"error: {error}"
        )
        print(formatted_output)
        return ToolResult(
            output=formatted_output,
            error=error,
            tool_name=self.name,
            command=task_description,
        )

    def _get_system_info(self) -> str:
        """
        Get system information to provide context to the interpreter.
        """
        import platform
        import subprocess
        
        system_info = []
        
        # Basic system info
        system_info.append(f"OS: {platform.system()} {platform.release()}")
        system_info.append(f"Architecture: {platform.machine()}")
        system_info.append(f"Python: {platform.python_version()}")
        
        # Current working directory
        cwd = os.getcwd()
        system_info.append(f"Current Directory: {cwd}")
        
        # Available commands
        try:
            # Check for common commands
            commands = ['ls', 'pwd', 'python', 'python3', 'pip', 'pip3']
            available_commands = []
            for cmd in commands:
                try:
                    subprocess.run([cmd, '--version'], capture_output=True, timeout=1)
                    available_commands.append(cmd)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            system_info.append(f"Available Commands: {', '.join(available_commands)}")
        except Exception:
            system_info.append("Available Commands: Unable to determine")
        
        return "\n".join(system_info)

    def to_params(self) -> dict:
        logger.debug(f"OpenInterpreterTool.to_params called with api_type: {self.api_type}")
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "A description of the task to be executed using open-interpreter. This should include what needs to be done and any relevant context about the system it will run on.",
                        }
                    },
                    "required": ["task_description"],
                },
            },
        }
        logger.debug(f"OpenInterpreterTool params: {params}")
        return params

