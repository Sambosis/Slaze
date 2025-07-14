from typing import ClassVar, Literal, Union
import logging
try:
    from interpreter import interpreter
except Exception:  # pragma: no cover - optional dependency
    interpreter = None

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
        input_schema = {
            "type": "object",
            "properties": {
                "instructions": {
                    "type": "string",
                    "description": "Instructions to execute via interpreter.chat",
                }
            },
            "required": ["instructions"],
        }
        super().__init__(input_schema=input_schema, display=display)

    async def __call__(self, instructions: str | None = None, **kwargs):
        if instructions is None:
            raise ToolError("no message provided")

        if self.display is not None:
            try:
                self.display.add_message("user", f"Interpreter request: {instructions}")
            except Exception as e:
                return ToolResult(error=str(e), tool_name=self.name, command=instructions)

        try:
            global interpreter
            if interpreter is None:
                from interpreter import interpreter as _interp
                interpreter = _interp
            result = interpreter.chat(instructions, display=False, stream=False, blocking=True)
            return ToolResult(
                output=str(result),
                tool_name=self.name,
                command=instructions,
            )
        except Exception as e:
            logger.error("Interpreter execution failed", exc_info=True)
            return ToolResult(
                output="",
                error=str(e),
                tool_name=self.name,
                command=instructions,
            )

    def to_params(self) -> dict:
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
        return params

