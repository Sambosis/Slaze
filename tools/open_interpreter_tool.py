from __future__ import annotations

import asyncio
from typing import ClassVar, Literal, Union

from interpreter import interpreter

from .base import BaseTool, ToolError, ToolResult
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole


class OpenInterpreterTool(BaseTool):
    """Execute natural language instructions using open-interpreter."""

    name: ClassVar[Literal["open_interpreter"]] = "open_interpreter"
    api_type: ClassVar[Literal["open_interpreter_20250124"]] = "open_interpreter_20250124"

    description = (
        "A tool that uses open-interpreter to execute natural language instructions."
    )

    def __init__(self, display: Union[WebUI, AgentDisplayConsole] | None = None):
        self.display = display
        super().__init__(input_schema=None, display=display)

    async def __call__(self, instruction: str | None = None, **_: object) -> ToolResult:
        """Run open-interpreter with the given instruction."""
        if instruction is None:
            raise ToolError("no instruction provided.")

        if self.display is not None:
            try:
                self.display.add_message("user", f"Executing instruction: {instruction}")
            except Exception as exc:
                return ToolResult(error=str(exc), tool_name=self.name, command=instruction)

        try:
            result = await asyncio.to_thread(interpreter.chat, instruction)
            output = result if isinstance(result, str) else str(result)
            return ToolResult(output=output, tool_name=self.name, command=instruction)
        except Exception as exc:
            return ToolResult(error=str(exc), tool_name=self.name, command=instruction)

    def to_params(self) -> dict:
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "Natural language instruction to execute.",
                        }
                    },
                    "required": ["instruction"],
                },
            },
        }
        return params
