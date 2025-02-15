## collection.py
"""Collection classes for managing multiple tools."""

from typing import Any
import json
from anthropic.types.beta import BetaToolUnionParam
from icecream import ic
from .base import (
    BaseAnthropicTool,
    ToolError,
    ToolFailure,
    ToolResult,
)

from load_constants import write_to_file


class ToolCollection:
    """A collection of anthropic-defined tools."""

    def __init__(self, *tools: BaseAnthropicTool, display=None):
        self.display = display
        # Update tools that support display
        updated_tools = []
        for tool in tools:
            if hasattr(tool, 'set_display'):
                tool.set_display(display)
            updated_tools.append(tool)
        self.tools = updated_tools
        self.tool_map = {tool.to_params()["name"]: tool for tool in self.tools}
        # ic(self.tools)
        # ic(self.tool_map)
    def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        
        params = [tool.to_params() for tool in self.tools]
        if params:
            params[-1]["cache_control"] = {"type": "ephemeral"}
        return params

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        ic.configureOutput(includeContext=True, outputFunction=write_to_file)

        tool = self.tool_map.get(name)
    
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            # ic(tool_input)
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)
    def get_tool_names_as_string(self) -> str:
        return ", ".join(self.tool_map.keys())
