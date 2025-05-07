"""
Tool modules index file - Consolidates all tool imports and provides a central registry
"""

from typing import Dict, List, Type, Optional
from .base import ToolResult, BaseAnthropicTool
from .bash import BashTool
from .write_code import WriteCodeTool
from .create_picture import PictureGenerationTool
from .envsetup import ProjectSetupTool
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from utils.context_provider import get_context_provider, context_provider


class ToolCollection:
    """Tool collection for managing and executing multiple tools"""

    def __init__(self, *tools, display: Optional[AgentDisplayWebWithPrompt] = None):
        self.tools: Dict[str, BaseAnthropicTool] = {}
        self.display = display
        self.context_provider = get_context_provider()

        for tool in tools:
            self.add_tool(tool)

    def add_tool(self, tool: BaseAnthropicTool) -> None:
        """Add a tool to the collection"""
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool from the collection"""
        if tool_name in self.tools:
            del self.tools[tool_name]

    # def get_tool(self, tool_name: str) -> Optional
