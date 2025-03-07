## collection.py
"""Collection classes for managing multiple tools."""

from typing import Dict, Any, List, Optional
import asyncio
import json
import inspect
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
    """Collection of tools for the agent to use."""
    
    def __init__(self, *tools, display=None):
        """
        Initialize the tool collection.
        
        Args:
            *tools: Tools to add to the collection
            display: Optional display object for UI interaction
        """
        self.tools: Dict[str, BaseAnthropicTool] = {}
        self.display = display
        
        # Add each tool to the collection
        for tool in tools:
            if hasattr(tool, 'name'):
                tool_name = tool.name
                self.tools[tool_name] = tool
    
    def to_params(self) -> List[Dict[str, Any]]:
        """
        Convert all tools to a list of parameter dictionaries.
        
        Returns:
            List[Dict[str, Any]]: List of tool parameters
        """
        ic("---- COLLECTING TOOL PARAMS ----")
        tool_params = []
        
        for tool_name, tool in self.tools.items():
            ic(f"Tool: {tool_name}")
            try:
                params = tool.to_params()
                ic(f"Tool params for {tool_name}:")
                ic(params)
                tool_params.append(params)
            except Exception as e:
                ic(f"Error getting params for tool {tool_name}: {str(e)}")
        
        ic(f"Total tools collected: {len(tool_params)}")
        ic("---- END COLLECTING TOOL PARAMS ----")
        return tool_params
    
    async def run(self, name: str, tool_input: Dict[str, Any]) -> ToolResult:
        """
        Run a tool with the given name and input.
        
        Args:
            name: Name of the tool to run
            tool_input: Input parameters for the tool
            
        Returns:
            ToolResult: Result of the tool execution
            
        Raises:
            ValueError: If the tool is not found
        """
        if name not in self.tools:
            return ToolResult(
                error=f"Tool '{name}' not found. Available tools: {', '.join(self.tools.keys())}",
                tool_name=name,
                command=tool_input.get("command", "unknown")
            )
        
        tool = self.tools[name]
        
        try:
            # Execute the tool and get the result
            result = await tool(**tool_input)
            
            # If the result is None or not a ToolResult, create a proper one
            if result is None:
                command = tool_input.get("command", "unknown")
                command_str = command.value if hasattr(command, 'value') else str(command)
                return ToolResult(
                    error="Tool execution returned None",
                    tool_name=name,
                    command=command_str
                )
                
            # If it's already a ToolResult but missing attributes, add them
            if not hasattr(result, 'tool_name') or result.tool_name is None:
                result = ToolResult(
                    output=result.output if hasattr(result, 'output') else None,
                    error=result.error if hasattr(result, 'error') else None,
                    base64_image=result.base64_image if hasattr(result, 'base64_image') else None,
                    system=result.system if hasattr(result, 'system') else None,
                    message=result.message if hasattr(result, 'message') else None,
                    tool_name=name,
                    command=result.command if hasattr(result, 'command') else tool_input.get("command", "unknown")
                )
                
            if not hasattr(result, 'command') or result.command is None:
                command = tool_input.get("command", "unknown")
                command_str = command.value if hasattr(command, 'value') else str(command)
                result = ToolResult(
                    output=result.output if hasattr(result, 'output') else None,
                    error=result.error if hasattr(result, 'error') else None,
                    base64_image=result.base64_image if hasattr(result, 'base64_image') else None,
                    system=result.system if hasattr(result, 'system') else None,
                    message=result.message if hasattr(result, 'message') else None,
                    tool_name=result.tool_name,
                    command=command_str
                )
                    
            return result
            
        except Exception as e:
            # Return ToolResult with error message
            command = tool_input.get("command", "unknown")
            command_str = command.value if hasattr(command, 'value') else str(command)
            return ToolResult(
                error=f"Error executing tool '{name}': {str(e)}",
                tool_name=name,
                command=command_str
            )
