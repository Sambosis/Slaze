## base.py
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields, replace
from typing import Any, Optional, Dict

from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from icecream import ic


@dataclass(kw_only=True, frozen=True)
class ToolResult:
    """
    Result from executing a tool.

    Attributes:
        output: The output of the tool execution
        error: Optional error message if the tool execution failed
        base64_image: Optional base64-encoded image data
        system: Optional system message
        message: Optional message
        tool_name: Name of the tool that was executed
        command: Command that was executed
    """

    output: Optional[str] = None
    error: Optional[str] = None
    base64_image: Optional[str] = None
    system: Optional[str] = None
    message: Optional[str] = None
    tool_name: Optional[str] = None
    command: Optional[str] = None

    def __bool__(self):
        """Returns True if the tool execution was successful."""
        return any(getattr(self, field.name) for field in fields(self))

    def __add__(self, other: "ToolResult"):
        def combine_fields(
            field: str | None, other_field: str | None, concatenate: bool = True
        ):
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("Cannot combine tool results")
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
            tool_name=self.tool_name or other.tool_name,
            command=self.command or other.command,
        )

    def replace(self, **kwargs):
        """Returns a new ToolResult with the given fields replaced."""
        return replace(self, **kwargs)


class ToolError(Exception):
    """Exception raised when a tool fails to execute."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return self.message


class BaseAnthropicTool(metaclass=ABCMeta):
    """Base class for all tools."""

    name: str = "base_tool"
    api_type: str = "custom"
    description: str = "Base tool implementation"

    def __init__(
        self,
        input_schema: Optional[Dict[str, Any]] = None,
        display: Optional[AgentDisplayWebWithPrompt] = None,
    ):
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {},
            "required": [],
        }
        self.display = display

    def set_display(self, display: AgentDisplayWebWithPrompt):
        """Set the display instance for the tool."""
        self.display = display

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A description of what the tool does."""
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> Any:
        """Execute the tool with the given arguments."""
        pass

    def to_params(self) -> Dict[str, Any]:
        """Convert the tool to xAI API parameters."""
        ic(f"BaseAnthropicTool.to_params called for {self.name}")
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }
        ic(f"BaseAnthropicTool params for {self.name}: {params}")
        return params


class CLIResult(ToolResult):
    """A ToolResult that can be rendered as a CLI output."""

    pass


class ToolFailure(ToolResult):
    """A ToolResult that represents a failure."""

    pass
