from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str]
    metadata: Optional[dict]

@dataclass
class AgentMessage:
    role: str
    content: str
    timestamp: Optional[float]
    tool_calls: Optional[list[ToolCall]]

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass
class AgentConfig:
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str
    tools_enabled: Optional[list[str]]

@dataclass
class FileOperation:
    path: str
    operation: str
    content: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]


class Agent:
    messages: list["AgentMessage"]
    tools: object
    output_manager: object
    def __init__(self, config: AgentConfig, client: object) -> None: ...
    def process_message(self, message: str) -> AgentMessage: ...
    def execute_tool(self, tool_call: ToolCall) -> ToolResult: ...
    def get_context(self) -> str: ...


class BaseTool:
    enabled: bool
    def __init__(self, name: str, description: str) -> None: ...
    def execute(self, args: dict) -> ToolResult: ...
    def validate_args(self, args: dict) -> bool: ...


class WriteCodeTool(BaseTool):
    def __init__(self, name: str, description: str) -> None: ...
    def write_file(self, path: str, content: str) -> ToolResult: ...
    def edit_file(self, path: str, changes: list["FileOperation"]) -> ToolResult: ...


class BashTool(BaseTool):
    def __init__(self, name: str, description: str) -> None: ...
    def run_command(self, command: str, timeout: Optional[int]) -> ToolResult: ...


class EditTool(BaseTool):
    def __init__(self, name: str, description: str) -> None: ...
    def apply_edit(self, file_path: str, old_content: str, new_content: str) -> ToolResult: ...

def format_messages(messages: list["AgentMessage"], max_length: Optional[int]) -> str: ...

def validate_tool_call(tool_call: ToolCall, available_tools: list[str]) -> bool: ...

def extract_code_blocks(content: str, language: Optional[str]) -> list[str]: ...

def sanitize_input(input_text: str, max_length: Optional[int]) -> str: ...

def load_system_prompt(prompt_path: str) -> str: ...

def track_tokens(input_tokens: int, output_tokens: int, model: str) -> dict: ...

DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"
MAX_TOKENS: int = 4096
DEFAULT_TEMPERATURE: float = 0.1
TOOL_TIMEOUT: int = 30
MAX_MESSAGE_LENGTH: int = 50000