from dataclasses import dataclass

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict