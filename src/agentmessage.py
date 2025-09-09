from dataclasses import dataclass
from typing import List, Dict
from typing import Optional

@dataclass
class AgentMessage:
    role: str
    content: str
    timestamp: Optional[float]
    tool_calls: Optional[list[ToolCall]]