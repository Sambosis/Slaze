from dataclasses import dataclass
from typing import Optional

@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str]
    metadata: Optional[dict]