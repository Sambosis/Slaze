from dataclasses import dataclass
from typing import List, Dict
from typing import Optional

@dataclass
class AgentConfig:
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str
    tools_enabled: Optional[list[str]]