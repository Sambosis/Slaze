from dataclasses import dataclass
from typing import Optional

@dataclass
class FileOperation:
    path: str
    operation: str
    content: Optional[str]
    line_start: Optional[int]
    line_end: Optional[int]