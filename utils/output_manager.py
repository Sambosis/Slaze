import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING, Union

from typing import Dict
import logging

from .web_ui import WebUI
from .agent_display_console import AgentDisplayConsole
from config import get_constant  # Updated import

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tools.base import ToolResult


class OutputManager:

    def __init__(self,
                 display: Union[WebUI, AgentDisplayConsole],
                 image_dir: Optional[Path] = None):
        LOGS_DIR = Path(get_constant("LOGS_DIR"))
        self.image_dir = LOGS_DIR / "computer_tool_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.image_counter = 0
        self.display = display

    def save_image(self, base64_data: str) -> Optional[Path]:
        """Save base64 image data to file and return path."""
        if not base64_data:
            logger.error("No base64 data provided to save_image")
            return None

        try:
            self.image_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:8]
            image_path = self.image_dir / f"image_{timestamp}_{image_hash}.png"

            image_data = base64.b64decode(base64_data)
            with open(image_path, "wb") as f:
                f.write(image_data)
            return image_path
        except Exception as e:
            logger.error(f"Error saving image: {e}", exc_info=True)
            return None

    def format_tool_output(self, result: "ToolResult", tool_name: str):
        """Format and display tool output."""
        if result is None:
            logger.error("None result provided to format_tool_output")
            return

        output_text = f"Used Tool: {tool_name}\n"

        if isinstance(result, str):
            output_text += f"{result}"
        else:
            text = self._truncate_string(
                str(result.output) if result.output is not None else "")
            output_text += f"Output: {text}\n"
            if result.base64_image:
                image_path = self.save_image(result.base64_image)
                if image_path:
                    output_text += (
                        f"[green]📸 Screenshot saved to {image_path}[/green]\n")
                else:
                    output_text += "[red]Failed to save screenshot[/red]\n"

        # self.display., output_text)











    def _truncate_string(self, text: str, max_length: int = 500) -> str:
        """Truncate a string to a max length with ellipsis."""
        if text is None:
            return ""

        if len(text) > max_length:
            return text[:200] + "\n...\n" + text[-200:]
        return text
