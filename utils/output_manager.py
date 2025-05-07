import base64
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, TYPE_CHECKING

from anthropic import APIResponse
from anthropic.types.beta import BetaContentBlock, BetaMessageParam
from icecream import ic

from .agent_display_web_with_prompt import (
    AgentDisplayWebWithPrompt,
)  # Relative import for AgentDisplay
from config import get_constant  # Updated import

if TYPE_CHECKING:
    from tools.base import ToolResult


class OutputManager:
    def __init__(
        self, display: AgentDisplayWebWithPrompt, image_dir: Optional[Path] = None
    ):
        LOGS_DIR = Path(get_constant("LOGS_DIR"))
        self.image_dir = LOGS_DIR / "computer_tool_images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.image_counter = 0
        self.display = display

    def save_image(self, base64_data: str) -> Optional[Path]:
        """Save base64 image data to file and return path."""
        if not base64_data:
            ic("Error: No base64 data provided to save_image")
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
            ic(f"Error saving image: {e}")
            return None

    def format_tool_output(self, result: "ToolResult", tool_name: str):
        """Format and display tool output."""
        if result is None:
            ic("Error: None result provided to format_tool_output")
            return

        output_text = f"Used Tool: {tool_name}\n"

        if isinstance(result, str):
            output_text += f"{result}"
        else:
            text = self._truncate_string(
                str(result.output) if result.output is not None else ""
            )
            output_text += f"Output: {text}\n"
            if result.base64_image:
                image_path = self.save_image(result.base64_image)
                if image_path:
                    output_text += (
                        f"[green]📸 Screenshot saved to {image_path}[/green]\n"
                    )
                else:
                    output_text += "[red]Failed to save screenshot[/red]\n"

        # self.display., output_text)

    def format_api_response(self, response: APIResponse):
        """Format and display API response."""
        if response is None or not hasattr(response, "content") or not response.content:
            ic("Error: Invalid API response in format_api_response")
            return

        if response.content and hasattr(response.content[0], "text"):
            self._truncate_string(response.content[0].text)

    def format_content_block(self, block: BetaContentBlock) -> None:
        """Format and display content block."""
        if block is None:
            ic("Error: None block provided to format_content_block")
            return

        if getattr(block, "type", None) == "tool_use":
            safe_input = {
                k: v
                for k, v in block.input.items()
                if not isinstance(v, str) or len(v) < 1000
            }
            json.dumps(safe_input) if isinstance(safe_input, dict) else str(safe_input)

    def format_recent_conversation(
        self, messages: List[BetaMessageParam], num_recent: int = 10
    ):
        """Format and display recent conversation."""
        if messages is None or not messages:
            ic("Error: No messages provided to format_recent_conversation")
            return

        # recent_messages = messages[:num_recent] if len(messages) > num_recent else messages
        recent_messages = messages[-num_recent:]
        for msg in recent_messages:
            if msg["role"] == "user":
                self._format_user_content(msg["content"])
            elif msg["role"] == "assistant":
                self._format_assistant_content(msg["content"])

    def _format_user_content(self, content: Any):
        """Format and display user content."""
        if content is None:
            ic("Error: None content provided to _format_user_content")
            return

        if isinstance(content, list):
            for content_block in content:
                if isinstance(content_block, dict):
                    if content_block.get("type") == "tool_result":
                        for item in content_block.get("content", []):
                            if item.get("type") == "text":
                                self._truncate_string(item.get("text", ""))
                            #     self.display., text)
                            # elif item.get("type") == "image":
                            #     self.display., "📸 Screenshot captured")
        elif isinstance(content, str):
            self._truncate_string(content)
            # self.display., text)

    def _format_assistant_content(self, content: Any):
        """Format and display assistant content."""
        if content is None:
            ic("Error: None content provided to _format_assistant_content")
            return

        if isinstance(content, list):
            for content_block in content:
                if isinstance(content_block, dict):
                    if content_block.get("type") == "text":
                        self._truncate_string(content_block.get("text", ""))
                    elif content_block.get("type") == "tool_use":
                        content_block.get("name")
                        tool_input = content_block.get("input", "")
                        if isinstance(tool_input, dict):
                            "\n".join(f"{k}: {v}" for k, v in tool_input.items())
                        else:
                            try:
                                tool_input = json.loads(tool_input)
                                "\n".join(f"{k}: {v}" for k, v in tool_input.items())
                            except json.JSONDecodeError:
                                str(tool_input)
                        # self.display., (tool_name, f"Input: {input_text}"))
        elif isinstance(content, str):
            self._truncate_string(content)

    def _truncate_string(self, text: str, max_length: int = 500) -> str:
        """Truncate a string to a max length with ellipsis."""
        if text is None:
            return ""

        if len(text) > max_length:
            return text[:200] + "\n...\n" + text[-200:]
        return text
