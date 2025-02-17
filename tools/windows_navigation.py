# tools/windows_navigation.py

# import pyautogui
import time
from typing import Literal, Optional, Dict, Any
from pathlib import Path
import json
import logging
from .base import BaseAnthropicTool, ToolResult, ToolError
from rich import print as rr

# Configure logging
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

class WindowsNavigationTool:
    """
    A tool specializing in navigating Windows and Windows applications using keyboard shortcuts.
    """

    name: Literal["windows_navigate"] = "windows_navigate"
    api_type: Literal["custom"] = "custom"
    description: str = (
        "A comprehensive tool for Windows navigation using keyboard shortcuts. "
        "Supports window management, file operations, system controls, and accessibility features. "
        "Uses pyautogui to simulate keyboard and mouse inputs for automated Windows control."
    )

    def __init__(self):
        """Initialize the WindowsNavigationTool with shortcuts configuration."""
        self.shortcuts_file = Path(__file__).parent / "windows_shortcuts.json"
        self.shortcuts = self._load_shortcuts()
        self.session_history = []  # Track navigation actions
        logging.info("WindowsNavigationTool initialized with shortcuts from '%s'", self.shortcuts_file)

    def _load_shortcuts(self) -> Dict[str, Any]:
        """Load keyboard shortcuts from JSON file."""
        try:
            if self.shortcuts_file.exists():
                with open(self.shortcuts_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Shortcuts file not found at {self.shortcuts_file}")
                return {}
        except Exception as e:
            logger.error(f"Error loading shortcuts: {e}")
            return {}

    def to_params(self) -> dict:
        """Define the parameters for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.api_type,
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            # Window Management
                            "switch_window", "open_start_menu", "minimize_window", 
                            "maximize_window", "restore_window", "close_window", 
                            "take_screenshot", "go_to_desktop",
                            "switch_virtual_desktop_left", "switch_virtual_desktop_right",
                            
                            # File Explorer
                            "open_file_explorer", "refresh_explorer",
                            
                            # Taskbar
                            "open_task_manager", "lock_workstation", "sign_out", 
                            "hibernate", "sleep",
                            
                            # Clipboard Operations
                            "copy", "paste", "cut", "select_all",
                            
                            # System Controls
                            "open_run_dialog", "open_settings", "open_search",
                            
                            # Accessibility
                            "toggle_high_contrast", "toggle_narrator", "toggle_magnifier"
                        ],
                        "description": "The Windows action to perform"
                    },
                    "modifier": {
                        "type": "string",
                        "enum": ["ctrl", "alt", "shift", "win"],
                        "description": "Optional modifier key(s)"
                    },
                    "target": {
                        "type": "string",
                        "description": "Optional target for the action (e.g., window title)"
                    }
                },
                "required": ["action"]
            }
        }

    async def __call__(
        self,
        action: str,
        modifier: Optional[str] = None,
        target: Optional[str] = None
    ) -> ToolResult:
        """
        Execute the requested Windows action.
        
        Args:
            action (str): The action to perform (must be one of the defined actions)
            modifier (Optional[str]): Optional modifier key(s)
            target (Optional[str]): Optional target for the action
            
        Returns:
            ToolResult: Contains either the success output or error message
        """
        try:
            # Get shortcut configuration
            shortcut = self.shortcuts.get(action)
            if not shortcut:
                return ToolResult(error=f"Unknown action: {action}")

            # Prepare keys sequence
            keys = shortcut["keys"]
            if modifier:
                keys = [modifier] + keys

            # Execute the shortcut
            result = await self._execute_action(action, keys, target)
            
            # Log the action
            self.session_history.append(f"{action}: {result.output if result.output else result.error}")
            
            return result

        except Exception as e:
            error_msg = f"Error executing {action}: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)

    async def _execute_action(self, action: str, keys: list, target: Optional[str] = None) -> ToolResult:
        """Execute a Windows action with the given keys and target."""
        try:
            # Activate window if needed
            if target and "window" in action.lower():
                windows = pyautogui.getWindowsWithTitle(target)
                if not windows:
                    return ToolResult(error=f"No window found with title '{target}'")
                windows[0].activate()
                time.sleep(0.2)

            # Execute the key combination
            # pyautogui.hotkey(*keys)
            time.sleep(0.1)  # Small delay for action to complete

            # Handle any follow-up input
            if target and self.shortcuts.get(action, {}).get("requires_target", False):
                # pyautogui.typewrite(target)
                # pyautogui.press('enter')

            success_msg = f"Successfully executed '{action}'"
            if target:
                success_msg += f" with target '{target}'"
            
            logger.info(success_msg)
            return ToolResult(output=success_msg)

        except Exception as e:
            error_msg = f"Failed to execute {action}: {str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)

    def get_session_history(self) -> str:
        """Returns the session history of navigation actions."""
        return "\n".join(self.session_history)