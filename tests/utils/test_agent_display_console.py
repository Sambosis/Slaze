import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock

from utils.agent_display_console import AgentDisplayConsole
from rich.panel import Panel
from rich.syntax import Syntax

@pytest.fixture
def console_display():
    # Reset interactive_tool_calls to False by default for other tests if AgentDisplayConsole is used elsewhere
    return AgentDisplayConsole(interactive_tool_calls=True)

@pytest.mark.asyncio
async def test_prompt_approve_directly(console_display: AgentDisplayConsole):
    tool_name = "test_tool"
    tool_args = {"param1": "value1"}
    tool_id = "id_123"

    with patch('asyncio.to_thread') as mock_to_thread, \
         patch.object(console_display, 'wait_for_user_input', new_callable=AsyncMock) as mock_wait_for_input:

        # Simulate Prompt.ask for action choice ("approve")
        # Simulate Confirm.ask for final confirmation (True)
        mock_to_thread.side_effect = [
            "approve", # User chooses 'approve' action
            True       # User confirms execution
        ]

        result = await console_display.prompt_for_tool_call_approval(tool_name, tool_args, tool_id)

        assert result == {"name": tool_name, "args": tool_args, "approved": True}
        # Check that Prompt.ask and Confirm.ask were called via asyncio.to_thread
        assert mock_to_thread.call_count == 2
        mock_wait_for_input.assert_not_called() # Not called if not editing

@pytest.mark.asyncio
async def test_prompt_cancel_action(console_display: AgentDisplayConsole):
    tool_name = "test_tool"
    tool_args = {"param1": "value1"}
    tool_id = "id_123"

    with patch('asyncio.to_thread') as mock_to_thread:
        # Simulate Prompt.ask for action choice ("cancel")
        mock_to_thread.return_value = "cancel" # First call to Prompt.ask

        result = await console_display.prompt_for_tool_call_approval(tool_name, tool_args, tool_id)

        assert result == {"name": tool_name, "args": tool_args, "approved": False}
        mock_to_thread.assert_called_once() # Only Prompt.ask for action, no Confirm.ask

@pytest.mark.asyncio
async def test_prompt_edit_then_approve(console_display: AgentDisplayConsole):
    tool_name = "test_tool"
    original_args = {"param1": "value1"}
    modified_args_str = '{"param1": "edited_value", "param2": 42}'
    modified_args_dict = {"param1": "edited_value", "param2": 42}
    tool_id = "id_123"

    with patch('asyncio.to_thread') as mock_to_thread, \
         patch.object(console_display, 'wait_for_user_input', new_callable=AsyncMock) as mock_wait_for_input:

        # Simulate Prompt.ask for action choice ("edit")
        # Simulate user input for new JSON
        # Simulate Confirm.ask for final confirmation (True)
        mock_to_thread.side_effect = [
            "edit", # User chooses 'edit' action
            True    # User confirms execution of modified args
        ]
        # Simulate multi-line input for JSON
        mock_wait_for_input.side_effect = [modified_args_str, EOFError]


        result = await console_display.prompt_for_tool_call_approval(tool_name, original_args, tool_id)

        assert result == {"name": tool_name, "args": modified_args_dict, "approved": True}
        assert mock_to_thread.call_count == 2 # Prompt.ask for action, Confirm.ask for approval
        mock_wait_for_input.assert_called()

@pytest.mark.asyncio
async def test_prompt_edit_invalid_json_then_approve_original(console_display: AgentDisplayConsole):
    tool_name = "test_tool"
    original_args = {"param1": "value1"}
    invalid_json_str = '{"param1": "broken value"' # Missing closing brace
    tool_id = "id_123"

    with patch('asyncio.to_thread') as mock_to_thread, \
         patch.object(console_display, 'wait_for_user_input', new_callable=AsyncMock) as mock_wait_for_input, \
         patch.object(console_display.console, 'print') as mock_console_print: # To check error messages

        mock_to_thread.side_effect = [
            "edit", # User chooses 'edit' action
            True    # User confirms execution (should be original args)
        ]
        mock_wait_for_input.side_effect = [invalid_json_str, EOFError]

        result = await console_display.prompt_for_tool_call_approval(tool_name, original_args, tool_id)

        assert result == {"name": tool_name, "args": original_args, "approved": True}
        # Check that an error message about invalid JSON was printed
        assert any("Invalid JSON" in call.args[0] for call in mock_console_print.call_args_list if call.args)


@pytest.mark.asyncio
async def test_prompt_edit_empty_json_then_approve_original(console_display: AgentDisplayConsole):
    tool_name = "test_tool"
    original_args = {"param1": "value1"}
    empty_json_str = ''
    tool_id = "id_123"

    with patch('asyncio.to_thread') as mock_to_thread, \
         patch.object(console_display, 'wait_for_user_input', new_callable=AsyncMock) as mock_wait_for_input, \
         patch.object(console_display.console, 'print') as mock_console_print:

        mock_to_thread.side_effect = [
            "edit", # User chooses 'edit' action
            True    # User confirms execution (should be original args)
        ]
        mock_wait_for_input.side_effect = [empty_json_str, EOFError]
        # Also simulate empty lines if wait_for_user_input is called multiple times before EOF
        # For this test, one call returning empty string then EOF is enough.

        result = await console_display.prompt_for_tool_call_approval(tool_name, original_args, tool_id)

        assert result == {"name": tool_name, "args": original_args, "approved": True}
        assert any("No input received" in call.args[0] for call in mock_console_print.call_args_list if call.args)


@pytest.mark.asyncio
async def test_prompt_edit_then_reject(console_display: AgentDisplayConsole):
    tool_name = "test_tool"
    original_args = {"param1": "value1"}
    modified_args_str = '{"param1": "edited_value"}'
    modified_args_dict = {"param1": "edited_value"}
    tool_id = "id_123"

    with patch('asyncio.to_thread') as mock_to_thread, \
         patch.object(console_display, 'wait_for_user_input', new_callable=AsyncMock) as mock_wait_for_input:

        mock_to_thread.side_effect = [
            "edit", # User chooses 'edit' action
            False   # User then REJECTS execution
        ]
        mock_wait_for_input.side_effect = [modified_args_str, EOFError]

        result = await console_display.prompt_for_tool_call_approval(tool_name, original_args, tool_id)

        # Args should be the modified ones, but approved is False
        assert result == {"name": tool_name, "args": modified_args_dict, "approved": False}

@pytest.mark.asyncio
async def test_prompt_edit_cancelled_with_keyboard_interrupt(console_display: AgentDisplayConsole):
    tool_name = "test_tool"
    original_args = {"param1": "value1"}
    tool_id = "id_123"

    with patch('asyncio.to_thread') as mock_to_thread, \
         patch.object(console_display, 'wait_for_user_input', new_callable=AsyncMock) as mock_wait_for_input, \
         patch.object(console_display.console, 'print') as mock_console_print:

        mock_to_thread.side_effect = [
            "edit", # User chooses 'edit' action
            # No second call to mock_to_thread for Confirm.ask because KeyboardInterrupt should bypass it
        ]
        # Simulate KeyboardInterrupt during the first call to wait_for_user_input
        mock_wait_for_input.side_effect = KeyboardInterrupt

        result = await console_display.prompt_for_tool_call_approval(tool_name, original_args, tool_id)

        # Should default to approving original parameters if edit is interrupted
        assert result == {"name": tool_name, "args": original_args, "approved": True}
        assert any("Edit cancelled" in call.args[0] for call in mock_console_print.call_args_list if call.args)
        # Confirm.ask should not have been reached
        assert mock_to_thread.call_count == 1 # Only for the initial "action" prompt
