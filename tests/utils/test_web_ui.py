import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from queue import Queue

# Mock the config module before importing WebUI
# This is to prevent issues with get_constant, set_constant etc. during WebUI init
mock_config = MagicMock()
mock_config.LOGS_DIR = "mock_logs"
mock_config.PROMPTS_DIR = MagicMock()
mock_config.PROMPTS_DIR.glob.return_value = []
mock_config.PROMPTS_DIR.exists.return_value = True # Assume prompts dir exists
mock_config.get_constant.return_value = "mock_value"

# IMPORTANT: Patch 'config' module specifically for 'utils.web_ui' before it's imported by tests
# Also patch tools if their import in WebUI __init__ is problematic
# For now, let's try patching config and see.
# If tools are an issue, they are imported lazily in WebUI, so might be okay for init.

# Patch 'ftfy' as it's imported by utils.web_ui
# If it's not a direct dependency of the tested method, this might not be strictly necessary
# but good for isolating the WebUI class for testing.
with patch.dict('sys.modules', {'config': mock_config, 'ftfy': MagicMock()}):
    from utils.web_ui import WebUI


@pytest.fixture
def web_ui_instance():
    # Mock agent_runner, which is called by WebUI
    mock_agent_runner = AsyncMock()

    # Path for Flask templates - adjust if your structure is different or mock render_template
    # For unit testing prompt_for_tool_call_approval, Flask routes might not be directly hit.
    with patch('utils.web_ui.Flask', MagicMock()) as mock_flask, \
         patch('utils.web_ui.SocketIO', MagicMock()) as mock_socketio, \
         patch('utils.web_ui.Path.mkdir'), \
         patch('utils.web_ui.ToolCollection', MagicMock()): # Mock ToolCollection if its init is complex

        ui = WebUI(agent_runner=mock_agent_runner, interactive_tool_calls=True)
        # Replace the actual socketio instance with a MagicMock for easier testing of emits
        ui.socketio = MagicMock()
        ui.app = MagicMock() # Mock Flask app if needed for other parts, not strictly for this method
        return ui

@pytest.mark.asyncio
async def test_web_prompt_approve(web_ui_instance: WebUI):
    tool_name = "web_tool"
    tool_args = {"data": "send"}
    tool_id = "web_id_1"

    # Store approval_request_id to simulate client sending it back
    captured_emit_data = {}

    def capture_emit(*args, **kwargs):
        if args[0] == "request_tool_approval":
            captured_emit_data.update(args[1])

    web_ui_instance.socketio.emit.side_effect = capture_emit

    async def simulate_client_response():
        # Give a slight delay for the server to emit and set up the event
        await asyncio.sleep(0.01)
        request_id = captured_emit_data.get("approval_request_id")
        assert request_id is not None
        # Simulate the socketio handler being called
        web_ui_instance.handle_tool_approval_response({
            "approval_request_id": request_id,
            "name": tool_name,
            "args": tool_args,
            "approved": True
        })

    # Run the method and the simulation concurrently
    main_task = asyncio.create_task(web_ui_instance.prompt_for_tool_call_approval(tool_name, tool_args, tool_id))
    client_sim_task = asyncio.create_task(simulate_client_response())

    result = await main_task
    await client_sim_task # ensure simulation completes

    assert result == {"name": tool_name, "args": tool_args, "approved": True}
    web_ui_instance.socketio.emit.assert_called_once_with(
        "request_tool_approval",
        {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_id": tool_id,
            "approval_request_id": captured_emit_data.get("approval_request_id") # Check it's the same id
        }
    )

@pytest.mark.asyncio
async def test_web_prompt_edit_and_approve(web_ui_instance: WebUI):
    tool_name = "web_tool_edit"
    original_args = {"data": "original"}
    modified_args = {"data": "modified", "extra": True}
    tool_id = "web_id_edit"

    captured_emit_data = {}
    def capture_emit(*args, **kwargs):
        if args[0] == "request_tool_approval":
            captured_emit_data.update(args[1])
    web_ui_instance.socketio.emit.side_effect = capture_emit

    async def simulate_client_response_edit():
        await asyncio.sleep(0.01)
        request_id = captured_emit_data.get("approval_request_id")
        assert request_id is not None
        web_ui_instance.handle_tool_approval_response({
            "approval_request_id": request_id,
            "name": tool_name, # Name could also be editable on client, assume same for this test
            "args": modified_args, # Client sends back modified args
            "approved": True
        })

    main_task = asyncio.create_task(web_ui_instance.prompt_for_tool_call_approval(tool_name, original_args, tool_id))
    client_sim_task = asyncio.create_task(simulate_client_response_edit())
    result = await main_task
    await client_sim_task

    assert result == {"name": tool_name, "args": modified_args, "approved": True}
    assert captured_emit_data["tool_args"] == original_args # Ensure original args were sent

@pytest.mark.asyncio
async def test_web_prompt_reject(web_ui_instance: WebUI):
    tool_name = "web_tool_reject"
    tool_args = {"data": "reject_this"}
    tool_id = "web_id_reject"

    captured_emit_data = {}
    def capture_emit(*args, **kwargs):
        if args[0] == "request_tool_approval":
            captured_emit_data.update(args[1])
    web_ui_instance.socketio.emit.side_effect = capture_emit

    async def simulate_client_response_reject():
        await asyncio.sleep(0.01)
        request_id = captured_emit_data.get("approval_request_id")
        assert request_id is not None
        web_ui_instance.handle_tool_approval_response({
            "approval_request_id": request_id,
            "name": tool_name,
            "args": tool_args, # Args might be original or modified, doesn't matter much if rejected
            "approved": False # Client rejects
        })

    main_task = asyncio.create_task(web_ui_instance.prompt_for_tool_call_approval(tool_name, tool_args, tool_id))
    client_sim_task = asyncio.create_task(simulate_client_response_reject())
    result = await main_task
    await client_sim_task

    assert result == {"name": tool_name, "args": tool_args, "approved": False}

@pytest.mark.asyncio
async def test_web_prompt_timeout(web_ui_instance: WebUI):
    tool_name = "web_tool_timeout"
    tool_args = {"data": "timeout_test"}
    tool_id = "web_id_timeout"

    # No client response will be simulated for timeout

    # Patch the timeout value for faster testing
    with patch('utils.web_ui.asyncio.wait_for') as mock_wait_for:
        # Make wait_for raise TimeoutError immediately
        mock_wait_for.side_effect = asyncio.TimeoutError

        result = await web_ui_instance.prompt_for_tool_call_approval(tool_name, tool_args, tool_id)

    assert result == {"name": tool_name, "args": tool_args, "approved": False, "error": "timeout"}
    web_ui_instance.socketio.emit.assert_called_once() # Check that emit was attempted
    # Ensure the event was added and then removed due to timeout
    assert not web_ui_instance.tool_approval_events # Should be empty after timeout
    assert not web_ui_instance.tool_approval_data

# Note: To run these tests, you might need to ensure that the asyncio event loop
# is properly managed, especially with pytest-asyncio.
# The `handle_tool_approval_response` is synchronous but sets an asyncio.Event.
# The way it's called in `simulate_client_response` should work fine with pytest-asyncio.
# The patch for 'config' and 'ftfy' at the top is crucial.
