import pytest
from unittest.mock import patch, MagicMock

from tools.open_interpreter_tool import OpenInterpreterTool
from tools.base import ToolResult, ToolError

@pytest.fixture
def interpreter_tool():
    return OpenInterpreterTool()

@pytest.mark.asyncio
async def test_call_success(interpreter_tool):
    with patch('tools.open_interpreter_tool.interpreter') as mock_interp:
        mock_interp.chat.return_value = 'ok'
        mock_chat = mock_interp.chat
        result = await interpreter_tool('print(1)')
    assert isinstance(result, ToolResult)
    assert result.output == 'ok'
    assert result.tool_name == 'open_interpreter'
    assert result.command == 'print(1)'
    mock_chat.assert_called_with('print(1)', display=False, stream=False, blocking=True)

@pytest.mark.asyncio
async def test_no_message(interpreter_tool):
    with pytest.raises(ToolError):
        await interpreter_tool()

@pytest.mark.asyncio
async def test_display_error(interpreter_tool):
    mock_display = MagicMock()
    mock_display.add_message.side_effect = Exception('fail')
    interpreter_tool.display = mock_display
    with patch('tools.open_interpreter_tool.interpreter') as mock_interp:
        mock_interp.chat.return_value = 'ok'
        result = await interpreter_tool('cmd')
    assert result.error == 'fail'

@pytest.mark.asyncio
async def test_chat_exception(interpreter_tool):
    with patch('tools.open_interpreter_tool.interpreter') as mock_interp:
        mock_interp.chat.side_effect = Exception('boom')
        result = await interpreter_tool('cmd')
    assert result.error == 'boom'
