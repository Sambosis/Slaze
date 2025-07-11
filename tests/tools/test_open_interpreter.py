import pytest
from unittest.mock import patch, MagicMock

from tools.open_interpreter_tool import OpenInterpreterTool
from tools.base import ToolResult, ToolError


@pytest.fixture
def oi_tool():
    return OpenInterpreterTool()


@pytest.mark.asyncio
async def test_basic_instruction_execution(oi_tool):
    with patch('tools.open_interpreter_tool.interpreter.chat', return_value='done'):
        result = await oi_tool("list files")

    assert isinstance(result, ToolResult)
    assert result.output == 'done'
    assert result.tool_name == "open_interpreter"
    assert result.command == "list files"


@pytest.mark.asyncio
async def test_no_instruction_provided(oi_tool):
    with pytest.raises(ToolError):
        await oi_tool()


@pytest.mark.asyncio
async def test_display_integration(oi_tool):
    mock_display = MagicMock()
    oi_tool.display = mock_display
    with patch('tools.open_interpreter_tool.interpreter.chat', return_value='ok'):
        result = await oi_tool("check version")

    assert isinstance(result, ToolResult)
    mock_display.add_message.assert_called_with("user", "Executing instruction: check version")

