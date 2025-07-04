import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from utils.command_converter import CommandConverter, convert_command_for_system


@pytest.fixture
def command_converter():
    """Fixture to create a CommandConverter instance."""
    return CommandConverter()


@pytest.fixture
def mock_llm_client():
    """Fixture to create a mock LLM client."""
    client = MagicMock()
    client.call = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_command_converter_init(command_converter: CommandConverter):
    """Test that CommandConverter initializes properly with system info."""
    assert command_converter.system_info is not None
    assert "os_name" in command_converter.system_info
    assert "architecture" in command_converter.system_info
    assert command_converter.conversion_prompt is not None
    assert "You are a bash command converter" in command_converter.conversion_prompt


@pytest.mark.asyncio
async def test_system_info_gathering(command_converter: CommandConverter):
    """Test that system information is gathered correctly."""
    system_info = command_converter.system_info
    
    required_keys = [
        "os_name", "os_version", "architecture", "python_version",
        "shell", "home_dir", "current_working_dir", "path_separator",
        "file_separator", "environment_vars"
    ]
    
    for key in required_keys:
        assert key in system_info, f"Missing required key: {key}"
    
    assert isinstance(system_info["environment_vars"], dict)
    assert "PATH" in system_info["environment_vars"]


@pytest.mark.asyncio
async def test_conversion_prompt_generation(command_converter: CommandConverter):
    """Test that the conversion prompt includes system information."""
    prompt = command_converter.conversion_prompt
    
    # Check that system info is included in prompt
    assert command_converter.system_info["os_name"] in prompt
    assert "CRITICAL OUTPUT FORMAT" in prompt
    assert "EXAMPLES:" in prompt
    assert "RULES:" in prompt
    assert "find /path -type f" in prompt  # Example command


@pytest.mark.asyncio
async def test_convert_command_success(command_converter: CommandConverter, mock_llm_client):
    """Test successful command conversion."""
    # Mock the LLM response
    mock_llm_client.call.return_value = "find /path -type f -not -path '*/.*'"
    
    with patch('utils.command_converter.create_llm_client', return_value=mock_llm_client):
        with patch('utils.command_converter.get_constant', return_value="anthropic/claude-sonnet-4"):
            result = await command_converter.convert_command("find /path -type f")
    
    assert result == "find /path -type f -not -path '*/.*'"
    mock_llm_client.call.assert_called_once()


@pytest.mark.asyncio
async def test_convert_command_with_cleaning(command_converter: CommandConverter, mock_llm_client):
    """Test command conversion with response cleaning."""
    # Mock LLM response with markdown and extra text
    mock_llm_client.call.return_value = """```bash
ls -la /directory | grep -v "^\\."
```

This command will list files excluding hidden ones."""
    
    with patch('utils.command_converter.create_llm_client', return_value=mock_llm_client):
        with patch('utils.command_converter.get_constant', return_value="anthropic/claude-sonnet-4"):
            result = await command_converter.convert_command("ls -la /directory")
    
    assert result == 'ls -la /directory | grep -v "^\\."'


@pytest.mark.asyncio
async def test_convert_command_fallback_on_error(command_converter: CommandConverter, mock_llm_client):
    """Test that conversion falls back to original command on error."""
    # Mock LLM client to raise an exception
    mock_llm_client.call.side_effect = Exception("API Error")
    
    with patch('utils.command_converter.create_llm_client', return_value=mock_llm_client):
        with patch('utils.command_converter.get_constant', return_value="anthropic/claude-sonnet-4"):
            result = await command_converter.convert_command("echo hello")
    
    # Should return original command on error
    assert result == "echo hello"


@pytest.mark.asyncio
async def test_clean_response_basic():
    """Test basic response cleaning functionality."""
    converter = CommandConverter()
    
    # Test basic cleaning
    response = "  ls -la  \n"
    cleaned = converter._clean_response(response)
    assert cleaned == "ls -la"


@pytest.mark.asyncio
async def test_clean_response_markdown():
    """Test cleaning response with markdown code blocks."""
    converter = CommandConverter()
    
    # Test markdown removal
    response = "```bash\nfind /path -type f\n```"
    cleaned = converter._clean_response(response)
    assert cleaned == "find /path -type f"


@pytest.mark.asyncio
async def test_clean_response_multiline():
    """Test cleaning multiline response (takes first line)."""
    converter = CommandConverter()
    
    # Test multiline response
    response = "find /path -type f\nThis is an explanation\nMore text"
    cleaned = converter._clean_response(response)
    assert cleaned == "find /path -type f"


@pytest.mark.asyncio
async def test_clean_response_empty():
    """Test cleaning empty response raises error."""
    converter = CommandConverter()
    
    with pytest.raises(ValueError, match="Empty response from LLM"):
        converter._clean_response("")


@pytest.mark.asyncio
async def test_clean_response_too_long():
    """Test cleaning very long response raises error."""
    converter = CommandConverter()
    
    very_long_command = "x" * 1001  # Over 1000 char limit
    with pytest.raises(ValueError, match="Invalid command format"):
        converter._clean_response(very_long_command)


@pytest.mark.asyncio
async def test_global_convert_function():
    """Test the global convert_command_for_system function."""
    mock_converter = MagicMock()
    mock_converter.convert_command = AsyncMock(return_value="converted_command")
    
    with patch('utils.command_converter.CommandConverter', return_value=mock_converter):
        result = await convert_command_for_system("original_command")
    
    assert result == "converted_command"
    mock_converter.convert_command.assert_called_once_with("original_command")


@pytest.mark.asyncio
async def test_global_convert_function_reuses_instance():
    """Test that the global function reuses the same converter instance."""
    # Clear any existing instance
    import utils.command_converter
    utils.command_converter._converter_instance = None
    
    mock_converter = MagicMock()
    mock_converter.convert_command = AsyncMock(return_value="converted_command")
    
    with patch('utils.command_converter.CommandConverter', return_value=mock_converter) as mock_class:
        # Call twice
        await convert_command_for_system("command1")
        await convert_command_for_system("command2")
    
    # Should only create one instance
    mock_class.assert_called_once()
    assert mock_converter.convert_command.call_count == 2


@pytest.mark.asyncio
async def test_llm_client_integration(command_converter: CommandConverter):
    """Test integration with LLM client creation."""
    with patch('utils.command_converter.create_llm_client') as mock_create:
        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value="converted command")
        mock_create.return_value = mock_client
        
        with patch('utils.command_converter.get_constant', return_value="test-model"):
            result = await command_converter.convert_command("test command")
        
        mock_create.assert_called_once_with("test-model")
        mock_client.call.assert_called_once()
        assert result == "converted command"


@pytest.mark.asyncio
async def test_call_llm_parameters(command_converter: CommandConverter, mock_llm_client):
    """Test that _call_llm passes correct parameters to LLM client."""
    mock_llm_client.call.return_value = "test response"
    
    with patch('utils.command_converter.create_llm_client', return_value=mock_llm_client):
        await command_converter._call_llm("test-model", "test command")
    
    # Verify the call parameters
    call_args = mock_llm_client.call.call_args
    assert call_args.kwargs["max_tokens"] == 200
    assert call_args.kwargs["temperature"] == 0.1
    
    messages = call_args.kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "test command"


if __name__ == "__main__":
    pytest.main([__file__])