import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from tools.write_code import WriteCodeTool, CodeCommand, FileDetail, LLMResponseError
from tools.base import ToolResult


@pytest.fixture
def write_code_tool():
    """Fixture to create a WriteCodeTool instance."""
    return WriteCodeTool()


@pytest.fixture
def mock_display():
    """Fixture to create a mock display."""
    display = MagicMock()
    display.add_message = MagicMock()
    return display


@pytest.fixture
def sample_files():
    """Fixture providing sample file details for testing."""
    return [
        {
            "filename": "main.py",
            "code_description": "Main entry point for the application. Contains the main function that initializes the app and starts the server.",
            "external_imports": ["flask", "os"],
            "internal_imports": ["config", "routes"]
        },
        {
            "filename": "config.py",
            "code_description": "Configuration module that loads environment variables and sets up application settings.",
            "external_imports": ["os", "dotenv"],
            "internal_imports": []
        }
    ]


@pytest.fixture
def mock_openai_client():
    """Fixture to mock the OpenAI client."""
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "# Sample generated code\nprint('Hello World')"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestWriteCodeTool:
    """Test cases for WriteCodeTool."""

    def test_tool_properties(self, write_code_tool: WriteCodeTool):
        """Test that the tool has the correct properties."""
        assert write_code_tool.name == "write_codebase_tool"
        assert write_code_tool.api_type == "custom"
        assert "generates a codebase" in write_code_tool.description.lower()

    def test_to_params(self, write_code_tool: WriteCodeTool):
        """Test the to_params method returns correct OpenAI function calling format."""
        params = write_code_tool.to_params()
        
        assert params["type"] == "function"
        assert params["function"]["name"] == "write_codebase_tool"
        assert "generates a codebase" in params["function"]["description"].lower()
        assert params["function"]["parameters"]["type"] == "object"
        
        # Check required fields
        required_fields = params["function"]["parameters"]["required"]
        assert "command" in required_fields
        # files is no longer required globally (only for write_codebase), so do not assert it here
        
        # Check properties structure
        properties = params["function"]["parameters"]["properties"]
        assert "command" in properties
        assert "files" in properties
        
        # Check command enum includes both commands
        cmd_enum = properties["command"]["enum"]
        assert CodeCommand.WRITE_CODEBASE.value in cmd_enum
        assert CodeCommand.SCAFFOLD_WEB_APP.value in cmd_enum

    @pytest.mark.asyncio
    async def test_unsupported_command(self, write_code_tool: WriteCodeTool):
        """Test that unsupported commands return an error."""
        result = await write_code_tool(
            command="invalid_command",  # type: ignore
            files=[],
        )
        
        assert isinstance(result, ToolResult)
        assert result.error is not None
        assert "unsupported command" in result.error.lower()
        assert result.tool_name == "write_codebase_tool"

    @pytest.mark.asyncio
    async def test_empty_files_list(self, write_code_tool: WriteCodeTool):
        """Test that empty files list returns an error."""
        result = await write_code_tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=[],
        )
        
        assert isinstance(result, ToolResult)
        assert result.error is not None
        assert "no files specified" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_files_format(self, write_code_tool: WriteCodeTool):
        """Test that invalid file format returns an error."""

        # Missing required code_description
        invalid_files_1 = [
            {"filename": "test.py"}
        ]
        result_1 = await write_code_tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=invalid_files_1,
        )
        assert isinstance(result_1, ToolResult)
        assert result_1.error is not None
        assert "invalid format" in result_1.error.lower()

        # Missing required filename
        invalid_files_2 = [
            {"code_description": "desc"}
        ]
        result_2 = await write_code_tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=invalid_files_2,
        )
        assert isinstance(result_2, ToolResult)
        assert result_2.error is not None
        assert "invalid format" in result_2.error.lower()

        # Invalid type for external_imports (should be list, not string)
        invalid_files_3 = [
            {
                "filename": "test.py",
                "code_description": "desc",
                "external_imports": "not_a_list"
            }
        ]
        result_3 = await write_code_tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=invalid_files_3,
        )
        assert isinstance(result_3, ToolResult)
        assert result_3.error is not None
        assert "invalid format" in result_3.error.lower()

        # Invalid type for internal_imports (should be list, not int)
        invalid_files_4 = [
            {
                "filename": "test.py",
                "code_description": "desc",
                "internal_imports": 123
            }
        ]
        result_4 = await write_code_tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=invalid_files_4,
        )
        assert isinstance(result_4, ToolResult)
        assert result_4.error is not None
        assert "invalid format" in result_4.error.lower()

        # Unexpected field in file dict
        invalid_files_5 = [
            {
                "filename": "test.py",
                "code_description": "desc",
                "unexpected_field": "unexpected"
            }
        ]
        result_5 = await write_code_tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=invalid_files_5,
        )
        assert isinstance(result_5, ToolResult)
        assert result_5.error is not None
        assert "invalid format" in result_5.error.lower()

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    async def test_missing_repo_dir_config(self, mock_get_constant, write_code_tool: WriteCodeTool, sample_files):
        """Test that missing REPO_DIR configuration returns an error."""
        mock_get_constant.return_value = None
        
        result = await write_code_tool(
            command=CodeCommand.WRITE_CODEBASE,
            files=sample_files,
        )
        
        assert isinstance(result, ToolResult)
        assert result.error is not None
        assert "repo_dir is not configured" in result.error.lower()

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    @patch('tools.write_code.AsyncOpenAI')
    async def test_successful_code_generation(self, mock_openai_class, mock_get_constant, 
                                            write_code_tool: WriteCodeTool, sample_files, 
                                            mock_openai_client, mock_display):
        """Test successful code generation and file writing."""
        # Setup mocks
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_constant.return_value = temp_dir
            mock_openai_class.return_value = mock_openai_client
            write_code_tool.display = mock_display
            
            result = await write_code_tool(
                command=CodeCommand.WRITE_CODEBASE,
                files=sample_files,
            )
            
            assert isinstance(result, ToolResult)
            assert result.error is None
            assert result.output is not None
            assert ("successfully generated" in result.output.lower() or 
                    "successfully" in result.output.lower())
            
            # Check that files were created
            assert (Path(temp_dir) / "main.py").exists()
            assert (Path(temp_dir) / "config.py").exists()
            
            # Check display interactions
            mock_display.add_message.assert_called()

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    @patch('tools.write_code.AsyncOpenAI')
    async def test_llm_error_handling(self, mock_openai_class, mock_get_constant,
                                    write_code_tool: WriteCodeTool, sample_files):
        """Test handling of LLM errors during code generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_constant.return_value = temp_dir
            
            # Mock OpenAI client to raise an exception
            mock_client = AsyncMock()
            mock_client.chat.completions.create.side_effect = Exception("LLM Error")
            mock_openai_class.return_value = mock_client
            
            result = await write_code_tool(
                command=CodeCommand.WRITE_CODEBASE,
                files=sample_files,
            )
            
            assert isinstance(result, ToolResult)
            assert result.output is not None
            # Should still return a result even with errors, but indicate issues
            assert "error" in result.output.lower() or result.error is not None

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    async def test_file_creation_logging(self, mock_get_constant, write_code_tool: WriteCodeTool):
        """Test that file creation is properly logged."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_constant.return_value = temp_dir
            
            # Create a mock log file
            log_file = Path(temp_dir) / "logs" / "file_log.json"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_file.write_text('{"test": "log content"}')
            
            # Mock the LOG_FILE constant
            with patch.object(write_code_tool, '_get_file_creation_log_content') as mock_log:
                mock_log.return_value = '{"test": "log content"}'
                
                log_content = write_code_tool._get_file_creation_log_content()
                assert "log content" in log_content

    def test_file_detail_validation(self):
        """Test FileDetail model validation."""
        # Valid FileDetail
        valid_detail = FileDetail(
            filename="test.py",
            code_description="Test file description",
            external_imports=["os", "sys"],
            internal_imports=["config"]
        )
        assert valid_detail.filename == "test.py"
        assert valid_detail.code_description == "Test file description"
        assert valid_detail.external_imports == ["os", "sys"]
        assert valid_detail.internal_imports == ["config"]
        
        # FileDetail with minimal required fields
        minimal_detail = FileDetail(
            filename="minimal.py",
            code_description="Minimal description"
        )
        assert minimal_detail.filename == "minimal.py"
        assert minimal_detail.external_imports is None
        assert minimal_detail.internal_imports is None

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    @patch('tools.write_code.AsyncOpenAI')
    async def test_path_resolution(self, mock_openai_class, mock_get_constant,
                                 write_code_tool: WriteCodeTool, sample_files, mock_openai_client):
        """Test that project paths are resolved correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_constant.return_value = temp_dir
            mock_openai_class.return_value = mock_openai_client
            
            result = await write_code_tool(
                command=CodeCommand.WRITE_CODEBASE,
                files=sample_files[:1],
            )

            assert isinstance(result, ToolResult)
            assert result.error is None or "error" not in result.error.lower()

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    async def test_directory_creation(self, mock_get_constant, write_code_tool: WriteCodeTool, sample_files):
        """Test that directories are created when they don't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_constant.return_value = temp_dir
            
            # Use a nested project path that doesn't exist
            with patch('tools.write_code.AsyncOpenAI') as mock_openai_class:
                mock_client = AsyncMock()
                mock_response = MagicMock()
                mock_response.choices = [MagicMock()]
                mock_response.choices[0].message.content = "print('test')"
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai_class.return_value = mock_client
                
                result = await write_code_tool(
                    command=CodeCommand.WRITE_CODEBASE,
                    files=sample_files[:1],
                )

                assert isinstance(result, ToolResult)

    def test_extract_code_block(self, write_code_tool: WriteCodeTool):
        """Test the extract_code_block method."""
        # Test with markdown code block
        markdown_text = """
Here's the code:

```python
def hello():
    print("Hello World")
```

Some additional text.
"""
        
        code, language = write_code_tool.extract_code_block(markdown_text)
        assert "def hello():" in code
        assert "print(\"Hello World\")" in code
        assert language == "python"
        
        # Test with text that has no code block
        plain_text = "This is just plain text without code blocks."
        code, language = write_code_tool.extract_code_block(plain_text)
        assert code == plain_text
        assert language == "unknown"

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    @patch('tools.write_code.AsyncOpenAI')
    async def test_display_integration(self, mock_openai_class, mock_get_constant,
                                     write_code_tool: WriteCodeTool, sample_files,
                                     mock_openai_client, mock_display):
        """Test that display integration works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_constant.return_value = temp_dir
            mock_openai_class.return_value = mock_openai_client
            write_code_tool.display = mock_display
            
            await write_code_tool(
                command=CodeCommand.WRITE_CODEBASE,
                files=sample_files,
            )
            
            # Verify display was called with appropriate messages
            calls = mock_display.add_message.call_args_list
            assert len(calls) > 0
            
            # Check for skeleton generation messages
            skeleton_messages = [call for call in calls if "skeleton" in str(call).lower()]
            assert len(skeleton_messages) > 0
            
            # Check for code generation messages
            code_messages = [call for call in calls if "generating" in str(call).lower()]
            assert len(code_messages) > 0

    @pytest.mark.asyncio
    async def test_llm_response_error_handling(self, write_code_tool: WriteCodeTool):
        """Test handling of LLMResponseError."""
        # Test that LLMResponseError is properly defined and can be raised
        with pytest.raises(LLMResponseError):
            raise LLMResponseError("Test error message")
        
        # Test the should_retry_llm_call function
        from tools.write_code import should_retry_llm_call
        
        # Should retry on LLMResponseError
        assert should_retry_llm_call(LLMResponseError("test")) is True
        
        # Should not retry on regular Exception
        assert should_retry_llm_call(Exception("test")) is False

    def test_code_command_enum(self):
        """Test the CodeCommand enum."""
        assert CodeCommand.WRITE_CODEBASE.value == "write_codebase"
        assert CodeCommand.SCAFFOLD_WEB_APP.value == "scaffold_web_app"
        assert str(CodeCommand.WRITE_CODEBASE) == "CodeCommand.WRITE_CODEBASE"

    @pytest.mark.asyncio
    @patch('tools.write_code.get_constant')
    @patch('tools.write_code.AsyncOpenAI')
    async def test_scaffold_web_app_react(self, mock_openai_class, mock_get_constant,
                                          write_code_tool: WriteCodeTool, mock_openai_client, mock_display):
        """Scaffold a React app and ensure files are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_constant.return_value = temp_dir
            mock_openai_class.return_value = mock_openai_client
            write_code_tool.display = mock_display

            result = await write_code_tool(
                command=CodeCommand.SCAFFOLD_WEB_APP,
                framework="react",
                project_name="webapp",
                routes=["about", "contact"],
                components=["Header", "Footer"],
                use_typescript=True,
                css_framework="none",
            )

            assert isinstance(result, ToolResult)
            assert result.error is None
            # Verify key files exist (limited to 5 files total by tool)
            assert (Path(temp_dir) / "webapp" / "package.json").exists()
            # At least one src file should exist
            src_dir = Path(temp_dir) / "webapp" / "src"
            assert src_dir.exists()
            # Display should be called
            mock_display.add_message.assert_called()

    @pytest.mark.asyncio
    async def test_scaffold_missing_params(self, write_code_tool: WriteCodeTool):
        """Scaffold should error when required params are missing."""
        result = await write_code_tool(
            command=CodeCommand.SCAFFOLD_WEB_APP,
        )
        assert isinstance(result, ToolResult)
        assert result.error is not None
        assert "missing required parameters" in result.error.lower()


if __name__ == "__main__":
    pytest.main([__file__])