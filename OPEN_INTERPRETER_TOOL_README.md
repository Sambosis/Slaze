# OpenInterpreterTool

## Overview

The `OpenInterpreterTool` is an alternative to the traditional bash tool that uses the `open-interpreter` library's `interpreter.chat()` method to execute commands and tasks. This provides enhanced AI-powered command interpretation and execution capabilities.

## Features

- **AI-Powered Execution**: Uses open-interpreter's chat interface for intelligent command execution
- **System Context Awareness**: Automatically provides system information to the interpreter
- **Enhanced Error Handling**: Better error reporting and fallback mechanisms
- **Flexible Task Descriptions**: Accepts natural language task descriptions instead of raw commands

## Installation

### Prerequisites

1. Install open-interpreter:
```bash
pip install open-interpreter --break-system-packages
```

Note: The `--break-system-packages` flag may be required on some systems due to Python environment restrictions.

### Dependencies

The tool requires the following Python packages:
- `open-interpreter` - The main interpreter library
- Standard library modules: `os`, `platform`, `subprocess`, `pathlib`

## Usage

### Basic Usage

```python
from tools.open_interpreter_tool import OpenInterpreterTool

# Create an instance of the tool
tool = OpenInterpreterTool()

# Execute a task
result = await tool(task_description="List all files in the current directory")
```

### Task Description Format

The tool accepts natural language descriptions of tasks to be executed. Examples:

- `"List all Python files in the current directory"`
- `"Create a new directory called 'test' and navigate to it"`
- `"Install the requests package using pip"`
- `"Check the system memory usage and disk space"`

### System Information

The tool automatically provides system context to the interpreter, including:
- Operating system and version
- Architecture
- Python version
- Current working directory
- Available commands

## API Reference

### Class: OpenInterpreterTool

#### Constructor
```python
OpenInterpreterTool(display=None)
```

**Parameters:**
- `display`: Optional display interface for logging (WebUI or AgentDisplayConsole)

#### Methods

##### `__call__(task_description)`
Executes a task using open-interpreter.

**Parameters:**
- `task_description` (str): Natural language description of the task to execute

**Returns:**
- `ToolResult`: Object containing execution results

##### `_get_system_info()`
Gathers system information for context.

**Returns:**
- `str`: Formatted system information

##### `to_params()`
Returns the tool's API parameters for function calling.

**Returns:**
- `dict`: Tool parameters in OpenAI function calling format

## Comparison with BashTool

| Feature | BashTool | OpenInterpreterTool |
|---------|----------|-------------------|
| Command Input | Raw shell commands | Natural language descriptions |
| Execution | Direct subprocess | AI-powered interpretation |
| Error Handling | Basic error capture | Enhanced error reporting |
| System Context | Limited | Comprehensive system info |
| Flexibility | Command-specific | Task-oriented |

## Example Use Cases

### 1. File Operations
```python
result = await tool(task_description="Create a backup of all .txt files in the current directory")
```

### 2. System Administration
```python
result = await tool(task_description="Check system resources and list running processes")
```

### 3. Package Management
```python
result = await tool(task_description="Install numpy and pandas packages")
```

### 4. Development Tasks
```python
result = await tool(task_description="Set up a new Python virtual environment and install dependencies")
```

## Error Handling

The tool handles various error scenarios:

1. **Missing open-interpreter**: Provides clear installation instructions
2. **Import errors**: Graceful fallback with error messages
3. **Execution failures**: Detailed error reporting
4. **System information gathering**: Continues with partial information

## Integration

The tool is integrated into the existing tool collection and can be used alongside other tools:

```python
from tools import OpenInterpreterTool, BashTool

# Use both tools as needed
bash_tool = BashTool()
interpreter_tool = OpenInterpreterTool()

# Choose the appropriate tool for the task
if task_requires_raw_commands:
    result = await bash_tool(command="ls -la")
else:
    result = await interpreter_tool(task_description="List files with detailed information")
```

## Testing

Run the standalone test to verify functionality:

```bash
python3 standalone_open_interpreter_test.py
```

This will test:
- Tool properties and configuration
- System information gathering
- API parameter generation
- Basic execution (with expected failure when open-interpreter is not installed)

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure open-interpreter is installed
2. **Permission Errors**: Check system permissions for command execution
3. **Network Issues**: Some operations may require internet access
4. **Python Version**: Ensure compatibility with your Python version

### Debug Mode

Enable verbose logging by modifying the tool configuration:

```python
# In the tool implementation
interpreter.verbose = True  # Enable verbose output
```

## Future Enhancements

Potential improvements for future versions:

1. **Custom Interpreter Configuration**: Allow users to configure interpreter settings
2. **Result Parsing**: Enhanced parsing of interpreter output
3. **Caching**: Cache frequently used commands and results
4. **Plugin System**: Support for custom interpreter plugins
5. **Batch Operations**: Execute multiple tasks in sequence

## Contributing

To contribute to the OpenInterpreterTool:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This tool is part of the existing project and follows the same license terms.