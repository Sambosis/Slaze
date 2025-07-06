# Tool Management Interface

A comprehensive web-based interface for managing and executing various development tools through a modern, user-friendly web application.

## Overview

The Tool Management Interface provides a webpage where users can:
- Select tools from a visual grid of available tools
- Fill in parameters for each tool through dynamic forms
- Execute tools and view results in real-time
- Chain multiple tool executions for complex workflows

## Features

### Available Tools

1. **📄 Read File** - Read contents of files with optional line range selection
2. **⚡ Run Terminal Command** - Execute terminal commands with background execution option
3. **📁 List Directory** - List contents of directories
4. **🔍 Grep Search** - Search for patterns in files using regex
5. **✏️ Edit File** - Create new files or edit existing ones
6. **🔄 Search & Replace** - Find and replace text in files
7. **🔎 File Search** - Search for files by name using fuzzy matching
8. **🗑️ Delete File** - Delete files from the filesystem
9. **🌐 Web Search** - Search the web for information (placeholder)
10. **📓 Edit Notebook** - Edit Jupyter notebook cells (placeholder)

### User Interface Features

- **Interactive Tool Grid**: Visual cards for each tool with descriptions
- **Dynamic Forms**: Tool-specific input forms with validation
- **Real-time Results**: Live display of tool execution results
- **Loading Indicators**: Visual feedback during tool execution
- **Error Handling**: Clear error messages for failed operations
- **Responsive Design**: Works on desktop and mobile devices

## Installation and Setup

### Requirements

- Python 3.7+
- Flask
- Flask-SocketIO

### Installation

1. Install dependencies:
   ```bash
   pip3 install flask flask-socketio --break-system-packages
   ```

2. Run the standalone tool manager:
   ```bash
   python3 standalone_tool_manager.py
   ```

3. Open your browser to `http://localhost:5000`

### Alternative: Integration with Existing Application

If you want to integrate with the existing Slazy Agent application:

1. Install additional dependencies:
   ```bash
   pip3 install python-dotenv openai ftfy requests rich regex --break-system-packages
   ```

2. Use the integrated version:
   ```bash
   python3 run.py web --port 5000
   ```

3. Navigate to `http://localhost:5000/tools`

## Usage Guide

### Basic Tool Execution

1. **Select a Tool**: Click on any tool card in the grid
2. **Fill Parameters**: Complete the form fields that appear
3. **Execute**: Click the "Execute" button
4. **View Results**: Results appear in the results area below

### Tool-Specific Usage

#### Read File
- **File Path**: Enter relative or absolute path to file
- **Line Range**: Optional start and end line numbers
- **Read Entire File**: Check to ignore line range

#### Run Terminal Command
- **Command**: Enter the command to execute
- **Background**: Check for long-running processes

#### List Directory
- **Directory Path**: Enter path relative to workspace

#### Grep Search
- **Search Pattern**: Enter regex pattern (escape special characters)
- **Include/Exclude**: File patterns to include/exclude
- **Case Sensitive**: Check for case-sensitive search

#### Edit File
- **File Path**: Path to file to edit/create
- **Instructions**: Description of what you're doing
- **Code Edit**: File content (use `// ... existing code ...` for partial edits)

#### Search & Replace
- **File Path**: Path to file to modify
- **Old Text**: Text to replace (include context for uniqueness)
- **New Text**: Replacement text

#### File Search
- **Search Query**: Partial filename to search for

#### Delete File
- **File Path**: Path to file to delete
- **Confirmation**: Browser will prompt for confirmation

### Advanced Features

#### Error Handling
- Failed operations show clear error messages
- Timeouts are handled gracefully
- File permission errors are reported

#### Security Features
- File operations are sandboxed to workspace
- Command execution has timeout limits
- Dangerous operations require confirmation

## Architecture

### Backend Components

1. **ToolManager Class**: Handles tool execution and parameter validation
2. **Flask Application**: Web server and routing
3. **SocketIO Integration**: Real-time communication between frontend and backend
4. **Error Handling**: Comprehensive error catching and reporting

### Frontend Components

1. **Tool Grid**: Visual representation of available tools
2. **Dynamic Forms**: JavaScript-generated forms for each tool
3. **Real-time Updates**: SocketIO-based result display
4. **User Interface**: Modern, responsive design

### Tool Implementation

Each tool is implemented as a method in the `ToolManager` class:
- Input validation and sanitization
- Error handling and logging
- Security considerations
- Result formatting

## Development

### Adding New Tools

1. Add tool method to `ToolManager` class:
   ```python
   def _new_tool(self, params):
       """Implement new tool functionality."""
       # Tool implementation here
       return result
   ```

2. Add tool to `execute_tool` method dispatch

3. Add tool card to HTML template

4. Add form handling to JavaScript

### Customization

- **Styling**: Modify CSS in `templates/tools.html`
- **Tool Behavior**: Update methods in `ToolManager` class
- **UI Layout**: Modify HTML structure in template
- **Validation**: Add parameter validation in tool methods

## Security Considerations

- **Path Traversal**: All file paths are validated and sandboxed
- **Command Injection**: Terminal commands are executed with proper isolation
- **Resource Limits**: Timeouts prevent infinite execution
- **Permission Checks**: File operations respect system permissions

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change port in `standalone_tool_manager.py`
2. **Permission Errors**: Check file permissions in workspace
3. **Command Not Found**: Ensure required tools are installed
4. **Browser Not Connecting**: Check firewall settings

### Debug Mode

The standalone version runs in debug mode by default. For production:
```python
socketio.run(app, host='0.0.0.0', port=5000, debug=False)
```

## License

This tool management interface is part of the Slazy Agent project and follows the same licensing terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your tool implementation
4. Test thoroughly
5. Submit a pull request

## Future Enhancements

- **Tool Chaining**: Execute multiple tools in sequence
- **Result Persistence**: Save tool execution history
- **User Authentication**: Multi-user support
- **Tool Marketplace**: Plugin system for custom tools
- **API Integration**: REST API for programmatic access
- **Real Web Search**: Integration with search APIs
- **Full Notebook Support**: Complete Jupyter notebook editing

## Support

For support and questions, please refer to the main Slazy Agent documentation or create an issue in the repository.