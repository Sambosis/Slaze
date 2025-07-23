# VS Code-Style File Browser

A new VS Code-style file browser interface has been added to the Slazy Agent web UI. This interface provides a comprehensive view of project files with syntax highlighting and real-time updates.

## Features

### 🎯 **VS Code-Style Layout**
- **Left Sidebar**: File explorer with tree view showing all files in REPO_DIR
- **Main Editor Area**: Tabbed interface for viewing file contents with syntax highlighting
- **Bottom Console Panel**: Real-time display of assistant messages and tool outputs
- **Right Sidebar**: User messages and interaction history

### 📁 **File Explorer**
- Hierarchical tree view of project files and directories
- Click to expand/collapse folders
- File type icons with color coding:
  - 🐍 Python files (blue)
  - 🟨 JavaScript files (yellow)
  - 🔴 HTML files (red)
  - 🔵 CSS files (blue)
  - 📄 JSON files (yellow-green)
  - 📝 Markdown files (blue)
  - 📄 Text files (gray)

### 📝 **Code Editor**
- Syntax highlighting for multiple languages (Python, JavaScript, HTML, CSS, JSON, Markdown, etc.)
- Tabbed interface for multiple open files
- Read-only view of file contents
- Automatic language detection based on file extension

### 💬 **Real-time Communication**
- **Console Panel**: Shows assistant responses and tool execution results
- **User Messages Panel**: Displays user inputs and interactions
- WebSocket integration for real-time updates

## How to Access

### Via Web Interface
1. Start the web server: `python run.py web`
2. Navigate to the main page (usually `http://localhost:5002`)
3. Click on the "📁 File Browser" link in the navigation

### Direct Access
- Go directly to: `http://localhost:5002/browser`

### Test Server (Standalone)
For testing purposes, you can run the standalone file browser:
```bash
python3 test_server.py
```
Then visit: `http://localhost:5004/browser`

## Usage

1. **Browse Files**: Click on folders in the left sidebar to expand/collapse them
2. **Open Files**: Click on any file to open it in the main editor area
3. **Multiple Tabs**: Open multiple files - they'll appear as tabs at the top
4. **Close Tabs**: Click the × button on any tab to close it
5. **Switch Between Files**: Click on tab headers to switch between open files
6. **Monitor Activity**: Watch the console panel for real-time agent activity
7. **View Messages**: Check the right sidebar for user message history

## File Structure

The file browser displays files from the configured `REPO_DIR` directory, which is typically:
- `{PROJECT_ROOT}/repo/{PROMPT_NAME}/` for active agent sessions
- `{PROJECT_ROOT}/repo/` for the test server

## Technical Details

### API Endpoints
- `GET /browser` - Serves the file browser interface
- `GET /api/file-tree` - Returns JSON tree structure of files
- `GET /api/file-content?path=<file_path>` - Returns file content

### WebSocket Events
- `user_message` - User input messages
- `assistant_message` - Agent responses  
- `tool_result` - Tool execution results

### Security
- File access is restricted to the configured REPO_DIR
- Binary files are handled gracefully (shows "Binary file" message)
- Path traversal protection prevents access outside allowed directories

## Browser Compatibility

The interface works best in modern browsers with support for:
- CSS Grid
- WebSocket connections
- ES6 JavaScript features

Tested browsers:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Customization

The interface can be customized by modifying:
- `templates/file_browser.html` - Main template
- CSS styles within the template for appearance
- JavaScript functions for behavior

## Troubleshooting

### File Browser Not Loading
- Ensure the web server is running
- Check that templates/file_browser.html exists
- Verify Flask and Flask-SocketIO are installed

### Files Not Showing
- Check that REPO_DIR exists and contains files
- Verify permissions on the directory
- Look for errors in the server console

### Syntax Highlighting Not Working
- Ensure Prism.js CDN resources are loading
- Check browser console for JavaScript errors
- Verify file extensions are recognized

## Future Enhancements

Potential improvements for future versions:
- File editing capabilities
- File upload/download
- Search functionality
- Git integration
- Collaborative editing
- Custom themes