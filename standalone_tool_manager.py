#!/usr/bin/env python3

"""
Standalone Tool Management Interface
A simple Flask web application for managing and executing tools.
"""

import os
import subprocess
import logging
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ToolManager:
    """Handle tool execution for the web interface."""
    
    def __init__(self, workspace_path="/workspace"):
        self.workspace_path = workspace_path
    
    def execute_tool(self, tool_name, params):
        """Execute a tool with given parameters."""
        try:
            if tool_name == 'read_file':
                return self._read_file(params)
            elif tool_name == 'run_terminal_cmd':
                return self._run_terminal_cmd(params)
            elif tool_name == 'list_dir':
                return self._list_dir(params)
            elif tool_name == 'grep_search':
                return self._grep_search(params)
            elif tool_name == 'edit_file':
                return self._edit_file(params)
            elif tool_name == 'search_replace':
                return self._search_replace(params)
            elif tool_name == 'file_search':
                return self._file_search(params)
            elif tool_name == 'delete_file':
                return self._delete_file(params)
            elif tool_name == 'web_search':
                return self._web_search(params)
            elif tool_name == 'edit_notebook':
                return self._edit_notebook(params)
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            logging.error(f"Error executing {tool_name}: {str(e)}")
            return f"Error executing {tool_name}: {str(e)}"
    
    def _read_file(self, params):
        """Read file contents."""
        file_path = params.get('target_file', '')
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.workspace_path, file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if params.get('should_read_entire_file', False):
                return content
            else:
                lines = content.splitlines()
                start_line = params.get('start_line_one_indexed', 1) - 1
                end_line = params.get('end_line_one_indexed_inclusive', len(lines))
                return '\n'.join(lines[start_line:end_line])
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    def _run_terminal_cmd(self, params):
        """Execute terminal command."""
        command = params.get('command', '')
        is_background = params.get('is_background', False)
        
        try:
            if is_background:
                # For background processes, start and return immediately
                process = subprocess.Popen(
                    command,
                    shell=True,
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                return f"Command started in background with PID: {process.pid}"
            else:
                # For regular commands, wait for completion
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=self.workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = result.stdout
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                return output
        except subprocess.TimeoutExpired:
            return "Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _list_dir(self, params):
        """List directory contents."""
        dir_path = params.get('relative_workspace_path', '.')
        if not os.path.isabs(dir_path):
            dir_path = os.path.join(self.workspace_path, dir_path)
        
        try:
            items = []
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    items.append(f"[dir]  {item}/")
                else:
                    size = os.path.getsize(item_path)
                    items.append(f"[file] {item} ({size} bytes)")
            return '\n'.join(items)
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    def _grep_search(self, params):
        """Search for patterns in files."""
        query = params.get('query', '')
        include_pattern = params.get('include_pattern', '')
        exclude_pattern = params.get('exclude_pattern', '')
        case_sensitive = params.get('case_sensitive', False)
        
        try:
            cmd = ['grep', '-r', '-n']
            if not case_sensitive:
                cmd.append('-i')
            
            if include_pattern:
                cmd.extend(['--include', include_pattern])
            if exclude_pattern:
                cmd.extend(['--exclude', exclude_pattern])
            
            cmd.extend([query, self.workspace_path])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                return result.stdout
            elif result.returncode == 1:
                return "No matches found"
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error executing grep: {str(e)}"
    
    def _edit_file(self, params):
        """Edit or create a file."""
        file_path = params.get('target_file', '')
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.workspace_path, file_path)
        
        content = params.get('code_edit', '')
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return f"File successfully written to {file_path}"
        except Exception as e:
            return f"Error editing file: {str(e)}"
    
    def _search_replace(self, params):
        """Search and replace text in file."""
        file_path = params.get('file_path', '')
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.workspace_path, file_path)
        
        old_string = params.get('old_string', '')
        new_string = params.get('new_string', '')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if old_string in content:
                new_content = content.replace(old_string, new_string, 1)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return f"Successfully replaced text in {file_path}"
            else:
                return f"Old string not found in {file_path}"
        except Exception as e:
            return f"Error with search and replace: {str(e)}"
    
    def _file_search(self, params):
        """Search for files by name."""
        query = params.get('query', '')
        
        try:
            cmd = ['find', self.workspace_path, '-name', f'*{query}*', '-type', 'f']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error searching for files: {str(e)}"
    
    def _delete_file(self, params):
        """Delete a file."""
        file_path = params.get('target_file', '')
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.workspace_path, file_path)
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return f"File {file_path} deleted successfully"
            else:
                return f"File {file_path} does not exist"
        except Exception as e:
            return f"Error deleting file: {str(e)}"
    
    def _web_search(self, params):
        """Search the web (placeholder)."""
        search_term = params.get('search_term', '')
        return f"Web search functionality not implemented. Search term: {search_term}"
    
    def _edit_notebook(self, params):
        """Edit notebook cell (placeholder)."""
        notebook_path = params.get('target_notebook', '')
        return f"Notebook editing functionality not implemented. Notebook: {notebook_path}"

# Create Flask app
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='threading')

# Create tool manager instance
tool_manager = ToolManager()

@app.route('/')
def index():
    """Main page."""
    return render_template('tools.html')

@app.route('/tools')
def tools():
    """Tools page."""
    return render_template('tools.html')

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logging.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logging.info('Client disconnected')

@socketio.on('execute_tool')
def handle_execute_tool(data):
    """Handle tool execution request."""
    tool_name = data.get('tool', '')
    params = data.get('params', {})
    logging.info(f'Executing tool: {tool_name}')
    
    try:
        result = tool_manager.execute_tool(tool_name, params)
        emit('tool_result', {'result': result})
    except Exception as e:
        logging.error(f'Error executing tool {tool_name}: {str(e)}')
        emit('tool_error', {'error': str(e)})

if __name__ == '__main__':
    print("Starting Standalone Tool Management Interface...")
    print("Web interface will be available at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)