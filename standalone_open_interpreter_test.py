#!/usr/bin/env python3
"""
Standalone test for the OpenInterpreterTool functionality
"""

import sys
import os
import platform
import subprocess

# Mock classes for testing
class MockDisplay:
    def add_message(self, role, content):
        print(f"[{role}] {content}")

class MockToolResult:
    def __init__(self, output=None, error=None, tool_name=None, command=None):
        self.output = output
        self.error = error
        self.tool_name = tool_name
        self.command = command

class MockBaseTool:
    def __init__(self, input_schema=None, display=None):
        self.input_schema = input_schema
        self.display = display

class MockToolError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

# Create a simplified version of the OpenInterpreterTool for testing
class OpenInterpreterTool(MockBaseTool):
    def __init__(self, display=None):
        self.display = display
        super().__init__(input_schema=None, display=display)

    @property
    def description(self) -> str:
        return """
        A tool that uses open-interpreter's interpreter.chat() method to execute commands
        and tasks. This provides an alternative to direct bash execution with enhanced
        AI-powered command interpretation and execution.
        """

    name = "open_interpreter"
    api_type = "open_interpreter_20250124"

    async def __call__(self, task_description: str | None = None, **kwargs):
        if task_description is not None:
            if self.display is not None:
                try:
                    self.display.add_message("user", f"Executing task with open-interpreter: {task_description}")
                except Exception as e:
                    return MockToolResult(error=str(e), tool_name=self.name, command=task_description)

            return await self._execute_with_interpreter(task_description)
        raise MockToolError("no task description provided.")

    async def _execute_with_interpreter(self, task_description: str):
        """
        Execute a task using open-interpreter's interpreter.chat() method.
        """
        output = ""
        error = ""
        success = False
        cwd = None
        
        try:
            # Get the current working directory
            cwd = os.getcwd()
            
            # Try to import and use open-interpreter
            try:
                import interpreter
                
                # Configure interpreter settings
                interpreter.offline = False  # Allow online operations
                interpreter.auto_run = True  # Auto-run commands
                interpreter.verbose = False  # Reduce verbosity for tool usage
                
                # Create system information context
                system_info = self._get_system_info()
                full_task = f"{task_description}\n\nSystem Information: {system_info}"
                
                # Execute the task using interpreter.chat()
                result = interpreter.chat(full_task)
                
                # Extract output from the result
                if hasattr(result, 'messages'):
                    # Get the last assistant message which should contain the execution result
                    for message in reversed(result.messages):
                        if message.get('role') == 'assistant':
                            output = message.get('content', '')
                            break
                else:
                    output = str(result)
                
                success = True
                
            except ImportError:
                error = "open-interpreter package is not installed. Please install it with: pip install open-interpreter"
                success = False
                
        except Exception as e:
            error = str(e)
            success = False

        formatted_output = (
            f"task_description: {task_description}\n"
            f"working_directory: {cwd}\n"
            f"success: {str(success).lower()}\n"
            f"output: {output}\n"
            f"error: {error}"
        )
        print(formatted_output)
        return MockToolResult(
            output=formatted_output,
            error=error,
            tool_name=self.name,
            command=task_description,
        )

    def _get_system_info(self) -> str:
        """
        Get system information to provide context to the interpreter.
        """
        system_info = []
        
        # Basic system info
        system_info.append(f"OS: {platform.system()} {platform.release()}")
        system_info.append(f"Architecture: {platform.machine()}")
        system_info.append(f"Python: {platform.python_version()}")
        
        # Current working directory
        cwd = os.getcwd()
        system_info.append(f"Current Directory: {cwd}")
        
        # Available commands
        try:
            # Check for common commands
            commands = ['ls', 'pwd', 'python', 'python3', 'pip', 'pip3']
            available_commands = []
            for cmd in commands:
                try:
                    subprocess.run([cmd, '--version'], capture_output=True, timeout=1)
                    available_commands.append(cmd)
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
            system_info.append(f"Available Commands: {', '.join(available_commands)}")
        except Exception:
            system_info.append("Available Commands: Unable to determine")
        
        return "\n".join(system_info)

    def to_params(self) -> dict:
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": "A description of the task to be executed using open-interpreter. This should include what needs to be done and any relevant context about the system it will run on.",
                        }
                    },
                    "required": ["task_description"],
                },
            },
        }
        return params

def test_open_interpreter_tool():
    """Test the OpenInterpreterTool functionality"""
    
    print("Testing OpenInterpreterTool...")
    
    # Create an instance of the tool
    tool = OpenInterpreterTool()
    
    # Test 1: Tool properties
    print("\n=== Test 1: Tool properties ===")
    print(f"Tool name: {tool.name}")
    print(f"API type: {tool.api_type}")
    print(f"Description: {tool.description}")
    
    # Test 2: System information
    print("\n=== Test 2: System information ===")
    system_info = tool._get_system_info()
    print(f"System Info: {system_info}")
    
    # Test 3: Tool parameters
    print("\n=== Test 3: Tool parameters ===")
    params = tool.to_params()
    print(f"Tool parameters: {params}")
    
    # Test 4: Call method (will fail without open-interpreter installed)
    print("\n=== Test 4: Call method ===")
    try:
        import asyncio
        result = asyncio.run(tool(task_description="List the current directory contents"))
        print(f"Result: {result}")
    except Exception as e:
        print(f"Expected error (open-interpreter not installed): {e}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_open_interpreter_tool()