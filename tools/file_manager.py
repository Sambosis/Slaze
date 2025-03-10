import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from .base import BaseAnthropicTool, ToolResult, ToolError
from utils.file_logger import log_file_operation, extract_code_skeleton, get_all_current_code, get_all_current_skeleton
from utils.docker_service import DockerService

class FileManagerTool(BaseAnthropicTool):
    name = "file_manager"
    description = "A tool for managing files, including viewing, creating, editing, and listing files."

    def __init__(self, display=None):
        super().__init__(input_schema=None, display=display)
        self.display = display
        self.docker_service = DockerService()
        self.files_to_create = {}  # Dictionary to store files to be created and their status

    def to_params(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": "custom",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "edit", "list", "replace", "skeleton", "mark_created", "edit_section", "replace_function"],
                        "description": "Command to execute: view, create, edit, list, replace, skeleton, mark_created, edit_section, replace_function"
                    },
                    "path": {
                        "type": "string",
                        "description": "Path of the file or directory"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for creating or editing a file"
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Old string to be replaced"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "New string to replace the old string"
                    },
                    "section": {
                        "type": "string",
                        "description": "Section of the file to edit"
                    },
                    "function_name": {
                        "type": "string",
                        "description": "Function or method name to replace code"
                    }
                },
                "required": ["command", "path"]
            }
        }

    def view_file(self, path: str) -> ToolResult:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            return ToolResult(output=content, tool_name=self.name, command="view")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="view")

    def create_file(self, path: str, content: str) -> ToolResult:
        try:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(content)
            log_file_operation(path, "create", content)
            self.files_to_create[path] = False  # Mark the file as not created yet
            return ToolResult(output="File created successfully", tool_name=self.name, command="create")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="create")

    def edit_file(self, path: str, content: str) -> ToolResult:
        try:
            with open(path, 'w', encoding='utf-8') as file:
                file.write(content)
            log_file_operation(path, "edit", content)
            return ToolResult(output="File edited successfully", tool_name=self.name, command="edit")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="edit")

    def list_files(self, path: str) -> ToolResult:
        try:
            files = [str(p) for p in Path(path).rglob('*') if p.is_file()]
            return ToolResult(output="\n".join(files), tool_name=self.name, command="list")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="list")

    def replace_code(self, path: str, old_str: str, new_str: str) -> ToolResult:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            content = content.replace(old_str, new_str)
            with open(path, 'w', encoding='utf-8') as file:
                file.write(content)
            log_file_operation(path, "replace", content)
            return ToolResult(output="Code replaced successfully", tool_name=self.name, command="replace")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="replace")

    def get_skeleton(self, path: str) -> ToolResult:
        try:
            skeleton = extract_code_skeleton(path)
            return ToolResult(output=skeleton, tool_name=self.name, command="skeleton")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="skeleton")

    def mark_file_created(self, path: str) -> ToolResult:
        if path in self.files_to_create:
            self.files_to_create[path] = True  # Mark the file as created
            return ToolResult(output="File marked as created", tool_name=self.name, command="mark_created")
        else:
            return ToolResult(error="File not found in the list of files to be created", tool_name=self.name, command="mark_created")

    def edit_section(self, path: str, section: str, new_content: str) -> ToolResult:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            start_index = content.find(section)
            if start_index == -1:
                return ToolResult(error="Section not found", tool_name=self.name, command="edit_section")
            end_index = start_index + len(section)
            new_content = content[:start_index] + new_content + content[end_index:]
            with open(path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            log_file_operation(path, "edit_section", new_content)
            return ToolResult(output="Section edited successfully", tool_name=self.name, command="edit_section")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="edit_section")

    def replace_function(self, path: str, function_name: str, new_code: str) -> ToolResult:
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            function_start = content.find(f"def {function_name}(")
            if function_start == -1:
                return ToolResult(error="Function not found", tool_name=self.name, command="replace_function")
            function_end = content.find("def ", function_start + 1)
            if function_end == -1:
                function_end = len(content)
            new_content = content[:function_start] + new_code + content[function_end:]
            with open(path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            log_file_operation(path, "replace_function", new_content)
            return ToolResult(output="Function replaced successfully", tool_name=self.name, command="replace_function")
        except Exception as e:
            return ToolResult(error=str(e), tool_name=self.name, command="replace_function")

    def __call__(self, *, command: str, path: str, content: str = None, old_str: str = None, new_str: str = None, section: str = None, function_name: str = None, **kwargs) -> ToolResult:
        if command == "view":
            return self.view_file(path)
        elif command == "create":
            return self.create_file(path, content)
        elif command == "edit":
            return self.edit_file(path, content)
        elif command == "list":
            return self.list_files(path)
        elif command == "replace":
            return self.replace_code(path, old_str, new_str)
        elif command == "skeleton":
            return self.get_skeleton(path)
        elif command == "mark_created":
            return self.mark_file_created(path)
        elif command == "edit_section":
            return self.edit_section(path, section, content)
        elif command == "replace_function":
            return self.replace_function(path, function_name, content)
        else:
            return ToolResult(error=f"Unknown command: {command}", tool_name=self.name, command=command)
