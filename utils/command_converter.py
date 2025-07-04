import logging
import platform
import os
import re
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
from config import get_constant
from .llm_client import create_llm_client

logger = logging.getLogger(__name__)

class CommandConverter:
    """
    LLM-based command converter that transforms bash commands to be appropriate
    for the current system environment.
    """
    
    def __init__(self):
        self.system_info = self._get_system_info()
        self.conversion_prompt = self._build_conversion_prompt()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Gather system information for command conversion context."""
        return {
            "os_name": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "shell": os.environ.get("SHELL", "/bin/bash"),
            "home_dir": str(Path.home()),
            "current_working_dir": str(Path.cwd()),
            "path_separator": os.pathsep,
            "file_separator": os.sep,
            "environment_vars": {
                "PATH": os.environ.get("PATH", ""),
                "USER": os.environ.get("USER", ""),
                "HOME": os.environ.get("HOME", ""),
            }
        }
    
    def _build_conversion_prompt(self) -> str:
        """Build the system prompt for command conversion."""
        os_name = self.system_info['os_name']
        
        # Build OS-specific examples and rules
        if os_name == "Windows":
            examples = """EXAMPLES:
Input: "dir"
Output: dir

Input: "dir C:\\path"
Output: dir C:\\path

Input: "find /path -type f"
Output: dir /s /b C:\\path\\* | findstr /v "\\\\\\."

Input: "ls -la"
Output: dir

Input: "echo hello"
Output: echo hello"""
            
            rules = f"""RULES:
- Keep Windows commands (dir, type, copy, etc.) as-is - do NOT convert to Linux equivalents
- If a Linux command is used, convert it to the Windows equivalent
- For file listing: use "dir" not "ls"
- For finding files: use "dir /s /b" not "find"
- Use Windows path separators (\\) when needed
- Hidden files on Windows start with . - filter them when appropriate
- Ensure the command will work on {os_name}
- Return ONLY the command, no other text"""
        else:
            examples = """EXAMPLES:
Input: "find /path -type f"
Output: find /path -type f -not -path "*/.*"

Input: "ls -la /directory"  
Output: ls -la /directory | grep -v "^\\."

Input: "dir"
Output: ls

Input: "echo hello"
Output: echo hello"""
            
            rules = f"""RULES:
- Keep Linux/Unix commands as-is when they work correctly
- If a Windows command is used, convert it to the Linux equivalent  
- For file listing: use "ls" not "dir"
- Always exclude hidden files/directories in find and ls operations
- Use Linux path separators (/) when needed
- Ensure the command will work on {os_name}
- Return ONLY the command, no other text"""
        
        return f"""You are a command converter that adapts commands for different system environments.

SYSTEM INFORMATION:
- OS: {os_name} {self.system_info['os_version']}
- Architecture: {self.system_info['architecture']}
- Shell: {self.system_info['shell']}
- Working Directory: {self.system_info['current_working_dir']}
- Path Separator: {self.system_info['path_separator']}
- File Separator: {self.system_info['file_separator']}

CONVERSION GOALS:
1. Ensure commands work properly on the current system ({os_name})
2. Filter out hidden files/directories when listing or finding files
3. Convert between Windows and Linux command equivalents as needed
4. Use appropriate flags and options for the target system
5. Handle cross-platform compatibility issues

CRITICAL OUTPUT FORMAT:
You MUST respond with ONLY the converted command, nothing else. No explanations, no markdown, no additional text.
The response should be a single line containing only the executable command.

{examples}

{rules}
"""

    async def convert_command(self, original_command: str) -> str:
        """
        Convert a command using LLM to be appropriate for the current system.
        
        Args:
            original_command: The original bash command to convert
            
        Returns:
            The converted command appropriate for the current system
        """
        try:
            # Get the model from config
            model = get_constant("MAIN_MODEL", "anthropic/claude-sonnet-4")
            
            # Prepare the conversion request
            converted_command = await self._call_llm(model, original_command)
            
            # Validate and clean the response
            cleaned_command = self._clean_response(converted_command)
            
            logger.info(f"Command converted: '{original_command}' -> '{cleaned_command}'")
            return cleaned_command
            
        except Exception as e:
            logger.warning(f"Command conversion failed for '{original_command}': {e}")
            # Fallback to original command if conversion fails
            return original_command
    
    async def _call_llm(self, model: str, command: str) -> str:
        """
        Call the LLM API to convert the command.
        
        Args:
            model: The model to use for conversion
            command: The original command
            
        Returns:
            The LLM response containing the converted command
        """
        # Prepare the messages for the LLM
        messages = [
            {
                "role": "system", 
                "content": self.conversion_prompt
            },
            {
                "role": "user", 
                "content": command
            }
        ]
        
        # Create LLM client and call it
        client = create_llm_client(model)
        return await client.call(
            messages=messages,
            max_tokens=200,  # Keep response short
            temperature=0.1  # Low temperature for consistent output
        )
    
    def _clean_response(self, response: str) -> str:
        """
        Clean the LLM response to extract just the command.
        
        Args:
            response: The raw LLM response
            
        Returns:
            The cleaned command string
        """
        # Remove any markdown code blocks
        response = re.sub(r'^```.*?\n|```$', '', response, flags=re.MULTILINE)
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # Split by lines and take the first non-empty line
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if not lines:
            raise ValueError("Empty response from LLM")
        
        command = lines[0]
        
        # Basic validation - ensure it looks like a command
        if not command or len(command) > 1000:  # Reasonable length limit
            raise ValueError(f"Invalid command format: {command}")
        
        return command

# Global instance for reuse
_converter_instance: Optional[CommandConverter] = None

async def convert_command_for_system(original_command: str) -> str:
    """
    Convert a bash command to be appropriate for the current system.
    
    Args:
        original_command: The original bash command
        
    Returns:
        The converted command appropriate for the current system
    """
    global _converter_instance
    
    if _converter_instance is None:
        _converter_instance = CommandConverter()
    
    return await _converter_instance.convert_command(original_command)