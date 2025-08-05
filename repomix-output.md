This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where line numbers have been added.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Line numbers have been added to the beginning of each line
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
__init__.py
agent_display_console.py
command_converter.py
context_helpers.py
file_logger.py
llm_client.py
output_manager.py
web_ui.py
```

# Files

## File: __init__.py
````python
1: """Utility functions and classes used throughout Slaze."""
2: 
3: __all__ = []
````

## File: agent_display_console.py
````python
  1: import asyncio
  2: import json
  3: import os
  4: import sys
  5: from rich.console import Console
  6: from rich.panel import Panel
  7: from rich.prompt import Prompt, Confirm, IntPrompt
  8: from rich.syntax import Syntax
  9: from pathlib import Path
 10: from config import (
 11:     PROMPTS_DIR,
 12:     get_constant,
 13:     set_constant,
 14:     set_prompt_name,
 15:     write_constants_to_file,
 16: )
 17: 
 18: class AgentDisplayConsole:
 19:     def __init__(self):
 20:         # Configure console for Windows Unicode support
 21:         if os.name == 'nt':  # Windows
 22:             # Set environment variables for UTF-8 support
 23:             os.environ['PYTHONIOENCODING'] = 'utf-8'
 24:             # Try to set Windows console to UTF-8 mode
 25:             try:
 26:                 # Enable UTF-8 mode on Windows
 27:                 os.system('chcp 65001 > nul')
 28:                 # Reconfigure stdout/stderr for UTF-8 (Python 3.7+)
 29:                 if hasattr(sys.stdout, 'reconfigure') and callable(getattr(sys.stdout, 'reconfigure', None)):
 30:                     sys.stdout.reconfigure(encoding='utf-8', errors='replace')
 31:                 if hasattr(sys.stderr, 'reconfigure') and callable(getattr(sys.stderr, 'reconfigure', None)):
 32:                     sys.stderr.reconfigure(encoding='utf-8', errors='replace')
 33:             except Exception:
 34:                 pass
 35:         
 36:         # Initialize Rich console with safe settings for Windows
 37:         self.console = Console(
 38:             force_terminal=True,
 39:             legacy_windows=False,
 40:             file=sys.stdout,
 41:             width=120,
 42:             safe_box=True,
 43:             highlight=False
 44:         )
 45: 
 46:     def add_message(self, msg_type, content):
 47:         # Use safer characters for Windows compatibility
 48:         if msg_type == "user":
 49:             self.console.print(f"[User]: {content}")
 50:         elif msg_type == "assistant":
 51:             self.console.print(f"[Assistant]: {content}")
 52:         elif msg_type == "tool":
 53:             self.console.print(f"[Tool]: {content}")
 54:         else:
 55:             self.console.print(f"[{msg_type}]: {content}")
 56: 
 57:     async def wait_for_user_input(self, prompt_message=">> Your input: "):
 58:         return await asyncio.to_thread(input, prompt_message)
 59: 
 60:     async def select_prompt_console(self):
 61:         self.console.print("--- Select a Prompt ---")
 62:         if not PROMPTS_DIR.exists():
 63:             self.console.print(f"[bold yellow]Warning: Prompts directory '{PROMPTS_DIR}' not found. Creating it now.[/bold yellow]")
 64:             try:
 65:                 PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
 66:                 self.console.print(f"[green]Successfully created prompts directory: {PROMPTS_DIR}[/green]")
 67:             except Exception as e:
 68:                 self.console.print(f"[bold red]Error creating prompts directory {PROMPTS_DIR}: {e}[/bold red]")
 69:                 return "Default task due to directory creation error."
 70: 
 71:         options = {}
 72:         prompt_files = sorted([f for f in PROMPTS_DIR.iterdir() if f.is_file() and f.suffix == '.md'])
 73: 
 74:         prompt_lines = []
 75:         for i, prompt_file in enumerate(prompt_files):
 76:             options[str(i + 1)] = prompt_file
 77:             prompt_lines.append(f"{i + 1}. {prompt_file.name}")
 78:         
 79:         if prompt_lines:
 80:             self.console.print("\n".join(prompt_lines))
 81: 
 82:         create_new_option_num = len(options) + 1
 83:         self.console.print(f"{create_new_option_num}. Create a new prompt")
 84: 
 85:         choice = IntPrompt.ask("Enter your choice", choices=[str(i) for i in range(1, create_new_option_num + 1)])
 86: 
 87:         if choice != create_new_option_num:
 88:             prompt_path = options[str(choice)]
 89:             task = prompt_path.read_text(encoding="utf-8")
 90:             prompt_name = prompt_path.stem
 91:             self.console.print(
 92:                 Panel(
 93:                     Syntax(task, "markdown", theme="dracula", line_numbers=True),
 94:                     title="Current Prompt Content",
 95:                 )
 96:             )
 97:             if Confirm.ask(f"Do you want to edit '{prompt_path.name}'?", default=False):
 98:                 new_lines = []
 99:                 while True:
100:                     try:
101:                         line = await self.wait_for_user_input("")
102:                         new_lines.append(line)
103:                     except EOFError:
104:                         break
105:                 if new_lines:
106:                     task = "\n".join(new_lines)
107:                     prompt_path.write_text(task, encoding="utf-8")
108:         else:
109:             new_filename_input = Prompt.ask("Enter a filename for the new prompt", default="custom_prompt")
110:             filename_stem = new_filename_input.strip().replace(" ", "_").replace(".md", "")
111:             prompt_name = filename_stem
112:             new_prompt_path = PROMPTS_DIR / f"{filename_stem}.md"
113:             new_prompt_lines = []
114:             while True:
115:                 try:
116:                     line = await self.wait_for_user_input("")
117:                     new_prompt_lines.append(line)
118:                 except EOFError:
119:                     break
120:             if new_prompt_lines:
121:                 task = "\n".join(new_prompt_lines)
122:                 new_prompt_path.write_text(task, encoding="utf-8")
123:             else:
124:                 task = ""
125: 
126:         # Configure repository directory for this prompt
127:         base_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo"
128:         repo_dir = base_repo_dir / prompt_name
129:         repo_dir.mkdir(parents=True, exist_ok=True)
130:         set_prompt_name(prompt_name)
131:         set_constant("REPO_DIR", repo_dir)
132:         write_constants_to_file()
133: 
134:         return task
135: 
136:     async def confirm_tool_call(self, tool_name: str, args: dict, schema: dict) -> dict | None:
137:         """Display tool call parameters and allow the user to edit them."""
138:         self.console.print(Panel(f"Tool call: [bold]{tool_name}[/bold]", title="Confirm Tool"))
139:         updated_args = dict(args)
140:         properties = schema.get("properties", {}) if schema else {}
141: 
142:         for param in properties:
143:             current_val = args.get(param, "")
144:             default_str = str(current_val) if current_val is not None else ""
145:             user_val = Prompt.ask(param, default=default_str)
146:             if user_val != default_str:
147:                 pinfo = properties.get(param, {})
148:                 if pinfo.get("type") == "integer":
149:                     try:
150:                         updated_args[param] = int(user_val)
151:                     except ValueError:
152:                         updated_args[param] = user_val
153:                 elif pinfo.get("type") == "array":
154:                     try:
155:                         updated_args[param] = json.loads(user_val)
156:                     except Exception:
157:                         updated_args[param] = [v.strip() for v in user_val.split(',') if v.strip()]
158:                 else:
159:                     updated_args[param] = user_val
160: 
161:         if Confirm.ask("Execute tool with these parameters?", default=True):
162:             return updated_args
163:         return None
````

## File: command_converter.py
````python
  1: import logging
  2: import platform
  3: import os
  4: import re
  5: from typing import Optional, Dict, Any
  6: from pathlib import Path
  7: from config import get_constant
  8: from .llm_client import create_llm_client
  9: 
 10: logger = logging.getLogger(__name__)
 11: 
 12: class CommandConverter:
 13:     """
 14:     LLM-based command converter that transforms bash commands to be appropriate
 15:     for the current system environment.
 16:     """
 17:     
 18:     def __init__(self):
 19:         self.system_info = self._get_system_info()
 20:         self.conversion_prompt = self._build_conversion_prompt()
 21:     
 22:     def _get_system_info(self) -> Dict[str, Any]:
 23:         """Gather system information for command conversion context."""
 24:         return {
 25:             "os_name": platform.system(),
 26:             "os_version": platform.version(),
 27:             "architecture": platform.machine(),
 28:             "python_version": platform.python_version(),
 29:             "shell": os.environ.get("SHELL", "/bin/bash"),
 30:             "home_dir": str(Path.home()),
 31:             "current_working_dir": str(Path.cwd()),
 32:             "path_separator": os.pathsep,
 33:             "file_separator": os.sep,
 34:             "environment_vars": {
 35:                 "PATH": os.environ.get("PATH", ""),
 36:                 "USER": os.environ.get("USER", ""),
 37:                 "HOME": os.environ.get("HOME", ""),
 38:             }
 39:         }
 40:     
 41:     def _build_conversion_prompt(self) -> str:
 42:         """Build the system prompt for command conversion."""
 43:         os_name = self.system_info['os_name']
 44:         
 45:         # Build OS-specific examples and rules
 46:         if os_name == "Windows":
 47:             examples = """EXAMPLES:
 48: Input: "dir"
 49: Output: dir
 50: 
 51: Input: "dir C:\\path"
 52: Output: dir C:\\path
 53: 
 54: Input: "find /path -type f"
 55: Output: dir /s /b C:\\path\\* | findstr /v "\\\\\\."
 56: 
 57: Input: "ls -la"
 58: Output: dir
 59: 
 60: Input: "echo hello"
 61: Output: echo hello"""
 62:             
 63:             rules = f"""RULES:
 64: - Keep Windows commands (dir, type, copy, etc.) as-is - do NOT convert to Linux equivalents
 65: - If a Linux command is used, convert it to the Windows equivalent
 66: - For file listing: use "dir" not "ls"
 67: - For finding files: use "dir /s /b" not "find"
 68: - Use Windows path separators (\\) when needed
 69: - Hidden files on Windows start with . - filter them when appropriate
 70: - Ensure the command will work on {os_name}
 71: - Return ONLY the command, no other text"""
 72:         else:
 73:             examples = """EXAMPLES:
 74: Input: "find /path -type f"
 75: Output: find /path -type f -not -path "*/.*"
 76: 
 77: Input: "ls -la /directory"  
 78: Output: ls -la /directory | grep -v "^\\."
 79: 
 80: Input: "dir"
 81: Output: ls
 82: 
 83: Input: "echo hello"
 84: Output: echo hello"""
 85:             
 86:             rules = f"""RULES:
 87: - Keep Linux/Unix commands as-is when they work correctly
 88: - If a Windows command is used, convert it to the Linux equivalent  
 89: - For file listing: use "ls" not "dir"
 90: - Always exclude hidden files/directories in find and ls operations
 91: - Use Linux path separators (/) when needed
 92: - Ensure the command will work on {os_name}
 93: - Return ONLY the command, no other text"""
 94:         
 95:         return f"""You are a command converter that adapts commands for different system environments.
 96: 
 97: SYSTEM INFORMATION:
 98: - OS: {os_name} {self.system_info['os_version']}
 99: - Architecture: {self.system_info['architecture']}
100: - Shell: {self.system_info['shell']}
101: - Working Directory: {self.system_info['current_working_dir']}
102: - Path Separator: {self.system_info['path_separator']}
103: - File Separator: {self.system_info['file_separator']}
104: 
105: CONVERSION GOALS:
106: 1. Ensure commands work properly on the current system ({os_name})
107: 2. Filter out hidden files/directories when listing or finding files
108: 3. Convert between Windows and Linux command equivalents as needed
109: 4. Use appropriate flags and options for the target system
110: 5. Handle cross-platform compatibility issues
111: 
112: CRITICAL OUTPUT FORMAT:
113: You MUST respond with ONLY the converted command, nothing else. No explanations, no markdown, no additional text.
114: The response should be a single line containing only the executable command.
115: 
116: {examples}
117: 
118: {rules}
119: """
120: 
121:     async def convert_command(self, original_command: str) -> str:
122:         """
123:         Convert a command using LLM to be appropriate for the current system.
124:         
125:         Args:
126:             original_command: The original bash command to convert
127:             
128:         Returns:
129:             The converted command appropriate for the current system
130:         """
131:         try:
132:             # Get the model from config
133:             model = get_constant("MAIN_MODEL", "anthropic/claude-sonnet-4")
134:             
135:             # Prepare the conversion request
136:             converted_command = await self._call_llm(model, original_command)
137:             
138:             # Validate and clean the response
139:             cleaned_command = self._clean_response(converted_command)
140:             
141:             logger.info(f"Command converted: '{original_command}' -> '{cleaned_command}'")
142:             return cleaned_command
143:             
144:         except Exception as e:
145:             logger.warning(f"Command conversion failed for '{original_command}': {e}")
146:             # Fallback to original command if conversion fails
147:             return original_command
148:     
149:     async def _call_llm(self, model: str, command: str) -> str:
150:         """
151:         Call the LLM API to convert the command.
152:         
153:         Args:
154:             model: The model to use for conversion
155:             command: The original command
156:             
157:         Returns:
158:             The LLM response containing the converted command
159:         """
160:         # Prepare the messages for the LLM
161:         messages = [
162:             {
163:                 "role": "system", 
164:                 "content": self.conversion_prompt
165:             },
166:             {
167:                 "role": "user", 
168:                 "content": command
169:             }
170:         ]
171:         
172:         # Create LLM client and call it
173:         client = create_llm_client(model)
174:         return await client.call(
175:             messages=messages,
176:             max_tokens=200,  # Keep response short
177:             temperature=0.1  # Low temperature for consistent output
178:         )
179:     
180:     def _clean_response(self, response: str) -> str:
181:         """
182:         Clean the LLM response to extract just the command.
183:         
184:         Args:
185:             response: The raw LLM response
186:             
187:         Returns:
188:             The cleaned command string
189:         """
190:         # Remove any markdown code blocks
191:         response = re.sub(r'^```.*?\n|```$', '', response, flags=re.MULTILINE)
192:         
193:         # Remove leading/trailing whitespace
194:         response = response.strip()
195:         
196:         # Split by lines and take the first non-empty line
197:         lines = [line.strip() for line in response.split('\n') if line.strip()]
198:         if not lines:
199:             raise ValueError("Empty response from LLM")
200:         
201:         command = lines[0]
202:         
203:         # Basic validation - ensure it looks like a command
204:         if not command or len(command) > 1000:  # Reasonable length limit
205:             raise ValueError(f"Invalid command format: {command}")
206:         
207:         return command
208: 
209: # Global instance for reuse
210: _converter_instance: Optional[CommandConverter] = None
211: 
212: async def convert_command_for_system(original_command: str) -> str:
213:     """
214:     Convert a bash command to be appropriate for the current system.
215:     
216:     Args:
217:         original_command: The original bash command
218:         
219:     Returns:
220:         The converted command appropriate for the current system
221:     """
222:     global _converter_instance
223:     
224:     if _converter_instance is None:
225:         _converter_instance = CommandConverter()
226:     
227:     return await _converter_instance.convert_command(original_command)
````

## File: context_helpers.py
````python
  1: from typing import Any, Dict, List, Union
  2: from pathlib import Path
  3: from datetime import datetime
  4: 
  5: import os
  6: from utils.web_ui import WebUI
  7: from utils.agent_display_console import AgentDisplayConsole
  8: # from config import write_to_file # Removed as it was for ic
  9: # Removed: from load_constants import *
 10: from config import MAIN_MODEL, get_constant, googlepro # Import get_constant
 11: from utils.file_logger import aggregate_file_states
 12: from openai import OpenAI
 13: import logging
 14: import json
 15: from tenacity import retry, stop_after_attempt, wait_random_exponential
 16: # from icecream import ic # Removed
 17: # from rich import print as rr # Removed
 18: 
 19: # ic.configureOutput(includeContext=True, outputFunction=write_to_file) # Removed
 20: 
 21: logger = logging.getLogger(__name__)
 22: 
 23: QUICK_SUMMARIES = []
 24: 
 25: 
 26: 
 27: 
 28: 
 29: def format_messages_to_string(messages):
 30:     """Return a human readable string for a list of messages."""
 31: 
 32:     def _val(obj, key, default=None):
 33:         if isinstance(obj, dict):
 34:             return obj.get(key, default)
 35:         return getattr(obj, key, default)
 36: 
 37:     try:
 38:         output_pieces = []
 39:         for msg in messages:
 40:             role = msg.get("role", "unknown").upper()
 41:             output_pieces.append(f"\n{role}:")
 42: 
 43:             if "tool_call_id" in msg:
 44:                 output_pieces.append(f"\nTool Call ID: {msg['tool_call_id']}")
 45: 
 46:             if msg.get("tool_calls"):
 47:                 for tc in msg["tool_calls"]:
 48:                     name = _val(_val(tc, "function"), "name")
 49:                     args = _val(_val(tc, "function"), "arguments")
 50:                     tc_id = _val(tc, "id")
 51:                     output_pieces.append(
 52:                         f"\nTool Call -> {name or 'unknown'} (ID: {tc_id or 'n/a'})"
 53:                     )
 54:                     if args:
 55:                         try:
 56:                             parsed = json.loads(args) if isinstance(args, str) else args
 57:                             formatted = json.dumps(parsed, indent=2)
 58:                         except Exception:
 59:                             formatted = str(args)
 60:                         output_pieces.append(f"\nArguments: {formatted}")
 61: 
 62:             content = msg.get("content")
 63:             if isinstance(content, list):
 64:                 for block in content:
 65:                     if isinstance(block, dict):
 66:                         btype = block.get("type")
 67:                         if btype == "text":
 68:                             output_pieces.append(f"\n{block.get('text', '')}")
 69:                         elif btype == "image":
 70:                             output_pieces.append("\n[Image content omitted]")
 71:                         elif btype == "tool_use":
 72:                             output_pieces.append(f"\nTool Call: {block.get('name')}")
 73:                             if "input" in block:
 74:                                 inp = block["input"]
 75:                                 if isinstance(inp, (dict, list)):
 76:                                     output_pieces.append(
 77:                                         f"\nInput: {json.dumps(inp, indent=2)}"
 78:                                     )
 79:                                 else:
 80:                                     output_pieces.append(f"\nInput: {inp}")
 81:                         elif btype == "tool_result":
 82:                             output_pieces.append(
 83:                                 f"\nTool Result [ID: {block.get('tool_use_id', 'unknown')}]"
 84:                             )
 85:                             if block.get("is_error"):
 86:                                 output_pieces.append("\nError: True")
 87:                             for item in block.get("content", []):
 88:                                 if item.get("type") == "text":
 89:                                     output_pieces.append(f"\n{item.get('text', '')}")
 90:                                 elif item.get("type") == "image":
 91:                                     output_pieces.append("\n[Image content omitted]")
 92:                         else:
 93:                             for key, value in block.items():
 94:                                 if key == "cache_control":
 95:                                     continue
 96:                                 output_pieces.append(f"\n{key}: {value}")
 97:                     else:
 98:                         output_pieces.append(f"\n{block}")
 99:             elif content is not None:
100:                 output_pieces.append(f"\n{content}")
101: 
102:             output_pieces.append("\n" + "-" * 80)
103: 
104:         return "".join(output_pieces)
105:     except Exception as e:
106:         return f"Error during formatting: {str(e)}"
107: 
108: 
109: 
110: 
111: 
112: def filter_messages(messages: List[Dict]) -> List[Dict]:
113:     """
114:     Keep only messages with role 'user' or 'assistant'.
115:     Also keep any tool_result messages that contain errors.
116:     """
117:     keep_roles = {"user", "assistant"}
118:     filtered = []
119:     for msg in messages:
120:         if msg.get("role") in keep_roles:
121:             filtered.append(msg)
122:         elif isinstance(msg.get("content"), list):
123:             for block in msg["content"]:
124:                 if isinstance(block, dict) and block.get("type") == "tool_result":
125:                     # Check if any text in the tool result indicates an error
126:                     text = ""
127:                     for item in block.get("content", []):
128:                         if isinstance(item, dict) and item.get("type") == "text":
129:                             text += item.get("text", "")
130:                     if "error" in text.lower():
131:                         filtered.append(msg)
132:                         break
133:     return filtered
134: 
135: 
136: def extract_text_from_content(content: Any) -> str:
137:     if isinstance(content, str):
138:         return content
139:     elif isinstance(content, list):
140:         text_parts = []
141:         for item in content:
142:             if isinstance(item, dict):
143:                 if item.get("type") == "text":
144:                     text_parts.append(item.get("text", ""))
145:                 elif item.get("type") == "tool_result":
146:                     for sub_item in item.get("content", []):
147:                         if sub_item.get("type") == "text":
148:                             text_parts.append(sub_item.get("text", ""))
149:         return " ".join(text_parts)
150:     return ""
151: 
152: 
153: 
154: 
155: 
156: 
157: 
158: 
159: def get_all_summaries() -> str:
160:     """Combine all summaries into a chronological narrative."""
161:     if not QUICK_SUMMARIES:
162:         return "No summaries available yet."
163: 
164:     combined = "\n"
165:     for entry in QUICK_SUMMARIES:
166:         combined += f"{entry}\n"
167:     return combined
168: 
169: 
170: async def reorganize_context(messages: List[Dict[str, Any]], summary: str) -> str:
171:     """Reorganize the context by filtering and summarizing messages."""
172:     conversation_text = ""
173: 
174:     # Look for tool results related to image generation
175:     image_generation_results = []
176: 
177:     for msg in messages:
178:         role = msg["role"].upper()
179:         if isinstance(msg["content"], list):
180:             for block in msg["content"]:
181:                 if isinstance(block, dict):
182:                     if block.get("type") == "text":
183:                         conversation_text += f"\n{role}: {block.get('text', '')}"
184:                     elif block.get("type") == "tool_result":
185:                         # Track image generation results
186:                         if any(
187:                             "picture_generation" in str(item)
188:                             for item in block.get("content", [])
189:                         ):
190:                             for item in block.get("content", []):
191:                                 if item.get(
192:                                     "type"
193:                                 ) == "text" and "Generated image" in item.get(
194:                                     "text", ""
195:                                 ):
196:                                     image_generation_results.append(
197:                                         item.get("text", "")
198:                                     )
199: 
200:                         for item in block.get("content", []):
201:                             if item.get("type") == "text":
202:                                 conversation_text += (
203:                                     f"\n{role} (Tool Result): {item.get('text', '')}"
204:                                 )
205:         else:
206:             conversation_text += f"\n{role}: {msg['content']}"
207: 
208:     # Add special section for image generation if we found any
209:     if image_generation_results:
210:         conversation_text += "\n\nIMAGE GENERATION RESULTS:\n" + "\n".join(
211:             image_generation_results
212:         )
213:     logger.debug(f"Conversation text for reorganize_context: {conversation_text[:500]}...") # Log snippet
214:     summary_prompt = f"""I need a summary of completed steps and next steps for a project that is ALREADY IN PROGRESS. 
215:     This is NOT a new project - you are continuing work on an existing codebase.
216: 
217:     VERY IMPORTANT INSTRUCTIONS:
218:     1. ALL FILES mentioned as completed or created ARE ALREADY CREATED AND FULLY FUNCTIONAL.
219:        - Do NOT suggest recreating these files.
220:        - Do NOT suggest checking if these files exist.
221:        - Assume all files mentioned in completed steps exist exactly where they are described.
222:     
223:     2. ALL STEPS listed as completed HAVE ALREADY BEEN SUCCESSFULLY DONE.
224:        - Do NOT suggest redoing any completed steps.
225:     
226:     3. Your summary should be in TWO clearly separated parts:
227:        a. COMPLETED: List all tasks/steps that have been completed so far
228:        b. NEXT STEPS: List 1-4 specific, actionable steps that should be taken next to complete the project
229:     
230:     4. List each completed item and next step ONLY ONCE, even if it appears multiple times in the context.
231:     
232:     5. If any images were generated, mention each image, its purpose, and its location in the COMPLETED section.
233:     
234:     Please format your response with:
235:     <COMPLETED>
236:     [List of ALL completed steps and created files - these are DONE and exist]
237:     </COMPLETED>
238: 
239:     <NEXT_STEPS>
240:     [Numbered list of 1-4 next steps to complete the project]
241:     </NEXT_STEPS>
242: 
243:     Here is the Summary part:
244:     {summary}
245:     
246:     Here is the messages part:
247:     <MESSAGES>
248:     {conversation_text}
249:     </MESSAGES>
250:     """
251: 
252:     try:
253:         OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
254:         if not OPENROUTER_API_KEY:
255:             raise ValueError("OPENROUTER_API_KEY environment variable is not set")
256: 
257:         sum_client = OpenAI(
258:             base_url="https://openrouter.ai/api/v1",
259:             api_key=OPENROUTER_API_KEY,
260:         )
261:         model = MAIN_MODEL
262:         response = sum_client.chat.completions.create(
263:             model=model, messages=[{"role": "user", "content": summary_prompt}]
264:         )
265:         logger.debug(f"Reorganize context API response: {response}")
266:         if not response or not response.choices:
267:             raise ValueError("No response received from OpenRouter API")
268: 
269:         summary = response.choices[0].message.content
270:         logger.debug(f"Reorganized context summary: {summary[:500]}...") # Log snippet
271:         if not summary:
272:             raise ValueError("Empty response content from OpenRouter API")
273: 
274:         start_tag = "<COMPLETED>"
275:         end_tag = "</COMPLETED>"
276:         if start_tag in summary and end_tag in summary:
277:             completed_items = summary[
278:                 summary.find(start_tag) + len(start_tag) : summary.find(end_tag)
279:             ]
280:         else:
281:             completed_items = "No completed items found."
282: 
283:         start_tag = "<NEXT_STEPS>"
284:         end_tag = "</NEXT_STEPS>"
285:         if start_tag in summary and end_tag in summary:
286:             steps = summary[
287:                 summary.find(start_tag) + len(start_tag) : summary.find(end_tag)
288:             ]
289:         else:
290:             steps = "No steps found."
291: 
292:         return completed_items, steps
293: 
294:     except Exception as e:
295:         logger.error(f"Error in reorganize_context: {str(e)}", exc_info=True)
296:         # Return default values in case of error
297:         return (
298:             "Error processing context. Please try again.",
299:             "Error processing steps. Please try again.",
300:         )
301: 
302: @retry(
303:     stop=stop_after_attempt(max_attempt_number=5),
304:     wait=wait_random_exponential(multiplier=2, min=4, max=10),
305: )
306: async def refresh_context_async(
307:     task: str, messages: List[Dict], display: Union[WebUI, AgentDisplayConsole], client
308: ) -> str:
309:     """
310:     Create a combined context string by filtering and (if needed) summarizing messages
311:     and appending current file contents.
312:     """
313:     filtered = filter_messages(messages)
314:     summary = get_all_summaries() # This is a local function in context_helpers
315:     completed, next_steps = await reorganize_context(filtered, summary)
316: 
317:     file_contents = aggregate_file_states()
318:     if len(file_contents) > 200000:
319:         file_contents = (
320:             file_contents[:70000] + " ... [TRUNCATED] ... " + file_contents[-70000:]
321:         )
322: 
323:     # Get code skeletons
324:     from utils.file_logger import get_all_current_skeleton
325:     from utils.file_logger import get_all_current_code
326:     code_skeletons = get_all_current_skeleton()
327:     current_code = get_all_current_code()
328:     # The logic is if there is code, then supply that, if not then supply the skeletons, if there is no code or skeletons, then say there are no code skeletons
329: 
330:     if current_code:
331:         code_skeletons = current_code
332:     elif not code_skeletons or code_skeletons == "No Python files have been tracked yet.":
333:         code_skeletons = "No code skeletons available."
334: 
335:     # Extract information about images generated
336:     images_info = ""
337:     if "## Generated Images:" in file_contents:
338:         images_section = file_contents.split("## Generated Images:")[1]
339:         if "##" in images_section:
340:             images_section = images_section.split("##")[0]
341:         images_info = "## Generated Images:\n" + images_section.strip()
342: 
343:     # call the LLM and pass it all current messages then the task and ask it to give an updated version of the task
344:     prompt = f""" Your job is to update the task based on the current state of the project.
345:     The task is: {task}
346:     The current state of the project is:
347:     {file_contents}
348:     {code_skeletons}
349:     {completed}
350:     {next_steps}
351:     {images_info}
352: 
353:     Once again, here is the task that I need you to give an updated version of.  
354:     Make sure that you give any tips, lessons learned,  what has been done, and what needs to be done.
355:     Make sure you give clear guidance on how to import various files and in general how they should work together.
356:     """
357: 
358:     messages_for_llm = [{"role": "user", "content": prompt}]
359:     response = client.chat.completions.create(
360:         model=MAIN_MODEL,
361:         messages=messages_for_llm, # Corrected variable name
362:         max_tokens=get_constant("MAX_SUMMARY_TOKENS", 20000) # Use get_constant
363:     )
364:     new_task = response.choices[0].message.content
365: 
366:     combined_content = f"""Original request: 
367:     {task}
368:     
369:     IMPORTANT: This is a CONTINUING PROJECT. All files listed below ALREADY EXIST and are FULLY FUNCTIONAL.
370:     DO NOT recreate any existing files or redo completed steps. Continue the work from where it left off.
371: 
372:     Current Project Files and Assets:
373:     {file_contents}
374: 
375:     Code Skeletons (Structure of Python files):
376:     {code_skeletons}
377: 
378:     COMPLETED STEPS (These have ALL been successfully completed - DO NOT redo these):
379:     {completed}
380: 
381:     NEXT STEPS (Continue the project by completing these):
382:     {next_steps}
383: 
384: 
385:     Updated Request:
386:     {new_task}
387:     NOTES: 
388:     - All files mentioned in completed steps ALREADY EXIST in the locations specified.
389:     - All completed steps have ALREADY BEEN DONE successfully.
390:     - Continue the project by implementing the next steps, building on the existing work.
391:     """
392:     logger.info(f"Refreshed context combined_content (first 500 chars): {combined_content[:500]}...")
393:     return combined_content
````

## File: file_logger.py
````python
  1: import json
  2: import datetime
  3: import shutil
  4: from pathlib import Path
  5: from config import get_constant, LOGS_DIR
  6: 
  7: import ast
  8: from typing import Union
  9: import os
 10: import mimetypes
 11: import base64
 12: import logging
 13: 
 14: logger = logging.getLogger(__name__)
 15: 
 16: try:
 17:     from config import get_constant
 18: 
 19:     # Import the function but don't redefine it
 20:     try:
 21:         from config import convert_to_docker_path
 22:     except ImportError:
 23:         # Define our own if not available in config
 24:         def convert_to_docker_path(path: Union[str, Path]) -> str:
 25:             """
 26:             Convert a local Windows path to a Docker container path.
 27:             No longer converts to Docker path, returns original path.
 28:             Args:
 29:                 path: The local path to convert
 30: 
 31:             Returns:
 32:                 The original path as a string
 33:             """
 34:             if isinstance(path, Path):
 35:                 return str(path)
 36:             return path if path is not None else ""
 37: except ImportError:
 38:     # Fallback if config module is not available
 39:     def get_constant(name):
 40:         # Default values for essential constants
 41:         defaults = {
 42:             "LOG_FILE": os.path.join(
 43:                 os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
 44:                 "logs",
 45:                 "file_log.json",
 46:             ),
 47:         }
 48:         return defaults.get(name)
 49: 
 50:     def convert_to_docker_path(path: Union[str, Path]) -> str:
 51:         """
 52:         Convert a local Windows path to a Docker container path.
 53:         No longer converts to Docker path, returns original path.
 54:         Args:
 55:             path: The local path to convert
 56: 
 57:         Returns:
 58:             The original path as a string
 59:         """
 60:         if isinstance(path, Path):
 61:             return str(path)
 62:         return path if path is not None else ""
 63: 
 64: 
 65: # File for logging operations
 66: try:
 67:     LOG_FILE = get_constant("LOG_FILE")
 68: except:
 69:     LOG_FILE = os.path.join(
 70:         os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
 71:         "logs",
 72:         "file_log.json",
 73:     )
 74: 
 75: # In-memory tracking of file operations # FILE_OPERATIONS removed
 76: # FILE_OPERATIONS = {} # Removed
 77: 
 78: # Ensure log directory exists
 79: os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
 80: 
 81: # If log file doesn't exist, create an empty one
 82: if not os.path.exists(LOG_FILE):
 83:     with open(LOG_FILE, "w") as f:
 84:         json.dump({"files": {}}, f)
 85: 
 86: # Track file operations # Removed unused global variables
 87: # file_operations = [] # Removed
 88: # tracked_files = set() # Removed
 89: # file_contents = {} # Removed
 90: 
 91: 
 92: def log_file_operation(
 93:     file_path: Path, operation: str, content: str = None, metadata: dict = None
 94: ):
 95:     """
 96:     Log a file operation (create, update, delete) with enhanced metadata handling.
 97: 
 98:     Args:
 99:         file_path: Path to the file
100:         operation: Type of operation ('create', 'update', 'delete')
101:         content: Optional content for the file
102:         metadata: Optional dictionary containing additional metadata (e.g., image generation prompt)
103:     """
104:     # Defensively initialize metadata to prevent NoneType errors
105:     if metadata is None:
106:         metadata = {}
107: 
108:     # Ensure file_path is a Path object
109:     if not isinstance(file_path, Path):
110:         file_path = Path(file_path)
111: 
112:     # Create a string representation of the file path for consistent logging
113:     file_path_str = str(file_path)
114: 
115:     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
116:     extension = file_path.suffix.lower() if file_path.suffix else ""
117: 
118:     # Determine if the file is an image
119:     is_image = extension in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"]
120:     mime_type = mimetypes.guess_type(file_path_str)[0]
121:     if mime_type and mime_type.startswith("image/"):
122:         is_image = True
123: 
124:     # Track file operations in memory # Removed FILE_OPERATIONS update logic
125:     # if file_path_str not in FILE_OPERATIONS:
126:     #     FILE_OPERATIONS[file_path_str] = {
127:     #         "operations": [],
128:     #         "last_updated": timestamp,
129:     #         "extension": extension,
130:     #         "is_image": is_image,
131:     #         "mime_type": mime_type,
132:     #     }
133:     #
134:     # # Update the in-memory tracking
135:     # FILE_OPERATIONS[file_path_str]["operations"].append(
136:     #     {"timestamp": timestamp, "operation": operation}
137:     # )
138:     # FILE_OPERATIONS[file_path_str]["last_updated"] = timestamp
139: 
140:     # Load existing log data or create a new one
141:     log_data = {"files": {}}
142: 
143:     if os.path.exists(LOG_FILE):
144:         try:
145:             with open(LOG_FILE, "r") as f:
146:                 log_data = json.load(f)
147:         except json.JSONDecodeError:
148:             # If the log file is corrupted, start fresh
149:             log_data = {"files": {}}
150: 
151:     # Create or update the file entry in the log
152:     if file_path_str not in log_data["files"]:
153:         log_data["files"][file_path_str] = {
154:             "operations": [],
155:             "metadata": {},
156:             "content": None,
157:             "extension": extension,
158:             "is_image": is_image,
159:             "mime_type": mime_type,
160:             "last_updated": timestamp,
161:         }
162: 
163:     # Add the operation to the log
164:     log_data["files"][file_path_str]["operations"].append(
165:         {"timestamp": timestamp, "operation": operation}
166:     )
167: 
168:     # Update the metadata if provided
169:     if metadata:
170:         # Ensure we have a metadata dictionary
171:         if "metadata" not in log_data["files"][file_path_str]:
172:             log_data["files"][file_path_str]["metadata"] = {}
173: 
174:         # Update with new metadata
175:         log_data["files"][file_path_str]["metadata"].update(metadata)
176: 
177:     # Store the content if provided, otherwise try to read it from the file
178:     file_content = content
179: 
180:     try:
181:         # Only try to read the file if it exists and content wasn't provided
182:         if file_content is None and file_path.exists() and file_path.is_file():
183:             try:
184:                 # Handle different file types appropriately
185:                 if is_image:
186:                     # For images, store base64 encoded content
187:                     with open(file_path, "rb") as f:
188:                         img_content = f.read()
189:                         log_data["files"][file_path_str]["content"] = base64.b64encode(
190:                             img_content
191:                         ).decode("utf-8")
192:                         # Add file size to metadata
193:                         if "metadata" not in log_data["files"][file_path_str]:
194:                             log_data["files"][file_path_str]["metadata"] = {}
195:                         log_data["files"][file_path_str]["metadata"]["size"] = len(
196:                             img_content
197:                         )
198: 
199:                 elif extension in [".py", ".js", ".html", ".css", ".json", ".md"]:
200:                     # For code and text files, store as text
201:                     with open(file_path, "r", encoding="utf-8") as f:
202:                         text_content = f.read()
203:                         log_data["files"][file_path_str]["content"] = text_content
204: 
205:                 else:
206:                     # For other binary files, store base64 encoded
207:                     with open(file_path, "rb") as f:
208:                         bin_content = f.read()
209:                         log_data["files"][file_path_str]["content"] = base64.b64encode(
210:                             bin_content
211:                         ).decode("utf-8")
212:                         # Add file size to metadata
213:                         if "metadata" not in log_data["files"][file_path_str]:
214:                             log_data["files"][file_path_str]["metadata"] = {}
215:                         log_data["files"][file_path_str]["metadata"]["size"] = len(
216:                             bin_content
217:                         )
218: 
219:             except Exception as read_error:
220:                 logger.error(f"Error reading file content for {file_path_str}: {read_error}", exc_info=True)
221:                 # Don't fail the entire operation, just log the error
222:                 if "metadata" not in log_data["files"][file_path_str]:
223:                     log_data["files"][file_path_str]["metadata"] = {}
224:                 log_data["files"][file_path_str]["metadata"]["read_error"] = str(
225:                     read_error
226:                 )
227: 
228:         elif file_content is not None:
229:             # Use the provided content
230:             log_data["files"][file_path_str]["content"] = file_content
231: 
232:     except Exception as e:
233:         logger.error(f"Error processing file content for {file_path_str}: {e}", exc_info=True)
234:         # Don't fail the entire operation, just log the error
235:         if "metadata" not in log_data["files"][file_path_str]:
236:             log_data["files"][file_path_str]["metadata"] = {}
237:         log_data["files"][file_path_str]["metadata"]["processing_error"] = str(e)
238: 
239:     # Update last_updated timestamp
240:     log_data["files"][file_path_str]["last_updated"] = timestamp
241: 
242:     # Write the updated log data back to the log file
243:     try:
244:         with open(LOG_FILE, "w") as f:
245:             json.dump(log_data, f, indent=2)
246:     except Exception as write_error:
247:         logger.error(f"Error writing to log file {LOG_FILE}: {write_error}", exc_info=True)
248: 
249: 
250: def aggregate_file_states() -> str:
251:     """
252:     Collect information about all tracked files and their current state.
253: 
254:     Returns:
255:         A formatted string with information about all files.
256:     """
257:     LOG_FILE = Path(get_constant("LOG_FILE"))
258:     if not LOG_FILE.exists():
259:         return "No files have been tracked yet."
260: 
261:     try:
262:         with open(LOG_FILE, "r", encoding="utf-8") as f:
263:             log_data = json.loads(f.read())
264:     except (json.JSONDecodeError, FileNotFoundError):
265:         return "Error reading log file."
266: 
267:     if not log_data:
268:         return "No files have been tracked yet."
269: 
270:     # Group files by type
271:     image_files = []
272:     code_files = []
273:     text_files = []
274:     other_files = []
275: 
276:     for file_path, file_info in log_data.items():
277:         file_type = file_info.get("file_type", "other")
278: 
279:         # Get the Docker path for display
280:         docker_path = file_info.get("docker_path", convert_to_docker_path(file_path))
281: 
282:         # Sort operations by timestamp to get the latest state
283:         operations = sorted(
284:             file_info.get("operations", []),
285:             key=lambda x: x.get("timestamp", ""),
286:             reverse=True,
287:         )
288: 
289:         latest_operation = operations[0] if operations else {"operation": "unknown"}
290: 
291:         if file_type == "image":
292:             image_metadata = file_info.get("image_metadata", {})
293:             image_files.append(
294:                 {
295:                     "path": docker_path,
296:                     "operation": latest_operation.get("operation"),
297:                     "prompt": image_metadata.get("prompt", "No prompt available"),
298:                     "dimensions": image_metadata.get("dimensions", "Unknown"),
299:                     "created_at": image_metadata.get(
300:                         "created_at", file_info.get("created_at", "Unknown")
301:                     ),
302:                 }
303:             )
304:         elif file_type == "code":
305:             code_files.append(
306:                 {
307:                     "path": docker_path,
308:                     "operation": latest_operation.get("operation"),
309:                     "content": file_info.get("content", ""),
310:                     "skeleton": file_info.get("skeleton", "No skeleton available"),
311:                 }
312:             )
313:         elif file_type == "text":
314:             text_files.append(
315:                 {
316:                     "path": docker_path,
317:                     "operation": latest_operation.get("operation"),
318:                     "content": file_info.get("content", ""),
319:                 }
320:             )
321:         else:
322:             basic_info = file_info.get("basic_info", {})
323:             other_files.append(
324:                 {
325:                     "path": docker_path,
326:                     "operation": latest_operation.get("operation"),
327:                     "mime_type": basic_info.get("mime_type", "Unknown"),
328:                     "size": basic_info.get("size", 0),
329:                 }
330:             )
331: 
332:     # Format the output
333:     output = []
334: 
335:     if image_files:
336:         output.append("## Image Files")
337:         for img in image_files:
338:             output.append(f"### {img['path']}")
339:             output.append(f"- **Operation**: {img['operation']}")
340:             output.append(f"- **Created**: {img['created_at']}")
341:             output.append(f"- **Prompt**: {img['prompt']}")
342:             output.append(f"- **Dimensions**: {img['dimensions']}")
343:             output.append("")
344: 
345:     if code_files:
346:         output.append("## Code Files")
347:         for code in code_files:
348:             output.append(f"### {code['path']}")
349:             output.append(f"- **Operation**: {code['operation']}")
350: 
351:             # Add syntax highlighting based on file extension
352:             extension = Path(code["path"]).suffix.lower()
353:             lang = get_language_from_extension(extension)
354: 
355:             output.append("- **Structure**:")
356:             output.append(f"```{lang}")
357:             output.append(code["skeleton"])
358:             output.append("```")
359: 
360:             output.append("- **Content**:")
361:             output.append(f"```{lang}")
362:             output.append(code["content"])
363:             output.append("```")
364:             output.append("")
365: 
366:     if text_files:
367:         output.append("## Text Files")
368:         for text in text_files:
369:             output.append(f"### {text['path']}")
370:             output.append(f"- **Operation**: {text['operation']}")
371: 
372:             # Add syntax highlighting based on file extension
373:             extension = Path(text["path"]).suffix.lower()
374:             lang = get_language_from_extension(extension)
375: 
376:             output.append("- **Content**:")
377:             output.append(f"```{lang}")
378:             output.append(text["content"])
379:             output.append("```")
380:             output.append("")
381: 
382:     if other_files:
383:         output.append("## Other Files")
384:         for other in other_files:
385:             output.append(f"### {other['path']}")
386:             output.append(f"- **Operation**: {other['operation']}")
387:             output.append(f"- **MIME Type**: {other['mime_type']}")
388:             output.append(f"- **Size**: {other['size']} bytes")
389:             output.append("")
390: 
391:     return "\n".join(output)
392: 
393: 
394: def extract_code_skeleton(source_code: Union[str, Path]) -> str:
395:     """
396:     Extract a code skeleton from existing Python code.
397: 
398:     This function takes Python code and returns just the structure: imports,
399:     class definitions, method/function signatures, and docstrings, with
400:     implementations replaced by 'pass' statements.
401: 
402:     Args:
403:         source_code: Either a path to a Python file or a string containing Python code
404: 
405:     Returns:
406:         str: The extracted code skeleton
407:     """
408:     # Load the source code
409:     if isinstance(source_code, (str, Path)) and Path(source_code).exists():
410:         with open(source_code, "r", encoding="utf-8") as file:
411:             code_str = file.read()
412:     else:
413:         code_str = str(source_code)
414: 
415:     # Parse the code into an AST
416:     try:
417:         tree = ast.parse(code_str)
418:     except SyntaxError as e:
419:         return f"# Error parsing code: {e}\n{code_str}"
420: 
421:     # Extract imports
422:     imports = []
423:     for node in ast.walk(tree):
424:         if isinstance(node, ast.Import):
425:             for name in node.names:
426:                 imports.append(
427:                     f"import {name.name}"
428:                     + (f" as {name.asname}" if name.asname else "")
429:                 )
430:         elif isinstance(node, ast.ImportFrom):
431:             module = node.module or ""
432:             names = ", ".join(
433:                 name.name + (f" as {name.asname}" if name.asname else "")
434:                 for name in node.names
435:             )
436:             imports.append(f"from {module} import {names}")
437: 
438:     # Helper function to handle complex attributes
439:     def format_attribute(node):
440:         """Helper function to recursively format attribute expressions"""
441:         if isinstance(node, ast.Name):
442:             return node.id
443:         elif isinstance(node, ast.Attribute):
444:             return f"{format_attribute(node.value)}.{node.attr}"
445:         # Add support for ast.Subscript nodes (like List[int])
446:         elif isinstance(node, ast.Subscript):
447:             # Use ast.unparse for Python 3.9+ or manual handling for earlier versions
448:             if hasattr(ast, "unparse"):
449:                 return ast.unparse(node)
450:             else:
451:                 # Simplified handling for common cases
452:                 if isinstance(node.value, ast.Name):
453:                     base = node.value.id
454:                 else:
455:                     base = format_attribute(node.value)
456:                 # Simple handling for slice
457:                 if isinstance(node.slice, ast.Index) and hasattr(node.slice, "value"):
458:                     if isinstance(node.slice.value, ast.Name):
459:                         return f"{base}[{node.slice.value.id}]"
460:                     else:
461:                         return f"{base}[...]"  # Fallback for complex slices
462:                 return f"{base}[...]"  # Fallback for complex cases
463:         else:
464:             # Fallback for other node types - use ast.unparse if available
465:             if hasattr(ast, "unparse"):
466:                 return ast.unparse(node)
467:             return str(node)
468: 
469:     # Get docstrings and function/class signatures
470:     class CodeSkeletonVisitor(ast.NodeVisitor):
471:         def __init__(self):
472:             self.skeleton = []
473:             self.indent_level = 0
474:             self.imports = []
475: 
476:         def visit_Import(self, node):
477:             # Already handled above
478:             pass
479: 
480:         def visit_ImportFrom(self, node):
481:             # Already handled above
482:             pass
483: 
484:         def visit_ClassDef(self, node):
485:             # Extract class definition with inheritance
486:             bases = []
487:             for base in node.bases:
488:                 if isinstance(base, ast.Name):
489:                     bases.append(base.id)
490:                 elif isinstance(base, ast.Attribute):
491:                     # Use the helper function to handle nested attributes
492:                     bases.append(format_attribute(base))
493:                 else:
494:                     # Fallback for other complex cases
495:                     if hasattr(ast, "unparse"):
496:                         bases.append(ast.unparse(base))
497:                     else:
498:                         bases.append("...")
499: 
500:             class_def = f"class {node.name}"
501:             if bases:
502:                 class_def += f"({', '.join(bases)})"
503:             class_def += ":"
504: 
505:             # Add class definition
506:             self.skeleton.append("\n" + "    " * self.indent_level + class_def)
507: 
508:             # Add docstring if it exists
509:             docstring = ast.get_docstring(node)
510:             if docstring:
511:                 doc_lines = docstring.split("\n")
512:                 if len(doc_lines) == 1:
513:                     self.skeleton.append(
514:                         "    " * (self.indent_level + 1) + f'"""{docstring}"""'
515:                     )
516:                 else:
517:                     self.skeleton.append("    " * (self.indent_level + 1) + '"""')
518:                     for line in doc_lines:
519:                         self.skeleton.append("    " * (self.indent_level + 1) + line)
520:                     self.skeleton.append("    " * (self.indent_level + 1) + '"""')
521: 
522:             # Increment indent for class members
523:             self.indent_level += 1
524: 
525:             # Visit all class members
526:             for item in node.body:
527:                 if not isinstance(item, ast.Expr) or not isinstance(
528:                     item.value, ast.Str
529:                 ):
530:                     self.visit(item)
531: 
532:             # If no members were added, add a pass statement
533:             if len(self.skeleton) > 0 and not self.skeleton[-1].strip().startswith(
534:                 "def "
535:             ):
536:                 if "pass" not in self.skeleton[-1]:
537:                     self.skeleton.append("    " * self.indent_level + "pass")
538: 
539:             # Restore indent
540:             self.indent_level -= 1
541: 
542:         def visit_FunctionDef(self, node):
543:             # Extract function signature
544:             args = []
545:             defaults = [None] * (
546:                 len(node.args.args) - len(node.args.defaults)
547:             ) + node.args.defaults
548: 
549:             # Process regular arguments
550:             for i, arg in enumerate(node.args.args):
551:                 arg_str = arg.arg
552:                 # Add type annotation if available
553:                 if arg.annotation:
554:                     # Use the helper function to handle complex types
555:                     if hasattr(ast, "unparse"):
556:                         arg_str += f": {ast.unparse(arg.annotation)}"
557:                     else:
558:                         if isinstance(arg.annotation, ast.Name):
559:                             arg_str += f": {arg.annotation.id}"
560:                         elif isinstance(arg.annotation, ast.Attribute):
561:                             arg_str += f": {format_attribute(arg.annotation)}"
562:                         elif isinstance(arg.annotation, ast.Subscript):
563:                             arg_str += f": {format_attribute(arg.annotation)}"
564:                         else:
565:                             arg_str += ": ..."  # Fallback for complex annotations
566: 
567:                 # Add default value if available
568:                 if defaults[i] is not None:
569:                     if hasattr(ast, "unparse"):
570:                         arg_str += f" = {ast.unparse(defaults[i])}"
571:                     else:
572:                         # Simplified handling for common default values
573:                         if isinstance(
574:                             defaults[i], (ast.Str, ast.Num, ast.NameConstant)
575:                         ):
576:                             arg_str += f" = {ast.literal_eval(defaults[i])}"
577:                         elif isinstance(defaults[i], ast.Name):
578:                             arg_str += f" = {defaults[i].id}"
579:                         elif isinstance(defaults[i], ast.Attribute):
580:                             arg_str += f" = {format_attribute(defaults[i])}"
581:                         else:
582:                             arg_str += " = ..."  # Fallback for complex defaults
583: 
584:                 args.append(arg_str)
585: 
586:             # Handle *args
587:             if node.args.vararg:
588:                 args.append(f"*{node.args.vararg.arg}")
589: 
590:             # Handle keyword-only args
591:             if node.args.kwonlyargs:
592:                 if not node.args.vararg:
593:                     args.append("*")
594:                 for i, kwarg in enumerate(node.args.kwonlyargs):
595:                     kw_str = kwarg.arg
596:                     if kwarg.annotation:
597:                         if hasattr(ast, "unparse"):
598:                             kw_str += f": {ast.unparse(kwarg.annotation)}"
599:                         else:
600:                             kw_str += f": {format_attribute(kwarg.annotation)}"
601:                     if (
602:                         i < len(node.args.kw_defaults)
603:                         and node.args.kw_defaults[i] is not None
604:                     ):
605:                         if hasattr(ast, "unparse"):
606:                             kw_str += f" = {ast.unparse(node.args.kw_defaults[i])}"
607:                         else:
608:                             kw_str += " = ..."  # Fallback for complex defaults
609:                     args.append(kw_str)
610: 
611:             # Handle **kwargs
612:             if node.args.kwarg:
613:                 args.append(f"**{node.args.kwarg.arg}")
614: 
615:             # Build function signature
616:             func_def = f"def {node.name}({', '.join(args)})"
617: 
618:             # Add return type if specified
619:             if node.returns:
620:                 if hasattr(ast, "unparse"):
621:                     func_def += f" -> {ast.unparse(node.returns)}"
622:                 else:
623:                     func_def += f" -> {format_attribute(node.returns)}"
624: 
625:             func_def += ":"
626: 
627:             # Add function definition
628:             self.skeleton.append("\n" + "    " * self.indent_level + func_def)
629: 
630:             # Add docstring if it exists
631:             docstring = ast.get_docstring(node)
632:             if docstring:
633:                 doc_lines = docstring.split("\n")
634:                 if len(doc_lines) == 1:
635:                     self.skeleton.append(
636:                         "    " * (self.indent_level + 1) + f'"""{docstring}"""'
637:                     )
638:                 else:
639:                     self.skeleton.append("    " * (self.indent_level + 1) + '"""')
640:                     for line in doc_lines:
641:                         self.skeleton.append("    " * (self.indent_level + 1) + line)
642:                     self.skeleton.append("    " * (self.indent_level + 1) + '"""')
643: 
644:             # Add pass statement in place of the function body
645:             self.skeleton.append("    " * (self.indent_level + 1) + "pass")
646: 
647:     # Run the visitor on the AST
648:     visitor = CodeSkeletonVisitor()
649:     visitor.visit(tree)
650: 
651:     # Combine imports and code skeleton
652:     result = []
653: 
654:     # Add all imports first
655:     if imports:
656:         result.extend(imports)
657:         result.append("")  # Add a blank line after imports
658: 
659:     # Add the rest of the code skeleton
660:     result.extend(visitor.skeleton)
661: 
662:     return "\n".join(result)
663: 
664: 
665: def get_all_current_code() -> str:
666:     """
667:     Returns all the current code in the project as a string.
668:     This function is used to provide context about the existing code to the LLM.
669: 
670:     Returns:
671:         A string with all the current code.
672:     """
673:     try:
674:         # Ensure log file exists
675:         if not os.path.exists(LOG_FILE):
676:             logger.warning(f"Log file not found: {LOG_FILE}")
677:             # Initialize a new log file so future operations work
678:             with open(LOG_FILE, "w") as f:
679:                 json.dump({"files": {}}, f)
680:             return "No code has been written yet."
681: 
682:         # Load log data with robust error handling
683:         try:
684:             with open(LOG_FILE, "r", encoding="utf-8") as f:
685:                 log_data = json.load(f)
686:         except json.JSONDecodeError:
687:             logger.error(f"Log file contains invalid JSON: {LOG_FILE}")
688:             # Reset the log file with valid JSON
689:             with open(LOG_FILE, "w") as f:
690:                 json.dump({"files": {}}, f)
691:             return "Error reading log file. Log file has been reset."
692:         except Exception as e:
693:             logger.error(f"Unexpected error reading log file: {e}", exc_info=True)
694:             return f"Error reading log file: {str(e)}"
695: 
696:         # Validate log structure
697:         if not isinstance(log_data, dict) or "files" not in log_data:
698:             logger.error("Log file has invalid format (missing 'files' key)")
699:             # Fix the format
700:             log_data = {"files": {}}
701:             with open(LOG_FILE, "w") as f:
702:                 json.dump(log_data, f)
703:             return "Error reading log file. Log file has been reset with correct structure."
704: 
705:         output = []
706:         code_files = []
707: 
708:         # Process each file in the log
709:         for file_path, file_data in log_data.get("files", {}).items():
710:             try:
711:                 # Skip files that have been deleted (last operation is 'delete')
712:                 operations = file_data.get("operations", [])
713:                 if operations and operations[-1].get("operation") == "delete":
714:                     continue
715: 
716:                 # Get content and metadata
717:                 content = file_data.get("content")
718:                 if content is None:
719:                     continue
720: 
721:                 # Skip images and binary files
722:                 is_image = file_data.get("is_image", False)
723:                 mime_type = file_data.get("mime_type", "")
724:                 extension = file_data.get("extension", "").lower()
725: 
726:                 if is_image or (mime_type and mime_type.startswith("image/")):
727:                     continue
728: 
729:                 # Only include code files
730:                 if extension in [
731:                     ".py",
732:                     ".js",
733:                     ".html",
734:                     ".css",
735:                     ".ts",
736:                     ".jsx",
737:                     ".tsx",
738:                     ".java",
739:                     ".cpp",
740:                     ".c",
741:                     ".h",
742:                     ".cs",
743:                 ]:
744:                     code_files.append(
745:                         {"path": file_path, "content": content, "extension": extension}
746:                     )
747:             except Exception as file_error:
748:                 logger.error(f"Error processing file {file_path}: {file_error}", exc_info=True)
749:                 # Continue processing other files
750: 
751:         # Sort files by path for consistent output
752:         code_files.sort(key=lambda x: x["path"])
753: 
754:         # Format the output
755:         output.append("# Code\n")
756: 
757:         if code_files:
758:             for code in code_files:
759:                 try:
760:                     output.append(f"## {code['path']}")
761:                     lang = get_language_from_extension(code["extension"])
762:                     output.append(f"```{lang}")
763:                     output.append(code["content"])
764:                     output.append("```\n")
765:                 except Exception as format_error:
766:                     logger.error( # Replaced print with logger.error
767:                         f"Error formatting code file {code.get('path')}: {format_error}", exc_info=True
768:                     )
769:                     # Add a simpler version without failing
770:                     output.append(f"## {code.get('path', 'Unknown file')}")
771:                     output.append("```")
772:                     output.append("Error displaying file content")
773:                     output.append("```\n")
774:         else:
775:             output.append("No code files have been created yet.\n")
776: 
777:         # Return the formatted output (This was missing)
778:         return "\n".join(output)
779: 
780:     except Exception as e:
781:         logger.critical(f"Critical error in get_all_current_code: {e}", exc_info=True)
782:         return "Error reading code files. Please check application logs for details."
783: 
784: 
785: def get_all_current_skeleton() -> str:
786:     """
787:     Get the skeleton of all Python code files.
788: 
789:     Returns:
790:         A formatted string with the skeleton of all Python code files.
791:     """
792:     LOG_FILE = Path(get_constant("LOG_FILE"))
793:     if not LOG_FILE.exists():
794:         return "No Python files have been tracked yet."
795: 
796:     try:
797:         with open(LOG_FILE, "r", encoding="utf-8") as f:
798:             log_data = json.load(f)
799:     except (json.JSONDecodeError, FileNotFoundError):
800:         return "Error reading log file."
801: 
802:     if not log_data or not log_data.get("files"):
803:         return "No Python files have been tracked yet."
804: 
805:     output = ["# All Python File Skeletons"]
806: 
807:     for file_path, file_info in log_data.get("files", {}).items():
808:         # Skip files that have been deleted (last operation is 'delete')
809:         operations = file_info.get("operations", [])
810:         if operations and operations[-1].get("operation") == "delete":
811:             continue
812: 
813:         # Only process Python files
814:         extension = file_info.get("extension", "").lower()
815:         if extension != ".py":
816:             continue
817: 
818:         # Get Docker path for display
819:         docker_path = convert_to_docker_path(file_path)
820: 
821:         # Look for skeleton in metadata
822:         metadata = file_info.get("metadata", {})
823:         skeleton = metadata.get("skeleton", "")
824: 
825:         # If no skeleton in metadata, try to extract it from the content
826:         if not skeleton and file_info.get("content"):
827:             try:
828:                 skeleton = extract_code_skeleton(file_info.get("content", ""))
829:             except Exception as e:
830:                 logger.error(f"Error extracting skeleton from {file_path}: {e}", exc_info=True)
831:                 skeleton = "# Failed to extract skeleton"
832: 
833:         if skeleton:
834:             # Add file header
835:             output.append(f"## {docker_path}")
836: 
837:             # Add skeleton with syntax highlighting
838:             output.append("```python")
839:             output.append(skeleton)
840:             output.append("```")
841:             output.append("")
842: 
843:     return "\n".join(output)
844: 
845: 
846: def get_language_from_extension(extension: str) -> str:
847:     """
848:     Map file extensions to programming languages for syntax highlighting.
849: 
850:     Args:
851:         extension: The file extension (e.g., '.py', '.js')
852: 
853:     Returns:
854:         The corresponding language name for syntax highlighting.
855:     """
856:     extension = extension.lower()
857:     mapping = {
858:         ".py": "python",
859:         ".js": "javascript",
860:         ".jsx": "javascript",
861:         ".ts": "typescript",
862:         ".tsx": "typescript",
863:         ".html": "html",
864:         ".css": "css",
865:         ".json": "json",
866:         ".md": "markdown",
867:         ".yaml": "yaml",
868:         ".yml": "yaml",
869:         ".toml": "toml",
870:         ".java": "java",
871:         ".cpp": "cpp",
872:         ".c": "c",
873:         ".cs": "csharp",
874:         ".sh": "bash",
875:         ".rb": "ruby",
876:         ".go": "go",
877:         ".php": "php",
878:         ".rs": "rust",
879:         ".swift": "swift",
880:         ".kt": "kotlin",
881:         ".sql": "sql",
882:     }
883:     return mapping.get(extension, "")
884: 
885: 
886: 
887: 
888: 
889: def archive_logs():
890:     """Archive all log files in LOGS_DIR by moving them to an archive folder with a timestamp."""
891:     try:
892:         # Create timestamp for the archive folder
893:         timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
894:         archive_dir = Path(LOGS_DIR, "archive", timestamp)
895:         archive_dir.mkdir(parents=True, exist_ok=True)
896: 
897:         # Get all files in LOGS_DIR
898:         log_path = Path(LOGS_DIR)
899:         log_files = [f for f in log_path.iterdir() if f.is_file()]
900: 
901:         # Skip archiving if there are no files
902:         if not log_files:
903:             return "No log files to archive"
904: 
905:         # Move each file to the archive directory
906:         for file_path in log_files:
907:             # Skip archive directory itself
908:             if "archive" in str(file_path):
909:                 continue
910: 
911:             # Create destination path
912:             dest_path = Path(archive_dir, file_path.name)
913: 
914:             # Copy the file if it exists (some might be created later)
915:             if file_path.exists():
916:                 shutil.copy2(file_path, dest_path)
917: 
918:                 # Clear the original file but keep it
919:                 with open(file_path, "w") as f:
920:                     f.write("")
921: 
922:         return f"Archived {len(log_files)} log files to {archive_dir}"
923:     except Exception as e:
924:         return f"Error archiving files: {str(e)}"
````

## File: llm_client.py
````python
  1: import logging
  2: import os
  3: import aiohttp
  4: from typing import Dict, List
  5: from abc import ABC, abstractmethod
  6: 
  7: logger = logging.getLogger(__name__)
  8: 
  9: class LLMClient(ABC):
 10:     """Abstract base class for LLM clients."""
 11:     
 12:     @abstractmethod
 13:     async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
 14:         """Call the LLM with messages and return response."""
 15:         pass
 16: 
 17: class OpenRouterClient(LLMClient):
 18:     """OpenRouter API client."""
 19:     
 20:     def __init__(self, model: str):
 21:         self.model = model
 22:         self.api_key = os.environ.get("OPENROUTER_API_KEY")
 23:         if not self.api_key:
 24:             raise ValueError("OPENROUTER_API_KEY environment variable not set")
 25:     
 26:     async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
 27:         headers = {
 28:             "Authorization": f"Bearer {self.api_key}",
 29:             "Content-Type": "application/json",
 30:             "HTTP-Referer": "https://github.com/command-converter",
 31:             "X-Title": "Command Converter"
 32:         }
 33:         
 34:         payload = {
 35:             "model": self.model,
 36:             "messages": messages,
 37:             "max_tokens": max_tokens,
 38:             "temperature": temperature,
 39:         }
 40:         
 41:         async with aiohttp.ClientSession() as session:
 42:             async with session.post(
 43:                 "https://openrouter.ai/api/v1/chat/completions",
 44:                 headers=headers,
 45:                 json=payload,
 46:                 timeout=aiohttp.ClientTimeout(total=30)
 47:             ) as response:
 48:                 if response.status != 200:
 49:                     error_text = await response.text()
 50:                     raise RuntimeError(f"OpenRouter API error: {response.status} - {error_text}")
 51:                 
 52:                 result = await response.json()
 53:                 
 54:                 if "choices" not in result or not result["choices"]:
 55:                     raise RuntimeError("Invalid OpenRouter response format")
 56:                 
 57:                 return result["choices"][0]["message"]["content"]
 58: 
 59: class OpenAIClient(LLMClient):
 60:     """OpenAI API client."""
 61:     
 62:     def __init__(self, model: str):
 63:         self.model = model
 64:         self.api_key = os.environ.get("OPENAI_API_KEY")
 65:         if not self.api_key:
 66:             raise ValueError("OPENAI_API_KEY environment variable not set")
 67:     
 68:     async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
 69:         headers = {
 70:             "Authorization": f"Bearer {self.api_key}",
 71:             "Content-Type": "application/json",
 72:         }
 73:         
 74:         payload = {
 75:             "model": self.model,
 76:             "messages": messages,
 77:             "max_tokens": max_tokens,
 78:             "temperature": temperature,
 79:         }
 80:         
 81:         async with aiohttp.ClientSession() as session:
 82:             async with session.post(
 83:                 "https://api.openai.com/v1/chat/completions",
 84:                 headers=headers,
 85:                 json=payload,
 86:                 timeout=aiohttp.ClientTimeout(total=30)
 87:             ) as response:
 88:                 if response.status != 200:
 89:                     error_text = await response.text()
 90:                     raise RuntimeError(f"OpenAI API error: {response.status} - {error_text}")
 91:                 
 92:                 result = await response.json()
 93:                 
 94:                 if "choices" not in result or not result["choices"]:
 95:                     raise RuntimeError("Invalid OpenAI response format")
 96:                 
 97:                 return result["choices"][0]["message"]["content"]
 98: 
 99: class AnthropicClient(LLMClient):
100:     """Anthropic API client."""
101:     
102:     def __init__(self, model: str):
103:         self.model = model
104:         self.api_key = os.environ.get("ANTHROPIC_API_KEY")
105:         if not self.api_key:
106:             raise ValueError("ANTHROPIC_API_KEY environment variable not set")
107:     
108:     async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str:
109:         headers = {
110:             "x-api-key": self.api_key,
111:             "Content-Type": "application/json",
112:             "anthropic-version": "2023-06-01"
113:         }
114:         
115:         # Convert messages format for Anthropic
116:         system_content = ""
117:         user_messages = []
118:         
119:         for msg in messages:
120:             if msg["role"] == "system":
121:                 system_content = msg["content"]
122:             else:
123:                 user_messages.append(msg)
124:         
125:         payload = {
126:             "model": self.model,
127:             "max_tokens": max_tokens,
128:             "temperature": temperature,
129:             "messages": user_messages,
130:         }
131:         
132:         if system_content:
133:             payload["system"] = system_content
134:         
135:         async with aiohttp.ClientSession() as session:
136:             async with session.post(
137:                 "https://api.anthropic.com/v1/messages",
138:                 headers=headers,
139:                 json=payload,
140:                 timeout=aiohttp.ClientTimeout(total=30)
141:             ) as response:
142:                 if response.status != 200:
143:                     error_text = await response.text()
144:                     raise RuntimeError(f"Anthropic API error: {response.status} - {error_text}")
145:                 
146:                 result = await response.json()
147:                 
148:                 if "content" not in result or not result["content"]:
149:                     raise RuntimeError("Invalid Anthropic response format")
150:                 
151:                 return result["content"][0]["text"]
152: 
153: def create_llm_client(model: str) -> LLMClient:
154:     """Factory function to create appropriate LLM client based on model name."""
155:     if model.startswith("anthropic/"):
156:         return OpenRouterClient(model)
157:     elif model.startswith("openai/"):
158:         return OpenRouterClient(model)
159:     elif model.startswith("google/"):
160:         return OpenRouterClient(model)
161:     elif model.startswith("gpt-"):
162:         return OpenAIClient(model)
163:     elif model.startswith("claude-"):
164:         return AnthropicClient(model)
165:     else:
166:         # Default to OpenRouter for unknown models
167:         logger.warning(f"Unknown model format: {model}, defaulting to OpenRouter")
168:         return OpenRouterClient(model)
````

## File: output_manager.py
````python
 1: import base64
 2: import hashlib
 3: import json
 4: from datetime import datetime
 5: from pathlib import Path
 6: from typing import Any, List, Optional, TYPE_CHECKING, Union
 7: 
 8: from typing import Dict
 9: import logging
10: 
11: from .web_ui import WebUI
12: from .agent_display_console import AgentDisplayConsole
13: from config import get_constant  # Updated import
14: 
15: logger = logging.getLogger(__name__)
16: 
17: if TYPE_CHECKING:
18:     from tools.base import ToolResult
19: 
20: 
21: class OutputManager:
22: 
23:     def __init__(self,
24:                  display: Union[WebUI, AgentDisplayConsole],
25:                  image_dir: Optional[Path] = None):
26:         LOGS_DIR = Path(get_constant("LOGS_DIR"))
27:         self.image_dir = LOGS_DIR / "computer_tool_images"
28:         self.image_dir.mkdir(parents=True, exist_ok=True)
29:         self.image_counter = 0
30:         self.display = display
31: 
32:     def save_image(self, base64_data: str) -> Optional[Path]:
33:         """Save base64 image data to file and return path."""
34:         if not base64_data:
35:             logger.error("No base64 data provided to save_image")
36:             return None
37: 
38:         try:
39:             self.image_counter += 1
40:             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
41:             image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:8]
42:             image_path = self.image_dir / f"image_{timestamp}_{image_hash}.png"
43: 
44:             image_data = base64.b64decode(base64_data)
45:             with open(image_path, "wb") as f:
46:                 f.write(image_data)
47:             return image_path
48:         except Exception as e:
49:             logger.error(f"Error saving image: {e}", exc_info=True)
50:             return None
51: 
52:     def format_tool_output(self, result: "ToolResult", tool_name: str):
53:         """Format and display tool output."""
54:         if result is None:
55:             logger.error("None result provided to format_tool_output")
56:             return
57: 
58:         output_text = f"Used Tool: {tool_name}\n"
59: 
60:         if isinstance(result, str):
61:             output_text += f"{result}"
62:         else:
63:             text = self._truncate_string(
64:                 str(result.output) if result.output is not None else "")
65:             output_text += f"Output: {text}\n"
66:             if result.base64_image:
67:                 image_path = self.save_image(result.base64_image)
68:                 if image_path:
69:                     output_text += (
70:                         f"[green]📸 Screenshot saved to {image_path}[/green]\n")
71:                 else:
72:                     output_text += "[red]Failed to save screenshot[/red]\n"
73: 
74:         # self.display., output_text)
75: 
76: 
77: 
78: 
79: 
80: 
81: 
82: 
83: 
84: 
85: 
86:     def _truncate_string(self, text: str, max_length: int = 500) -> str:
87:         """Truncate a string to a max length with ellipsis."""
88:         if text is None:
89:             return ""
90: 
91:         if len(text) > max_length:
92:             return text[:200] + "\n...\n" + text[-200:]
93:         return text
````

## File: web_ui.py
````python
  1: import asyncio
  2: import os
  3: import logging
  4: import json
  5: from queue import Queue
  6: from flask import Flask, render_template, jsonify, request
  7: from flask_socketio import SocketIO
  8: from config import (
  9:     LOGS_DIR,
 10:     PROMPTS_DIR,
 11:     get_constant,
 12:     set_constant,
 13:     set_prompt_name,
 14:     write_constants_to_file,
 15: )
 16: from pathlib import Path
 17: 
 18: 
 19: logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
 20: 
 21: def log_message(msg_type, message):
 22:     """Log a message to a file."""
 23:     if msg_type == "user":
 24:         emojitag = "🤡 "
 25:     elif msg_type == "assistant":
 26:         emojitag = "🧞‍♀️ "
 27:     elif msg_type == "tool":
 28:         emojitag = "📎 "
 29:     else:
 30:         emojitag = "❓ "
 31:     log_file = os.path.join(LOGS_DIR, f"{msg_type}_messages.log")
 32:     with open(log_file, "a", encoding="utf-8") as file:
 33:         file.write(emojitag * 5)
 34:         file.write(f"\n{message}\n\n")
 35: 
 36: class WebUI:
 37:     def __init__(self, agent_runner):
 38:         logging.info("Initializing WebUI")
 39:         # More robust path for templates
 40:         template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
 41:         logging.info(f"Template directory set to: {template_dir}")
 42:         self.app = Flask(__name__, template_folder=template_dir)
 43:         self.app.config["SECRET_KEY"] = "secret!"
 44:         self.socketio = SocketIO(self.app, async_mode="threading", cookie=None)
 45:         self.user_messages = []
 46:         self.assistant_messages = []
 47:         self.tool_results = []
 48:         # Using a standard Queue for cross-thread communication
 49:         self.input_queue = Queue()
 50:         self.tool_queue = Queue()
 51:         self.agent_runner = agent_runner
 52:         # Import tools lazily to avoid circular imports
 53:         from tools import (
 54:             # BashTool,
 55:             # OpenInterpreterTool,
 56:             ProjectSetupTool,
 57:             WriteCodeTool,
 58:             PictureGenerationTool,
 59:             EditTool,
 60:             ToolCollection,
 61:             BashTool
 62:         )
 63: 
 64:         self.tool_collection = ToolCollection(
 65:             WriteCodeTool(display=self),
 66:             ProjectSetupTool(display=self),
 67:             BashTool(display=self),
 68:             # OpenInterpreterTool(display=self),  # Uncommented and enabled for testing
 69:             PictureGenerationTool(display=self),
 70:             EditTool(display=self),
 71:             display=self,
 72:         )
 73:         self.setup_routes()
 74:         self.setup_socketio_events()
 75:         logging.info("WebUI initialized")
 76: 
 77:     def setup_routes(self):
 78:         logging.info("Setting up routes")
 79: 
 80:         @self.app.route("/")
 81:         def select_prompt_route():
 82:             logging.info("Serving modern prompt selection page (default)")
 83:             prompt_files = list(PROMPTS_DIR.glob("*.md"))
 84:             options = [file.name for file in prompt_files]
 85:             return render_template("select_prompt_modern.html", options=options)
 86: 
 87:         @self.app.route("/classic")
 88:         def select_prompt_classic_route():
 89:             logging.info("Serving classic prompt selection page")
 90:             prompt_files = list(PROMPTS_DIR.glob("*.md"))
 91:             options = [file.name for file in prompt_files]
 92:             return render_template("select_prompt.html", options=options)
 93: 
 94:         @self.app.route("/modern")
 95:         def select_prompt_modern_route():
 96:             logging.info("Serving modern prompt selection page (redirect)")
 97:             prompt_files = list(PROMPTS_DIR.glob("*.md"))
 98:             options = [file.name for file in prompt_files]
 99:             return render_template("select_prompt_modern.html", options=options)
100: 
101:         @self.app.route("/run_agent", methods=["POST"])
102:         def run_agent_route():
103:             logging.info("Received request to run agent")
104:             try:
105:                 choice = request.form.get("choice")
106:                 filename = request.form.get("filename")
107:                 prompt_text = request.form.get("prompt_text")
108:                 logging.info(f"Form data: choice={choice}, filename={filename}, prompt_text length={len(prompt_text) if prompt_text else 0}")
109: 
110:                 if choice == "new":
111:                     logging.info("Creating new prompt")
112:                     if not filename:
113:                         logging.error("Filename is required for new prompts")
114:                         return jsonify({"error": "Filename is required for new prompts"}), 400
115:                     new_prompt_path = PROMPTS_DIR / f"{filename}.md"
116:                     prompt_name = Path(filename).stem
117:                     with open(new_prompt_path, "w", encoding="utf-8") as f:
118:                         f.write(prompt_text or "")
119:                     task = prompt_text or ""
120:                 else:
121:                     logging.info(f"Loading existing prompt: {choice}")
122:                     if not choice:
123:                         logging.error("Choice is required for existing prompts")
124:                         return jsonify({"error": "Choice is required for existing prompts"}), 400
125:                     prompt_path = PROMPTS_DIR / choice
126:                     if not prompt_path.exists():
127:                         logging.error(f"Prompt file not found: {prompt_path}")
128:                         return jsonify({"error": f"Prompt file not found: {choice}"}), 404
129:                     prompt_name = prompt_path.stem
130:                     if prompt_text:
131:                         logging.info("Updating existing prompt")
132:                         with open(prompt_path, "w", encoding="utf-8") as f:
133:                             f.write(prompt_text)
134:                     with open(prompt_path, "r", encoding="utf-8") as f:
135:                         task = f.read()
136:                     filename = prompt_path.stem
137: 
138:                 # Configure repository directory for this prompt
139:                 base_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo"
140:                 repo_dir = base_repo_dir / prompt_name
141:                 repo_dir.mkdir(parents=True, exist_ok=True)
142:                 set_prompt_name(prompt_name)
143:                 set_constant("REPO_DIR", repo_dir)
144:                 write_constants_to_file()
145:                 
146:                 logging.info(f"Starting agent runner in background thread for task: {prompt_name}")
147:                 coro = self.agent_runner(task, self)
148:                 self.socketio.start_background_task(asyncio.run, coro)
149:                 return render_template("web_ide.html")
150:             except FileNotFoundError as e:
151:                 logging.error(f"File not found in run_agent route: {e}", exc_info=True)
152:                 return jsonify({"error": f"File not found: {str(e)}"}), 404
153:             except PermissionError as e:
154:                 logging.error(f"Permission error in run_agent route: {e}", exc_info=True)
155:                 return jsonify({"error": f"Permission error: {str(e)}"}), 403
156:             except Exception as e:
157:                 logging.error(f"Unexpected error in run_agent route: {e}", exc_info=True)
158:                 return jsonify({"error": f"Unexpected error starting agent: {str(e)}"}), 500
159: 
160:         @self.app.route("/messages")
161:         def get_messages():
162:             logging.info("Serving messages")
163:             return jsonify(
164:                 {
165:                     "user": self.user_messages,
166:                     "assistant": self.assistant_messages,
167:                     "tool": self.tool_results,
168:                 }
169:             )
170: 
171:         @self.app.route("/api/prompts/<path:filename>")
172:         def api_get_prompt(filename):
173:             """Return the raw content of a prompt file."""
174:             logging.info(f"Serving prompt content for: {filename}")
175:             try:
176:                 prompt_path = PROMPTS_DIR / filename
177:                 with open(prompt_path, "r", encoding="utf-8") as f:
178:                     data = f.read()
179:                 return data, 200, {"Content-Type": "text/plain; charset=utf-8"}
180:             except FileNotFoundError:
181:                 logging.error(f"Prompt not found: {filename}")
182:                 return "Prompt not found", 404
183: 
184:         @self.app.route("/api/tasks")
185:         def api_get_tasks():
186:             """Return the list of available tasks."""
187:             logging.info("Serving tasks list")
188:             try:
189:                 prompt_files = list(PROMPTS_DIR.glob("*.md"))
190:                 tasks = [file.name for file in prompt_files]
191:                 return jsonify(tasks)
192:             except Exception as e:
193:                 logging.error(f"Error loading tasks: {e}")
194:                 return jsonify([]), 500
195: 
196:         @self.app.route("/api/files")
197:         def api_get_files():
198:             """Return the file tree."""
199:             repo_dir = get_constant("REPO_DIR")
200:             if not repo_dir:
201:                 return jsonify({"error": "REPO_DIR not configured"}), 500
202: 
203:             def get_file_tree(path):
204:                 tree = []
205:                 for item in sorted(Path(path).iterdir()):
206:                     node = {"name": item.name, "path": str(item.relative_to(repo_dir))}
207:                     if item.is_dir():
208:                         node["type"] = "directory"
209:                         node["children"] = get_file_tree(item)
210:                     else:
211:                         node["type"] = "file"
212:                     tree.append(node)
213:                 return tree
214: 
215:         @self.app.route("/api/file_tree")
216:         def api_file_list():
217:             """Return a list of files under the current repository."""
218:             repo_dir = Path(get_constant("REPO_DIR"))
219:             files = [
220:                 str(p.relative_to(repo_dir))
221:                 for p in repo_dir.rglob("*")
222:                 if p.is_file()
223:             ]
224:             return jsonify(files)
225: 
226:         @self.app.route("/api/file")
227:         def api_get_file():
228:             """Return the contents of a file within the repo."""
229:             rel_path = request.args.get("path", "")
230:             repo_dir = Path(get_constant("REPO_DIR"))
231:             safe_path = os.path.normpath(rel_path)
232:             
233:             try:
234:                 file_path = repo_dir / safe_path
235:                 if file_path.is_file():
236:                     with open(file_path, 'r', encoding='utf-8') as f:
237:                         content = f.read()
238:                     return jsonify({"content": content})
239:                 else:
240:                     return jsonify({"error": "File not found"}), 404
241:             except Exception as e:
242:                 logging.error(f"Error getting file: {e}")
243:                 return jsonify({"error": "Error getting file"}), 500
244: 
245:         @self.app.route("/api/files/content")
246:         def api_get_file_content():
247:             """Return the content of a file."""
248:             repo_dir = get_constant("REPO_DIR")
249:             if not repo_dir:
250:                 return jsonify({"error": "REPO_DIR not configured"}), 500
251: 
252:             file_path = request.args.get("path")
253:             if not file_path:
254:                 return jsonify({"error": "Missing path parameter"}), 400
255: 
256:             try:
257:                 abs_path = Path(repo_dir) / file_path
258:                 if not abs_path.is_file():
259:                     return jsonify({"error": "File not found"}), 404
260:                 with open(abs_path, "r", encoding="utf-8") as f:
261:                     content = f.read()
262:                 return jsonify({"content": content})
263:             except Exception as e:
264:                 logging.error(f"Error getting file content: {e}")
265:                 return jsonify({"error": "Error getting file content"}), 500
266: 
267: 
268:         @self.app.route("/tools")
269:         def tools_route():
270:             """Display available tools."""
271:             tool_list = []
272:             for tool in self.tool_collection.tools.values():
273:                 info = tool.to_params()["function"]
274:                 tool_list.append({"name": info["name"], "description": info["description"]})
275:             return render_template("tool_list.html", tools=tool_list)
276: 
277:         @self.app.route("/tools/<tool_name>", methods=["GET", "POST"])
278:         def run_tool_route(tool_name):
279:             """Run an individual tool from the toolbox."""
280:             tool = self.tool_collection.tools.get(tool_name)
281:             if not tool:
282:                 return "Tool not found", 404
283:             params = tool.to_params()["function"]["parameters"]
284:             result_text = None
285:             if request.method == "POST":
286:                 tool_input = {}
287:                 for param in params.get("properties", {}):
288:                     value = request.form.get(param)
289:                     if value:
290:                         pinfo = params["properties"].get(param, {})
291:                         if pinfo.get("type") == "integer":
292:                             try:
293:                                 tool_input[param] = int(value)
294:                             except ValueError:
295:                                 tool_input[param] = value
296:                         elif pinfo.get("type") == "array":
297:                             try:
298:                                 tool_input[param] = json.loads(value)
299:                             except Exception:
300:                                 tool_input[param] = [v.strip() for v in value.split(',') if v.strip()]
301:                         else:
302:                             tool_input[param] = value
303:                 try:
304:                     result = asyncio.run(self.tool_collection.run(tool_name, tool_input))
305:                     result_text = result.output or result.error
306:                 except Exception as exc:
307:                     result_text = str(exc)
308:             return render_template(
309:                 "tool_form.html",
310:                 tool_name=tool_name,
311:                 params=params,
312:                 result=result_text,
313:             )
314: 
315:         @self.app.route("/browser")
316:         def file_browser_route():
317:             """Serve the VS Code-style file browser interface."""
318:             logging.info("Serving file browser interface")
319:             return render_template("file_browser.html")
320: 
321:         @self.app.route("/api/file-tree")
322:         def api_file_tree():
323:             """Return the file tree structure for the current REPO_DIR."""
324:             logging.info("Serving file tree")
325:             try:
326:                 repo_dir = Path(get_constant("REPO_DIR"))
327:                 if not repo_dir.exists():
328:                     return jsonify([])
329:                 
330:                 def build_tree(path):
331:                     items = []
332:                     try:
333:                         for item in sorted(path.iterdir()):
334:                             # Skip hidden files and directories
335:                             if item.name.startswith('.'):
336:                                 continue
337:                             
338:                             if item.is_dir():
339:                                 items.append({
340:                                     'name': item.name,
341:                                     'path': str(item),
342:                                     'type': 'directory',
343:                                     'children': build_tree(item)
344:                                 })
345:                             else:
346:                                 items.append({
347:                                     'name': item.name,
348:                                     'path': str(item),
349:                                     'type': 'file'
350:                                 })
351:                     except PermissionError:
352:                         pass
353:                     return items
354:                 
355:                 tree = build_tree(repo_dir)
356:                 return jsonify(tree)
357:             except Exception as e:
358:                 logging.error(f"Error building file tree: {e}")
359:                 return jsonify([])
360: 
361:         @self.app.route("/api/file-content")
362:         def api_file_content():
363:             """Return the content of a specific file."""
364:             file_path = request.args.get('path')
365:             if not file_path:
366:                 return "File path is required", 400
367:             
368:             try:
369:                 path = Path(file_path)
370:                 # Security check - ensure the path is within REPO_DIR
371:                 repo_dir = Path(get_constant("REPO_DIR"))
372:                 if not path.resolve().is_relative_to(repo_dir.resolve()):
373:                     return "Access denied", 403
374:                 
375:                 if not path.exists():
376:                     return "File not found", 404
377:                 
378:                 if not path.is_file():
379:                     return "Path is not a file", 400
380:                 
381:                 # Try to read as text, handle binary files gracefully
382:                 try:
383:                     with open(path, 'r', encoding='utf-8') as f:
384:                         content = f.read()
385:                 except UnicodeDecodeError:
386:                     # If it's a binary file, return a message instead
387:                     return "Binary file - cannot display content", 200
388:                 
389:                 return content, 200, {"Content-Type": "text/plain; charset=utf-8"}
390:                 
391:             except Exception as e:
392:                 logging.error(f"Error reading file {file_path}: {e}")
393:                 return f"Error reading file: {str(e)}", 500
394: 
395:         logging.info("Routes set up")
396: 
397:     def setup_socketio_events(self):
398:         logging.info("Setting up SocketIO events")
399: 
400:         @self.socketio.on("connect")
401:         def handle_connect():
402:             logging.info("Client connected")
403: 
404:         @self.socketio.on("disconnect")
405:         def handle_disconnect():
406:             logging.info("Client disconnected")
407: 
408:         @self.socketio.on("user_input")
409:         def handle_user_input(data):
410:             user_input = data.get("message", "") or data.get("input", "")
411:             logging.info(f"Received user input: {user_input}")
412:             # Queue is thread-safe; use blocking put to notify waiting tasks
413:             self.input_queue.put(user_input)
414: 
415:         @self.socketio.on("tool_response")
416:         def handle_tool_response(data):
417:             params = data.get("input", {}) if data.get("action") != "cancel" else {}
418:             logging.info(f"Received tool response: {data.get('action', 'execute')}")
419:             self.tool_queue.put(params)
420: 
421:         @self.socketio.on("interrupt_agent")
422:         def handle_interrupt_agent():
423:             logging.info("Received interrupt agent request")
424:             # This could be used to signal the agent to stop processing
425:             self.input_queue.put("INTERRUPT")
426:         logging.info("SocketIO events set up")
427: 
428:     def start_server(self, host="0.0.0.0", port=5002):
429:         logging.info(f"Starting server on {host}:{port}")
430:         self.socketio.run(self.app, host=host, port=port, use_reloader=False, allow_unsafe_werkzeug=True)
431: 
432:     def add_message(self, msg_type, content):
433:         logging.info(f"Adding message of type {msg_type}")
434:         log_message(msg_type, content)
435:         if msg_type == "user":
436:             self.user_messages.append(content)
437:             # Also emit to file browser
438:             self.socketio.emit("user_message", {"content": content})
439:         elif msg_type == "assistant":
440:             self.assistant_messages.append(content)
441:             # Also emit to file browser
442:             self.socketio.emit("assistant_message", {"content": content})
443:         elif msg_type == "tool":
444:             self.tool_results.append(content)
445:             # Parse tool result for file browser
446:             if isinstance(content, str):
447:                 lines = content.split('\n')
448:                 tool_name = "Unknown"
449:                 if lines:
450:                     first_line = lines[0].strip()
451:                     if first_line.startswith('Tool:'):
452:                         tool_name = first_line.replace('Tool:', '').strip()
453:                 self.socketio.emit("tool_result", {"tool_name": tool_name, "result": content})
454:                 
455:                 # Check if this tool might have created/modified files
456:                 if any(keyword in content.lower() for keyword in ['created', 'wrote', 'generated', 'saved', 'modified', 'updated']):
457:                     # Emit file tree update after a short delay asynchronously
458:                     self.socketio.start_background_task(self._emit_file_tree_update)
459:                     
460:         self.broadcast_update()
461: 
462:     def broadcast_update(self):
463:         logging.info("Broadcasting update to clients")
464:         self.socketio.emit(
465:             "update",
466:             {
467:                 "user": self.user_messages,
468:                 "assistant": self.assistant_messages,
469:                 "tool": self.tool_results,
470:             },
471:         )
472: 
473:     async def wait_for_user_input(self, prompt_message: str | None = None) -> str:
474:         """Await the next user input sent via the web UI input queue."""
475:         if prompt_message:
476:             logging.info(f"Emitting agent_prompt: {prompt_message}")
477:             self.socketio.emit("agent_prompt", {"message": prompt_message})
478: 
479:         loop = asyncio.get_running_loop()
480:         user_response = await loop.run_in_executor(None, self.input_queue.get)
481: 
482:         # Clear the prompt after input is received
483:         logging.info("Emitting agent_prompt_clear")
484:         self.socketio.emit("agent_prompt_clear")
485: 
486:         return user_response
487: 
488:     async def confirm_tool_call(self, tool_name: str, args: dict, schema: dict) -> dict | None:
489:         """Send a tool prompt to the web UI and wait for edited parameters."""
490:         self.socketio.emit(
491:             "tool_prompt",
492:             {"tool": tool_name, "values": args, "schema": schema},
493:         )
494:         loop = asyncio.get_running_loop()
495:         params = await loop.run_in_executor(None, self.tool_queue.get)
496:         self.socketio.emit("tool_prompt_clear")
497:         return params
````
