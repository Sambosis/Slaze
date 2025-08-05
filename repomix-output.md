This file is a merged representation of the entire codebase, combined into a single document by Repomix.
The content has been processed where content has been compressed (code blocks are separated by ⋮---- delimiter).

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
- Content has been compressed - code blocks are separated by ⋮---- delimiter
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
__init__.py
agent_display_console.py
command_converter.py
context_helpers.py
file_logger.py
llm_client.py
miniOR.py
output_manager.py
web_ui.py
```

# Files

## File: __init__.py
````python
"""Utility functions and classes used throughout Slaze."""
⋮----
__all__ = []
````

## File: agent_display_console.py
````python
class AgentDisplayConsole
⋮----
def __init__(self)
⋮----
# Configure console for Windows Unicode support
if os.name == 'nt':  # Windows
# Set environment variables for UTF-8 support
⋮----
# Try to set Windows console to UTF-8 mode
⋮----
# Enable UTF-8 mode on Windows
⋮----
# Reconfigure stdout/stderr for UTF-8 (Python 3.7+)
⋮----
# Initialize Rich console with safe settings for Windows
⋮----
def add_message(self, msg_type, content)
⋮----
# Use safer characters for Windows compatibility
⋮----
async def wait_for_user_input(self, prompt_message=">> Your input: ")
⋮----
async def select_prompt_console(self)
⋮----
options = {}
prompt_files = sorted([f for f in PROMPTS_DIR.iterdir() if f.is_file() and f.suffix == '.md'])
⋮----
prompt_lines = []
⋮----
create_new_option_num = len(options) + 1
⋮----
choice = IntPrompt.ask("Enter your choice", choices=[str(i) for i in range(1, create_new_option_num + 1)])
⋮----
prompt_path = options[str(choice)]
task = prompt_path.read_text(encoding="utf-8")
prompt_name = prompt_path.stem
⋮----
new_lines = []
⋮----
line = await self.wait_for_user_input("")
⋮----
task = "\n".join(new_lines)
⋮----
new_filename_input = Prompt.ask("Enter a filename for the new prompt", default="custom_prompt")
filename_stem = new_filename_input.strip().replace(" ", "_").replace(".md", "")
prompt_name = filename_stem
new_prompt_path = PROMPTS_DIR / f"{filename_stem}.md"
new_prompt_lines = []
⋮----
task = "\n".join(new_prompt_lines)
⋮----
task = ""
⋮----
# Configure repository directory for this prompt
base_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo"
repo_dir = base_repo_dir / prompt_name
⋮----
async def confirm_tool_call(self, tool_name: str, args: dict, schema: dict) -> dict | None
⋮----
"""Display tool call parameters and allow the user to edit them."""
⋮----
updated_args = dict(args)
properties = schema.get("properties", {}) if schema else {}
⋮----
current_val = args.get(param, "")
default_str = str(current_val) if current_val is not None else ""
user_val = Prompt.ask(param, default=default_str)
⋮----
pinfo = properties.get(param, {})
````

## File: command_converter.py
````python
logger = logging.getLogger(__name__)
⋮----
class CommandConverter
⋮----
"""
    LLM-based command converter that transforms bash commands to be appropriate
    for the current system environment.
    """
⋮----
def __init__(self)
⋮----
def _get_system_info(self) -> Dict[str, Any]
⋮----
"""Gather system information for command conversion context."""
⋮----
def _build_conversion_prompt(self) -> str
⋮----
"""Build the system prompt for command conversion."""
os_name = self.system_info['os_name']
⋮----
# Build OS-specific examples and rules
⋮----
examples = """EXAMPLES:
⋮----
rules = f"""RULES:
⋮----
async def convert_command(self, original_command: str) -> str
⋮----
"""
        Convert a command using LLM to be appropriate for the current system.
        
        Args:
            original_command: The original bash command to convert
            
        Returns:
            The converted command appropriate for the current system
        """
⋮----
# Get the model from config
model = get_constant("MAIN_MODEL", "anthropic/claude-sonnet-4")
⋮----
# Prepare the conversion request
converted_command = await self._call_llm(model, original_command)
⋮----
# Validate and clean the response
cleaned_command = self._clean_response(converted_command)
⋮----
# Fallback to original command if conversion fails
⋮----
async def _call_llm(self, model: str, command: str) -> str
⋮----
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
⋮----
# Create LLM client and call it
client = create_llm_client(model)
⋮----
max_tokens=200,  # Keep response short
temperature=0.1  # Low temperature for consistent output
⋮----
def _clean_response(self, response: str) -> str
⋮----
"""
        Clean the LLM response to extract just the command.
        
        Args:
            response: The raw LLM response
            
        Returns:
            The cleaned command string
        """
# Remove any markdown code blocks
response = re.sub(r'^```.*?\n|```$', '', response, flags=re.MULTILINE)
⋮----
# Remove leading/trailing whitespace
response = response.strip()
⋮----
# Split by lines and take the first non-empty line
lines = [line.strip() for line in response.split('\n') if line.strip()]
⋮----
command = lines[0]
⋮----
# Basic validation - ensure it looks like a command
if not command or len(command) > 1000:  # Reasonable length limit
⋮----
# Global instance for reuse
_converter_instance: Optional[CommandConverter] = None
⋮----
async def convert_command_for_system(original_command: str) -> str
⋮----
"""
    Convert a bash command to be appropriate for the current system.
    
    Args:
        original_command: The original bash command
        
    Returns:
        The converted command appropriate for the current system
    """
⋮----
_converter_instance = CommandConverter()
````

## File: context_helpers.py
````python
# from config import write_to_file # Removed as it was for ic
# Removed: from load_constants import *
from config import MAIN_MODEL, get_constant, googlepro # Import get_constant
⋮----
# from icecream import ic # Removed
# from rich import print as rr # Removed
⋮----
# ic.configureOutput(includeContext=True, outputFunction=write_to_file) # Removed
⋮----
logger = logging.getLogger(__name__)
⋮----
QUICK_SUMMARIES = []
⋮----
def format_messages_to_restart(messages)
⋮----
"""
    Format a list of messages into a formatted string.
    """
⋮----
output_pieces = []
⋮----
def format_messages_to_string(messages)
⋮----
"""Return a human readable string for a list of messages."""
⋮----
def _val(obj, key, default=None)
⋮----
role = msg.get("role", "unknown").upper()
⋮----
name = _val(_val(tc, "function"), "name")
args = _val(_val(tc, "function"), "arguments")
tc_id = _val(tc, "id")
⋮----
parsed = json.loads(args) if isinstance(args, str) else args
formatted = json.dumps(parsed, indent=2)
⋮----
formatted = str(args)
⋮----
content = msg.get("content")
⋮----
btype = block.get("type")
⋮----
inp = block["input"]
⋮----
"""
    Summarize the most recent messages.
    """
⋮----
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
sum_client = OpenAI(
all_summaries = get_all_summaries()
model = MAIN_MODEL
conversation_text = ""
⋮----
role = msg["role"].upper()
⋮----
content = block.get("text", "")
⋮----
content = (
⋮----
content = item.get("text", "")
⋮----
content = msg["content"]
⋮----
logger.debug(f"conversation_text for summary: {conversation_text[:500]}...") # Log snippet
summary_prompt = f"""Please provide your response in a concise markdown format with short statements that document what happened. Structure your response as a list with clear labels for each step, such as:
response = sum_client.chat.completions.create(
⋮----
max_tokens=get_constant("MAX_SUMMARY_TOKENS", 4000) # Use get_constant
⋮----
# Add error handling for response
⋮----
error_msg = "Error: No valid response received from summary API"
# print(response) # Replaced by logger
⋮----
summary = response.choices[0].message.content
⋮----
# Check if summary is None or empty
⋮----
error_msg = "Error: Empty summary received from API"
⋮----
logger.debug(f"Generated summary: {summary[:500]}...") # Log snippet
⋮----
error_msg = f"Error generating summary: {str(e)}"
⋮----
def filter_messages(messages: List[Dict]) -> List[Dict]
⋮----
"""
    Keep only messages with role 'user' or 'assistant'.
    Also keep any tool_result messages that contain errors.
    """
keep_roles = {"user", "assistant"}
filtered = []
⋮----
# Check if any text in the tool result indicates an error
text = ""
⋮----
def extract_text_from_content(content: Any) -> str
⋮----
text_parts = []
⋮----
def truncate_message_content(content: Any, max_length: int = 150_000) -> Any
⋮----
def add_summary(summary: str) -> None
⋮----
"""Add a new summary to the global list with timestamp and log it to a file."""
stripped_summary = summary.strip()
⋮----
summary_file_path = Path(get_constant("SUMMARY_FILE"))
⋮----
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = f"\n--------------------\n[{timestamp}]\n{stripped_summary}\n--------------------\n"
⋮----
def get_all_summaries() -> str
⋮----
"""Combine all summaries into a chronological narrative."""
⋮----
combined = "\n"
⋮----
async def reorganize_context(messages: List[Dict[str, Any]], summary: str) -> str
⋮----
"""Reorganize the context by filtering and summarizing messages."""
⋮----
# Look for tool results related to image generation
image_generation_results = []
⋮----
# Track image generation results
⋮----
# Add special section for image generation if we found any
⋮----
logger.debug(f"Conversation text for reorganize_context: {conversation_text[:500]}...") # Log snippet
summary_prompt = f"""I need a summary of completed steps and next steps for a project that is ALREADY IN PROGRESS.
⋮----
logger.debug(f"Reorganized context summary: {summary[:500]}...") # Log snippet
⋮----
start_tag = "<COMPLETED>"
end_tag = "</COMPLETED>"
⋮----
completed_items = summary[
⋮----
completed_items = "No completed items found."
⋮----
start_tag = "<NEXT_STEPS>"
end_tag = "</NEXT_STEPS>"
⋮----
steps = summary[
⋮----
steps = "No steps found."
⋮----
# Return default values in case of error
⋮----
"""
    Create a combined context string by filtering and (if needed) summarizing messages
    and appending current file contents.
    """
filtered = filter_messages(messages)
summary = get_all_summaries() # This is a local function in context_helpers
⋮----
file_contents = aggregate_file_states()
⋮----
file_contents = (
⋮----
# Get code skeletons
⋮----
code_skeletons = get_all_current_skeleton()
current_code = get_all_current_code()
# The logic is if there is code, then supply that, if not then supply the skeletons, if there is no code or skeletons, then say there are no code skeletons
⋮----
code_skeletons = current_code
⋮----
code_skeletons = "No code skeletons available."
⋮----
# Extract information about images generated
images_info = ""
⋮----
images_section = file_contents.split("## Generated Images:")[1]
⋮----
images_section = images_section.split("##")[0]
images_info = "## Generated Images:\n" + images_section.strip()
⋮----
# call the LLM and pass it all current messages then the task and ask it to give an updated version of the task
prompt = f""" Your job is to update the task based on the current state of the project.
⋮----
messages_for_llm = [{"role": "user", "content": prompt}]
response = client.chat.completions.create(
⋮----
messages=messages_for_llm, # Corrected variable name
max_tokens=get_constant("MAX_SUMMARY_TOKENS", 20000) # Use get_constant
⋮----
new_task = response.choices[0].message.content
⋮----
combined_content = f"""Original request:
````

## File: file_logger.py
````python
logger = logging.getLogger(__name__)
⋮----
# Import the function but don't redefine it
⋮----
# Define our own if not available in config
def convert_to_docker_path(path: Union[str, Path]) -> str
⋮----
"""
            Convert a local Windows path to a Docker container path.
            No longer converts to Docker path, returns original path.
            Args:
                path: The local path to convert

            Returns:
                The original path as a string
            """
⋮----
# Fallback if config module is not available
def get_constant(name)
⋮----
# Default values for essential constants
defaults = {
⋮----
def convert_to_docker_path(path: Union[str, Path]) -> str
⋮----
"""
        Convert a local Windows path to a Docker container path.
        No longer converts to Docker path, returns original path.
        Args:
            path: The local path to convert

        Returns:
            The original path as a string
        """
⋮----
# File for logging operations
⋮----
LOG_FILE = get_constant("LOG_FILE")
⋮----
LOG_FILE = os.path.join(
⋮----
# In-memory tracking of file operations # FILE_OPERATIONS removed
# FILE_OPERATIONS = {} # Removed
⋮----
# Ensure log directory exists
⋮----
# If log file doesn't exist, create an empty one
⋮----
# Track file operations # Removed unused global variables
# file_operations = [] # Removed
# tracked_files = set() # Removed
# file_contents = {} # Removed
⋮----
"""
    Log a file operation (create, update, delete) with enhanced metadata handling.

    Args:
        file_path: Path to the file
        operation: Type of operation ('create', 'update', 'delete')
        content: Optional content for the file
        metadata: Optional dictionary containing additional metadata (e.g., image generation prompt)
    """
# Defensively initialize metadata to prevent NoneType errors
⋮----
metadata = {}
⋮----
# Ensure file_path is a Path object
⋮----
file_path = Path(file_path)
⋮----
# Create a string representation of the file path for consistent logging
file_path_str = str(file_path)
⋮----
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
extension = file_path.suffix.lower() if file_path.suffix else ""
⋮----
# Determine if the file is an image
is_image = extension in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"]
mime_type = mimetypes.guess_type(file_path_str)[0]
⋮----
is_image = True
⋮----
# Track file operations in memory # Removed FILE_OPERATIONS update logic
# if file_path_str not in FILE_OPERATIONS:
#     FILE_OPERATIONS[file_path_str] = {
#         "operations": [],
#         "last_updated": timestamp,
#         "extension": extension,
#         "is_image": is_image,
#         "mime_type": mime_type,
#     }
#
# # Update the in-memory tracking
# FILE_OPERATIONS[file_path_str]["operations"].append(
#     {"timestamp": timestamp, "operation": operation}
# )
# FILE_OPERATIONS[file_path_str]["last_updated"] = timestamp
⋮----
# Load existing log data or create a new one
log_data = {"files": {}}
⋮----
log_data = json.load(f)
⋮----
# If the log file is corrupted, start fresh
⋮----
# Create or update the file entry in the log
⋮----
# Add the operation to the log
⋮----
# Update the metadata if provided
⋮----
# Ensure we have a metadata dictionary
⋮----
# Update with new metadata
⋮----
# Store the content if provided, otherwise try to read it from the file
file_content = content
⋮----
# Only try to read the file if it exists and content wasn't provided
⋮----
# Handle different file types appropriately
⋮----
# For images, store base64 encoded content
⋮----
img_content = f.read()
⋮----
# Add file size to metadata
⋮----
# For code and text files, store as text
⋮----
text_content = f.read()
⋮----
# For other binary files, store base64 encoded
⋮----
bin_content = f.read()
⋮----
# Don't fail the entire operation, just log the error
⋮----
# Use the provided content
⋮----
# Don't fail the entire operation, just log the error
⋮----
# Update last_updated timestamp
⋮----
# Write the updated log data back to the log file
⋮----
def aggregate_file_states() -> str
⋮----
"""
    Collect information about all tracked files and their current state.

    Returns:
        A formatted string with information about all files.
    """
LOG_FILE = Path(get_constant("LOG_FILE"))
⋮----
log_data = json.loads(f.read())
⋮----
# Group files by type
image_files = []
code_files = []
text_files = []
other_files = []
⋮----
file_type = file_info.get("file_type", "other")
⋮----
# Get the Docker path for display
docker_path = file_info.get("docker_path", convert_to_docker_path(file_path))
⋮----
# Sort operations by timestamp to get the latest state
operations = sorted(
⋮----
latest_operation = operations[0] if operations else {"operation": "unknown"}
⋮----
image_metadata = file_info.get("image_metadata", {})
⋮----
basic_info = file_info.get("basic_info", {})
⋮----
# Format the output
output = []
⋮----
# Add syntax highlighting based on file extension
extension = Path(code["path"]).suffix.lower()
lang = get_language_from_extension(extension)
⋮----
extension = Path(text["path"]).suffix.lower()
⋮----
def extract_code_skeleton(source_code: Union[str, Path]) -> str
⋮----
"""
    Extract a code skeleton from existing Python code.

    This function takes Python code and returns just the structure: imports,
    class definitions, method/function signatures, and docstrings, with
    implementations replaced by 'pass' statements.

    Args:
        source_code: Either a path to a Python file or a string containing Python code

    Returns:
        str: The extracted code skeleton
    """
# Load the source code
⋮----
code_str = file.read()
⋮----
code_str = str(source_code)
⋮----
# Parse the code into an AST
⋮----
tree = ast.parse(code_str)
⋮----
# Extract imports
imports = []
⋮----
module = node.module or ""
names = ", ".join(
⋮----
# Helper function to handle complex attributes
def format_attribute(node)
⋮----
"""Helper function to recursively format attribute expressions"""
⋮----
# Add support for ast.Subscript nodes (like List[int])
⋮----
# Use ast.unparse for Python 3.9+ or manual handling for earlier versions
⋮----
# Simplified handling for common cases
⋮----
base = node.value.id
⋮----
base = format_attribute(node.value)
# Simple handling for slice
⋮----
return f"{base}[...]"  # Fallback for complex slices
return f"{base}[...]"  # Fallback for complex cases
⋮----
# Fallback for other node types - use ast.unparse if available
⋮----
# Get docstrings and function/class signatures
class CodeSkeletonVisitor(ast.NodeVisitor)
⋮----
def __init__(self)
⋮----
def visit_Import(self, node)
⋮----
# Already handled above
⋮----
def visit_ImportFrom(self, node)
⋮----
def visit_ClassDef(self, node)
⋮----
# Extract class definition with inheritance
bases = []
⋮----
# Use the helper function to handle nested attributes
⋮----
# Fallback for other complex cases
⋮----
class_def = f"class {node.name}"
⋮----
# Add class definition
⋮----
# Add docstring if it exists
docstring = ast.get_docstring(node)
⋮----
doc_lines = docstring.split("\n")
⋮----
# Increment indent for class members
⋮----
# Visit all class members
⋮----
# If no members were added, add a pass statement
⋮----
# Restore indent
⋮----
def visit_FunctionDef(self, node)
⋮----
# Extract function signature
args = []
defaults = [None] * (
⋮----
# Process regular arguments
⋮----
arg_str = arg.arg
# Add type annotation if available
⋮----
# Use the helper function to handle complex types
⋮----
arg_str += ": ..."  # Fallback for complex annotations
⋮----
# Add default value if available
⋮----
# Simplified handling for common default values
⋮----
arg_str += " = ..."  # Fallback for complex defaults
⋮----
# Handle *args
⋮----
# Handle keyword-only args
⋮----
kw_str = kwarg.arg
⋮----
kw_str += " = ..."  # Fallback for complex defaults
⋮----
# Handle **kwargs
⋮----
# Build function signature
func_def = f"def {node.name}({', '.join(args)})"
⋮----
# Add return type if specified
⋮----
# Add function definition
⋮----
# Add pass statement in place of the function body
⋮----
# Run the visitor on the AST
visitor = CodeSkeletonVisitor()
⋮----
# Combine imports and code skeleton
result = []
⋮----
# Add all imports first
⋮----
result.append("")  # Add a blank line after imports
⋮----
# Add the rest of the code skeleton
⋮----
def get_all_current_code() -> str
⋮----
"""
    Returns all the current code in the project as a string.
    This function is used to provide context about the existing code to the LLM.

    Returns:
        A string with all the current code.
    """
⋮----
# Ensure log file exists
⋮----
# Initialize a new log file so future operations work
⋮----
# Load log data with robust error handling
⋮----
# Reset the log file with valid JSON
⋮----
# Validate log structure
⋮----
# Fix the format
⋮----
# Process each file in the log
⋮----
# Skip files that have been deleted (last operation is 'delete')
operations = file_data.get("operations", [])
⋮----
# Get content and metadata
content = file_data.get("content")
⋮----
# Skip images and binary files
is_image = file_data.get("is_image", False)
mime_type = file_data.get("mime_type", "")
extension = file_data.get("extension", "").lower()
⋮----
# Only include code files
⋮----
# Continue processing other files
⋮----
# Sort files by path for consistent output
⋮----
# Format the output
⋮----
lang = get_language_from_extension(code["extension"])
⋮----
logger.error( # Replaced print with logger.error
⋮----
# Add a simpler version without failing
⋮----
# Return the formatted output (This was missing)
⋮----
def get_all_current_skeleton() -> str
⋮----
"""
    Get the skeleton of all Python code files.

    Returns:
        A formatted string with the skeleton of all Python code files.
    """
⋮----
output = ["# All Python File Skeletons"]
⋮----
# Skip files that have been deleted (last operation is 'delete')
operations = file_info.get("operations", [])
⋮----
# Only process Python files
extension = file_info.get("extension", "").lower()
⋮----
# Get Docker path for display
docker_path = convert_to_docker_path(file_path)
⋮----
# Look for skeleton in metadata
metadata = file_info.get("metadata", {})
skeleton = metadata.get("skeleton", "")
⋮----
# If no skeleton in metadata, try to extract it from the content
⋮----
skeleton = extract_code_skeleton(file_info.get("content", ""))
⋮----
skeleton = "# Failed to extract skeleton"
⋮----
# Add file header
⋮----
# Add skeleton with syntax highlighting
⋮----
def get_language_from_extension(extension: str) -> str
⋮----
"""
    Map file extensions to programming languages for syntax highlighting.

    Args:
        extension: The file extension (e.g., '.py', '.js')

    Returns:
        The corresponding language name for syntax highlighting.
    """
extension = extension.lower()
mapping = {
⋮----
def should_skip_for_zip(path)
⋮----
"""
    Determine if a file or directory should be skipped when creating a ZIP file.

    Args:
        path: Path to check

    Returns:
        bool: True if the path should be skipped, False otherwise
    """
path_str = str(path).lower()
⋮----
# Skip virtual environment files and directories
⋮----
# On Windows, particularly skip Linux-style virtual env paths
⋮----
# Skip common directories not needed in the ZIP
dirs_to_skip = [
⋮----
def archive_logs()
⋮----
"""Archive all log files in LOGS_DIR by moving them to an archive folder with a timestamp."""
⋮----
# Create timestamp for the archive folder
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
archive_dir = Path(LOGS_DIR, "archive", timestamp)
⋮----
# Get all files in LOGS_DIR
log_path = Path(LOGS_DIR)
log_files = [f for f in log_path.iterdir() if f.is_file()]
⋮----
# Skip archiving if there are no files
⋮----
# Move each file to the archive directory
⋮----
# Skip archive directory itself
⋮----
# Create destination path
dest_path = Path(archive_dir, file_path.name)
⋮----
# Copy the file if it exists (some might be created later)
⋮----
# Clear the original file but keep it
````

## File: llm_client.py
````python
logger = logging.getLogger(__name__)
⋮----
class LLMClient(ABC)
⋮----
"""Abstract base class for LLM clients."""
⋮----
@abstractmethod
    async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str
⋮----
"""Call the LLM with messages and return response."""
⋮----
class OpenRouterClient(LLMClient)
⋮----
"""OpenRouter API client."""
⋮----
def __init__(self, model: str)
⋮----
async def call(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.1) -> str
⋮----
headers = {
⋮----
payload = {
⋮----
error_text = await response.text()
⋮----
result = await response.json()
⋮----
class OpenAIClient(LLMClient)
⋮----
"""OpenAI API client."""
⋮----
class AnthropicClient(LLMClient)
⋮----
"""Anthropic API client."""
⋮----
# Convert messages format for Anthropic
system_content = ""
user_messages = []
⋮----
system_content = msg["content"]
⋮----
def create_llm_client(model: str) -> LLMClient
⋮----
"""Factory function to create appropriate LLM client based on model name."""
⋮----
# Default to OpenRouter for unknown models
````

## File: miniOR.py
````python
openai41mini : str = "openai/gpt-4.1-mini"
gemma3n4b : str = "google/gemma-3n-e4b-it"
sonnet4 : str = "anthropic/claude-sonnet-4"
openai41 : str = "openai/gpt-4.1"
openaio3 : str = "openai/o3"
openaio3pro : str = "openai/o3-pro"
googlepro : str = "google/gemini-2.5-pro-preview"
googleflash : str = "google/gemini-2.5-flash-preview"
googleflashlite : str = "google/gemini-2.5-flash-lite-preview-06-17"
grok4 : str = "x-ai/grok-4"
SUMMARY_MODEL : str = googleflashlite  # Model for summaries
MAIN_MODEL : str = f"{googleflashlite}"  # Primary model for main agent operations
CODE_MODEL : str = f"{googleflashlite}:web"  # Model for code generation tasks
BASE_URL : str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL : str = MAIN_MODEL
⋮----
BASE_SYSTEM_PROMPT : str = "You are a helpful AI assistant. "
⋮----
"""
    Get the appropriate LLM instance based on the async flag.
    """
⋮----
def chat(prompt_str, model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL))
⋮----
"""
    Send a chat message to the LLM.
    
    """
llm = get_llm()
messages = [{"role": "user", "content": prompt_str}]
⋮----
# Example usage
response = chat(prompt_str="Hello, how can you assist me today?")
````

## File: output_manager.py
````python
from config import get_constant  # Updated import
⋮----
logger = logging.getLogger(__name__)
⋮----
class OutputManager
⋮----
LOGS_DIR = Path(get_constant("LOGS_DIR"))
⋮----
def save_image(self, base64_data: str) -> Optional[Path]
⋮----
"""Save base64 image data to file and return path."""
⋮----
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
image_hash = hashlib.md5(base64_data.encode()).hexdigest()[:8]
image_path = self.image_dir / f"image_{timestamp}_{image_hash}.png"
⋮----
image_data = base64.b64decode(base64_data)
⋮----
def format_tool_output(self, result: "ToolResult", tool_name: str)
⋮----
"""Format and display tool output."""
⋮----
output_text = f"Used Tool: {tool_name}\n"
⋮----
text = self._truncate_string(
⋮----
image_path = self.save_image(result.base64_image)
⋮----
# self.display., output_text)
⋮----
def format_api_response(self, response: Dict[str, Any])
⋮----
"""Format and display API response."""
⋮----
def format_content_block(self, block: Dict[str, Any]) -> None
⋮----
"""Format and display content block."""
⋮----
safe_input = {
⋮----
"""Format and display recent conversation."""
⋮----
# recent_messages = messages[:num_recent] if len(messages) > num_recent else messages
recent_messages = messages[-num_recent:]
⋮----
def _format_user_content(self, content: Any)
⋮----
"""Format and display user content."""
⋮----
#     self.display., text)
# elif item.get("type") == "image":
#     self.display., "📸 Screenshot captured")
⋮----
# self.display., text)
⋮----
def _format_assistant_content(self, content: Any)
⋮----
"""Format and display assistant content."""
⋮----
tool_input = content_block.get("input", "")
⋮----
tool_input = json.loads(tool_input)
⋮----
# self.display., (tool_name, f"Input: {input_text}"))
⋮----
def _truncate_string(self, text: str, max_length: int = 500) -> str
⋮----
"""Truncate a string to a max length with ellipsis."""
````

## File: web_ui.py
````python
def log_message(msg_type, message)
⋮----
"""Log a message to a file."""
⋮----
emojitag = "🤡 "
⋮----
emojitag = "🧞‍♀️ "
⋮----
emojitag = "📎 "
⋮----
emojitag = "❓ "
log_file = os.path.join(LOGS_DIR, f"{msg_type}_messages.log")
⋮----
class WebUI
⋮----
def __init__(self, agent_runner)
⋮----
# More robust path for templates
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
⋮----
# Using a standard Queue for cross-thread communication
⋮----
# Import tools lazily to avoid circular imports
⋮----
# BashTool,
# OpenInterpreterTool,
⋮----
# OpenInterpreterTool(display=self),  # Uncommented and enabled for testing
⋮----
def setup_routes(self)
⋮----
@self.app.route("/")
        def select_prompt_route()
⋮----
prompt_files = list(PROMPTS_DIR.glob("*.md"))
options = [file.name for file in prompt_files]
⋮----
@self.app.route("/classic")
        def select_prompt_classic_route()
⋮----
@self.app.route("/modern")
        def select_prompt_modern_route()
⋮----
@self.app.route("/run_agent", methods=["POST"])
        def run_agent_route()
⋮----
choice = request.form.get("choice")
filename = request.form.get("filename")
prompt_text = request.form.get("prompt_text")
⋮----
new_prompt_path = PROMPTS_DIR / f"{filename}.md"
prompt_name = Path(filename).stem
⋮----
task = prompt_text or ""
⋮----
prompt_path = PROMPTS_DIR / choice
⋮----
prompt_name = prompt_path.stem
⋮----
task = f.read()
filename = prompt_path.stem
⋮----
# Configure repository directory for this prompt
base_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo"
repo_dir = base_repo_dir / prompt_name
⋮----
coro = self.agent_runner(task, self)
⋮----
@self.app.route("/messages")
        def get_messages()
⋮----
@self.app.route("/api/prompts/<path:filename>")
        def api_get_prompt(filename)
⋮----
"""Return the raw content of a prompt file."""
⋮----
prompt_path = PROMPTS_DIR / filename
⋮----
data = f.read()
⋮----
@self.app.route("/api/tasks")
        def api_get_tasks()
⋮----
"""Return the list of available tasks."""
⋮----
tasks = [file.name for file in prompt_files]
⋮----
@self.app.route("/api/files")
        def api_get_files()
⋮----
"""Return the file tree."""
repo_dir = get_constant("REPO_DIR")
⋮----
def get_file_tree(path)
⋮----
tree = []
⋮----
node = {"name": item.name, "path": str(item.relative_to(repo_dir))}
⋮----
@self.app.route("/api/file_tree")
        def api_file_list()
⋮----
"""Return a list of files under the current repository."""
repo_dir = Path(get_constant("REPO_DIR"))
files = [
⋮----
@self.app.route("/api/file")
        def api_get_file()
⋮----
"""Return the contents of a file within the repo."""
rel_path = request.args.get("path", "")
⋮----
safe_path = os.path.normpath(rel_path)
⋮----
file_path = repo_dir / safe_path
⋮----
content = f.read()
⋮----
@self.app.route("/api/files/content")
        def api_get_file_content()
⋮----
"""Return the content of a file."""
⋮----
file_path = request.args.get("path")
⋮----
abs_path = Path(repo_dir) / file_path
⋮----
@self.app.route("/tools")
        def tools_route()
⋮----
"""Display available tools."""
tool_list = []
⋮----
info = tool.to_params()["function"]
⋮----
@self.app.route("/tools/<tool_name>", methods=["GET", "POST"])
        def run_tool_route(tool_name)
⋮----
"""Run an individual tool from the toolbox."""
tool = self.tool_collection.tools.get(tool_name)
⋮----
params = tool.to_params()["function"]["parameters"]
result_text = None
⋮----
tool_input = {}
⋮----
value = request.form.get(param)
⋮----
pinfo = params["properties"].get(param, {})
⋮----
result = asyncio.run(self.tool_collection.run(tool_name, tool_input))
result_text = result.output or result.error
⋮----
result_text = str(exc)
⋮----
@self.app.route("/browser")
        def file_browser_route()
⋮----
"""Serve the VS Code-style file browser interface."""
⋮----
@self.app.route("/api/file-tree")
        def api_file_tree()
⋮----
"""Return the file tree structure for the current REPO_DIR."""
⋮----
def build_tree(path)
⋮----
items = []
⋮----
# Skip hidden files and directories
⋮----
tree = build_tree(repo_dir)
⋮----
@self.app.route("/api/file-content")
        def api_file_content()
⋮----
"""Return the content of a specific file."""
file_path = request.args.get('path')
⋮----
path = Path(file_path)
# Security check - ensure the path is within REPO_DIR
⋮----
# Try to read as text, handle binary files gracefully
⋮----
# If it's a binary file, return a message instead
⋮----
def setup_socketio_events(self)
⋮----
@self.socketio.on("connect")
        def handle_connect()
⋮----
@self.socketio.on("disconnect")
        def handle_disconnect()
⋮----
@self.socketio.on("user_input")
        def handle_user_input(data)
⋮----
user_input = data.get("message", "") or data.get("input", "")
⋮----
# Queue is thread-safe; use blocking put to notify waiting tasks
⋮----
@self.socketio.on("tool_response")
        def handle_tool_response(data)
⋮----
params = data.get("input", {}) if data.get("action") != "cancel" else {}
⋮----
@self.socketio.on("interrupt_agent")
        def handle_interrupt_agent()
⋮----
# This could be used to signal the agent to stop processing
⋮----
def start_server(self, host="0.0.0.0", port=5002)
⋮----
def add_message(self, msg_type, content)
⋮----
# Also emit to file browser
⋮----
# Parse tool result for file browser
⋮----
lines = content.split('\n')
tool_name = "Unknown"
⋮----
first_line = lines[0].strip()
⋮----
tool_name = first_line.replace('Tool:', '').strip()
⋮----
# Check if this tool might have created/modified files
⋮----
# Emit file tree update after a short delay asynchronously
⋮----
def broadcast_update(self)
⋮----
async def wait_for_user_input(self, prompt_message: str | None = None) -> str
⋮----
"""Await the next user input sent via the web UI input queue."""
⋮----
loop = asyncio.get_running_loop()
user_response = await loop.run_in_executor(None, self.input_queue.get)
⋮----
# Clear the prompt after input is received
⋮----
async def confirm_tool_call(self, tool_name: str, args: dict, schema: dict) -> dict | None
⋮----
"""Send a tool prompt to the web UI and wait for edited parameters."""
⋮----
params = await loop.run_in_executor(None, self.tool_queue.get)
````
