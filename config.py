import json
import logging.handlers
import os
import platform
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from flask import g
import openai

# Load environment variables from .env file
load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
except Exception:
    pass

# Configure UTF-8 for Windows console and subprocess environments
if platform.system() == "Windows":
    # Set UTF-8 environment variables for subprocess calls
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONLEGACYWINDOWSFSENCODING', '0')
    os.environ.setdefault('PYTHONUTF8', '1')
    
    # Try to set Windows console to UTF-8 mode
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], capture_output=True, check=False)
    except Exception:
        pass

# Logging constants
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
LOG_FILE_APP = "logs/app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
OS_NAME = platform.system()

# Define the top-level directory
TOP_LEVEL_DIR = Path.cwd()
WORKER_DIR = TOP_LEVEL_DIR # For worker processes, from load_constants.py

# Define the repository directory
REPO_DIR = TOP_LEVEL_DIR / "repo"

# Define other relevant paths
SYSTEM_PROMPT_DIR = TOP_LEVEL_DIR / "system_prompt"
SYSTEM_PROMPT_FILE = SYSTEM_PROMPT_DIR / "system_prompt.md"
BASH_PROMPT_DIR = TOP_LEVEL_DIR / "tools"
BASH_PROMPT_FILE = BASH_PROMPT_DIR / "bash.md"
LLM_GEN_CODE_DIR = None  # Initialize as None
TOOLS_DIR = TOP_LEVEL_DIR / "tools"
SCRIPTS_DIR = TOP_LEVEL_DIR / "scripts"
TESTS_DIR = TOP_LEVEL_DIR / "tests"
LOGS_DIR = TOP_LEVEL_DIR / "logs"
PROMPTS_DIR = TOP_LEVEL_DIR / "prompts"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "file_creation_log.json" # For file operations audit
MESSAGES_FILE = LOGS_DIR / "messages.md" # For conversation history
SUMMARY_FILE = LOGS_DIR / "summaries/summary.md" # For storing summaries, from load_constants.py
CODE_FILE = LOGS_DIR / "code_messages.py" # For WriteCodeTool code logging
USER_LOG_FILE = LOGS_DIR / "user_messages.log"
ASSISTANT_LOG_FILE = LOGS_DIR / "assistant_messages.log"
TOOL_LOG_FILE = LOGS_DIR / "tool_messages.log"

kimi2 = "moonshotai/kimi-k2"
openai41mini = "openai/gpt-4.1-mini"
gemma3n4b = "google/gemma-3n-e4b-it"
sonnet4 = "anthropic/claude-sonnet-4"
openai41 = "openai/gpt-4.1"
openaio3 = "openai/o3"
openaio3pro = "openai/o3-pro"
googlepro = "google/gemini-2.5-pro-preview"
googleflash = "google/gemini-2.5-flash"
googleflashlite = "google/gemini-2.5-flash-lite-preview-06-17"
grok4 = "x-ai/grok-4"
qwen3 = "qwen/qwen3-235b-a22b-07-25"
qwencoder = "qwen/qwen3-coder:nitro"
zai45 = "z-ai/glm-4.5"
halfa = "openrouter/horizon-beta"
autor = "meta-llama/llama-4-maverick"
openaiopen = "openai/gpt-oss-120b"
open5 = "openai/gpt-5"
open5mini="openai/gpt-5-mini"
open5nano = "openai/gpt-5-nano"

# Model constants
ALL_MODELS= open5
SUMMARY_MODEL = ALL_MODELS  # Model for summaries
MAIN_MODEL = ALL_MODELS  # Primary model for main agent operations
CODE_MODEL = ALL_MODELS  # Model for code generation tasks

# Feature flag constants
COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"

# Limits
MAX_SUMMARY_MESSAGES = 40 # Max messages for context summarization input
MAX_SUMMARY_TOKENS = 65000 # Max tokens for context summarization output (aligning with config.py's original value)

# Create a cache directory if it does not exist
CACHE_DIR = TOP_LEVEL_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Global variable for current prompt name (if needed by any module) ---
PROMPT_NAME: Optional[str] = None

def set_prompt_name(name: str):
    """Sets the global PROMPT_NAME."""
    global PROMPT_NAME
    PROMPT_NAME = name
    # Optionally, also save to constants.json if it needs to be persisted across sessions/restarts
    # set_constant("PROMPT_NAME", name)

# --- System Prompt Loading ---
try:
    with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
        _prompt_contents = f.read()
    if "{{OS_NAME}}" in _prompt_contents:
        SYSTEM_PROMPT = _prompt_contents.replace("{{OS_NAME}}", OS_NAME)
    else:
        SYSTEM_PROMPT = _prompt_contents + (
            f"\n\nHost operating system: {OS_NAME}. "
            "Use appropriate commands for this environment."
        )
except FileNotFoundError:
    SYSTEM_PROMPT = "System prompt file not found. Please ensure 'system_prompt/system_prompt.md' exists."
    logging.error(f"CRITICAL: System prompt file not found at {SYSTEM_PROMPT_FILE}")

def reload_system_prompt() -> str:
    """
    Reloads the system prompt from SYSTEM_PROMPT_FILE.
    Returns the new system prompt.
    """
    global SYSTEM_PROMPT
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        if "{{OS_NAME}}" in contents:
            SYSTEM_PROMPT = contents.replace("{{OS_NAME}}", OS_NAME)
        else:
            SYSTEM_PROMPT = contents + (
                f"\n\nHost operating system: {OS_NAME}. "
                "Use appropriate commands for this environment."
            )
        logging.info(f"System prompt reloaded from {SYSTEM_PROMPT_FILE}")
        return SYSTEM_PROMPT
    except FileNotFoundError:
        logging.error(f"Failed to reload system prompt: File not found at {SYSTEM_PROMPT_FILE}")
        # Keep the old SYSTEM_PROMPT if reload fails
        return SYSTEM_PROMPT

# --- Constants Management (using JSON cache file) ---
def write_constants_to_file():
    """Writes all current (exportable) constants to the JSON cache file."""
    constants = {
        "TOP_LEVEL_DIR": str(TOP_LEVEL_DIR),
        "WORKER_DIR": str(WORKER_DIR),
        "REPO_DIR": str(REPO_DIR),
        "SYSTEM_PROMPT_DIR": str(SYSTEM_PROMPT_DIR),
        "SYSTEM_PROMPT_FILE": str(SYSTEM_PROMPT_FILE),
        "BASH_PROMPT_DIR": str(BASH_PROMPT_DIR),
        "BASH_PROMPT_FILE": str(BASH_PROMPT_FILE),
        "LLM_GEN_CODE_DIR": str(LLM_GEN_CODE_DIR) if LLM_GEN_CODE_DIR else "",
        "TOOLS_DIR": str(TOOLS_DIR),
        "SCRIPTS_DIR": str(SCRIPTS_DIR),
        "TESTS_DIR": str(TESTS_DIR),
        "LOGS_DIR": str(LOGS_DIR),
        "PROMPTS_DIR": str(PROMPTS_DIR),
        "CACHE_DIR": str(CACHE_DIR),
        "LOG_FILE": str(LOG_FILE),
        "MESSAGES_FILE": str(MESSAGES_FILE),
        "SUMMARY_FILE": str(SUMMARY_FILE),
        "CODE_FILE": str(CODE_FILE),
        "USER_LOG_FILE": str(USER_LOG_FILE),
        "ASSISTANT_LOG_FILE": str(ASSISTANT_LOG_FILE),
        "TOOL_LOG_FILE": str(TOOL_LOG_FILE),
        "LOG_LEVEL_CONSOLE": LOG_LEVEL_CONSOLE,
        "LOG_LEVEL_FILE": LOG_LEVEL_FILE,
        "LOG_FILE_APP": LOG_FILE_APP,
        "LOG_MAX_BYTES": LOG_MAX_BYTES,
        "LOG_BACKUP_COUNT": LOG_BACKUP_COUNT,
        "OS_NAME": OS_NAME,
        "SUMMARY_MODEL": SUMMARY_MODEL,
        "MAIN_MODEL": MAIN_MODEL,
        "COMPUTER_USE_BETA_FLAG": COMPUTER_USE_BETA_FLAG,
        "PROMPT_CACHING_BETA_FLAG": PROMPT_CACHING_BETA_FLAG,
        "MAX_SUMMARY_MESSAGES": MAX_SUMMARY_MESSAGES,
        "MAX_SUMMARY_TOKENS": MAX_SUMMARY_TOKENS,
        "PROMPT_NAME": PROMPT_NAME if PROMPT_NAME else "",
        "TASK": "NOT YET CREATED", # Default task, can be updated by set_constant
    }
    # Ensure CACHE_DIR exists before writing
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "constants.json", "w") as f:
        json.dump(constants, f, indent=4)

def get_constants(): # Renamed from original get_constants to avoid conflict during transition
    """Loads all constants from the JSON cache file."""
    # Ensure the constants file exists by calling write_constants_to_file if it doesn't.
    # This also ensures that if new constants are added to `write_constants_to_file`,
    # they get persisted on the next call to `get_constants` or `get_constant`.
    constants_file_path = CACHE_DIR / "constants.json"
    if not constants_file_path.exists():
        write_constants_to_file()

    with open(constants_file_path, "r") as f:
        constants = json.load(f)
    return constants


# Function to set a constant and persist it
def get_constant(name: str, default: Any = None) -> Any:
    # `get_constants()` now ensures the file is written if it doesn't exist.
    constants = get_constants()
    value = constants.get(name, default)

    # Convert path strings to Path objects if applicable, with exclusions
    if (
        value is not None
        and isinstance(value, str)
        and ("PATH" in name.upper() or "DIR" in name.upper() or "FILE" in name.upper())
        and not any(x in name for x in ["MODEL", "FLAG", "LEVEL", "TASK", "NAME"])
    ):
        try:
            return Path(value)
        except TypeError:  # Handle cases where value might not be a valid path string
            return value
    return value

# Function to set a constant and persist it
def set_constant(name: str, value: Any):
    constants = get_constants() # Load current constants

    # Convert Path objects to strings for JSON serialization
    if isinstance(value, Path):
        constants[name] = str(value)
    else:
        constants[name] = value

    # Write all constants (including the updated one) back to the file
    # This uses the same structure as write_constants_to_file to keep it consistent
    # We update the dictionary `constants` and then dump it.
    # For simplicity, we'll call write_constants_to_file which uses the global Python vars.
    # So, if we want set_constant to be robust for *any* key, we might need to update globals first,
    # or make write_constants_to_file accept a dictionary.

    # Update the global variable if it exists (e.g. MAIN_MODEL etc.)
    # This makes the change immediately available to the current session.
    if name in globals():
        globals()[name] = value

    with open(CACHE_DIR / "constants.json", "w") as f:
        json.dump(constants, f, indent=4)
    logging.info(f"Constant '{name}' set to '{value}' and persisted.")
    return True


def write_to_file(s: str, file_path: Path): # Modified to take Path object
    """Write debug output to a file in a compact, VS Code collapsible format."""
    # datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") # Removed as it's not used
    lines = s.split("\n")
    output = []

    # Save the first line to a variable
    first_line = lines[0]
    # Remove the first line from the lines list
    lines = lines[1:]

    output.append(f"# ENTRY {first_line}: ")
    output.append("Details: ")
    # Join and clean multi-line strings
    for line in lines:
        if line.strip() == "'":  # Skip standalone quote marks
            continue
        # Remove trailing quotes and clean up the line
        cleaned_line = line.strip().strip("'")
        if not cleaned_line:  # Skip empty lines
            continue

        if "tool_input:" in line:
            try:
                json_part = line.split("tool_input: ")[1]
                if json_part.strip().startswith("{") and json_part.strip().endswith(
                    "}"
                ):
                    json_obj = json.loads(json_part)
                    output.append(
                        f"tool_input: {json.dumps(json_obj, separators=(',', ':'))}"
                    )
                else:
                    output.append(f"{cleaned_line}")
            except (IndexError, json.JSONDecodeError):
                output.append(f"{cleaned_line}")
        else:
            # If line contains JSON-like content, try to parse and format it
            if cleaned_line.strip().startswith("{") and cleaned_line.strip().endswith(
                "}"
            ):
                try:
                    json_obj = json.loads(cleaned_line)
                    output.append(json.dumps(json_obj, separators=(",", ":")))
                except json.JSONDecodeError:
                    output.append(f"{cleaned_line}")
            else:
                output.append(f"{cleaned_line}")
    output.append("\n")

    with open(file_path, "a", encoding="utf-8") as f:
        f.write("\n".join(output) + "\n")


# Call write_constants_to_file once at import time to ensure the file is populated
# with defaults if it doesn't exist or is empty.
write_constants_to_file()

# Function to load the constants from a file (original name, now calls the new get_constants)
def load_constants():
    return get_constants()


# --- Logging Setup ---
# This setup needs to happen after basic constants like LOG_LEVEL_CONSOLE are defined.
# The temporary override mechanism for get_constant during logging setup is kept.

def setup_logging():
    """Set up logging for the application."""
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.DEBUG)  # Set root logger level

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')

    # Console Handler
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(getattr(logging, _get_constant_for_logging_setup("LOG_LEVEL_CONSOLE").upper(), logging.INFO))
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # File Handler
    # Use _get_constant_for_logging_setup for LOGS_DIR during setup
    logs_dir_for_setup = _get_constant_for_logging_setup("LOGS_DIR")
    logs_dir_for_setup.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists

    log_file_app_str = _get_constant_for_logging_setup("LOG_FILE_APP")
    log_file_path = Path(log_file_app_str)

    if not log_file_path.is_absolute():
        # Use _get_constant_for_logging_setup for TOP_LEVEL_DIR during setup
        top_level_dir_for_setup = _get_constant_for_logging_setup("TOP_LEVEL_DIR")
        log_file_path = top_level_dir_for_setup / log_file_path

    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=_get_constant_for_logging_setup("LOG_MAX_BYTES"), # Use overridden getter
        backupCount=_get_constant_for_logging_setup("LOG_BACKUP_COUNT"), # Use overridden getter
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, _get_constant_for_logging_setup("LOG_LEVEL_FILE").upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Suppress verbose LiteLLM debug logs from open-interpreter
    # These logs are very verbose and clutter the console output
    litellm_logger = logging.getLogger('litellm')
    litellm_logger.setLevel(logging.WARNING)  # Only show warnings and errors
    
    # Also suppress other noisy loggers that might be used by open-interpreter
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)
    
    openai_logger = logging.getLogger('openai')
    openai_logger.setLevel(logging.WARNING)

# Define these specific constants directly for setup_logging to use initially.
# These are the Python global variables defined at the top of config.py
_initial_constants_for_logging_setup = {
    "LOG_LEVEL_CONSOLE": LOG_LEVEL_CONSOLE, # String
    "LOG_LEVEL_FILE": LOG_LEVEL_FILE,       # String
    "LOG_FILE_APP": LOG_FILE_APP,           # String (path relative to LOGS_DIR or absolute)
    "LOG_MAX_BYTES": LOG_MAX_BYTES,         # Integer
    "LOG_BACKUP_COUNT": LOG_BACKUP_COUNT,   # Integer
    "LOGS_DIR": LOGS_DIR,                   # Path object
    "TOP_LEVEL_DIR": TOP_LEVEL_DIR          # Path object
}

_original_get_constant_func = get_constant # Store original get_constant

def _get_constant_for_logging_setup(name: str) -> Any:
    """Special getter for logging setup to use direct global vars instead of JSON file."""
    val = _initial_constants_for_logging_setup.get(name)
    # This special getter returns values as they are (Path objects remain Path objects)
    # because setup_logging expects them in their correct types.
    if val is not None:
        return val
    # Fallback to original get_constant if not in initial set (should not happen for logging keys)
    return _original_get_constant_func(name)

# Temporarily override get_constant for the duration of setup_logging
_config_get_constant_backup = get_constant
get_constant = _get_constant_for_logging_setup

setup_logging()

# Restore original get_constant function
get_constant = _config_get_constant_backup

# Final write to ensure all potentially new constants (like WORKER_DIR, SUMMARY_FILE)
# and any modifications during setup (like PROMPT_NAME if set_prompt_name was called)
# are saved to constants.json.
write_constants_to_file()

logging.info("Logging setup complete. Constants file up-to-date.")
