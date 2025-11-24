import json
import logging.handlers
import os
import platform
import sys
import subprocess
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

# Import all model constants
from models import *

# Load environment variables from .env file
load_dotenv()

# --- Environment Setup ---
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
        subprocess.run(['chcp', '65001'], capture_output=True, check=False)
    except Exception:
        pass

# --- Path Constants ---
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

# Create Logs Directory
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Log Files
LOG_FILE = LOGS_DIR / "file_creation_log.json" # For file operations audit
MESSAGES_FILE = LOGS_DIR / "messages.md" # For conversation history
SUMMARY_FILE = LOGS_DIR / "summaries/summary.md" # For storing summaries, from load_constants.py
CODE_FILE = LOGS_DIR / "code_messages.py" # For WriteCodeTool code logging
USER_LOG_FILE = LOGS_DIR / "user_messages.log"
ASSISTANT_LOG_FILE = LOGS_DIR / "assistant_messages.log"
TOOL_LOG_FILE = LOGS_DIR / "tool_messages.log"

# Cache Directory
CACHE_DIR = TOP_LEVEL_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging Constants ---
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
LOG_FILE_APP = "logs/app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
OS_NAME = platform.system()

# --- Feature Flags & Limits ---
COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"

MAX_SUMMARY_MESSAGES = 40 # Max messages for context summarization input
MAX_SUMMARY_TOKENS = 65000 # Max tokens for context summarization output

# --- Global State ---
PROMPT_NAME: Optional[str] = None

def set_prompt_name(name: str):
    """Sets the global PROMPT_NAME."""
    global PROMPT_NAME
    PROMPT_NAME = name

# --- System Prompt Loading ---
SYSTEM_PROMPT = ""

def load_system_prompt_content() -> str:
    try:
        with open(SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        if "{{OS_NAME}}" in contents:
            return contents.replace("{{OS_NAME}}", OS_NAME)
        else:
            return contents + (
                f"\n\nHost operating system: {OS_NAME}. "
                "Use appropriate commands for this environment."
            )
    except FileNotFoundError:
        msg = f"System prompt file not found. Please ensure '{SYSTEM_PROMPT_FILE}' exists."
        logging.error(msg)
        return msg

SYSTEM_PROMPT = load_system_prompt_content()

def reload_system_prompt() -> str:
    """
    Reloads the system prompt from SYSTEM_PROMPT_FILE.
    Returns the new system prompt.
    """
    global SYSTEM_PROMPT
    SYSTEM_PROMPT = load_system_prompt_content()
    logging.info(f"System prompt reloaded from {SYSTEM_PROMPT_FILE}")
    return SYSTEM_PROMPT

# --- Constants Management ---

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
        "TASK": "NOT YET CREATED", 
    }
    # Ensure CACHE_DIR exists before writing
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "constants.json", "w") as f:
        json.dump(constants, f, indent=4)

def get_constants():
    """Loads all constants from the JSON cache file."""
    constants_file_path = CACHE_DIR / "constants.json"
    if not constants_file_path.exists():
        write_constants_to_file()

    with open(constants_file_path, "r") as f:
        constants = json.load(f)
    return constants

def get_constant(name: str, default: Any = None) -> Any:
    constants = get_constants()
    value = constants.get(name, default)

    # Convert path strings to Path objects if applicable
    if (
        value is not None
        and isinstance(value, str)
        and ("PATH" in name.upper() or "DIR" in name.upper() or "FILE" in name.upper())
        and not any(x in name for x in ["MODEL", "FLAG", "LEVEL", "TASK", "NAME"])
    ):
        try:
            return Path(value)
        except TypeError:
            return value
    return value

def set_constant(name: str, value: Any):
    constants = get_constants()
    
    if isinstance(value, Path):
        constants[name] = str(value)
    else:
        constants[name] = value

    # Update global variable if it exists
    if name in globals():
        globals()[name] = value

    with open(CACHE_DIR / "constants.json", "w") as f:
        json.dump(constants, f, indent=4)
    logging.info(f"Constant '{name}' set to '{value}' and persisted.")
    return True

def write_to_file(s: str, file_path: Path):
    """Write debug output to a file in a compact, VS Code collapsible format."""
    lines = s.split("\n")
    output = []

    if not lines:
        return

    first_line = lines[0]
    lines = lines[1:]

    output.append(f"# ENTRY {first_line}: ")
    output.append("Details: ")
    
    for line in lines:
        if line.strip() == "'":
            continue
        cleaned_line = line.strip().strip("'")
        if not cleaned_line:
            continue

        if "tool_input:" in line:
            try:
                json_part = line.split("tool_input: ")[1]
                if json_part.strip().startswith("{") and json_part.strip().endswith("}"):
                    json_obj = json.loads(json_part)
                    output.append(f"tool_input: {json.dumps(json_obj, separators=(',', ':'))}")
                else:
                    output.append(f"{cleaned_line}")
            except (IndexError, json.JSONDecodeError):
                output.append(f"{cleaned_line}")
        else:
            if cleaned_line.strip().startswith("{") and cleaned_line.strip().endswith("}"):
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

# --- Logging Setup ---

def setup_logging():
    """Set up logging for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')

    # File Handler
    log_file_path = Path(LOG_FILE_APP)
    if not log_file_path.is_absolute():
        log_file_path = TOP_LEVEL_DIR / log_file_path

    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    
    file_level = getattr(logging, LOG_LEVEL_FILE.upper(), logging.DEBUG)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Suppress verbose logs
    logging.getLogger('litellm').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)

# Initialize constants file
write_constants_to_file()

# Initialize logging
setup_logging()

# Function to load constants (alias for backward compatibility)
def load_constants():
    return get_constants()

logging.info("Logging setup complete. Constants file up-to-date.")
