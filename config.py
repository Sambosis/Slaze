from pathlib import Path
import json
from datetime import datetime
import subprocess

import logging
import logging.handlers
from cycler import V

global PROJECT_DIR
PROJECT_DIR = None

# Logging constants
LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"
LOG_FILE_APP = "logs/app.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Define the top-level directory
TOP_LEVEL_DIR = Path.cwd()
googlepro = "google/gemini-2.5-pro-preview"
googleflash = "google/gemini-2.5-flash-preview"
# Define the repository directory based on PROJECT_DIR
REPO_DIR = TOP_LEVEL_DIR / "repo"

# Define other relevant paths based on PROJECT_DIR
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
# ICECREAM_OUTPUT_FILE = LOGS_DIR / "debug_log.md" # Removed as per requirement
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "file_creation_log.json"
MESSAGES_FILE = LOGS_DIR / "messages.md"
SUMMARY_MODEL = googlepro
MAIN_MODEL = googlepro
COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"
CODE_FILE = LOGS_DIR / "code_messages.py"
USER_LOG_FILE = LOGS_DIR / "user_messages.log"
ASSISTANT_LOG_FILE = LOGS_DIR / "assistant_messages.log"
TOOL_LOG_FILE = LOGS_DIR / "tool_messages.log"
MAX_SUMMARY_MESSAGES = 40
MAX_SUMMARY_TOKENS = 20000

# Create a cache directory if it does not exist
CACHE_DIR = TOP_LEVEL_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Function to write the constants to a file
def write_constants_to_file():
    constants = {
        "TOP_LEVEL_DIR": str(TOP_LEVEL_DIR),
        "REPO_DIR": str(REPO_DIR),
        "SYSTEM_PROMPT_DIR": str(SYSTEM_PROMPT_DIR),
        "SYSTEM_PROMPT_FILE": str(SYSTEM_PROMPT_FILE),
        "BASH_PROMPT_DIR": str(BASH_PROMPT_DIR),
        "BASH_PROMPT_FILE": str(BASH_PROMPT_FILE),
        "LLM_GEN_CODE_DIR": str(LLM_GEN_CODE_DIR) if LLM_GEN_CODE_DIR else "",
        "TOOLS_DIR": str(TOOLS_DIR),
        "SCRIPTS_DIR": str(SCRIPTS_DIR),
        "TESTS_DIR": str(TESTS_DIR),
        "SUMMARY_MODEL": SUMMARY_MODEL,
        "MAIN_MODEL": MAIN_MODEL,
        "COMPUTER_USE_BETA_FLAG": COMPUTER_USE_BETA_FLAG,
        "PROMPT_CACHING_BETA_FLAG": PROMPT_CACHING_BETA_FLAG,
        "MAX_SUMMARY_MESSAGES": MAX_SUMMARY_MESSAGES,
        "MAX_SUMMARY_TOKENS": MAX_SUMMARY_TOKENS,
        "LOGS_DIR": str(LOGS_DIR),
        "PROJECT_DIR": str(PROJECT_DIR) if PROJECT_DIR else "",
        "PROMPTS_DIR": str(PROMPTS_DIR),
        "LOG_FILE": str(LOG_FILE),
        "MESSAGES_FILE": str(MESSAGES_FILE),
        # "ICECREAM_OUTPUT_FILE": str(ICECREAM_OUTPUT_FILE), # Removed as per requirement
        "CODE_FILE": str(CODE_FILE),
        "TASK": "NOT YET CREATED",
    }
    with open(CACHE_DIR / "constants.json", "w") as f:
        json.dump(constants, f, indent=4)


def get_constants():
    with open(CACHE_DIR / "constants.json", "r") as f:
        constants = json.load(f)
    return constants


# Function to load the constants from a file
def load_constants():
    const_file = CACHE_DIR / "constants.json"
    try:
        # If file is empty, return an empty dict
        if const_file.stat().st_size == 0:
            return {}
        with open(const_file, "r") as f:
            constants = json.load(f)
        return constants
    except FileNotFoundError:
        return {}


# Get a constant by name
def get_constant(name):
    write_constants_to_file()
    constants = load_constants()
    if constants:
        return_constant = constants.get(name)
        # If return_constant contains PATH, DIR or FILE then return as Path
        if (
            return_constant
            and ("PATH" in name or "DIR" in name or "FILE" in name)
            and isinstance(return_constant, str)
        ):
            return Path(return_constant)
        else:
            return return_constant
    else:
        return None


# Function to set a constant
def set_constant(name, value):
    constants = load_constants() or {}
    # Convert Path objects to strings for JSON serialization
    if isinstance(value, Path):
        constants[name] = str(value)
    else:
        constants[name] = value
    with open(CACHE_DIR / "constants.json", "w") as f:
        json.dump(constants, f, indent=4)
    return True


def set_project_dir(project_name: str) -> Path:
    """
    Set up project directories for both local and Docker.
    This function also creates the project directory if it doesn't already exist.

    Args:
        project_name: The name of the project

    Returns:
        The Path to the project directory
    """
    global PROJECT_DIR, LLM_GEN_CODE_DIR
    PROJECT_DIR = REPO_DIR / project_name
    LLM_GEN_CODE_DIR = TOP_LEVEL_DIR / "llm_gen_code"

    # Create repo directory if it doesn't exist
    REPO_DIR.mkdir(parents=True, exist_ok=True)

    # Create the project directory if it doesn't exist
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # Create llm_gen_code directory if it doesn't exist
    LLM_GEN_CODE_DIR.mkdir(parents=True, exist_ok=True)

    set_constant("PROJECT_DIR", str(PROJECT_DIR))
    set_constant("LLM_GEN_CODE_DIR", str(LLM_GEN_CODE_DIR))
    return PROJECT_DIR


# Function to get the project directory
def get_project_dir():
    return PROJECT_DIR


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


with open(SYSTEM_PROMPT_DIR / "system_prompt.md", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

# ic.configureOutput(includeContext=True, outputFunction=write_to_file) # Removed icecream configuration

def setup_logging():
    """Set up logging for the application."""
    logger = logging.getLogger()  # Root logger
    logger.setLevel(logging.DEBUG)  # Set root logger level

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, get_constant("LOG_LEVEL_CONSOLE").upper(), logging.INFO))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    LOGS_DIR.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists
    log_file_path = get_constant("LOG_FILE_APP")
    # Ensure the log file path is absolute or relative to TOP_LEVEL_DIR if not already absolute
    if not Path(log_file_path).is_absolute():
        log_file_path = TOP_LEVEL_DIR / log_file_path
    else:
        log_file_path = Path(log_file_path)

    # Ensure the directory for the log file exists
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=get_constant("LOG_MAX_BYTES"),
        backupCount=get_constant("LOG_BACKUP_COUNT")
    )
    file_handler.setLevel(getattr(logging, get_constant("LOG_LEVEL_FILE").upper(), logging.DEBUG))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Call setup_logging() to apply the configuration when the module is imported.
# This needs to be after all constants are defined and functions like get_constant are available.
# However, get_constant relies on write_constants_to_file, which itself relies on constants being defined.
# We need to ensure LOG_LEVEL_CONSOLE, LOG_LEVEL_FILE, LOG_FILE_APP, LOG_MAX_BYTES, LOG_BACKUP_COUNT are available
# to get_constant before setup_logging is called.

# Define these specific constants directly for setup_logging to use initially.
_initial_constants_for_logging = {
    "LOG_LEVEL_CONSOLE": LOG_LEVEL_CONSOLE,
    "LOG_LEVEL_FILE": LOG_LEVEL_FILE,
    "LOG_FILE_APP": LOG_FILE_APP,
    "LOG_MAX_BYTES": LOG_MAX_BYTES,
    "LOG_BACKUP_COUNT": LOG_BACKUP_COUNT,
    "LOGS_DIR": LOGS_DIR, # Added for consistency, though LOGS_DIR is already a Path
    "TOP_LEVEL_DIR": TOP_LEVEL_DIR # Added for resolving log file path
}

_original_get_constant = get_constant

def _get_constant_for_logging_setup(name):
    if name in _initial_constants_for_logging:
        # Ensure Path objects are returned for directory/file constants if stored as strings
        val = _initial_constants_for_logging[name]
        if (
            isinstance(val, str)
            and ("PATH" in name or "DIR" in name or "FILE" in name)
            and name != "LOG_FILE_APP"
            and name != "LOG_LEVEL_FILE"  # Ensure log level strings are not converted to Path
            and name != "LOG_LEVEL_CONSOLE" # Ensure log level strings are not converted to Path
        ):
             # LOG_FILE_APP is handled specially for path resolution later
            return Path(val)
        return val
    return _original_get_constant(name)

# Temporarily override get_constant for setup_logging
get_constant_temp_override = get_constant
get_constant = _get_constant_for_logging_setup

setup_logging()

# Restore original get_constant
get_constant = get_constant_temp_override

# Now, write all constants to file, including the new logging ones.
# This ensures they are available for subsequent calls to get_constant() from other modules.
set_constant("LOG_LEVEL_CONSOLE", LOG_LEVEL_CONSOLE)
set_constant("LOG_LEVEL_FILE", LOG_LEVEL_FILE)
set_constant("LOG_FILE_APP", LOG_FILE_APP) # Stored as string, will be resolved by Path() later
set_constant("LOG_MAX_BYTES", LOG_MAX_BYTES)
set_constant("LOG_BACKUP_COUNT", LOG_BACKUP_COUNT)

logging.info("Logging setup complete.")
