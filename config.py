from pathlib import Path
import json
from datetime import datetime
import subprocess

from icecream import ic

global PROJECT_DIR
PROJECT_DIR = None

# Define the top-level directory
TOP_LEVEL_DIR = Path.cwd()

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
ICECREAM_OUTPUT_FILE = LOGS_DIR / "debug_log.md"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / "file_creation_log.json"
MESSAGES_FILE = LOGS_DIR / "messages.md"
SUMMARY_MODEL = "openai/gpt-4o"
MAIN_MODEL = "openai/gpt-4o"
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
        "ICECREAM_OUTPUT_FILE": str(ICECREAM_OUTPUT_FILE),
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


def write_to_file(s: str, file_path: str = ICECREAM_OUTPUT_FILE):
    """Write debug output to a file in a compact, VS Code collapsible format."""
    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
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

ic.configureOutput(includeContext=True, outputFunction=write_to_file)
