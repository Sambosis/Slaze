from dotenv import load_dotenv
from pathlib import Path
import os
from icecream import ic
from datetime import datetime
import json

# Remove the following line to prevent circular import
from config import TOP_LEVEL_DIR, REPO_DIR, SYSTEM_PROMPT_DIR, write_to_file , SYSTEM_PROMPT_FILE, SCRIPTS_DIR, LOGS_DIR, TESTS_DIR

# Get the directory where this script is located

# Load environment variables with error handling
try:
    load_dotenv()
except Exception as e:
    print(f"Error loading environment variables: {e}")

# Constants
MAX_SUMMARY_MESSAGES = 20
MAX_SUMMARY_TOKENS = 6000
WORKER_DIR = TOP_LEVEL_DIR
ICECREAM_OUTPUT_FILE =  LOGS_DIR / "debug_log.json"
COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"
SUMMARY_MODEL = "claude-3-5-haiku-latest"
MAIN_MODEL = "claude-3-5-sonnet-latest"
# Add near the top with other Path definitions
# PROJECT_DIR = TOP_LEVEL_DIR  # Default value

global PROMPT_NAME
PROMPT_NAME = None

# HOME = Path.home()
def update_project_dir(new_dir):
    global PROJECT_DIR
    PROJECT_DIR = REPO_DIR / new_dir


# Load system prompt
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = ""
    print(f"Warning: System prompt file not found at {SYSTEM_PROMPT_FILE}")

def reload_prompts():
    global SYSTEM_PROMPT
    try:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
    except FileNotFoundError:
        print(f"Warning: System prompt file not found at {SYSTEM_PROMPT_FILE}")

def update_paths(new_prompt_name):
    logs_dir = LOGS_DIR
    global PROMPT_NAME
    PROMPT_NAME = new_prompt_name
    return {
        'ICECREAM_OUTPUT_FILE': logs_dir / "debug_log.json",
        'SUMMARY_FILE': logs_dir / "summaries/summary.md",
        'SYSTEM_PROMPT_FILE': logs_dir / "prompts/system_prompt.md",
    }

def load_system_prompts():
    paths = update_paths()
    try:
        with open(paths['SYSTEM_PROMPT_FILE'], 'r', encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
        return SYSTEM_PROMPT
    except FileNotFoundError as e:
        raise Exception(f"Failed to load system prompts: {e}")

