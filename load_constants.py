"""
Module for loading and managing configuration constants and environment variables.

This module handles the loading of environment variables, system prompts, and various
configuration constants used throughout the application. It provides functionality for
updating project directories, managing system prompts, and handling logging configurations.

Dependencies:
    - python-dotenv: For loading environment variables
    - icecream: For debug logging
    - pathlib: For cross-platform path handling
"""

from dotenv import load_dotenv
from pathlib import Path
import os
from icecream import ic
from datetime import datetime
import json

# Import configuration constants from config module
# Note: These imports are used to prevent circular dependencies
from config import (
    TOP_LEVEL_DIR, REPO_DIR, SYSTEM_PROMPT_DIR, write_to_file,
    SYSTEM_PROMPT_FILE, SCRIPTS_DIR, LOGS_DIR, TESTS_DIR
)

# Get the directory where this script is located

# Load environment variables with error handling
try:
    load_dotenv()
except Exception as e:
    print(f"Error loading environment variables: {e}")

# Application-wide constants
MAX_SUMMARY_MESSAGES = 20  # Maximum number of messages to include in summaries
MAX_SUMMARY_TOKENS = 6000  # Maximum token limit for summaries
WORKER_DIR = TOP_LEVEL_DIR  # Directory where worker processes operate
ICECREAM_OUTPUT_FILE = LOGS_DIR / "debug_log.json"  # Path for debug logging output

# Feature flag constants for beta features
COMPUTER_USE_BETA_FLAG = "computer-use-2024-10-22"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"

# Model configuration constants
SUMMARY_MODEL = "claude-3-5-haiku-latest"  # Model used for generating summaries
MAIN_MODEL = "claude-3-5-sonnet-latest"    # Primary model for main operations
# Add near the top with other Path definitions
# PROJECT_DIR = TOP_LEVEL_DIR  # Default value

global PROMPT_NAME
PROMPT_NAME = None

# HOME = Path.home()
def update_project_dir(new_dir: str) -> None:
    """
    Update the project directory path based on the provided directory name.
    
    Args:
        new_dir (str): The name of the new directory to set as project directory.
                      This will be appended to REPO_DIR to create the full path.
    """
    global PROJECT_DIR
    PROJECT_DIR = REPO_DIR / new_dir


# Load system prompt with error handling
try:
    with open(SYSTEM_PROMPT_FILE, 'r', encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    SYSTEM_PROMPT = ""
    print(f"Warning: System prompt file not found at {SYSTEM_PROMPT_FILE}")

def reload_prompts() -> None:
    """
    Reload the system prompt from the configuration file.
    
    This function updates the global SYSTEM_PROMPT variable with the latest content
    from the system prompt file. It handles potential file not found errors gracefully.
    """
    global SYSTEM_PROMPT
    try:
        with open(SYSTEM_PROMPT_FILE, 'r', encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
    except FileNotFoundError:
        print(f"Warning: System prompt file not found at {SYSTEM_PROMPT_FILE}")

def update_paths(new_prompt_name: str) -> dict:
    """
    Update and return a dictionary of important file paths based on the new prompt name.
    
    Args:
        new_prompt_name (str): The name of the new prompt to set as current.
    
    Returns:
        dict: A dictionary containing paths for:
            - ICECREAM_OUTPUT_FILE: Debug logging output file
            - SUMMARY_FILE: File for storing summaries
            - SYSTEM_PROMPT_FILE: System prompt configuration file
    """
    logs_dir = LOGS_DIR
    global PROMPT_NAME
    PROMPT_NAME = new_prompt_name
    return {
        'ICECREAM_OUTPUT_FILE': logs_dir / "debug_log.json",
        'SUMMARY_FILE': logs_dir / "summaries/summary.md",
        'SYSTEM_PROMPT_FILE': logs_dir / "prompts/system_prompt.md",
    }

def load_system_prompts() -> str:
    """
    Load the system prompts from the configuration file.
    
    Returns:
        str: The content of the system prompt file.
    
    Raises:
        Exception: If the system prompt file cannot be found or read.
    """
    paths = update_paths()
    try:
        with open(paths['SYSTEM_PROMPT_FILE'], 'r', encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
        return SYSTEM_PROMPT
    except FileNotFoundError as e:
        raise Exception(f"Failed to load system prompts: {e}")

