#!/usr/bin/env python3
"""
Log File Initialization Script
This script initializes and validates the log file for the system.
"""

import os
import json
import sys


def init_log_file():
    """Initialize the log file with a valid structure."""
    # Determine the log file path
    try:
        from config import get_constant

        log_file = get_constant("LOG_FILE")
    except (ImportError, AttributeError):
        # Fallback if config is not available
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(script_dir, "logs", "file_log.json")

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Initializing log file at: {log_file}")

    # Create a valid empty log file
    initial_data = {"files": {}}

    # Check if file exists and is valid
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            # Check if structure is valid
            if not isinstance(existing_data, dict) or "files" not in existing_data:
                print("Existing log file has invalid structure. Resetting...")
                with open(log_file, "w", encoding="utf-8") as f:
                    json.dump(initial_data, f, indent=2)
            else:
                print("Existing log file is valid.")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Existing log file is corrupted ({e}). Resetting...")
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)
    else:
        # Create a new log file
        print("Creating new log file...")
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2)

    print("Log file initialization complete.")

    # Verify the log file is valid and readable
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(
            f"Verification successful. Log file contains {len(data.get('files', {}))} files."
        )
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        return False


if __name__ == "__main__":
    success = init_log_file()
    sys.exit(0 if success else 1)
