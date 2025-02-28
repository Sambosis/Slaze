#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Start Xvfb on display :99
if command_exists Xvfb; then
    Xvfb :99 -screen 0 1024x768x24 &
    XVFBPID=$!
    sleep 1  # Give Xvfb a moment to initialize

    # Check if Xvfb is running
    if ! kill -0 $XVFBPID 2>/dev/null; then
        echo "Error: Xvfb failed to start."
        exit 1
    fi
else
    echo "Error: Xvfb command not found."
    exit 1
fi

# Start fluxbox in the background
if command_exists fluxbox; then
    fluxbox &
    FLUXBOXPID=$!
    sleep 0.5
else
    echo "Error: fluxbox command not found."
    exit 1
fi

# Run the Python application
if command_exists python3; then
    python3 hello.py
else
    echo "Error: python3 command not found."
    exit 1
fi

# Keep the script running (important if python3 hello.py exits quickly)
wait

# Improved signal handling (example)
trap "kill -TERM $XVFBPID $FLUXBOXPID" EXIT SIGINT SIGTERM

# Additional improvements/comments:
#  The chosen approach includes robust error handling and background processes
#     - `command_exists()` checks for command availability such as Xvfb, fluxbox, and python3.
#     - Xvfb startup verification is performed by sending a signal (kill -0).
#  `sleep` is included to give processes time to intialize.
#  `wait` is added to wait for all background process.
#  Basic signal handling is present in the code.

# Further Considerations Beyond this Scope (Do not want to change the code any further, just comments):
# 1. More sophisticated logging:  Consider logging to a file instead of just echoing to standard output/error.
# 2. More robust signal handling:  Handle signals like SIGTERM and SIGINT more gracefully, perhaps by
#    cleaning up resources (e.g., explicitly closing windows) before exiting.
# 3. Consider using supervisord if the requirements become to large for a script, such as automatic restarts, complex dependencies..