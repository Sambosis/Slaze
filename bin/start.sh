#!/usr/bin/env bash
set -e

echo "Starting Slazy app..."

# Check if uv is available, if not try to find it
if ! command -v uv &> /dev/null; then
    echo "uv not found in PATH, searching for it..."
    
    # Try common installation locations
    if [ -f "$HOME/.cargo/bin/uv" ]; then
        echo "Found uv in .cargo/bin, adding to PATH"
        export PATH="$HOME/.cargo/bin:$PATH"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        echo "Found uv in .local/bin, adding to PATH" 
        export PATH="$HOME/.local/bin:$PATH"
    elif [ -f "/app/bin/uv" ]; then
        echo "Found uv in /app/bin, adding to PATH"
        export PATH="/app/bin:$PATH"
    else
        echo "uv not found, attempting to install..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
fi

# Verify uv is now available
if command -v uv &> /dev/null; then
    echo "uv is available: $(uv --version)"
else
    echo "WARNING: uv still not available"
fi

echo "Starting gunicorn..."
exec gunicorn -k eventlet -w 1 --bind 0.0.0.0:$PORT wsgi:app
