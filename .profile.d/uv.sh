#!/usr/bin/env bash
# This script ensures uv is available in the runtime environment

# Add uv to PATH from multiple possible locations
if [ -d "$HOME/.cargo/bin" ] && [ -f "$HOME/.cargo/bin/uv" ]; then
    export PATH="$HOME/.cargo/bin:$PATH"
elif [ -d "$HOME/.local/bin" ] && [ -f "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Also check app-level bin directory
if [ -d "$HOME/bin" ] && [ -f "$HOME/bin/uv" ]; then
    export PATH="$HOME/bin:$PATH"
fi

# Fallback: add both paths regardless
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$HOME/bin:$PATH"