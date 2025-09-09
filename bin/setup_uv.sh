#!/usr/bin/env bash
# Runtime script to ensure uv is available

# Add uv to PATH from both possible locations
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Verify uv is available
if command -v uv &> /dev/null; then
    echo "uv is available: $(uv --version)"
else
    echo "Warning: uv not found in PATH"
    echo "PATH: $PATH"
    echo "Checking common locations:"
    ls -la $HOME/.cargo/bin/ 2>/dev/null || echo "No .cargo/bin directory"
    ls -la $HOME/.local/bin/ 2>/dev/null || echo "No .local/bin directory"
fi
