# LLM-Based Command Converter

This project now includes a robust LLM-based command conversion system that replaces the previous regex-based approach for adapting bash commands to different system environments.

## Overview

The command converter uses an LLM to intelligently transform bash commands to be appropriate for the current system, ensuring:
- Hidden files/directories are filtered out when appropriate
- Commands work correctly on the target operating system
- Cross-platform compatibility is maintained
- Proper error handling and fallback mechanisms

## Architecture

### Core Components

1. **CommandConverter** (`utils/command_converter.py`)
   - Main conversion logic
   - System information gathering
   - LLM prompt generation and response processing

2. **LLMClient** (`utils/llm_client.py`)
   - Flexible client supporting multiple LLM providers
   - OpenRouter, OpenAI, and Anthropic API support

3. **BashTool** (`tools/bash.py`)
   - Updated to use LLM-based conversion
   - Fallback to legacy regex-based method if LLM fails

## Configuration

### Environment Variables

You need to set up API keys for your preferred LLM provider:

```bash
# For OpenRouter (supports many models including Claude, GPT, etc.)
export OPENROUTER_API_KEY="your_openrouter_key"

# For direct OpenAI access
export OPENAI_API_KEY="your_openai_key"

# For direct Anthropic access
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Model Configuration

The system uses the model specified in your `config.py`:

```python
MAIN_MODEL = "anthropic/claude-sonnet-4"  # Example model
```

Supported model formats:
- `anthropic/claude-*` (via OpenRouter)
- `openai/gpt-*` (via OpenRouter)
- `google/gemini-*` (via OpenRouter)
- `gpt-*` (direct OpenAI)
- `claude-*` (direct Anthropic)

## Usage

### Basic Usage

The conversion happens automatically when using the bash tool:

```python
bash_tool = BashTool()
result = await bash_tool("find /path -type f")
# Command is automatically converted to exclude hidden files
```

### Direct Conversion

You can also use the converter directly:

```python
from utils.command_converter import convert_command_for_system

converted = await convert_command_for_system("ls -la /directory")
print(converted)  # Output: ls -la /directory | grep -v "^\."
```

## Features

### System-Aware Conversion

The converter includes detailed system information in its prompts:
- Operating system and version
- Architecture
- Shell environment
- Current working directory
- Environment variables

### Intelligent Response Processing

- Removes markdown code blocks from LLM responses
- Extracts only the command (ignoring explanations)
- Validates command format and length
- Handles multi-line responses appropriately

### Robust Fallback

If LLM conversion fails:
1. Falls back to legacy regex-based modification
2. If that fails, returns the original command
3. Logs warnings for debugging

### Comprehensive Testing

- Unit tests for all conversion logic
- Integration tests with mock LLM clients
- Tests for various command patterns
- Error handling and fallback testing

## Examples

### Find Commands

```bash
# Input
find /home -type f

# LLM Output
find /home -type f -not -path "*/.*"
```

### List Commands

```bash
# Input
ls -la /directory

# LLM Output
ls -la /directory | grep -v "^\."
```

### No Change Needed

```bash
# Input
echo "hello world"

# LLM Output
echo "hello world"
```

## Installation

1. Add the required dependency to `requirements.txt`:
   ```
   aiohttp==3.10.11
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys (see Configuration section)

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   ValueError: OPENROUTER_API_KEY environment variable not set
   ```
   Solution: Set the appropriate API key environment variable

2. **LLM API Errors**
   - Check your API key validity
   - Verify your account has sufficient credits
   - Check network connectivity

3. **Command Not Modified**
   - The LLM may determine no modification is needed
   - Check logs for conversion attempts
   - Verify the model is responding correctly

### Debugging

Enable debug logging to see conversion details:

```python
import logging
logging.getLogger('utils.command_converter').setLevel(logging.DEBUG)
```

### Performance Considerations

- The converter caches a single instance for reuse
- LLM calls have a 30-second timeout
- Failed conversions fall back quickly to legacy methods
- Short responses (max 200 tokens) keep costs low

## Future Enhancements

Potential improvements:
- Caching of common command conversions
- Support for additional LLM providers
- Custom conversion rules per project
- Integration with shell history for learning user patterns