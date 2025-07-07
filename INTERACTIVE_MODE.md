# Interactive Mode Documentation

## Overview

The interactive mode allows users to review and edit tool calls before they are executed. This provides greater control over the agent's actions and allows for real-time parameter modification.

## Usage

```bash
python run.py interactive
```

## Features

### Tool Call Review
- View detailed parameter tables for each tool call
- See JSON representation of tool calls
- Review all parameters before execution

### Parameter Editing
- Edit individual parameters
- Support for strings, numbers, booleans, and JSON structures
- Validation for JSON parameters
- Preview changes before confirmation

### User Options
1. **Execute** - Run the tool with current parameters
2. **Edit** - Modify parameters before execution
3. **View JSON** - See the complete JSON structure
4. **Skip** - Skip this tool call
5. **Exit** - Stop the agent

## Example Flow

1. Agent proposes a tool call
2. User sees formatted parameter table
3. User can choose to edit parameters
4. User reviews and confirms changes
5. Tool executes with modified parameters

## Benefits

- **Safety**: Review dangerous operations before execution
- **Customization**: Fine-tune parameters for specific needs
- **Learning**: Understand what tools the agent is using
- **Control**: Skip unwanted operations