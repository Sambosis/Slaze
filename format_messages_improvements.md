# format_messages_to_string Function Improvements

## Overview
The `format_messages_to_string` function in `utils/context_helpers.py` has been enhanced to capture all types of content in messages, including tool calls, tool responses, and various content types that were previously missing.

## Issues Found and Fixed

### 1. Missing Tool Use (Tool Calls) Support
**Problem**: The function was not handling `tool_use` content blocks, which contain information about tool calls made by the assistant.

**Solution**: Added comprehensive handling for `tool_use` content blocks that extracts:
- Tool name (`name` field)
- Tool ID (`id` field) 
- Tool input parameters (`input` field)

### 2. Incorrect Tool Result ID Field
**Problem**: The function was using `content_block.get('name', 'unknown')` for tool result IDs, but tool results use `tool_use_id` field.

**Solution**: Changed to use `tool_use_id` field for proper tool result identification.

### 3. Missing Tool Result Error Information
**Problem**: The function wasn't capturing the `is_error` field from tool results, which indicates whether a tool execution succeeded or failed.

**Solution**: Added extraction and display of the `is_error` field for tool results.

### 4. Incomplete Content Type Handling
**Problem**: The function had limited handling for different content types within tool results and messages.

**Solution**: Enhanced to handle:
- Text content (`type: "text"`)
- Image content (`type: "image"`)
- Tool use content (`type: "tool_use"`)
- Tool result content (`type: "tool_result"`)
- Generic content blocks with unknown types

### 5. Better Content Structure Handling
**Problem**: Tool result content could be strings, lists, or None, but the function didn't handle all cases properly.

**Solution**: Added comprehensive handling for:
- List content (iterating through items)
- String content (direct display)
- None content (explicit "None" display)
- Nested content structures

## Content Types Now Supported

### Tool Use (Tool Calls)
```
Tool Call: function_name
Tool ID: call_id_123
Input: {'param1': 'value1', 'param2': 42}
```

### Tool Results
```
Tool Result [ID: call_id_123]:
Error: False
Text: Tool execution result here
```

### Text Content
```
Text: Regular text content
```

### Image Content
```
Image: base64 source too big
```

### Mixed Content
The function now properly handles messages with multiple content blocks of different types.

## Benefits

1. **Complete Information Capture**: All message content is now preserved in the formatted output
2. **Tool Call Traceability**: Tool calls and their corresponding results can be traced through their IDs
3. **Error Visibility**: Tool execution errors are clearly indicated
4. **Extensible**: The function can handle new content types gracefully
5. **Backward Compatible**: Existing functionality for text and basic content is preserved

## Usage Examples

The improved function handles complex message structures like:

```python
messages = [
    {
        'role': 'assistant',
        'content': [
            {'type': 'text', 'text': 'I will use a tool now.'},
            {
                'type': 'tool_use',
                'name': 'file_read',
                'id': 'call_789',
                'input': {'filename': 'test.txt'}
            }
        ]
    },
    {
        'role': 'user',
        'content': [
            {
                'type': 'tool_result',
                'tool_use_id': 'call_789',
                'is_error': False,
                'content': [{'type': 'text', 'text': 'File contents here'}]
            }
        ]
    }
]
```

This provides a complete picture of the conversation flow including all tool interactions, which is essential for debugging, logging, and context understanding.