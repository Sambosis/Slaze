# LLM-Enhanced Edit Tool Documentation

## Overview

The edit tool has been enhanced to use Large Language Model (LLM) calls for intelligent text replacement and insertion operations. Instead of requiring exact string matches, the tool can now understand natural language instructions and make context-aware modifications to files.

## Key Changes

### 1. Modified `edit.py`

The main edit tool (`/workspace/tools/edit.py`) has been updated with the following enhancements:

#### New Dependencies
- `asyncio` - For async LLM calls
- `utils.llm_client` - LLM client interface
- `config.MAIN_MODEL` - Configuration for the LLM model to use

#### New Match Mode
- Added `"llm"` as a new match mode option (default)
- Existing modes (`"exact"`, `"regex"`, `"fuzzy"`) are still supported

#### Enhanced Methods

1. **`_cmd_str_replace_enhanced`**: Routes to LLM or traditional replacement based on mode
2. **`_cmd_str_replace_llm`**: Handles LLM-based replacements
3. **`_cmd_insert_enhanced`**: Routes to LLM or traditional insertion
4. **`_cmd_insert_llm`**: Handles LLM-based insertions

#### LLM Helper Methods

1. **`_create_replacement_prompt`**: Creates prompts for replacement operations
2. **`_create_insertion_prompt`**: Creates prompts for insertion operations
3. **`_call_llm_for_edit`**: Handles LLM API calls and response cleaning
4. **`_create_change_summary`**: Generates summaries of changes made

## How It Works

### Replacement Operation

When using `str_replace` with `match_mode="llm"`:

1. The tool reads the entire file content
2. Creates a prompt with:
   - The complete file content
   - Natural language description of what to find (`old_str`)
   - Natural language description of changes (`new_str`)
3. Sends the prompt to the LLM
4. The LLM returns the complete modified file
5. The tool saves the modified content and creates a backup

Example:
```python
await editor(
    command="str_replace",
    path="example.py",
    old_str="The function that calculates sum",
    new_str="Add error handling and type checking",
    match_mode="llm"
)
```

### Insertion Operation

When using `insert` with natural language instructions:

1. The tool reads the file and gets context around the insertion point
2. Creates a prompt with:
   - The complete file content
   - The line number for insertion
   - Context before and after the insertion point
   - Natural language instruction for what to insert
3. Sends the prompt to the LLM
4. The LLM returns the complete file with new content inserted
5. The tool saves the modified content

Example:
```python
await editor(
    command="insert",
    path="example.py",
    insert_line=10,
    new_str="Add a function to calculate the product of two numbers with error handling"
)
```

## Prompt Engineering

The prompts are carefully designed to:

1. **Be explicit**: Clear instructions that the LLM should output ONLY the modified file content
2. **Provide context**: Full file content and surrounding context for accurate modifications
3. **Avoid formatting issues**: Instructions to not include markdown code blocks or explanations
4. **Maintain precision**: Low temperature (0.1) for consistent, accurate edits

## Response Processing

The tool includes robust response processing:

1. **Strips whitespace**: Removes leading/trailing whitespace
2. **Removes markdown blocks**: Automatically detects and removes ```python``` wrappers if present
3. **Preserves formatting**: Maintains original file's newline conventions

## Fallback Behavior

The tool maintains backward compatibility:

1. **Traditional modes**: `exact`, `regex`, and `fuzzy` modes work as before
2. **Automatic fallback**: If exact/regex match fails, automatically tries fuzzy matching
3. **Smart detection**: For insertions, simple text uses traditional insert, complex instructions use LLM

## Configuration

The LLM model is configured via:
- `config.MAIN_MODEL` - Sets which LLM model to use
- `utils.llm_client.create_llm_client()` - Creates appropriate client based on model

## Benefits

1. **Natural Language**: Use plain English to describe changes instead of exact code snippets
2. **Context Awareness**: LLM understands the code context and makes appropriate modifications
3. **Intelligent Generation**: Can generate new code based on descriptions
4. **Error Handling**: LLM can add proper error handling, type checking, documentation
5. **Refactoring**: Can understand and implement complex refactoring instructions

## Examples

### Adding Error Handling
```python
old_str="The calculate function without validation"
new_str="Add input validation and error handling"
```

### Refactoring Code
```python
old_str="The nested if statements in the validation logic"
new_str="Refactor to use early returns for better readability"
```

### Adding Documentation
```python
old_str="Functions without docstrings"
new_str="Add comprehensive docstrings with parameter and return descriptions"
```

## Testing

A demonstration script (`test_llm_edit_simple.py`) shows the functionality with a mock LLM client, demonstrating:
- Error handling addition
- New function generation
- Context-aware modifications

## Future Enhancements

Potential improvements could include:
- Diff visualization before applying changes
- Multi-file refactoring support
- Code review suggestions
- Automatic test generation
- Style guide enforcement