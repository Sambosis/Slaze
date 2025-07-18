# Bash Tool Display Changes

## Summary
Modified the `BashTool` class in `/workspace/tools/bash.py` to display command execution results in the same console-style format as the `ProjectSetupTool`.

## Changes Made

### 1. Added `_format_terminal_output` method
- **Purpose**: Format command execution to look like terminal input/output
- **Functionality**: 
  - Creates markdown code blocks with `console` syntax highlighting
  - Shows working directory with `$ cd {cwd}`
  - Shows command execution with `$ {command}`
  - Displays stdout and stderr output
  - Shows exit code for failed commands

### 2. Modified `_run_command` method
- **Added terminal display**: After command execution, calls `_format_terminal_output` and displays result as an "assistant" message
- **Error handling**: Improved error handling to properly display terminal output for different exception types:
  - `subprocess.CalledProcessError`: Uses the exception object directly (has stdout, stderr, returncode attributes)
  - `subprocess.TimeoutExpired`: Extracts output and stderr from exception
  - Other exceptions: Creates mock result object for display

### 3. Display Integration
- **User message**: Still shows "Executing command: {command}" as before
- **Assistant message**: Now shows formatted terminal output using `display.add_message("assistant", formatted_output)`
- **Error handling**: Gracefully handles display errors with logging warnings

## Example Output Format

Before (only user message):
```
User: Executing command: ls -la
```

After (user + assistant messages):
```
User: Executing command: ls -la
Assistant: 
```console
$ cd /workspace
$ ls -la
total 48
drwxr-xr-x 8 user user 4096 Jan 15 10:30 .
drwxr-xr-x 3 user user 4096 Jan 15 10:25 ..
-rw-r--r-- 1 user user 1234 Jan 15 10:30 README.md
...
```

## Consistency with ProjectSetup
The changes ensure that Bash tool output is displayed identically to ProjectSetup tool output:
- Same markdown console formatting
- Same command prompt style (`$ command`)
- Same error handling and display
- Same assistant message pattern

## Backward Compatibility
- All existing functionality preserved
- No breaking changes to the API
- Display is optional (gracefully handles when display is None)
- Tool result format unchanged for programmatic usage