# Windows Command Conversion Fix

## Issue Identified

The LLM-based command converter was incorrectly converting Windows commands to Linux equivalents, causing critical failures:

### Problem Example
```
Input: "dir"
LLM Output: "ls | grep -v '^\.'"
Windows Error: 'ls' is not recognized as an internal or external command
```

## Root Cause

The original implementation had **Linux-biased prompts** that:
1. Only included Linux/Unix command examples (`ls`, `find`, `grep`)
2. Didn't distinguish between Windows and Linux environments
3. Applied Linux-style hidden file filtering regardless of OS

## Solution Implemented

### 1. **OS-Aware Prompt Generation**

The `_build_conversion_prompt()` method now generates different prompts based on the detected OS:

#### Windows Prompt
```
EXAMPLES:
Input: "dir"
Output: dir

Input: "dir C:\path"
Output: dir C:\path

Input: "ls -la"
Output: dir

RULES:
- Keep Windows commands (dir, type, copy, etc.) as-is - do NOT convert to Linux equivalents
- If a Linux command is used, convert it to the Windows equivalent
- For file listing: use "dir" not "ls"
```

#### Linux Prompt
```
EXAMPLES:
Input: "dir"
Output: ls

Input: "find /path -type f"
Output: find /path -type f -not -path "*/.*"

RULES:
- If a Windows command is used, convert it to the Linux equivalent
- For file listing: use "ls" not "dir"
```

### 2. **Enhanced Legacy Fallback**

The `_legacy_modify_command()` method now includes platform detection:

#### Windows Behavior
```python
if os_name == "Windows":
    if command.startswith("dir"):
        return command  # Keep dir as-is
    
    if command.startswith("ls"):
        return "dir"  # Convert ls to dir
    
    # Convert find to Windows equivalent
    if command.startswith("find"):
        return "dir /s /b"  # Recursive directory listing
```

#### Linux Behavior
```python
else:  # Linux/Unix
    if command.startswith("dir"):
        return "ls -la"  # Convert dir to ls
    
    # Apply Linux-style hidden file filtering
    if command.startswith("ls -la"):
        return "ls -la | grep -v '^\.'"
```

### 3. **Comprehensive Testing**

Added platform-specific tests to verify correct behavior:

```python
@pytest.mark.asyncio
async def test_windows_command_handling():
    with patch('platform.system', return_value='Windows'):
        test_cases = [
            ("dir", "dir"),  # Keep dir as-is
            ("ls -la", "dir"),  # Convert ls to dir
            ("find /path -type f", "dir /s /b \\path\\*"),  # Convert find
        ]
```

## Result

### Before Fix (Broken)
```
Windows Command: dir
LLM Converts to: ls | grep -v "^\\."
Error: 'ls' is not recognized as an internal or external command
```

### After Fix (Working)
```
Windows Command: dir
LLM Keeps as: dir
Success: Directory listing works correctly
```

## Key Benefits

1. **Platform Intelligence**: Commands are converted appropriately for the target OS
2. **Backward Compatibility**: Linux systems continue to work as before
3. **Robust Fallback**: Multiple layers of protection prevent failures
4. **Comprehensive Testing**: Both Windows and Linux scenarios are tested

## Files Modified

- `utils/command_converter.py` - OS-aware prompt generation
- `tools/bash.py` - Platform-aware legacy fallback
- `tests/utils/test_command_converter.py` - Windows/Linux-specific tests
- `tests/tools/test_bash.py` - Platform-aware test cases

## Environment Detection

The system automatically detects the OS using `platform.system()`:
- Returns `"Windows"` on Windows systems
- Returns `"Linux"` on Linux systems
- Returns `"Darwin"` on macOS systems

## Future Improvements

1. **macOS Support**: Add specific handling for macOS commands
2. **PowerShell Integration**: Support PowerShell commands on Windows
3. **Command Caching**: Cache converted commands for better performance
4. **Custom Rules**: Allow users to define custom conversion rules

This fix ensures that the LLM-based command converter works correctly across all supported operating systems, preventing the critical Windows command conversion failures.