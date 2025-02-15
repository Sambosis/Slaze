from test_navigation_tool import shortcuts

windows_navigate_function = {
    "name": "windows_navigate",
    "description": "A tool for Windows navigation using keyboard shortcuts.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(shortcuts.keys()),
                "description": "The Windows action to perform."
            },
            "modifier": {
                "type": ["string", "null"],
                "enum": ["ctrl", "alt", "shift", "win"],
                "description": "Optional modifier key."
            },
            "target": {
                "type": ["string", "null"],
                "description": "Optional target for the action (e.g., window title)."
            }
        },
        "required": ["action"],
    },
}
edit_file_function =  {
        "name": "edit_file",
        "description": "Custom editing tool for viewing, creating and editing files. State is persistent across command calls and discussions with the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "description": "The command to run."
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path to file or directory, e.g. '/repo/file.py' or '/repo'."
                },
                "file_text": {
                    "type": "string",
                    "description": "Required parameter of 'create' command, with the content of the file to be created."
                },
                "view_range": {
                    "type": "array",
                    "items": {
                        "type": "integer"
                    },
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Optional parameter of 'view' command when 'path' points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting [start_line, -1] shows all lines from start_line to the end of the file."
                },
                "old_str": {
                    "type": "string",
                    "description": "Required parameter of 'str_replace' command containing the string in 'path' to replace."
                },
                "new_str": {
                    "type": "string",
                    "description": "Optional parameter of 'str_replace' command containing the new string (if not given, no string will be added). Required parameter of 'insert' command containing the string to insert."
                },
                "insert_line": {
                    "type": "integer",
                    "description": "Required parameter of 'insert' command. The 'new_str' will be inserted AFTER the line 'insert_line' of 'path'."
                }
            },
            "required": ["command", "path"]
        }
}
bash_command_function = {
    "name": "bash_command",
    "description": "Run commands in a bash shell. When invoking this tool, the contents of the 'command' parameter does NOT need to be XML-escaped. You have access to a mirror of common linux and python packages via apt and pip. State is persistent across command calls and discussions with the user. To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'. Please avoid commands that may produce a very large amount of output. Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.",
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to be executed."
            }
        },
        "required": ["command"]
    }
}