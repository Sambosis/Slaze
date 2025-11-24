import asyncio
import sys
import os

# Add root to sys.path to allow imports from config and tools
sys.path.append(os.getcwd())

from tools.ast_code_editor import ASTCodeEditorTool, EditorCommand
from config import get_constant

async def main():
    tool = ASTCodeEditorTool()
    target = "test_ast_editor_target.py" # Relative to REPO_DIR
    
    print(f"REPO_DIR: {get_constant('REPO_DIR')}")
    
    print("\n--- Test 1: Show Nested Symbol ---")
    res = await tool(command="show_symbol", path=target, symbol="outer_func.inner_func")
    if res.error:
        print(f"ERROR: {res.error}")
    else:
        print("SUCCESS")
        # print(res.output)

    print("\n--- Test 2: Show Variable ---")
    res = await tool(command="show_symbol", path=target, symbol="GLOBAL_VAR")
    if res.error:
        print(f"ERROR: {res.error}")
    else:
        print(f"SUCCESS: {res.output.strip()}")

    print("\n--- Test 3: Insert Before Class ---")
    res = await tool(command="insert_before", path=target, symbol="MyClass", text="# This is a class comment", dry_run=True)
    if res.error:
        print(f"ERROR: {res.error}")
    else:
        print("SUCCESS (Dry Run)")
        print(res.output)
    
    print("\n--- Test 4: Add Import ---")
    res = await tool(command="add_import", path=target, text="import math", dry_run=True)
    if res.error:
        print(f"ERROR: {res.error}")
    else:
        print("SUCCESS (Dry Run)")
        print(res.output)

    print("\n--- Test 5: Remove Import ---")
    # Note: target file has 'import os'
    res = await tool(command="remove_import", path=target, text="import os", dry_run=True)
    if res.error:
        print(f"ERROR: {res.error}")
    else:
        print("SUCCESS (Dry Run)")
        print(res.output)

if __name__ == "__main__":
    asyncio.run(main())
