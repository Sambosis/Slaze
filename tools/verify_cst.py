print("Starting verify_cst.py")
import asyncio
import sys
import os

# Add root to sys.path
sys.path.append(os.getcwd())

from tools.cst_code_editor import CSTCodeEditorTool
from config import get_constant

async def main():
    tool = CSTCodeEditorTool()
    # Create a temporary test file
    target = "test_cst_target.py"
    with open(os.path.join(get_constant("REPO_DIR"), target), "w", encoding="utf-8") as f:
        f.write('''
import os

class MyClass:
    """Docstring."""
    def method(self):
        pass

def my_func(x):
    return x + 1
''')

    print(f"REPO_DIR: {get_constant('REPO_DIR')}")
    
    print("\n--- Test 1: List Symbols ---")
    res = await tool(command="list_symbols", path=target)
    if res.error: print(f"ERROR: {res.error}")
    else: print(res.output)

    print("\n--- Test 2: Show Symbol ---")
    res = await tool(command="show_symbol", path=target, symbol="MyClass.method")
    if res.error: print(f"ERROR: {res.error}")
    else: print(f"SUCCESS: {len(res.output)} chars")

    print("\n--- Test 3: Replace Body (Preserve Indentation) ---")
    new_body = """
        print("Hello")
        return True
"""
    res = await tool(command="replace_body", path=target, symbol="MyClass.method", text=new_body, dry_run=True)
    if res.error: print(f"ERROR: {res.error}")
    else: 
        print("SUCCESS (Dry Run)")
        print(res.output)

    print("\n--- Test 4: Add Import ---")
    res = await tool(command="add_import", path=target, text="import sys", dry_run=True)
    if res.error: print(f"ERROR: {res.error}")
    else:
        print("SUCCESS (Dry Run)")
        print(res.output)

    print("\n--- Test 5: Insert Before ---")
    res = await tool(command="insert_before", path=target, symbol="MyClass", text="# Class Comment", dry_run=True)
    if res.error: print(f"ERROR: {res.error}")
    else:
        print("SUCCESS (Dry Run)")
        print(res.output)

    print("\n--- Test 6: Refactor (Rename Symbol and References) ---")
    # First, create a test file with function calls
    with open(os.path.join(get_constant("REPO_DIR"), target), "w", encoding="utf-8") as f:
        f.write('''
import os

def old_func(x):
    return x + 1

def caller():
    result = old_func(5)
    return old_func(result)

class MyClass:
    def use_func(self):
        return old_func(10)
''')
    
    res = await tool(command="refactor", path=target, symbol="old_func", text="new_func", dry_run=True)
    if res.error: 
        print(f"ERROR: {res.error}")
    else:
        print("SUCCESS (Dry Run)")
        # Check if both definition and references were renamed
        diff_output = res.output
        if "new_func" in diff_output and "+def new_func" in diff_output:
            print("✓ Function definition renamed")
        if "+    result = new_func(5)" in diff_output or "new_func(5)" in diff_output:
            print("✓ Function references renamed")
        print(diff_output)

if __name__ == "__main__":
    asyncio.run(main())
