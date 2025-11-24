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
    target = "test_cst_target_2.py"
    with open(os.path.join(get_constant("REPO_DIR"), target), "w", encoding="utf-8") as f:
        f.write('''
import os

class MyClass:
    """Docstring."""
    def method(self):
        print("Original")

@existing_decorator
def my_func(x):
    return x + 1
''')

    print(f"REPO_DIR: {get_constant('REPO_DIR')}")
    
    print("\n--- Test 1: Add Decorator ---")
    res = await tool(command="add_decorator", path=target, symbol="MyClass.method", text="@staticmethod", dry_run=True)
    if res.error: print(f"ERROR: {res.error}")
    else: print(res.output)

    print("\n--- Test 2: Remove Decorator ---")
    res = await tool(command="remove_decorator", path=target, symbol="my_func", text="@existing_decorator", dry_run=True)
    if res.error: print(f"ERROR: {res.error}")
    else: print(res.output)

    print("\n--- Test 3: Wrap Body ---")
    wrapper = """
    try:
        pass
    except Exception:
        print("Error")
    """
    res = await tool(command="wrap_body", path=target, symbol="MyClass.method", text=wrapper, dry_run=True)
    if res.error: print(f"ERROR: {res.error}")
    else: print(res.output)

    print("\n--- Test 4: Rename ---")
    res = await tool(command="rename", path=target, symbol="MyClass", text="RenamedClass", dry_run=True)
    if res.error: print(f"ERROR: {res.error}")
    else: print(res.output)

if __name__ == "__main__":
    asyncio.run(main())
