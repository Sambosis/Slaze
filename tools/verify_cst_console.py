import asyncio
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Mock openai to avoid import errors in other tools
sys.modules["openai"] = MagicMock()

# Add repo root to path
sys.path.append(str(Path(__file__).parent.parent))

from tools.cst_code_editor import CSTCodeEditorTool, CSTEditorCommand

class MockDisplay:
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append((role, content))
        print(f"[{role}] {content}")

from config import get_constant

async def main():
    display = MockDisplay()
    tool = CSTCodeEditorTool(display=display)
    
    repo_dir = Path(get_constant("REPO_DIR"))
    if not repo_dir.exists():
        repo_dir.mkdir(parents=True, exist_ok=True)
        
    # Create a dummy file in REPO_DIR
    test_file = repo_dir / "test_cst_console.py"
    test_file.write_text("def foo():\n    pass\n")
    
    try:
        print("Testing LIST command...")
        await tool(command="list_symbols", path="test_cst_console.py")
        
        print("\nTesting SHOW command...")
        await tool(command="show_symbol", path="test_cst_console.py", symbol="foo")
        
        print("\nTesting REPLACE_BODY command...")
        await tool(command="replace_body", path="test_cst_console.py", symbol="foo", text="    print('hello')")
        
        # Verify messages
        print("\n--- Verification ---")
        if len(display.messages) >= 3:
            print("SUCCESS: Display messages received.")
        else:
            print(f"FAILURE: Expected at least 3 messages, got {len(display.messages)}")
            
    finally:
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    asyncio.run(main())
