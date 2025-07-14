import asyncio
from tools.interpreter import InterpreterTool
from tools.base import ToolError
from dotenv import load_dotenv
import os

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

async def main():
    print("Starting InterpreterTool instantiation test (no display)...")
    # Ensure OPENAI_API_KEY is set for the interpreter's LLM
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the test, e.g., in a .env file.")
        # Attempting to set a dummy key if not found, for basic non-LLM dependent tests
        # This is risky as open-interpreter might still try to use it.
        # os.environ["OPENAI_API_KEY"] = "dummy-key-for-testing"
        # print("Warning: OPENAI_API_KEY was not set. Using a dummy key. LLM calls will likely fail.")
        # For now, let's require it.
        return

    try:
        # Instantiate without display to reduce dependencies for this test
        tool = InterpreterTool(display=None)
        print("InterpreterTool instantiated successfully (no display).")

        # Test __call__ method with a simple command
        print("\nTesting __call__ with a simple command: 'what is 2+2?'")
        try:
            # Use a very simple, non-filesystem command first
            result = await tool(command="What is 2+2? Reply with just the number.")
            print("\n__call__ method executed.")
            print(f"Result Output:\n{result.output}")
            if result.error:
                print(f"Result Error:\n{result.error}")

            print("\nTesting __call__ with a command: 'list files in current directory'")
            result_ls = await tool(command="List files in the current directory. Only show names.")
            print("\n__call__ method for 'list files' executed.")
            print(f"Result Output for 'list files':\n{result_ls.output}")
            if result_ls.error:
                print(f"Result Error for 'list files':\n{result_ls.error}")

        except ToolError as e:
            print(f"ToolError during __call__: {e}")
        except Exception as e:
            print(f"Unexpected error during __call__: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error during InterpreterTool instantiation or test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for OPENAI_API_KEY before running asyncio.run
    # This is a bit redundant as it's also checked in main(), but good for early exit.
    if not os.getenv("OPENAI_API_KEY"):
        print("Critical Error: OPENAI_API_KEY environment variable not set. Exiting.")
    else:
        asyncio.run(main())
