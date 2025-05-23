# run.py - console version
import asyncio
import os # Keep os if needed, e.g. by loaded modules, or for future use
# from dotenv import load_dotenv # Optional: Uncomment if you use .env files for API keys

from utils.agent_display_console import AgentDisplayConsole
from main import run_sampling_loop # Assuming main.py has run_sampling_loop and is importable

# Docker and web server related functions have been removed.

if __name__ == "__main__":
    # Optional: Load environment variables if you use .env files
    # For example, if OPENROUTER_API_KEY is in a .env file:
    # load_dotenv() 
    # print(f"OPENROUTER_API_KEY loaded: {os.getenv('OPENROUTER_API_KEY') is not None}")


    display = AgentDisplayConsole()

    async def run_console_app():
        # AgentDisplayConsole.select_prompt_console() is an async method.
        task = await display.select_prompt_console()
        
        if task:
            print("\n--- Starting Agent with Task ---")
            # AgentDisplayConsole is expected to be compatible with run_sampling_loop
            # due to its inheritance from AgentDisplayWeb and provision of necessary methods
            # like .loop, .add_message(), and .wait_for_user_input().
            await run_sampling_loop(task=task, display=display)
            print("\n--- Agent finished ---")
        else:
            print("No task selected. Exiting.")

    try:
        asyncio.run(run_console_app())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # import traceback # Uncomment for detailed traceback
        # traceback.print_exc() # Uncomment for detailed traceback
