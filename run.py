# run.py - console version
import asyncio
from typing import List, Dict, Any
import os  # Keep os if needed, e.g. by loaded modules, or for future use
from dotenv import load_dotenv  # Optional: Uncomment if you use .env files for API keys
import logging
from utils.agent_display_console import AgentDisplayConsole
from agent import Agent

# from main import run_sampling_loop # Assuming main.py has run_sampling_loop and is importable

# Docker and web server related functions have been removed.

import sys # Add this import

# Configure root logger for UTF-8 and file output only
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Test logger with a unicode character
logger = logging.getLogger(__name__) # This line should remain to keep the local logger
logger.info("Logging configured. Testing with Unicode: │")

# ----------------  The main Agent Loop ----------------
async def sampling_loop(
    *,
    agent: Agent,
    max_tokens: int = 180000,
) -> List[Dict[str, Any]]:
    """Main loop for agentic sampling."""
    running = True
    while running:
        try:
            running = await agent.step()

        except UnicodeEncodeError as ue:
            logger.error(f"UnicodeEncodeError: {ue}", exc_info=True)
            # rr(f"Unicode encoding error: {ue}") # Replaced by logger
            # rr(f"ascii: {ue.args[1].encode('ascii', errors='replace').decode('ascii')}") # Replaced by logger
            break
        except Exception as e:
            logger.error(
                f"Error in sampling loop: {str(e).encode('ascii', errors='replace').decode('ascii')}",
                exc_info=True
            )
            logger.error(
                f"The error occurred at the following message: {agent.messages[-1]} and line: {e.__traceback__.tb_lineno if e.__traceback__ else 'N/A'}",
            )
            logger.debug(f"Locals at error: {e.__traceback__.tb_frame.f_locals if e.__traceback__ else 'N/A'}")
            agent.display.add_message("user", ("Error", str(e)))
            raise
    return agent.messages


async def run_sampling_loop(
    task: str, display: AgentDisplayConsole
) -> List[Dict[str, Any]]:
    """Run the sampling loop with clean output handling."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please set the OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
        )
    # set the task as a constant in config
    agent = Agent(task=task, display=display)
    agent.messages.append({"role": "user", "content": task})
    messages = await sampling_loop(
        agent=agent,
        max_tokens=28000,
    )
    return messages


if __name__ == "__main__":
    # Optional: Load environment variables if you use .env files
    # For example, if OPENROUTER_API_KEY is in a .env file:
    load_dotenv()
    print(f"OPENROUTER_API_KEY loaded: {os.getenv('OPENROUTER_API_KEY') is not None}")

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
        import traceback  # Uncomment for detailed traceback

        traceback.print_exc()  # Uncomment for detailed traceback
