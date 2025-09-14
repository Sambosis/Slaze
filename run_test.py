# run_test.py - automatically run the agent with a predefined task
"""Run the agent using the task defined in prompts/heyworld.md without
requiring interactive prompt selection."""

import asyncio
from pathlib import Path
import os
from dotenv import load_dotenv
from utils.agent_display_console import AgentDisplayConsole
from agent import Agent
from icecream import install

install()

async def sampling_loop(*, agent: Agent, max_tokens: int = 180000):
    """Main loop for agentic sampling."""
    running = True
    while running:
        try:
            running = await agent.step()
        except UnicodeEncodeError as ue:
            print(f"UnicodeEncodeError: {ue}")
            break
        except Exception as e:
            print(f"Error in sampling loop: {e}")
            agent.display.add_message("user", ("Error", str(e)))
            raise
    return agent.messages

async def run_sampling_loop(task: str, display: AgentDisplayConsole):
    """Run the sampling loop with clean output handling."""
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not found. Please set the OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
        )

    agent = Agent(task=task, display=display)
    agent.messages.append({"role": "user", "content": task})
    return await sampling_loop(agent=agent, max_tokens=28000)

if __name__ == "__main__":
    load_dotenv()
    display = AgentDisplayConsole()

    async def run_console_app():
        task_path = Path("prompts/heyworld.md")
        task = task_path.read_text(encoding="utf-8")
        print(f"Using task from {task_path}")
        await run_sampling_loop(task=task, display=display)
        print("\n--- Agent finished ---")

    try:
        asyncio.run(run_console_app())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
