import asyncio
import click
import threading # Added
import webbrowser

from dotenv import load_dotenv

from agent import Agent
from utils.agent_display_console import AgentDisplayConsole
from utils.web_ui import WebUI
from utils.file_logger import archive_logs

@click.group()
def cli():
    """Slazy Agent CLI"""
    load_dotenv()
    archive_logs()

async def run_agent_async(task, display):
    """Asynchronously runs the agent with a given task and display."""
    agent = Agent(task=task, display=display)
    agent.messages.append({"role": "user", "content": task})
    await sampling_loop(agent=agent)

async def sampling_loop(agent: Agent):
    """Main loop for the agent."""
    running = True
    while running:
        running = await agent.step()

@cli.command()
def console():
    """Run the agent in console mode."""
    display = AgentDisplayConsole()
    
    async def run_console_app():
        task = await display.select_prompt_console()
        if task:
            print("\n--- Starting Agent with Task ---")
            await run_agent_async(task, display)
            print("\n--- Agent finished ---")
        else:
            print("No task selected. Exiting.")

    try:
        asyncio.run(run_console_app())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")

@cli.command()
@click.option('--port', default=5000, help='Port to run the web server on.')
def web(port):
    """Run the agent with a web interface."""

    # 1. Create an asyncio loop that will run in a separate thread
    loop = asyncio.new_event_loop()
    # It's good practice to set the loop for the current context before starting the thread,
    # though the thread itself will also call set_event_loop.
    # asyncio.set_event_loop(loop) # This might not be strictly necessary here

    # 2. Define a wrapper for run_agent_async to be used with run_coroutine_threadsafe
    #    This wrapper will be called by WebUI from a Flask/SocketIO thread
    def run_agent_async_web_wrapper(task, display_ref):
        # This function is called from a non-asyncio thread (SocketIO thread)
        # Schedule run_agent_async to run in the dedicated asyncio loop
        future = asyncio.run_coroutine_threadsafe(run_agent_async(task, display_ref), loop)
        try:
            # If you need to wait for a result or catch exceptions from the coroutine
            # future.result(timeout=None) # Or some appropriate timeout
            pass # For now, just fire and forget, errors handled within run_agent_async
        except Exception as e:
            print(f"Error scheduling or running agent task: {e}")


    # 3. Create WebUI, passing the loop and the new wrapper
    #    The WebUI will create its asyncio.Queue in this loop.
    display = WebUI(run_agent_async_web_wrapper, loop)

    # 4. Start the asyncio event loop in a separate thread
    def run_loop_forever(l):
        asyncio.set_event_loop(l)
        try:
            l.run_forever()
        finally:
            # Optional: Clean up before the loop closes
            # Gather all tasks and cancel them
            # all_tasks = asyncio.all_tasks(l)
            # for task in all_tasks:
            #     task.cancel()
            # l.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
            l.close()
            print("Asyncio event loop closed.")

    loop_thread = threading.Thread(target=run_loop_forever, args=(loop,), daemon=True)
    loop_thread.start()

    url = f"http://localhost:{port}"
    print(f"Web server started on port {port}. Opening your browser to {url}")
    webbrowser.open(url)
    print("Waiting for user to start a task from the web interface.")
    print(f"Asyncio event loop running in thread: {loop_thread.name}")

    # 5. Start the Flask-SocketIO server (blocking)
    #    This runs in the main thread.
    try:
        display.start_server(port=port)
    finally:
        print("Shutting down Flask server...")
        if loop.is_running():
            print("Stopping asyncio event loop...")
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=5) # Wait for the loop thread to finish
            if loop_thread.is_alive():
                print("Loop thread did not stop in time.")
        print("Shutdown complete.")

if __name__ == "__main__":
    cli()
