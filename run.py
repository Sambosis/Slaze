import asyncio
import click

from dotenv import load_dotenv
import webbrowser
import socket

from agent import Agent
from utils.agent_display_console import AgentDisplayConsole
from utils.web_ui import WebUI
from utils.file_logger import archive_logs

@click.group()
@click.option('--interactive-tool-calls', is_flag=True, help='Enable interactive inspection and modification of tool calls.')
def cli(interactive_tool_calls):
    """Slazy Agent CLI"""
    # Store the flag value in a way that can be accessed later,
    # e.g., by passing it to the agent or display.
    # For now, we can store it in a global or pass it down.
    # A simple approach is to make it available to the commands.
    # Click's context object can be used for this.
    click.get_current_context().obj = interactive_tool_calls
    load_dotenv()
    archive_logs()

async def run_agent_async(task, display, interactive_tool_calls=False):
    """Asynchronously runs the agent with a given task and display."""
    agent = Agent(task=task, display=display, interactive_tool_calls=interactive_tool_calls)
    await agent._revise_and_save_task(agent.task)
    agent.messages.append({"role": "user", "content": agent.task})
    await sampling_loop(agent=agent)

async def sampling_loop(agent: Agent):
    """Main loop for the agent."""
    running = True
    while running:
        running = await agent.step()

@cli.command()
@click.pass_context
def console(ctx):
    """Run the agent in console mode."""
    interactive_tool_calls = ctx.obj
    display = AgentDisplayConsole(interactive_tool_calls=interactive_tool_calls)
    
    async def run_console_app():
        task = await display.select_prompt_console()
        if task:
            print("\n--- Starting Agent with Task ---")
            await run_agent_async(task, display, interactive_tool_calls)
            print("\n--- Agent finished ---")
        else:
            print("No task selected. Exiting.")

    try:
        asyncio.run(run_console_app())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")

@cli.command()
@click.option('--port', default=5000, help='Port to run the web server on.')
@click.pass_context
def web(ctx, port):
    """Run the agent with a web interface."""
    interactive_tool_calls = ctx.obj
    # Pass interactive_tool_calls to WebUI and/or run_agent_async_wrapper
    # WebUI might need it if interaction happens there, or it might just pass it along.
    # For now, let's assume run_agent_async (and its wrapper for WebUI) will handle it.

    # Wrapper function to pass interactive_tool_calls to run_agent_async
    async def run_agent_async_wrapper(task, display):
        await run_agent_async(task, display, interactive_tool_calls)

    display = WebUI(run_agent_async_wrapper, interactive_tool_calls=interactive_tool_calls)

    # Determine the local IP address for convenience when accessing from
    # another machine on the network. Fallback to localhost if detection fails.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            host_ip = s.getsockname()[0]
    except Exception:
        host_ip = "localhost"

    url = f"http://{host_ip}:{port}"
    print(f"Web server started on port {port}. Opening your browser to {url}")
    webbrowser.open(url)
    print("Waiting for user to start a task from the web interface.")
    
    display.start_server(port=port)
    
    # The Flask-SocketIO server is a blocking call that will keep the application
    # alive. It's started in display.start_server().

if __name__ == "__main__":
    print("Starting Slazy Agent CLI...")
    cli()
