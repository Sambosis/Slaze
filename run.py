import asyncio
import click
import eventlet

eventlet.monkey_patch()

from dotenv import load_dotenv
import webbrowser
import socket

from agent import Agent
from utils.agent_display_console import AgentDisplayConsole
from utils.web_ui import WebUI
from utils.file_logger import archive_logs

@click.group()
def cli():
    """Slazy Agent CLI"""
    load_dotenv()
    archive_logs()

async def run_agent_async(task, display, manual_tools=False):
    """Asynchronously runs the agent with a given task and display."""
    output = f"The users original task is: {task}\n"
    display.add_message("user", output)
    output = "# Refining Now..."
    display.add_message("user", output)
    agent = Agent(task=task, display=display, manual_tool_confirmation=manual_tools)
    await agent._revise_and_save_task(agent.task)
    agent.messages.append({"role": "user", "content": agent.task})
    await sampling_loop(agent=agent)

async def sampling_loop(agent: Agent):
    """Main loop for the agent."""
    running = True
    while running:
        running = await agent.step()

@cli.command()
@click.option('--manual-tools', is_flag=True, help='Confirm tool parameters before execution')
def console(manual_tools):
    """Run the agent in console mode."""
    display = AgentDisplayConsole()
    
    async def run_console_app():
        task = await display.select_prompt_console()
        if task:
            print("\n--- Starting Agent with Task ---")
            await run_agent_async(task, display, manual_tools)
            print("\n--- Agent finished ---")
        else:
            print("No task selected. Exiting.")

    try:
        asyncio.run(run_console_app())
    except KeyboardInterrupt:
        print("\nApplication interrupted by user. Exiting.")

@cli.command()
@click.option('--port', default=5002, help='Port to run the web server on.')
@click.option('--manual-tools', is_flag=True, help='Confirm tool parameters before execution')
def web(port, manual_tools):
    """Run the agent with a web interface."""

    display = WebUI(lambda task, disp: run_agent_async(task, disp, manual_tools))

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
