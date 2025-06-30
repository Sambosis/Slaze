import asyncio
import click

from dotenv import load_dotenv
import webbrowser

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
    await agent._revise_and_save_task(agent.task)
    agent.messages.append({"role": "user", "content": agent.task})
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
@click.option('--host', default='0.0.0.0', help='Host interface to bind the web server.')
def web(port, host):
    """Run the agent with a web interface."""
    
    display = WebUI(run_agent_async)
    
    url_host = 'localhost' if host in ['0.0.0.0', ''] else host
    url = f"http://{url_host}:{port}"
    print(f"Web server started on {host}:{port}. Opening your browser to {url}")
    webbrowser.open(url)
    print("Waiting for user to start a task from the web interface.")
    
    display.start_server(host=host, port=port)
    
    # The Flask-SocketIO server is a blocking call that will keep the application
    # alive. It's started in display.start_server().

if __name__ == "__main__":
    print("Starting Slazy Agent CLI...")
    cli()
