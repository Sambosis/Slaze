#!/usr/bin/env python
"""
Local development runner with proper eventlet support for WebSocket connections.
This ensures that the local environment behaves similarly to the Heroku deployment.
"""

import eventlet
# Monkey patch standard library to work with eventlet
eventlet.monkey_patch()

import asyncio
import click
import webbrowser
import socket
from dotenv import load_dotenv

from agent import Agent
from utils.web_ui import WebUI
from utils.file_logger import archive_logs

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

@click.command()
@click.option('--port', default=5002, help='Port to run the web server on.')
@click.option('--manual-tools', is_flag=True, help='Confirm tool parameters before execution')
@click.option('--no-browser', is_flag=True, help='Do not open browser automatically')
def main(port, manual_tools, no_browser):
    """Run the agent with a web interface using eventlet for proper WebSocket support."""
    load_dotenv()
    archive_logs()
    
    print("🚀 Starting Slazy Agent with eventlet support...")
    
    display = WebUI(lambda task, disp: run_agent_async(task, disp, manual_tools))

    # Determine the local IP address
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            host_ip = s.getsockname()[0]
    except Exception:
        host_ip = "localhost"

    url = f"http://{host_ip}:{port}"
    print(f"✅ Web server starting on port {port}")
    print(f"📍 Local URL: http://localhost:{port}")
    print(f"📍 Network URL: {url}")
    
    if not no_browser:
        print("🌐 Opening browser...")
        webbrowser.open(f"http://localhost:{port}")
    
    print("⏳ Waiting for user to start a task from the web interface...")
    print("💡 Tip: If messages aren't showing, try refreshing the page after starting a task.")
    
    # Use eventlet's wsgi server for better WebSocket support
    import socketio
    import eventlet.wsgi
    
    # Get the underlying SocketIO server
    sio_server = display.socketio.server
    
    # Run with eventlet's WSGI server (similar to production)
    eventlet.wsgi.server(
        eventlet.listen(('0.0.0.0', port)),
        display.app,
        log_output=False
    )

if __name__ == "__main__":
    main()