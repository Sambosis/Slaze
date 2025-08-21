"""
Vercel serverless function entry point for the Slazy application.
This wraps the Flask application for deployment on Vercel.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up environment
os.environ.setdefault('FLASK_ENV', 'production')

# Import and configure the Flask app
from utils.web_ui import WebUI
from agent import Agent
import asyncio
from functools import wraps

def create_app():
    """Create and configure the Flask application for Vercel."""
    
    # Create a simple async runner for the agent
    async def run_agent_async(task, display, manual_tools=False):
        """Asynchronously runs the agent with a given task and display."""
        output = f"The users original task is: {task}\n"
        display.add_message("user", output)
        output = "# Refining Now..."
        display.add_message("user", output)
        agent = Agent(task=task, display=display, manual_tool_confirmation=manual_tools)
        await agent._revise_and_save_task(agent.task)
        agent.messages.append({"role": "user", "content": agent.task})
        
        # Run the agent
        running = True
        while running:
            running = await agent.step()
    
    # Initialize the WebUI with the agent runner
    display = WebUI(lambda task, disp: run_agent_async(task, disp, False))
    
    # Return the Flask app instance
    return display.app

# Create the app instance
app = create_app()

# This is the handler that Vercel will call
def handler(request, context):
    """Vercel serverless function handler."""
    with app.test_request_context(
        path=request.path,
        method=request.method,
        headers=request.headers,
        data=request.body
    ):
        try:
            response = app.full_dispatch_request()
            return {
                'statusCode': response.status_code,
                'headers': dict(response.headers),
                'body': response.get_data(as_text=True)
            }
        except Exception as e:
            return {
                'statusCode': 500,
                'body': str(e)
            }

# For local testing
if __name__ == "__main__":
    app.run(debug=True, port=5002)