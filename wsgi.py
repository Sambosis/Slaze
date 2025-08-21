import asyncio
from run import run_agent_async
from utils.web_ui import WebUI

# Create the WebUI and expose the Flask app for Gunicorn
web_ui = WebUI(lambda task, disp: run_agent_async(task, disp, False))
app = web_ui.app