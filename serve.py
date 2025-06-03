# serve.py is the entry point for the web server. It creates a new event loop and runs the web server on it.
# The server is started using the Waitress WSGI server, which is a production-quality server that can handle multiple requests concurrently.
import asyncio

from numpy import test
from utils.agent_display_web_with_prompt import create_app
from rich import print as rr
from rich.console import Console
if __name__ == "__main__":
    console = Console()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = create_app(loop)

    try:
        # from waitress import serve
        # serve(app, host="0.0.0.0", port=5001, threads=16) # Comment out Waitress
        print("Starting Flask development server...")
        app.run(host="0.0.0.0", port=5001, debug=True)  # Use Flask's dev server

    except Exception as e:
        console.print_exception(show_locals=True)
