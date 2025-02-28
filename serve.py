# serve.py is the entry point for the web server. It creates a new event loop and runs the web server on it.
# The server is started using the Waitress WSGI server, which is a production-quality server that can handle multiple requests concurrently.
import asyncio
from utils.agent_display_web_with_prompt import create_app

# Create new event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Create app with the loop
app = create_app(loop)

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5001, threads=16)
