import asyncio
import uvicorn
import webbrowser
import time
from utils.agent_display_web_with_prompt import create_app

# Create new event loop
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Create app with the loop
app = create_app(loop)

# Start the server
server_process = asyncio.create_task(uvicorn.run(app, host="0.0.0.0", port=5001))

# Wait a moment for the server to start
time.sleep(2)

# Open the browser
webbrowser.open("http://127.0.0.1:5001/select_prompt")

try:
    # Keep the main thread running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    server_process.cancel()
