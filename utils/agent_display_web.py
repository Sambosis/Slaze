# agent_display_web.py
import os
import threading
import asyncio
from queue import Queue
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from config import USER_LOG_FILE, ASSISTANT_LOG_FILE, TOOL_LOG_FILE, LOGS_DIR

def log_message(msg_type, message):
    """Log a message to a file."""
    if msg_type == "user":
        emojitag = "🤡 "
    elif msg_type == "assistant":
        emojitag = "🧞‍♀️ "
    elif msg_type == "tool":
        emojitag = "📎 "
    else:
        emojitag = "❓ "
    log_file = os.path.join(LOGS_DIR, f"{msg_type}_messages.log")
    with open(log_file, "a", encoding="utf-8") as file:
        file.write(emojitag * 5)
        file.write(f"\n{message}\n\n")

class AgentDisplayWeb:
    """
    A class for managing and displaying messages on a web page using FastAPI and WebSocket.
    """
    def __init__(self):
        # Assume that templates folder is at the project root.
        template_dir = os.path.join(os.getcwd(), "templates")
        self.app = FastAPI()
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
        self.templates = Jinja2Templates(directory=template_dir)
        self.user_messages = []
        self.assistant_messages = []
        self.tool_results = []
        self.message_queue = Queue()
        # We'll use an asyncio.Queue to deliver user input.
        self.input_queue = asyncio.Queue()
        self.loop = None  # This should be set by the main async function.
        self.setup_routes()
        self.setup_websocket_events()

    def setup_routes(self):
        @self.app.get('/')
        async def index():
            try:
                return self.templates.TemplateResponse("index.html", {"request": {}})
            except Exception as e:
                return HTMLResponse(f"Error rendering index: {e}", status_code=500)

        @self.app.get('/messages')
        async def get_messages():
            return {
                'user': self.user_messages,
                'assistant': self.assistant_messages,
                'tool': self.tool_results
            }

    def setup_websocket_events(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    data = await websocket.receive_text()
                    await self.input_queue.put(data)
                    await self.broadcast_update()
            except WebSocketDisconnect:
                print("[DEBUG] Client disconnected")

    async def wait_for_user_input(self, user_input = None):
        """
        Wait for user input sent from the client via WebSocket.
        Returns:
            The user input as a string.
        """
        # Wait for an item to appear in the queue.
        if user_input:
            print("[DEBUG] wait_for_user_input returning from main.py:", user_input)
            return user_input
        else:
            user_input = await self.input_queue.get()
            print("[DEBUG] wait_for_user_input returning from index.html:", user_input)
        return user_input

    async def broadcast_update(self):
        # Emit an update event to all connected clients
        for connection in self.app.websocket_connections:
            await connection.send_json({
                'user': self.user_messages,  # Only send the last eight messages
                'assistant': self.assistant_messages, # Only send the last five messages
                'tool': self.tool_results  # Simply pass the list directly
            })

    def add_message(self, msg_type, content):
        log_message(msg_type, content)
        if msg_type == "user":
            self.user_messages.append(content)
        elif msg_type == "assistant":
            self.assistant_messages.append(content)
        elif msg_type == "tool":
            self.tool_results.append(content)
        asyncio.create_task(self.broadcast_update())

    def clear_messages(self, panel):
        if panel in ("user", "all"):
            self.user_messages.clear()
        if panel in ("assistant", "all"):
            self.assistant_messages.clear()
        if panel in ("tool", "all"):
            self.tool_results.clear()
        asyncio.create_task(self.broadcast_update())

    def start_server(self, host='0.0.0.0', port=None): # Remove default port here
        if port is None: # Get port from environment or default to 5001 if not set (for local dev)
            port = int(os.environ.get('PORT', 5001))
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)
