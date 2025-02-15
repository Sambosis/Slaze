# agent_display_web.py
import os
import threading
import asyncio
from queue import Queue
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
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
    A class for managing and displaying messages on a web page using Flask and SocketIO.
    """
    def __init__(self):
        # Assume that templates folder is at the project root.
        template_dir = os.path.join(os.getcwd(), "templates")
        self.app = Flask(__name__, template_folder=template_dir)
        self.app.config['SECRET_KEY'] = 'secret!'
        self.app.debug = True  # Enable debug mode for detailed errors.
        self.socketio = SocketIO(self.app, async_mode='threading')
        self.user_messages = []
        self.assistant_messages = []
        self.tool_results = []
        self.message_queue = Queue()
        # We'll use an asyncio.Queue to deliver user input.
        self.input_queue = asyncio.Queue()
        self.loop = None  # This should be set by the main async function.
        self.setup_routes()
        self.setup_socketio_events()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            try:
                return render_template("index.html")
            except Exception as e:
                return f"Error rendering index: {e}", 500

        @self.app.route('/messages')
        def get_messages():
            return jsonify({
                'user': self.user_messages,
                'assistant': self.assistant_messages,
                'tool': self.tool_results
            })

    def setup_socketio_events(self):
        @self.socketio.on('user_input')
        def handle_user_input(data):
            print("[DEBUG] Received user_input event with data:", data)
            user_input = data.get('input', '')
            if self.loop is not None:
                # Enqueue the input on the main event loop
                self.loop.call_soon_threadsafe(self.input_queue.put_nowait, user_input)
                print("[DEBUG] Enqueued user input:", user_input)
            else:
                print("[DEBUG] No event loop set; cannot enqueue user input")
            return None

    async def wait_for_user_input(self, user_input = None):
        """
        Wait for user input sent from the client via SocketIO.
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

    def broadcast_update(self):
        # Emit an update event to all connected clients
        self.socketio.emit('update', {
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
        self.broadcast_update()

    def clear_messages(self, panel):
        if panel in ("user", "all"):
            self.user_messages.clear()
        if panel in ("assistant", "all"):
            self.assistant_messages.clear()
        if panel in ("tool", "all"):
            self.tool_results.clear()
        self.broadcast_update()

    def start_server(self, host='0.0.0.0', port=5000):
        thread = threading.Thread(
            target=self.socketio.run,
            args=(self.app,),
            kwargs={'host': host, 'port': port, 'use_reloader': False}
        )
        thread.daemon = True
        thread.start()
