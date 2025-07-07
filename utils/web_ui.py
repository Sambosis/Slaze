import asyncio
import os
import threading
import logging
import json
from queue import Queue
from flask import Flask, render_template, jsonify, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, disconnect
from config import (
    LOGS_DIR,
    PROMPTS_DIR,
    get_constant,
    set_constant,
    set_prompt_name,
    write_constants_to_file,
)
from pathlib import Path
from openai import OpenAI
import ftfy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class WebUI:
    def __init__(self, agent_runner):
        logging.info("Initializing WebUI")
        # More robust path for templates
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
        logging.info(f"Template directory set to: {template_dir}")
        self.app = Flask(__name__, template_folder=template_dir)
        self.app.config["SECRET_KEY"] = "secret!"
        self.socketio = SocketIO(self.app, async_mode="threading", cookie=None)
        self.user_messages = []
        self.assistant_messages = []
        self.tool_results = []
        # Using a standard Queue for cross-thread communication
        self.input_queue = Queue()
        self.agent_runner = agent_runner
        # Import tools lazily to avoid circular imports
        from tools import (
            BashTool,
            ProjectSetupTool,
            WriteCodeTool,
            PictureGenerationTool,
            EditTool,
            ToolCollection,
        )

        self.tool_collection = ToolCollection(
            WriteCodeTool(display=self),
            ProjectSetupTool(display=self),
            BashTool(display=self),
            PictureGenerationTool(display=self),
            EditTool(display=self),
            display=self,
        )
        self.setup_routes()
        self.setup_socketio_events()
        logging.info("WebUI initialized")

    def setup_routes(self):
        logging.info("Setting up routes")

        @self.app.route("/")
        def select_prompt_route():
            logging.info("Serving prompt selection page")
            prompt_files = list(PROMPTS_DIR.glob("*.md"))
            options = [file.name for file in prompt_files]
            return render_template("select_prompt.html", options=options)

        @self.app.route("/run_agent", methods=["POST"])
        def run_agent_route():
            logging.info("Received request to run agent")
            choice = request.form.get("choice")
            filename = request.form.get("filename")
            prompt_text = request.form.get("prompt_text")
            logging.info(f"Form data: choice={choice}, filename={filename}")

            if choice == "new":
                logging.info("Creating new prompt")
                new_prompt_path = PROMPTS_DIR / f"{filename}.md"
                prompt_name = Path(filename).stem
                with open(new_prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt_text)
                task = prompt_text
            else:
                logging.info(f"Loading existing prompt: {choice}")
                prompt_path = PROMPTS_DIR / choice
                prompt_name = prompt_path.stem
                if prompt_text:
                    logging.info("Updating existing prompt")
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write(prompt_text)
                with open(prompt_path, "r", encoding="utf-8") as f:
                    task = f.read()
                filename = prompt_path.stem

            # Configure repository directory for this prompt
            base_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo"
            repo_dir = base_repo_dir / prompt_name
            repo_dir.mkdir(parents=True, exist_ok=True)
            set_prompt_name(prompt_name)
            set_constant("PROJECT_DIR", repo_dir)
            set_constant("REPO_DIR", repo_dir)
            write_constants_to_file()
            
            logging.info("Starting agent runner in background thread")
            coro = self.agent_runner(task, self)
            self.socketio.start_background_task(asyncio.run, coro)
            return render_template("index.html")

        @self.app.route("/messages")
        def get_messages():
            logging.info("Serving messages")
            return jsonify(
                {
                    "user": self.user_messages,
                    "assistant": self.assistant_messages,
                    "tool": self.tool_results,
                }
            )

        @self.app.route("/api/prompts/<path:filename>")
        def api_get_prompt(filename):
            """Return the raw content of a prompt file."""
            logging.info(f"Serving prompt content for: {filename}")
            try:
                prompt_path = PROMPTS_DIR / filename
                with open(prompt_path, "r", encoding="utf-8") as f:
                    data = f.read()
                return data, 200, {"Content-Type": "text/plain; charset=utf-8"}
            except FileNotFoundError:
                logging.error(f"Prompt not found: {filename}")
                return "Prompt not found", 404

        @self.app.route("/tools")
        def tools_route():
            """Display available tools."""
            tool_list = []
            for tool in self.tool_collection.tools.values():
                info = tool.to_params()["function"]
                tool_list.append({"name": info["name"], "description": info["description"]})
            return render_template("tool_list.html", tools=tool_list)

        @self.app.route("/tools/<tool_name>", methods=["GET", "POST"])
        def run_tool_route(tool_name):
            """Run an individual tool from the toolbox."""
            tool = self.tool_collection.tools.get(tool_name)
            if not tool:
                return "Tool not found", 404
            params = tool.to_params()["function"]["parameters"]
            result_text = None
            if request.method == "POST":
                tool_input = {}
                for param in params.get("properties", {}):
                    value = request.form.get(param)
                    if value:
                        pinfo = params["properties"].get(param, {})
                        if pinfo.get("type") == "integer":
                            try:
                                tool_input[param] = int(value)
                            except ValueError:
                                tool_input[param] = value
                        elif pinfo.get("type") == "array":
                            try:
                                tool_input[param] = json.loads(value)
                            except Exception:
                                tool_input[param] = [v.strip() for v in value.split(',') if v.strip()]
                        else:
                            tool_input[param] = value
                try:
                    result = asyncio.run(self.tool_collection.run(tool_name, tool_input))
                    result_text = result.output or result.error
                except Exception as exc:
                    result_text = str(exc)
            return render_template(
                "tool_form.html",
                tool_name=tool_name,
                params=params,
                result=result_text,
            )
        logging.info("Routes set up")

    def setup_socketio_events(self):
        logging.info("Setting up SocketIO events")

        @self.socketio.on("connect")
        def handle_connect():
            logging.info("Client connected")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            logging.info("Client disconnected")

        @self.socketio.on("user_input")
        def handle_user_input(data):
            user_input = data.get("input", "")
            logging.info(f"Received user input: {user_input}")
            # Queue is thread-safe; use blocking put to notify waiting tasks
            self.input_queue.put(user_input)
        logging.info("SocketIO events set up")

    def start_server(self, host="0.0.0.0", port=5000):
        logging.info(f"Starting server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, use_reloader=False, allow_unsafe_werkzeug=True)

    def add_message(self, msg_type, content):
        logging.info(f"Adding message of type {msg_type}")
        log_message(msg_type, content)
        if msg_type == "user":
            self.user_messages.append(content)
        elif msg_type == "assistant":
            self.assistant_messages.append(content)
        elif msg_type == "tool":
            self.tool_results.append(content)
        self.broadcast_update()

    def broadcast_update(self):
        logging.info("Broadcasting update to clients")
        self.socketio.emit(
            "update",
            {
                "user": self.user_messages,
                "assistant": self.assistant_messages,
                "tool": self.tool_results,
            },
        )

    async def wait_for_user_input(self, prompt_message: str = None) -> str:
        """Await the next user input sent via the web UI input queue."""
        if prompt_message:
            logging.info(f"Emitting agent_prompt: {prompt_message}")
            self.socketio.emit("agent_prompt", {"message": prompt_message})

        loop = asyncio.get_running_loop()
        user_response = await loop.run_in_executor(None, self.input_queue.get)

        # Clear the prompt after input is received
        logging.info("Emitting agent_prompt_clear")
        self.socketio.emit("agent_prompt_clear")

        return user_response

