import asyncio
import os
import logging
import json
from queue import Queue
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO
from config import (
    LOGS_DIR,
    PROMPTS_DIR,
    get_constant,
    set_constant,
    set_prompt_name,
    write_constants_to_file,
)
from pathlib import Path


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
        self.tool_queue = Queue()
        self.agent_runner = agent_runner
        # Import tools lazily to avoid circular imports
        from tools import (
            # BashTool,
            # OpenInterpreterTool,
            ProjectSetupTool,
            WriteCodeTool,
            PictureGenerationTool,
            EditTool,
            ToolCollection,
            BashTool
        )

        self.tool_collection = ToolCollection(
            WriteCodeTool(display=self),
            ProjectSetupTool(display=self),
            BashTool(display=self),
            # OpenInterpreterTool(display=self),  # Uncommented and enabled for testing
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
            logging.info("Serving modern prompt selection page (default)")
            prompt_files = list(PROMPTS_DIR.glob("*.md"))
            options = [file.name for file in prompt_files]
            return render_template("select_prompt_modern.html", options=options)

        @self.app.route("/classic")
        def select_prompt_classic_route():
            logging.info("Serving classic prompt selection page")
            prompt_files = list(PROMPTS_DIR.glob("*.md"))
            options = [file.name for file in prompt_files]
            return render_template("select_prompt.html", options=options)

        @self.app.route("/modern")
        def select_prompt_modern_route():
            logging.info("Serving modern prompt selection page (redirect)")
            prompt_files = list(PROMPTS_DIR.glob("*.md"))
            options = [file.name for file in prompt_files]
            return render_template("select_prompt_modern.html", options=options)

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

        @self.app.route("/api/tasks")
        def api_get_tasks():
            """Return the list of available tasks."""
            logging.info("Serving tasks list")
            try:
                prompt_files = list(PROMPTS_DIR.glob("*.md"))
                tasks = [file.name for file in prompt_files]
                return jsonify(tasks)
            except Exception as e:
                logging.error(f"Error loading tasks: {e}")
                return jsonify([]), 500

        @self.app.route("/vscode")
        def repo_browser():
            """Render a simple VS Code style viewer."""
            return render_template("vscode_view.html")

        @self.app.route("/api/file_tree")
        def api_file_tree():
            """Return a list of files under the current repository."""
            repo_dir = Path(get_constant("REPO_DIR"))
            files = [
                str(p.relative_to(repo_dir))
                for p in repo_dir.rglob("*")
                if p.is_file()
            ]
            return jsonify(files)

        @self.app.route("/api/file")
        def api_get_file():
            """Return the contents of a file within the repo."""
            rel_path = request.args.get("path", "")
            repo_dir = Path(get_constant("REPO_DIR"))
            safe_path = os.path.normpath(rel_path)
            file_path = repo_dir / safe_path
            try:
                file_path.resolve().relative_to(repo_dir.resolve())
            except ValueError:
                return jsonify({"error": "Invalid path"}), 400
            if not file_path.is_file():
                return jsonify({"error": "File not found"}), 404
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as exc:  # pragma: no cover - unlikely
                return jsonify({"error": str(exc)}), 500
            return jsonify({"content": content})

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

        @self.app.route("/browser")
        def file_browser_route():
            """Serve the VS Code-style file browser interface."""
            logging.info("Serving file browser interface")
            return render_template("file_browser.html")

        @self.app.route("/api/file-tree")
        def api_file_tree():
            """Return the file tree structure for the current REPO_DIR."""
            logging.info("Serving file tree")
            try:
                repo_dir = Path(get_constant("REPO_DIR"))
                if not repo_dir.exists():
                    return jsonify([])
                
                def build_tree(path):
                    items = []
                    try:
                        for item in sorted(path.iterdir()):
                            # Skip hidden files and directories
                            if item.name.startswith('.'):
                                continue
                            
                            if item.is_dir():
                                items.append({
                                    'name': item.name,
                                    'path': str(item),
                                    'type': 'directory',
                                    'children': build_tree(item)
                                })
                            else:
                                items.append({
                                    'name': item.name,
                                    'path': str(item),
                                    'type': 'file'
                                })
                    except PermissionError:
                        pass
                    return items
                
                tree = build_tree(repo_dir)
                return jsonify(tree)
            except Exception as e:
                logging.error(f"Error building file tree: {e}")
                return jsonify([])

        @self.app.route("/api/file-content")
        def api_file_content():
            """Return the content of a specific file."""
            file_path = request.args.get('path')
            if not file_path:
                return "File path is required", 400
            
            try:
                path = Path(file_path)
                # Security check - ensure the path is within REPO_DIR
                repo_dir = Path(get_constant("REPO_DIR"))
                if not str(path).startswith(str(repo_dir)):
                    return "Access denied", 403
                
                if not path.exists():
                    return "File not found", 404
                
                if not path.is_file():
                    return "Path is not a file", 400
                
                # Try to read as text, handle binary files gracefully
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                except UnicodeDecodeError:
                    # If it's a binary file, return a message instead
                    return "Binary file - cannot display content", 200
                
                return content, 200, {"Content-Type": "text/plain; charset=utf-8"}
                
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                return f"Error reading file: {str(e)}", 500

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

        @self.socketio.on("tool_response")
        def handle_tool_response(data):
            params = data.get("input", {})
            logging.info("Received tool response")
            self.tool_queue.put(params)
        logging.info("SocketIO events set up")

    def start_server(self, host="0.0.0.0", port=5002):
        logging.info(f"Starting server on {host}:{port}")
        self.socketio.run(self.app, host=host, port=port, use_reloader=False, allow_unsafe_werkzeug=True)

    def add_message(self, msg_type, content):
        logging.info(f"Adding message of type {msg_type}")
        log_message(msg_type, content)
        if msg_type == "user":
            self.user_messages.append(content)
            # Also emit to file browser
            self.socketio.emit("user_message", {"content": content})
        elif msg_type == "assistant":
            self.assistant_messages.append(content)
            # Also emit to file browser
            self.socketio.emit("assistant_message", {"content": content})
        elif msg_type == "tool":
            self.tool_results.append(content)
            # Parse tool result for file browser
            if isinstance(content, str):
                lines = content.split('\n')
                tool_name = "Unknown"
                if lines:
                    first_line = lines[0].strip()
                    if first_line.startswith('Tool:'):
                        tool_name = first_line.replace('Tool:', '').strip()
                self.socketio.emit("tool_result", {"tool_name": tool_name, "result": content})
                
                # Check if this tool might have created/modified files
                if any(keyword in content.lower() for keyword in ['created', 'wrote', 'generated', 'saved', 'modified', 'updated']):
                    # Emit file tree update after a short delay asynchronously
                    self.socketio.start_background_task(self._emit_file_tree_update)
                    
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

    async def confirm_tool_call(self, tool_name: str, args: dict, schema: dict) -> dict | None:
        """Send a tool prompt to the web UI and wait for edited parameters."""
        self.socketio.emit(
            "tool_prompt",
            {"tool": tool_name, "values": args, "schema": schema},
        )
        loop = asyncio.get_running_loop()
        params = await loop.run_in_executor(None, self.tool_queue.get)
        self.socketio.emit("tool_prompt_clear")
        return params

