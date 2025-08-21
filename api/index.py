import os
import sys
from pathlib import Path

# Add the parent directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, jsonify, request
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from agent import Agent
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole

# Create a simplified version of the web UI for Vercel
class VercelWebUI:
    def __init__(self):
        # More robust path for templates
        template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
        static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'public', 'static'))
        
        self.app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
        self.app.config["SECRET_KEY"] = "secret!"
        
        # Setup basic routes
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/select')
        def select_prompt():
            return render_template('select_prompt_modern.html')
        
        @self.app.route('/api/health')
        def health():
            return jsonify({"status": "ok", "message": "Slazy Agent API is running"})
        
        @self.app.route('/api/prompts')
        def get_prompts():
            # Return available prompts
            prompts_dir = os.path.join(os.path.dirname(__file__), '..', 'prompts')
            prompts = []
            if os.path.exists(prompts_dir):
                for file in os.listdir(prompts_dir):
                    if file.endswith('.md'):
                        prompts.append(file[:-3])  # Remove .md extension
            return jsonify({"prompts": prompts})
        
        @self.app.route('/api/start-task', methods=['POST'])
        def start_task():
            # For now, return a message that real-time execution needs WebSocket
            data = request.get_json()
            task = data.get('task', '')
            
            return jsonify({
                "message": "Task received. Note: Real-time execution requires WebSocket support which is limited on Vercel serverless. Consider using a different deployment platform for full functionality.",
                "task": task,
                "status": "received"
            })

# Create the web UI instance
web_ui = VercelWebUI()
app = web_ui.app

# This is the entry point for Vercel
# Export the Flask app for Vercel
application = app

# For local testing
if __name__ == '__main__':
    app.run(debug=True)