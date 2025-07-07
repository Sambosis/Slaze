import asyncio
import os
import threading
import logging
import json
import uuid
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
from .web_ui import WebUI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebUIInteractive(WebUI):
    """
    Interactive WebUI that shows tool calls to users for review and editing
    before execution.
    """
    
    def __init__(self, agent_runner):
        super().__init__(agent_runner)
        self.tool_review_mode = True
        self.pending_tool_calls = {}  # Store pending tool calls by ID
        self.tool_call_responses = {}  # Store user responses by ID
        self.setup_interactive_socketio_events()
        logging.info("WebUIInteractive initialized")
    
    def setup_interactive_routes(self):
        """Set up additional routes for interactive mode."""
        
        @self.app.route("/interactive")
        def interactive_mode_route():
            """Display the interactive mode interface."""
            logging.info("Serving interactive mode page")
            return render_template("interactive.html")
        
        @self.app.route("/api/tool_call/<tool_call_id>", methods=["GET", "POST"])
        def handle_tool_call_api(tool_call_id):
            """Handle tool call review and editing."""
            if tool_call_id not in self.pending_tool_calls:
                return jsonify({"error": "Tool call not found"}), 404
            
            tool_call = self.pending_tool_calls[tool_call_id]
            
            if request.method == "GET":
                # Return tool call details for review
                return jsonify({
                    "id": tool_call_id,
                    "tool_name": tool_call["tool_name"],
                    "parameters": tool_call["parameters"],
                    "formatted_parameters": self.format_parameters_for_display(tool_call["parameters"])
                })
            
            elif request.method == "POST":
                # Handle user decision
                data = request.get_json()
                action = data.get("action")
                modified_parameters = data.get("parameters", tool_call["parameters"])
                
                # Store the response
                self.tool_call_responses[tool_call_id] = {
                    "action": action,
                    "parameters": modified_parameters
                }
                
                # Remove from pending
                del self.pending_tool_calls[tool_call_id]
                
                return jsonify({"status": "success"})
    
    def setup_interactive_socketio_events(self):
        """Set up SocketIO events for interactive mode."""
        
        @self.socketio.on("tool_call_decision")
        def handle_tool_call_decision(data):
            """Handle user decision on tool call."""
            tool_call_id = data.get("id")
            action = data.get("action")
            parameters = data.get("parameters", {})
            
            if tool_call_id in self.pending_tool_calls:
                # Store the response
                self.tool_call_responses[tool_call_id] = {
                    "action": action,
                    "parameters": parameters
                }
                
                # Remove from pending
                del self.pending_tool_calls[tool_call_id]
                
                logging.info(f"Tool call decision received: {action} for {tool_call_id}")
    
    def format_parameters_for_display(self, parameters):
        """Format parameters for better display in the web UI."""
        formatted = []
        for key, value in parameters.items():
            param_info = {
                "name": key,
                "value": value,
                "type": type(value).__name__,
                "display_value": self.format_value_for_display(value)
            }
            formatted.append(param_info)
        return formatted
    
    def format_value_for_display(self, value):
        """Format a value for display in the web UI."""
        if isinstance(value, (dict, list)):
            return json.dumps(value, indent=2)
        elif isinstance(value, str) and len(value) > 100:
            return value[:100] + "... (truncated)"
        else:
            return str(value)
    
    def get_user_tool_decision(self, tool_name, tool_input):
        """
        Show the tool call to the user via web interface and get their decision.
        
        Returns:
            Dict with keys:
            - 'action': 'execute', 'edit', 'skip', or 'exit'
            - 'modified_input': The potentially modified tool input
        """
        tool_call_id = str(uuid.uuid4())
        
        # Store the tool call
        self.pending_tool_calls[tool_call_id] = {
            "tool_name": tool_name,
            "parameters": tool_input
        }
        
        # Emit the tool call to the web interface
        self.socketio.emit("tool_call_review", {
            "id": tool_call_id,
            "tool_name": tool_name,
            "parameters": tool_input,
            "formatted_parameters": self.format_parameters_for_display(tool_input)
        })
        
        # Wait for user response
        return self.wait_for_tool_call_response(tool_call_id)
    
    def wait_for_tool_call_response(self, tool_call_id):
        """Wait for user response to tool call review."""
        import time
        
        # Poll for response (blocking call for synchronous execution)
        while tool_call_id not in self.tool_call_responses:
            time.sleep(0.1)
        
        response = self.tool_call_responses[tool_call_id]
        del self.tool_call_responses[tool_call_id]
        
        return {
            "action": response["action"],
            "modified_input": response["parameters"]
        }
    
    async def wait_for_user_input(self, prompt_message: str = None) -> str:
        """Override to handle interactive mode prompts."""
        if prompt_message:
            logging.info(f"Emitting agent_prompt: {prompt_message}")
            self.socketio.emit("agent_prompt", {"message": prompt_message})

        loop = asyncio.get_running_loop()
        user_response = await loop.run_in_executor(None, self.input_queue.get)

        # Clear the prompt after input is received
        logging.info("Emitting agent_prompt_clear")
        self.socketio.emit("agent_prompt_clear")

        return user_response
    
    def add_message(self, msg_type, content):
        """Override to add interactive mode message handling."""
        super().add_message(msg_type, content)
        
        # If this is a tool call message, emit it for interactive review
        if msg_type == "tool" and self.tool_review_mode:
            self.socketio.emit("tool_message", {
                "content": content,
                "type": "tool_result"
            })