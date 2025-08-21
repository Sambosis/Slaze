"""
Simplified Vercel serverless function without WebSocket dependencies.
This provides a basic HTTP API for the Slazy application.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set up environment
os.environ.setdefault('FLASK_ENV', 'production')

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../public/static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Enable CORS
CORS(app)

# Store agent tasks in memory (note: this won't persist between function invocations)
tasks = {}

@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Slazy API is running'})

@app.route('/api/start_task', methods=['POST'])
def start_task():
    """Start a new agent task."""
    try:
        data = request.get_json()
        task_description = data.get('task', '')
        
        if not task_description:
            return jsonify({'error': 'No task provided'}), 400
        
        # In a real implementation, you would queue this task
        # For now, we'll just return a task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Store task (note: won't persist in serverless)
        tasks[task_id] = {
            'id': task_id,
            'description': task_description,
            'status': 'queued',
            'result': None
        }
        
        return jsonify({
            'task_id': task_id,
            'message': 'Task queued successfully',
            'note': 'WebSocket functionality is not available on Vercel. Consider using polling or SSE.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/task_status/<task_id>')
def task_status(task_id):
    """Get the status of a task."""
    task = tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(task)

@app.route('/api/prompts')
def get_prompts():
    """Get available prompts."""
    try:
        from pathlib import Path
        prompts_dir = Path(__file__).parent.parent / 'prompts'
        
        if prompts_dir.exists():
            prompts = [f.stem for f in prompts_dir.glob('*.txt')]
            return jsonify({'prompts': prompts})
        else:
            return jsonify({'prompts': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config')
def get_config():
    """Get application configuration (non-sensitive)."""
    return jsonify({
        'api_configured': bool(os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY')),
        'deployment': 'vercel',
        'limitations': [
            'No WebSocket support',
            'Maximum 60-second execution time (Pro accounts)',
            'No persistent file storage',
            'Cold start delays possible'
        ]
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# Export the app for Vercel
app = app

# For local testing
if __name__ == '__main__':
    app.run(debug=True, port=5002)