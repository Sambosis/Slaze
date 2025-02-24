# agent_display_web_with_prompt.py (excerpt)

import asyncio
from flask import render_template, request, redirect, url_for, send_from_directory
from dotenv import load_dotenv
load_dotenv()
from utils.agent_display_web import AgentDisplayWeb
from config import PROMPTS_DIR, LOGS_DIR
from openai import OpenAI, AsyncOpenAI
import os
import json

def start_sampling_loop(task, display):
    """
    Simple wrapper function that spins up a fresh event loop
    to run the async `run_sampling_loop`.
    """
    from main import run_sampling_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    display.loop = loop  # Set the loop for the display
    loop.run_until_complete(run_sampling_loop(task, display))

class AgentDisplayWebWithPrompt(AgentDisplayWeb):
    def __init__(self):
        super().__init__()
        # Ensure prompts directory exists
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        self.setup_prompt_routes()

    def load_prompt_selections(self):
        # Scan the prompts directory for .md files and return their names in a list
        md_files = sorted(PROMPTS_DIR.glob("*.md"))
        # Optionally, print to console for debugging
        print(f"[DEBUG] Found prompt files: {[file.name for file in md_files]}", flush=True)
        return [file.name for file in md_files]

    def setup_prompt_routes(self):
        @self.app.route('/select_prompt', methods=['GET', 'POST'])
        def select_prompt():
            if request.method == 'POST':
                try:
                    print("[DEBUG] Received POST in select_prompt", flush=True)
                    choice = request.form.get('choice')
                    prompt_text = ""
                    if choice == 'new':
                        # Handle new prompt creation
                        filename = request.form.get('filename')
                        prompt_text = request.form.get('prompt_text')
                        if not filename:
                            print("[ERROR] No filename provided", flush=True)
                            return "Error: No filename provided", 400
                        if not prompt_text:
                            print("[ERROR] No prompt text provided", flush=True)
                            return "Error: No prompt text provided", 400
                        if not filename.endswith('.md'):
                            filename += '.md'
                        new_prompt_path = PROMPTS_DIR / filename
                        with open(new_prompt_path, 'w', encoding='utf-8') as f:
                            f.write(prompt_text)
                        task = prompt_text
                        print(f"[DEBUG] Created new prompt file: {new_prompt_path}", flush=True)
                    else:
                        # Handle existing prompt selection
                        prompt_path = PROMPTS_DIR / choice
                        if not prompt_path.exists():
                            print(f"[ERROR] Prompt file {choice} not found", flush=True)
                            return f"Error: Prompt file {choice} not found", 404
                        filename = prompt_path.stem
                        with open(prompt_path, 'r', encoding='utf-8') as f:
                            task = f.read()
                        print(f"[DEBUG] Loaded prompt from file: {prompt_path}", flush=True)
                                
                    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=OPENROUTER_API_KEY,
                    )
                    model = "google/gemini-flash-1.5:nitro"
                    messages = [{"role": "user", "content": f"Please create a simple step by step plan to accomplish the following task. It should be a very high level plan without too many steps.  Each step should be no more than 2 sentences long.  After your plan, provide a directory structure that should be used a list of every file that will need to be created to complete the project. Task:  {task}"}] 
                    print(f"[DEBUG] Sending LLM request with task: {task[:100]}...", flush=True)
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages)
                    task = completion.choices[0].message.content
                    print(f"[DEBUG] Received LLM response. Task updated.", flush=True)
                    from config import set_project_dir, set_constant
                    project_dir = set_project_dir(filename)
                    set_constant("PROJECT_DIR", str(project_dir))
                    task += (
                        f"Use Pydantic. Start testing as soon as possible. DO NOT start making fixes or improvements until you have tested to see if it is working as is.  Your project directory is {project_dir}. "
                        "You need to make sure that all files you create and work you do is done in that directory.\n"
                    )
                    print(f"[DEBUG] Final task string (first 150 chars): {task[:150]}", flush=True)
                    try:
                        print(f"[DEBUG] socketio: {self.socketio}", flush=True)
                        if not hasattr(self.socketio, 'start_background_task'):
                            print("[ERROR] start_background_task is missing", flush=True)
                        else:
                            print("[DEBUG] Starting background task...", flush=True)
                        self.socketio.start_background_task(start_sampling_loop, task, self)
                        print("[DEBUG] Background task started", flush=True)
                    except Exception as e:
                        import traceback
                        print(f"[ERROR] Failed to start background task: {str(e)}", flush=True)
                        traceback.print_exc()
                        return f"Error starting background task: {str(e)}", 500
                    return redirect(url_for('index'))
                except Exception as e:
                    import traceback
                    print(f"[ERROR] Exception in select_prompt POST: {str(e)}", flush=True)
                    traceback.print_exc()
                    return f"Error processing prompt selection: {e}", 500
            else:
                try:
                    # Just list .md files from the prompts directory
                    prompt_files = sorted(PROMPTS_DIR.glob("*.md"))
                    if not prompt_files:
                        return render_template('select_prompt.html', options=[], error="No prompt files found")
                    options = [file.name for file in prompt_files]
                    return render_template('select_prompt.html', options=options)
                except Exception as e:
                    return f"Error loading prompts: {str(e)}", 500

        @self.app.route('/api/prompts/<filename>')
        def get_prompt_content(filename):
            try:
                prompt_path = PROMPTS_DIR / filename
                if not prompt_path.exists():
                    return "Prompt file not found", 404

                with open(prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except Exception as e:
                return f"Error reading prompt: {e}", 500

        @self.app.route('/download/<filename>')
        def download_file(filename):
            try:
                return send_from_directory(LOGS_DIR, filename, as_attachment=True)
            except Exception as e:
                return f"Error downloading file: {e}", 500

        @self.app.route('/download_project_zip')
        def download_project_zip():
            try:
                from flask import send_file
                import tempfile
                import zipfile
                import os
                from config import PROJECT_DIR  # Ensure PROJECT_DIR is defined in config.py
                if not os.path.exists(PROJECT_DIR):
                    return "Project directory not found", 404

                # Create a temporary zip file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                    zip_path = tmp.name

                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(PROJECT_DIR):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, PROJECT_DIR)
                            zipf.write(file_path, arcname)

                return send_file(zip_path, as_attachment=True, download_name='project_documents.zip')
            except Exception as e:
                return f"Error creating zip file: {e}", 500

def create_app(loop=None):
    """Create and configure the application with an event loop"""
    display = AgentDisplayWebWithPrompt()
    if loop:
        display.loop = loop
    return display.app
