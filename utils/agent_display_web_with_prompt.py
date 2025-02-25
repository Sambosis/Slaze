# agent_display_web_with_prompt.py (excerpt)

import asyncio
from flask import render_template, request, redirect, url_for, send_from_directory
from dotenv import load_dotenv
load_dotenv()
from utils.agent_display_web import AgentDisplayWeb
from config import PROMPTS_DIR, LOGS_DIR
from openai import OpenAI, AsyncOpenAI
import os
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
        self.setup_prompt_routes()

    def setup_prompt_routes(self):
        @self.app.route('/select_prompt', methods=['GET', 'POST'])
        def select_prompt():
            if request.method == 'POST':
                try:
                    choice = request.form.get('choice')
                    if choice == 'new':
                        filename = request.form.get('filename')
                        prompt_text = request.form.get('prompt_text')
                        new_prompt_path = PROMPTS_DIR / f"{filename}.md"
                        with open(new_prompt_path, 'w', encoding='utf-8') as f:
                            f.write(prompt_text)
                        task = prompt_text
                    else:
                        prompt_path = PROMPTS_DIR / choice
                        filename = prompt_path.stem
                        with open(prompt_path, 'r', encoding='utf-8') as f:
                            task = f.read()
                    # client = OpenAI()
                    # model = "o3-mini"

                                
                    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=OPENROUTER_API_KEY,
                        )
                    model = "google/gemini-flash-1.5:nitro"
                    messages = [{"role": "user", "content": f"Please create a simple step by step plan to accomplish the following task. It should be a very high level plan without too many steps.  Each step should be no more than 2 sentences long.  After your plan, provide a directory structure that should be used a list of every file that will need to be created to complete the project. Task:  {task}"}] 
                    completion =  client.chat.completions.create(
                        model=model,
                        messages=messages)
                    task = completion.choices[0].message.content
                    from config import set_project_dir, set_constant
                    # Remove .md extension if present when setting project dir
                    project_name = filename.replace('.md', '')
                    project_dir = set_project_dir(project_name)
                    set_constant("PROJECT_DIR", str(project_dir))
                    task += (
                        f"Do not use Flask for this project. Start testing as soon as possible. DO NOT start making fixes or improvements until you have tested to see if it is working as is.  Your project directory is {project_dir}. "
                        "You need to make sure that all files you create and work you do is done in that directory.\n"
                    )

                    # Schedule your async function in a background thread,
                    # and let that thread call `asyncio.run(...)`.
                    self.socketio.start_background_task(start_sampling_loop, task, self)

                    return redirect(url_for('index'))

                except Exception as e:
                    return f"Error processing prompt selection: {e}", 500
            else:
                try:
                    prompt_files = list(PROMPTS_DIR.glob("*.md"))
                    options = [file.name for file in prompt_files]
                    return render_template('select_prompt.html', options=options)
                except Exception as e:
                    return f"Error rendering prompt selection: {e}", 500

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
