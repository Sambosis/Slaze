# agent_display_web_with_prompt.py (excerpt)

import asyncio
from flask import render_template, request, redirect, url_for, send_from_directory

from utils.agent_display_web import AgentDisplayWeb
from config import PROMPTS_DIR, LOGS_DIR

def start_sampling_loop(task, display):
    """
    Simple wrapper function that spins up a fresh event loop
    to run the async `run_sampling_loop`.
    """
    from main import run_sampling_loop
    asyncio.run(run_sampling_loop(task, display))

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

                    from config import set_project_dir, set_constant
                    project_dir = set_project_dir(filename)
                    set_constant("PROJECT_DIR", str(project_dir))
                    task += (
                        f"Your project directory is {project_dir}. "
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

def create_app():
    display = AgentDisplayWebWithPrompt()
    return display.app
