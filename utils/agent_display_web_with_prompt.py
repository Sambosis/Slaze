# agent_display_web_with_prompt.py
import os
import asyncio
from flask import render_template, request, redirect, url_for
from utils.agent_display_web import AgentDisplayWeb
from config import PROMPTS_DIR, MAIN_MODEL, MAX_SUMMARY_TOKENS
from anthropic import Anthropic

display = AgentDisplayWebWithPrompt() # Create instance of AgentDisplayWebWithPrompt
app = display.app # Assign the Flask app instance to the module-level 'app' variable

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
                    # Set up project directory information.
                    # Replace 'your_project_module' with the actual module name.
                    from main import run_sampling_loop
                    from config import set_project_dir,    set_constant
                    project_dir = set_project_dir(filename)
                    set_constant("PROJECT_DIR", str(project_dir))
                    task += (
                        f"Your project directory is {project_dir}. "
                        "You need to make sure that all files you create and work you do is done in that directory.\n"
                    )

                    if self.loop is None:
                        # Instead of attempting to get a running loop here (which fails in this thread),
                        # return an error.
                        return "Error: Event loop not set", 500
                    # anthropic_client = Anthropic()
                    # # Set up the agent.
                    # messages = [
                    #     {"role": "user", "content": f"Baased on this description, please provide or refine proper software specification, often called a Software Requirements Specification (SRS). Do not give timelines or unit testing. keep it relativly concise and leave room for implementation details to be developed by the programmers. There is no need to implement security features at this point. The goal is just to create a basic working prototype.\nDescription: {task}"},
                    #     ]
                    # response  = anthropic_client.messages.create(
                    # max_tokens=MAX_SUMMARY_TOKENS,
                    # messages=messages,
                    # model=MAIN_MODEL,
                    # )
                    # task = response.content[0].text
                    print(task)
                    # Schedule the sampling loop on the pre-set event loop.
                    asyncio.run_coroutine_threadsafe(run_sampling_loop(task, self), self.loop)
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

    def create_app():
        display = AgentDisplayWebWithPrompt()
        return display.app