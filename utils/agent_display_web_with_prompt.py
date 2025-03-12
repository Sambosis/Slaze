# utils/agent_display_web_with_prompt.py
import asyncio
from flask import render_template, request, redirect, url_for, send_from_directory
from dotenv import load_dotenv
load_dotenv()
from utils.agent_display_web import AgentDisplayWeb
from config import PROMPTS_DIR, LOGS_DIR, get_project_dir, get_docker_project_dir, set_project_dir
from pathlib import Path
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
                    filename = request.form.get('filename')
                    prompt_text = request.form.get('prompt_text')

                    if choice == 'new':
                        new_prompt_path = PROMPTS_DIR / f"{filename}.md"
                        print(f"Creating new prompt at {new_prompt_path}")
                        with open(new_prompt_path, 'w', encoding='utf-8') as f:
                            f.write(prompt_text)
                        task = prompt_text
                    else:
                        # Handle editing existing prompt
                        prompt_path = PROMPTS_DIR / choice
                        print(f"Using/updating prompt at {prompt_path}")
                        # If we have prompt text, it means the user edited the prompt
                        if prompt_text:
                            with open(prompt_path, 'w', encoding='utf-8') as f:
                                f.write(prompt_text)

                        # Read the prompt content (possibly updated)
                        with open(prompt_path, 'r', encoding='utf-8') as f:
                            task = f.read()
                        filename = prompt_path.stem

                    # Get assistant to analyze the task
                    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
                    client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=OPENROUTER_API_KEY,
                        )
                    model = "anthropic/claude-3.7-sonnet:beta"
                    prompt = f"""Do what would make sense as far as creating the final task definition.  If the user asks a simple non coding question you may simpley just repeat the task.  If it something unexpected then use your judgement to create a task definition that would work for the circumstance.  If it sounds like a programming project follow these instruction. First, restate the problem in more detail.  At this stage if there are any decisons that were left for the developer, you should make the choices needed and include them in your restated description of the promlem. 
                    After you have restated the expanded description. You will provide a file tree for the program.  In general, lean towards creating less files and folders while still keeping things organized and manageable. You will use absolute imports.  Do not create any non code files such as pyproject, gitignore, readme, etc.. You will need to attempt to list every file that will be needed and be thorough. This is all code files that will need to be created,  all asset files etc. For each file, in additon to the path and filename, you should give a brief statement about the purpose of the file and explicitly give the correct way to import the file using absolut imports. Do not actually create any of the code for the project. Just the expanded description and the file tree with the extra info included. The structure should focus on simplicity and reduced subdirectories and files.
                    Task:  {task}"""
                    messages = [{"role": "user", "content": prompt}] 
                    completion =  client.chat.completions.create(
                        model=model,
                        messages=messages)
                    task = completion.choices[0].message.content
                    from config import set_project_dir, set_constant
                    project_dir = set_project_dir(filename)
                    set_constant("PROJECT_DIR", str(project_dir))
                    docker_project_dir = get_docker_project_dir()
                    task += (
                        f"Start testing as soon as possible. DO NOT start making fixes or improvements until you have tested to see if it is working as is.  Your project directory is {docker_project_dir}. "
                        "You need to make sure that all files you create and work you do is done in that directory.\n"
                    )

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
                from flask import send_file, Response
                import tempfile
                import zipfile
                import os
                from pathlib import Path
                from config import PROJECT_DIR
                from utils.docker_service import DockerService
                from utils.file_logger import should_skip_for_zip
                
                if not os.path.exists(PROJECT_DIR):
                    return Response(f"Project directory not found: {PROJECT_DIR}", status=404)
                    
                project_path = Path(PROJECT_DIR)
                
                # Create a temporary file for the zip
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
                temp_file.close()
                
                # Create a zip file
                with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(project_path):
                        # Skip .venv directory and other unnecessary directories
                        dirs[:] = [d for d in dirs if not d.startswith('.venv') and d not in ['.git', '__pycache__', 'node_modules']]
                        
                        for file in files:
                            file_path = os.path.join(root, file)
                            
                            # Skip files in virtual environment bin directory
                            if '.venv' in root.split(os.sep) and file in ['python', 'python3', 'pip', 'pip3', 'activate']:
                                continue
                                
                            try:
                                # Get the relative path from project directory
                                arcname = os.path.relpath(file_path, project_path)
                                zipf.write(file_path, arcname)
                            except (FileNotFoundError, PermissionError) as e:
                                print(f"Error adding {file_path} to zip: {e}")
                
                # Send the zip file
                project_name = project_path.name
                return send_file(
                    temp_file.name,
                    mimetype='application/zip',
                    as_attachment=True,
                    download_name=f"{project_name}_project.zip"
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                return Response(f"Error creating zip file: {str(e)}", status=500)

def create_app(loop=None):
    """Create and configure the application with an event loop"""
    display = AgentDisplayWebWithPrompt()
    if loop:
        display.loop = loop
    return display.app
