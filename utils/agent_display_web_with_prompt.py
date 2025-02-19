# agent_display_web_with_prompt.py (excerpt)

import asyncio
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket, WebSocketDisconnect
from pathlib import Path

from utils.agent_display_web import AgentDisplayWeb
from config import PROMPTS_DIR, LOGS_DIR

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
        @self.app.get('/select_prompt')
        async def select_prompt(request: Request):
            try:
                prompt_files = list(PROMPTS_DIR.glob("*.md"))
                options = [file.name for file in prompt_files]
                return self.templates.TemplateResponse('select_prompt.html', {'request': request, 'options': options})
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error rendering prompt selection: {e}")

        @self.app.post('/select_prompt')
        async def select_prompt_post(request: Request, choice: str = Form(...), filename: str = Form(None), prompt_text: str = Form(None)):
            try:
                if choice == 'new':
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
                self.loop.run_in_executor(None, start_sampling_loop, task, self)

                return RedirectResponse(url='/')
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error processing prompt selection: {e}")

        @self.app.get('/api/prompts/{filename}')
        async def get_prompt_content(filename: str):
            try:
                prompt_path = PROMPTS_DIR / filename
                if not prompt_path.exists():
                    raise HTTPException(status_code=404, detail="Prompt file not found")

                with open(prompt_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return JSONResponse(content=content)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading prompt: {e}")

        @self.app.get('/download/{filename}')
        async def download_file(filename: str):
            try:
                file_path = LOGS_DIR / filename
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail="File not found")
                return FileResponse(file_path, filename=filename)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error downloading file: {e}")

def create_app(loop=None):
    """Create and configure the application with an event loop"""
    display = AgentDisplayWebWithPrompt()
    if loop:
        display.loop = loop
    return display.app
