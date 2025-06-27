import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.syntax import Syntax
from pathlib import Path
from config import (
    PROMPTS_DIR,
    get_constant,
    set_constant,
    set_prompt_name,
)

class AgentDisplayConsole:
    def __init__(self):
        self.console = Console()

    def add_message(self, msg_type, content):
        if msg_type == "user":
            self.console.print(f"👤 User: {content}")
        elif msg_type == "assistant":
            self.console.print(f"🧞 Assistant: {content}")
        elif msg_type == "tool":
            self.console.print(f"🛠️ Tool: {content}")
        else:
            self.console.print(f"{msg_type}: {content}")

    async def wait_for_user_input(self, prompt_message="➡️ Your input: "):
        return await asyncio.to_thread(input, prompt_message)

    async def select_prompt_console(self):
        self.console.print("--- Select a Prompt ---")
        if not PROMPTS_DIR.exists():
            self.console.print(f"[bold yellow]Warning: Prompts directory '{PROMPTS_DIR}' not found. Creating it now.[/bold yellow]")
            try:
                PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
                self.console.print(f"[green]Successfully created prompts directory: {PROMPTS_DIR}[/green]")
            except Exception as e:
                self.console.print(f"[bold red]Error creating prompts directory {PROMPTS_DIR}: {e}[/bold red]")
                return "Default task due to directory creation error."

        options = {}
        prompt_files = sorted([f for f in PROMPTS_DIR.iterdir() if f.is_file() and f.suffix == '.md'])

        prompt_lines = []
        for i, prompt_file in enumerate(prompt_files):
            options[str(i + 1)] = prompt_file
            prompt_lines.append(f"{i + 1}. {prompt_file.name}")
        
        if prompt_lines:
            self.console.print("\n".join(prompt_lines))

        create_new_option_num = len(options) + 1
        self.console.print(f"{create_new_option_num}. Create a new prompt")

        choice = IntPrompt.ask("Enter your choice", choices=[str(i) for i in range(1, create_new_option_num + 1)])

        if choice != create_new_option_num:
            prompt_path = options[str(choice)]
            task = prompt_path.read_text(encoding="utf-8")
            prompt_name = prompt_path.stem
            self.console.print(
                Panel(
                    Syntax(task, "markdown", theme="dracula", line_numbers=True),
                    title="Current Prompt Content",
                )
            )
            if Confirm.ask(f"Do you want to edit '{prompt_path.name}'?", default=False):
                new_lines = []
                while True:
                    try:
                        line = await self.wait_for_user_input("")
                        new_lines.append(line)
                    except EOFError:
                        break
                if new_lines:
                    task = "\n".join(new_lines)
                    prompt_path.write_text(task, encoding="utf-8")
        else:
            new_filename_input = Prompt.ask("Enter a filename for the new prompt", default="custom_prompt")
            filename_stem = new_filename_input.strip().replace(" ", "_").replace(".md", "")
            prompt_name = filename_stem
            new_prompt_path = PROMPTS_DIR / f"{filename_stem}.md"
            new_prompt_lines = []
            while True:
                try:
                    line = await self.wait_for_user_input("")
                    new_prompt_lines.append(line)
                except EOFError:
                    break
            if new_prompt_lines:
                task = "\n".join(new_prompt_lines)
                new_prompt_path.write_text(task, encoding="utf-8")
            else:
                task = ""

        # Configure repository directory for this prompt
        base_repo_dir = Path(get_constant("TOP_LEVEL_DIR")) / "repo"
        repo_dir = base_repo_dir / prompt_name
        repo_dir.mkdir(parents=True, exist_ok=True)
        set_prompt_name(prompt_name)
        set_constant("PROJECT_DIR", repo_dir)
        set_constant("REPO_DIR", repo_dir)

        return task
