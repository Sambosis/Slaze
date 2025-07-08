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
    write_constants_to_file,
)

class AgentDisplayConsole:
    def __init__(self, interactive_tool_calls: bool = False):
        self.console = Console()
        self.interactive_tool_calls = interactive_tool_calls

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
        write_constants_to_file()

        return task

    async def prompt_for_tool_call_approval(self, tool_name: str, tool_args: dict, tool_id: str) -> dict:
        self.console.print(Panel(f"🧞 Assistant proposes to call tool: [bold cyan]{tool_name}[/bold cyan]", title="Tool Call Proposed", expand=False))

        current_args_json = json.dumps(tool_args, indent=2)
        self.console.print("Parameters:")
        self.console.print(Syntax(current_args_json, "json", theme="dracula", line_numbers=True))

        modified_args = tool_args
        action = await asyncio.to_thread(
            Prompt.ask,
            "Choose action",
            choices=["approve", "edit", "cancel"],
            default="approve",
        )

        if action == "edit":
            self.console.print("Enter new parameters as a JSON string. Press Ctrl+D or Ctrl+Z (Windows) then Enter to finish.")
            new_args_lines = []
            while True:
                try:
                    line = await self.wait_for_user_input("") # Assuming wait_for_user_input handles multiline
                    new_args_lines.append(line)
                except EOFError:
                    break
                except KeyboardInterrupt: # Allow cancelling edit
                    self.console.print("[yellow]Edit cancelled. Using original parameters.[/yellow]")
                    return {"name": tool_name, "args": tool_args, "approved": True} # Default to approve original if edit is cancelled mid-way

            new_args_str = "\n".join(new_args_lines)
            if not new_args_str.strip():
                self.console.print("[yellow]No input received. Using original parameters.[/yellow]")
            else:
                try:
                    modified_args = json.loads(new_args_str)
                    self.console.print("[green]Parameters updated.[/green]")
                    current_args_json = json.dumps(modified_args, indent=2)
                    self.console.print("New Parameters:")
                    self.console.print(Syntax(current_args_json, "json", theme="dracula", line_numbers=True))
                except json.JSONDecodeError as e:
                    self.console.print(f"[bold red]Invalid JSON. Error: {e}. Using original parameters.[/bold red]")
                    # Fallback to original args, but still need to confirm approval
                except Exception as e: # Catch any other unexpected errors during parsing
                    self.console.print(f"[bold red]Error processing input: {e}. Using original parameters.[/bold red]")


        if action == "cancel":
            self.console.print("[yellow]Tool call cancelled by user.[/yellow]")
            return {"name": tool_name, "args": modified_args, "approved": False}

        # If 'approve' or 'edit' (and edit was successful or fell-through to original)
        if await asyncio.to_thread(Confirm.ask, f"Execute tool '{tool_name}' with these parameters?", default=True):
            return {"name": tool_name, "args": modified_args, "approved": True}
        else:
            self.console.print("[yellow]Tool call rejected by user.[/yellow]")
            return {"name": tool_name, "args": modified_args, "approved": False}
