import asyncio
import os
from pathlib import Path
from openai import OpenAI # type: ignore
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.syntax import Syntax
# Assuming config.py is at the root and contains PROMPTS_DIR, LOGS_DIR, etc.
from config import PROMPTS_DIR, LOGS_DIR, MAIN_MODEL, set_project_dir, set_constant, get_constant # get_docker_project_dir removed
# Assuming agent_display_web.py is in utils and contains AgentDisplayWeb, log_message
from utils.agent_display_web import AgentDisplayWeb, log_message
from tenacity import retry, stop_after_attempt, wait_exponential # type: ignore
class AgentDisplayConsole(AgentDisplayWeb):
    def __init__(self):
        # Initialize attributes that would have been set by AgentDisplayWeb's __init__,
        # but without starting Flask server or SocketIO.
        self.user_messages = []
        self.assistant_messages = []
        self.tool_results = []

        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.input_queue = asyncio.Queue() # For compatibility if any inherited method uses it
        self.user_interupt = False # For compatibility

        # Explicitly do NOT call super().__init__() if it starts web services.
        # Also, do not call methods that set up web routes or socket events.
        self.console = Console() # Instantiate Console here
        self.console.print("AgentDisplayConsole initialized.") # Added for verification

    def add_message(self, msg_type, content):
        log_message(msg_type, content) # From utils.agent_display_web
        if msg_type == "user":
            self.console.print(f"👤 User: {content}")
        elif msg_type == "assistant":
            self.console.print(f"🧞 Assistant: {content}")
        elif msg_type == "tool":
            self.console.print(f"🛠️ Tool: {content}")
        else:
            self.console.print(f"{msg_type}: {content}")

    async def wait_for_user_input(self, prompt_message="➡️ Your input: "):
        if not self.loop: # Should be set in __init__
            try:
                self.loop = asyncio.get_event_loop()
            except RuntimeError: # pragma: no cover
                # This case should ideally not be hit if __init__ sets the loop.
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)

        user_input = await asyncio.to_thread(input, prompt_message)
        return user_input

    def broadcast_update(self):
        pass  # Console updates are real-time via print

    def clear_messages(self, panel):
        # This is a simple interpretation for console.
        # A more complex version could use os.system('cls') or 'clear'.
        self.console.print(f"--- Messages for {panel} would be cleared ---")

    def start_server(self, host="0.0.0.0", port=None):
        pass  # No server to start in console version

    # Override web-specific setup methods from AgentDisplayWeb to be no-ops
    def setup_routes(self):
        pass

    def setup_socketio_events(self):
        pass
    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=4, max=10))
    async def call_llm(self, client, model, messages):
        completion = await asyncio.to_thread(
            client.chat.completions.create, model=model, messages=messages
        )   
        return completion

    async def select_prompt_console(self):
        console = self.console # Use the instance console
        console.print("--- Select a Prompt ---")
        if not PROMPTS_DIR.exists():
            console.print(f"[bold yellow]Warning: Prompts directory '{PROMPTS_DIR}' not found. Creating it now.[/bold yellow]")
            try:
                PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
                console.print(f"[green]Successfully created prompts directory: {PROMPTS_DIR}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error creating prompts directory {PROMPTS_DIR}: {e}[/bold red]")
                return "Default task due to directory creation error."

        options = {}
        prompt_files = []
        try:
            prompt_files = sorted([f for f in PROMPTS_DIR.iterdir() if f.is_file() and f.suffix == '.md'])
        except Exception as e:
            console.print(f"[bold red]Error listing prompts in {PROMPTS_DIR}: {e}[/bold red]")
            # Handle error, maybe allow creation only or return default task.

        table = Table(title="Available Prompts")
        table.add_column("Index", style="dim", width=6)
        table.add_column("Prompt File", style="cyan")

        for i, prompt_file in enumerate(prompt_files):
            options[str(i + 1)] = prompt_file
            table.add_row(str(i + 1), prompt_file.name)

        console.print(table)

        create_new_option_num = len(options) + 1
        console.print(f"{create_new_option_num}. Create a new prompt")

        choice_prompt = f"Enter your choice (1-{create_new_option_num})"

        while True:
            # Use IntPrompt for choice
            raw_choice_int = IntPrompt.ask(choice_prompt, console=console, choices=[str(i) for i in range(1, create_new_option_num + 1)])
            raw_choice = str(raw_choice_int) # Convert to string to match existing logic for options dict keys

            if raw_choice in options or raw_choice == str(create_new_option_num):
                choice = raw_choice
                break
            # IntPrompt already validates if the input is an int and within choices if provided,
            # but an additional check for safety or more complex validation could go here.
            # console.print("[bold red]Invalid choice. Please try again.[/bold red]") # Should not be reached if choices are used in IntPrompt

        filename_stem = "custom_prompt"
        task = ""

        if choice != str(create_new_option_num): # Existing prompt chosen
            prompt_path = options[choice]
            filename_stem = prompt_path.stem
            console.print(f"[bold green]Using prompt: {prompt_path.name}[/bold green]")
            try:
                task = prompt_path.read_text(encoding='utf-8')
                # task = ftfy.fix_text(task) # ftfy removed

                console.print(Panel(Syntax(task, "markdown", theme="dracula", line_numbers=True), title="Current Prompt Content"))
                if Confirm.ask(f"Do you want to edit '{prompt_path.name}'?", default=False, console=console):
                    console.print("[cyan]Enter new prompt content. Press Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) to save.[/cyan]")

                    new_lines = []
                    # Rich's Prompt.ask doesn't handle multi-line input in the way needed here (EOF signal).
                    # So, falling back to the existing line-by-line input mechanism.
                    # This part remains similar, but ensure wait_for_user_input is compatible or replaced if needed.
                    while True:
                        try:
                            # Assuming wait_for_user_input prints its own prompt or none for multi-line.
                            # If wait_for_user_input uses Python's input(), it should work fine.
                            line = await self.wait_for_user_input("")
                            new_lines.append(line)
                        except EOFError: # User signaled end of input
                            break

                    if new_lines:
                        task = "\n".join(new_lines)
                        # task = ftfy.fix_text(task) # ftfy removed
                        console.print(Panel(Syntax(task, "markdown", theme="dracula", line_numbers=True), title="New Prompt Content"))
                        try:
                            prompt_path.write_text(task, encoding='utf-8')
                            console.print(f"[green]Prompt '{prompt_path.name}' updated.[/green]")
                        except Exception as e:
                            console.print(f"[bold red]Error writing updated prompt to {prompt_path}: {e}[/bold red]")
            except Exception as e:
                console.print(f"[bold red]Error reading or processing prompt {prompt_path}: {e}[/bold red]")
                task = "" # Reset task if error

        else: # Create a new prompt
            new_filename_input = Prompt.ask("Enter a filename for the new prompt (e.g., my_new_task, press Enter for default 'custom_prompt')", default="custom_prompt", console=console)
            if new_filename_input.strip():
                filename_stem = new_filename_input.strip().replace(" ", "_").replace(".md", "")

            new_prompt_path = PROMPTS_DIR / f"{filename_stem}.md"
            console.print(Panel(f"[cyan]Enter the prompt text for '{new_prompt_path.name}'.\nPress Ctrl+D (Unix) or Ctrl+Z then Enter (Windows) to save.[/cyan]", title="Create New Prompt"))

            new_prompt_lines = []
            # Similar to editing, using existing mechanism for multi-line input.
            while True:
                try:
                    line = await self.wait_for_user_input("")
                    new_prompt_lines.append(line)
                except EOFError:
                    break

            if new_prompt_lines:
                task = "\n".join(new_prompt_lines)
                # task = ftfy.fix_text(task) # ftfy removed
                console.print(Panel(Syntax(task, "markdown", theme="dracula", line_numbers=True), title="New Prompt Content"))
                try:
                    new_prompt_path.write_text(task, encoding='utf-8')
                    console.print(f"[green]New prompt '{new_prompt_path.name}' saved.[/green]")
                except Exception as e:
                    console.print(f"[bold red]Error saving new prompt to {new_prompt_path}: {e}[/bold red]")
            else:
                console.print("[yellow]No input received for the new prompt. Task is empty.[/yellow]")
                task = ""

        # OpenAI Task Processing
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if OPENROUTER_API_KEY and task:
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
                model = MAIN_MODEL

                ai_prompt_template = f"""Do what would make sense as far as creating the final task definition. If the user asks a simple non coding question you may simpley just repeat the task. If it something unexpected then use your judgement to create a task definition that would work for the circumstance. If it sounds like a programming project follow these instruction. First, restate the problem in more detail. At this stage if there are any decisons that were left for the developer, you should make the choices needed and include them in your restated description of the promlem.
After you have restated the expanded description. You will provide a file tree for the program. In general, lean towards creating less files and folders while still keeping things organized and manageable. You will use absolute imports. Do not create any non code files such as pyproject, gitignore, readme, etc.. You will need to attempt to list every file that will be needed and be thorough. This is all code files that will need to be created, all asset files etc. For each file, in additon to the path and filename, you should give a brief statement about the purpose of the file and explicitly give the correct way to import the file using absolut imports. Do not actually create any of the code for the project. Just the expanded description and the file tree with the extra info included. The structure should focus on simplicity and reduced subdirectories and files.
Task: {task}"""

                messages = [{"role": "user", "content": ai_prompt_template}]
                console.print("[yellow]Sending task to AI for analysis (this may take a moment)...[/yellow]")
                completion = await self.call_llm(client, model, messages)
                processed_task = completion.choices[0].message.content
                if processed_task:
                    task = processed_task # ftfy removed
                    console.print("[green]--- Task analyzed and updated by AI ---[/green]")
                else:
                    console.print("[yellow]AI returned empty content. Using the task as is.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error during OpenAI task processing: {e}. Using the task as is.[/bold red]")
        elif not task:
            console.print("[yellow]Task is empty. Skipping OpenAI processing.[/yellow]")
        else: # No API key, but task exists
            console.print("[yellow]Warning: OPENROUTER_API_KEY not set. Skipping task analysis with AI.[/yellow]")

        # Set up project and task constants
        project_dir = set_project_dir(filename_stem)
        set_constant("PROJECT_DIR", str(project_dir))
        set_constant("TASK", task) # Set task before potentially appending suffix
        project_dir_for_task = get_constant("PROJECT_DIR")

        # Finalize task string
        task_final_suffix = f"""
        Start testing as soon as possible. DO NOT start making fixes or improvements until you have tested to see if it is working as is. Your project directory is {project_dir_for_task}. You need to make sure that all files you create and work you do is done in that directory.
        """
        if task:
            task += "\n" + task_final_suffix.strip()
        else:
            console.print("[bold yellow]Warning: Task is empty. The final task will only contain the standard suffix.[/bold yellow]")
            task = task_final_suffix.strip()

        # Save task
        if not LOGS_DIR.exists():
            try:
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                console.print(f"[bold red]Error creating logs directory {LOGS_DIR}: {e}[/bold red]")

        task_file_path = LOGS_DIR / "task.txt"
        try:
            task_file_path.write_text(task, encoding='utf-8')
            console.print(f"Task definition saved to '{task_file_path}'")
        except Exception as e:
            console.print(f"[bold red]Error saving task to {task_file_path}: {e}[/bold red]")

        console.print(f"[bold blue]Project directory set to: '{project_dir}'[/bold blue]")

        return task

async def main_test():
    # This function uses self.console, so it needs access to an instance of AgentDisplayConsole
    display = AgentDisplayConsole() # Create an instance

    # Ensure PROMPTS_DIR and LOGS_DIR exist for testing
    if not PROMPTS_DIR.exists(): 
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        display.console.print(f"Test Helper: Created PROMPTS_DIR at {PROMPTS_DIR}")
    if not LOGS_DIR.exists(): 
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        display.console.print(f"Test Helper: Created LOGS_DIR at {LOGS_DIR}")

    prompt_files_exist = any(PROMPTS_DIR.glob("*.md"))
    if not prompt_files_exist:
        dummy_prompt_path = PROMPTS_DIR / "example_prompt.md"
        with open(dummy_prompt_path, "w", encoding="utf-8") as f:
            f.write("This is an example prompt for testing purposes.")
        display.console.print(f"Test Helper: Created dummy prompt at {dummy_prompt_path}")

    try:
        display.console.print("Attempting to run display.select_prompt_console()...")
        task = await display.select_prompt_console() 

        display.console.print("\n--- Selected Task (from main_test) ---")
        display.console.print(task if task else "No task was selected or created.")
        
        display.add_message("assistant", "Console agent ready for further interaction.")
        # Use display.console for prompts in test, or ensure wait_for_user_input is styled if it uses raw input()
        user_response = await display.wait_for_user_input("Test input (type something and press Enter): ")
        display.add_message("user", user_response)
        
        display.add_message("tool", "Example tool output content.")
        display.clear_messages("all")

    except Exception as e:
        display.console.print(f"[bold red]Error during main_test: {e}[/bold red]")
        import traceback
        # Use console.print_exception() for rich traceback
        display.console.print_exception(show_locals=True)


if __name__ == '__main__':
    asyncio.run(main_test())
