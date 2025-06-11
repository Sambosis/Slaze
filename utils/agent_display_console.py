import asyncio
import os
from pathlib import Path
import ftfy # type: ignore
from openai import OpenAI # type: ignore
# Assuming config.py is at the root and contains PROMPTS_DIR, LOGS_DIR, etc.
from config import PROMPTS_DIR, LOGS_DIR, set_project_dir, set_constant, get_constant # get_docker_project_dir removed
# Assuming agent_display_web.py is in utils and contains AgentDisplayWeb, log_message
from utils.agent_display_web import AgentDisplayWeb, log_message
from utils.logger import logger

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
        logger.info("AgentDisplayConsole initialized.")

    def add_message(self, msg_type, content):
        log_message(msg_type, content) # From utils.agent_display_web
        if msg_type == "user":
            print(f"👤 User: {content}")
        elif msg_type == "assistant":
            print(f"🧞 Assistant: {content}")
        elif msg_type == "tool":
            print(f"🛠️ Tool: {content}")
        else:
            print(f"{msg_type}: {content}")

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
        print(f"--- Messages for {panel} would be cleared ---")

    def start_server(self, host="0.0.0.0", port=None):
        pass  # No server to start in console version

    # Override web-specific setup methods from AgentDisplayWeb to be no-ops
    def setup_routes(self):
        pass

    def setup_socketio_events(self):
        pass

    async def select_prompt_console(self):
        print("--- Select a Prompt ---")
        if not PROMPTS_DIR.exists():
            print(f"Warning: Prompts directory '{PROMPTS_DIR}' not found. Creating it now.")
            try:
                PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
                print(f"Successfully created prompts directory: {PROMPTS_DIR}")
            except Exception as e:
                print(f"Error creating prompts directory {PROMPTS_DIR}: {e}")
                # Fallback or decide how to handle this error, perhaps exit or use a default task.
                return "Default task due to directory creation error."

        options = {}
        try:
            prompt_files = sorted([f for f in PROMPTS_DIR.iterdir() if f.is_file() and f.suffix == '.md'])
            for i, prompt_file in enumerate(prompt_files):
                options[str(i + 1)] = prompt_file
                print(f"{i + 1}. {prompt_file.name}")
        except Exception as e:
            print(f"Error listing prompts in {PROMPTS_DIR}: {e}")
            # Handle error, maybe allow creation only or return default task.

        create_new_option_num = len(options) + 1
        print(f"{create_new_option_num}. Create a new prompt")

        choice = None
        while True:
            raw_choice = await self.wait_for_user_input(f"Enter your choice (1-{create_new_option_num}): ")
            if raw_choice in options or raw_choice == str(create_new_option_num):
                choice = raw_choice
                break
            print("Invalid choice. Please try again.")

        filename_stem = "custom_prompt"
        task = ""

        if choice != str(create_new_option_num): # Existing prompt chosen
            prompt_path = options[choice]
            filename_stem = prompt_path.stem
            print(f"Using prompt: {prompt_path.name}")
            try:
                task = prompt_path.read_text(encoding='utf-8')
                task = ftfy.fix_text(task)
                
                edit_choice = await self.wait_for_user_input(f"Do you want to edit '{prompt_path.name}'? (yes/no, default no): ")
                if edit_choice.lower() == 'yes':
                    print("--- Current prompt content ---")
                    print(task)
                    print("--- End of current content (Ctrl+D or Ctrl+Z then Enter to finish editing) ---")
                    
                    new_lines = []
                    while True:
                        try:
                            line = await self.wait_for_user_input("") # Read line by line
                            new_lines.append(line)
                        except EOFError: # User signaled end of input
                            break
                    
                    if new_lines:
                        task = "\n".join(new_lines)
                        task = ftfy.fix_text(task)
                        try:
                            prompt_path.write_text(task, encoding='utf-8')
                            print(f"Prompt '{prompt_path.name}' updated.")
                        except Exception as e:
                            print(f"Error writing updated prompt to {prompt_path}: {e}")
            except Exception as e:
                print(f"Error reading or processing prompt {prompt_path}: {e}")
                task = "" # Reset task if error

        else: # Create a new prompt
            new_filename_input = await self.wait_for_user_input("Enter a filename for the new prompt (e.g., my_new_task, press Enter for default 'custom_prompt'): ")
            if new_filename_input.strip():
                filename_stem = new_filename_input.strip().replace(" ", "_").replace(".md", "")
            
            new_prompt_path = PROMPTS_DIR / f"{filename_stem}.md"
            print(f"Enter the prompt text for '{new_prompt_path.name}' (Ctrl+D or Ctrl+Z then Enter to finish):")
            
            new_prompt_lines = []
            while True:
                try:
                    line = await self.wait_for_user_input("")
                    new_prompt_lines.append(line)
                except EOFError:
                    break
            
            if new_prompt_lines:
                task = "\n".join(new_prompt_lines)
                task = ftfy.fix_text(task)
                try:
                    new_prompt_path.write_text(task, encoding='utf-8')
                    print(f"New prompt '{new_prompt_path.name}' saved.")
                except Exception as e:
                    print(f"Error saving new prompt to {new_prompt_path}: {e}")
            else:
                print("No input received for the new prompt. Task is empty.")
                task = ""

        # OpenAI Task Processing
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if OPENROUTER_API_KEY and task:
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
                model = "anthropic/claude-3.5-sonnet" # Using 3.5 as 3.7 might not be available or stable
                
                ai_prompt_template = f"""Do what would make sense as far as creating the final task definition. If the user asks a simple non coding question you may simpley just repeat the task. If it something unexpected then use your judgement to create a task definition that would work for the circumstance. If it sounds like a programming project follow these instruction. First, restate the problem in more detail. At this stage if there are any decisons that were left for the developer, you should make the choices needed and include them in your restated description of the promlem.
After you have restated the expanded description. You will provide a file tree for the program. In general, lean towards creating less files and folders while still keeping things organized and manageable. You will use absolute imports. Do not create any non code files such as pyproject, gitignore, readme, etc.. You will need to attempt to list every file that will be needed and be thorough. This is all code files that will need to be created, all asset files etc. For each file, in additon to the path and filename, you should give a brief statement about the purpose of the file and explicitly give the correct way to import the file using absolut imports. Do not actually create any of the code for the project. Just the expanded description and the file tree with the extra info included. The structure should focus on simplicity and reduced subdirectories and files.
Task: {task}"""
                
                messages = [{"role": "user", "content": ai_prompt_template}]
                print("Sending task to AI for analysis (this may take a moment)...")
                
                completion = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    messages=messages
                )
                processed_task = completion.choices[0].message.content
                if processed_task: # Ensure content is not None
                    task = ftfy.fix_text(processed_task)
                    print("--- Task analyzed and updated by AI ---")
                else:
                    print("AI returned empty content. Using the task as is.")
            except Exception as e:
                print(f"Error during OpenAI task processing: {e}. Using the task as is.")
        elif not task:
            print("Task is empty. Skipping OpenAI processing.")
        else: # No API key, but task exists
            print("Warning: OPENROUTER_API_KEY not set. Skipping task analysis with AI.")

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
            task += "\n" + task_final_suffix.strip() # Use strip to remove leading/trailing newlines from suffix
        else: # If task is empty (e.g. new prompt with no input and no AI processing)
            # This might happen if the user creates a new prompt but provides no content,
            # and AI processing is skipped or fails to populate the task.
            print("Warning: Task is empty. The final task will only contain the standard suffix.")
            task = task_final_suffix.strip()

        # Save task
        if not LOGS_DIR.exists():
            try:
                LOGS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Error creating logs directory {LOGS_DIR}: {e}")
                # Decide how to handle this, maybe can't save task.txt

        task_file_path = LOGS_DIR / "task.txt"
        try:
            task_file_path.write_text(task, encoding='utf-8')
            print(f"Task definition saved to '{task_file_path}'")
        except Exception as e:
            print(f"Error saving task to {task_file_path}: {e}")
            
        print(f"Project directory set to: '{project_dir}'")
        
        return task

async def main_test():
    # Ensure PROMPTS_DIR and LOGS_DIR exist for testing
    # It's good practice to ensure these directories exist before running tests.
    # PROMPTS_DIR and LOGS_DIR are imported from config.
    if not PROMPTS_DIR.exists(): 
        PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Test Helper: Created PROMPTS_DIR at {PROMPTS_DIR}")
    if not LOGS_DIR.exists(): 
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Test Helper: Created LOGS_DIR at {LOGS_DIR}")

    # Create a dummy prompt for testing if none exist, so the selection list isn't empty.
    prompt_files_exist = any(PROMPTS_DIR.glob("*.md"))
    if not prompt_files_exist:
        dummy_prompt_path = PROMPTS_DIR / "example_prompt.md"
        with open(dummy_prompt_path, "w", encoding="utf-8") as f:
            f.write("This is an example prompt for testing purposes.")
        print(f"Test Helper: Created dummy prompt at {dummy_prompt_path}")

    display = AgentDisplayConsole()
    try:
        print("Attempting to run display.select_prompt_console()...")
        # select_prompt_console is an async method, so it needs to be awaited.
        task = await display.select_prompt_console() 

        print("\n--- Selected Task (from main_test) ---")
        print(task if task else "No task was selected or created.")
        
        display.add_message("assistant", "Console agent ready for further interaction.")
        user_response = await display.wait_for_user_input("Test input (type something and press Enter): ")
        display.add_message("user", user_response)
        
        display.add_message("tool", "Example tool output content.")
        display.clear_messages("all") # panel 'all' is just an example

    except Exception as e:
        print(f"Error during main_test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # If OPENROUTER_API_KEY is sourced from a .env file, it should be loaded here
    # for standalone testing of this script.
    # Example:
    # from dotenv import load_dotenv
    # load_dotenv()
    # print("Attempting to run asyncio.run(main_test())")
    asyncio.run(main_test())
