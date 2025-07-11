from typing import ClassVar, Literal, Union
from interpreter import interpreter
from .base import BaseTool, ToolResult
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole
import logging
import os
from rich import print as rr
from config import get_constant
from pathlib import Path

logger = logging.getLogger(__name__)

class InterpreterTool(BaseTool):
    def __init__(self, display: Union[WebUI, AgentDisplayConsole] = None):
        self.display = display
        # Initialize open-interpreter
        self.interpreter = interpreter
        # Configure settings for open-interpreter
        self.interpreter.auto_run = True  # Automatically run code
        self.interpreter.llm.model = "gpt-4-turbo" # Example model, can be configured
        self.interpreter.llm.api_key = os.getenv("OPENAI_API_KEY") # Ensure API key is set
        self.interpreter.custom_instructions = """
        You are an AI assistant that executes commands.
        When given a command, execute it and return the output.
        If you need to install packages, please ask for confirmation before proceeding.
        Provide the output clearly. If there are errors, report them.
        Be concise and direct in your responses.
        """
        super().__init__(input_schema=None, display=display)

    description = """
        A tool that allows the agent to run commands using the open-interpreter.
        All commands are executed relative to the current project directory if one
        has been set via the configuration. The tool parameters follow the OpenAI
        function calling format.
        """

    name: ClassVar[Literal["interpreter"]] = "interpreter"
    api_type: ClassVar[Literal["interpreter_20240726"]] = "interpreter_20240726" # Using a new API version date

    async def __call__(self, command: str | None = None, **kwargs):
        if command is not None:
            if self.display is not None:
                try:
                    self.display.add_message("user", f"Executing command with open-interpreter: {command}")
                except Exception as e:
                    return ToolResult(error=str(e), tool_name=self.name, command=command)

            return await self._run_command_with_interpreter(command)
        raise ToolError("no command provided.")

    async def _run_command_with_interpreter(self, command: str):
        """Execute a command using open-interpreter."""
        output_str = ""
        error_str = ""
        success = False

        repo_dir = get_constant("REPO_DIR")
        cwd = str(repo_dir) if repo_dir and Path(repo_dir).exists() else os.getcwd()
        self.interpreter.system_message = f"You are an AI assistant. You are running in a context where the current working directory is '{cwd}'. All file operations should be relative to this directory unless an absolute path is explicitly given. Execute the following command: {command}"

        try:
            # Ensure interpreter runs in the correct CWD if possible
            # Open-interpreter's chat method doesn't directly take a CWD,
            # it inherits the CWD of the process it's running in.
            # We can try to set it via a message or ensure the main process CWD is correct.
            # For now, we'll rely on the system_message and the agent's CWD.

            # For open-interpreter, the `chat` method is synchronous when streamed like this.
            # We must run it in a thread pool executor to avoid blocking the asyncio event loop.
            import asyncio

            messages = []

            # Define the synchronous part that will be run in a thread
            def _sync_interpreter_chat():
                temp_messages = []
                for chunk in self.interpreter.chat(command, display=False, stream=True):
                    temp_messages.append(chunk)
                    # Displaying messages from here might be tricky if display object is not thread-safe
                    # For now, we'll collect all messages and then process.
                    # If self.display is thread-safe or designed for async, this could be done here.
                    # Example:
                    # if self.display and chunk.get("type") == "console" and chunk.get("content"):
                    #     asyncio.run_coroutine_threadsafe(
                    #         self.display.add_message("assistant", f"Interpreter output: {chunk['content']}"),
                    #         asyncio.get_event_loop() # This needs careful handling if loop is not running or different
                    #     )
                return temp_messages

            messages = await asyncio.to_thread(_sync_interpreter_chat)

            # Process messages to extract output and errors
            full_console_output = []
            for msg_block in messages:
                if self.display and msg_block.get("type") == "console" and msg_block.get("content"):
                     # If display has an async add_message, it should be called with await
                     # For now, assuming add_message is synchronous or handles its own async if needed.
                     # This part is tricky if display method itself is async and called from sync code.
                     # Best to do display updates after await asyncio.to_thread completes.
                    pass # Displaying will be handled after collecting all messages if needed.

                if msg_block.get("type") == "console" and "content" in msg_block:
                    full_console_output.append(msg_block["content"])
                elif msg_block.get("type") == "code" and "code" in msg_block:
                    full_console_output.append(f"Executed Code:\n{msg_block['code']}")
                    if "output" in msg_block:
                         full_console_output.append(f"Code Output:\n{msg_block['output']}")
                elif msg_block.get("type") == "message" and "content" in msg_block:
                    full_console_output.append(f"Assistant: {msg_block['content']}")

            # Example of how to update display after collecting messages (if display methods are sync)
            if self.display:
                for item_output in full_console_output:
                    # This assumes add_message is not async or handles its own async nature.
                    # If add_message is async, it cannot be called directly like this from here.
                    # The display object would need a thread-safe way to queue messages.
                    # For simplicity, let's assume display is handled by the agent based on ToolResult.
                    pass


            if full_console_output:
                output_str = "\n".join(full_console_output)
            else:
                output_str = "No direct output from interpreter. Full conversation logged."

            for msg_block in messages:
                if msg_block.get("type") == "error" or \
                   (msg_block.get("role") == "assistant" and "error" in msg_block.get("content", "").lower()):
                    error_str += msg_block.get("content", "Unknown error from interpreter") + "\n"

            if not error_str:
                success = True
            else:
                success = False
                # Append error to the main output string if not already there (e.g. from assistant message)
                if error_str.strip() not in output_str:
                     output_str += f"\nErrors reported by interpreter:\n{error_str}"

            if len(output_str) > 200000:
                output_str = f"{output_str[:100000]} ... [TRUNCATED] ... {output_str[-100000:]}"
            # error_str is now part of output_str if errors occurred and was not duplicative

            formatted_output = (
                f"command: {command}\n"
                f"working_directory: {cwd}\n"
                f"success: {str(success).lower()}\n"
                f"output: {output_str}\n"
            )
            rr(formatted_output)

            return ToolResult(
                output=formatted_output,
                error=error_str.strip() if error_str else None,
                tool_name=self.name,
                command=command,
            )

        except Exception as e:
            logger.error(f"Error running command with open-interpreter: {e}", exc_info=True)
            current_error_str = str(e) # Renamed to avoid conflict with outer scope error_str
            rr(current_error_str)

            formatted_output = (
                f"command: {command}\n"
                f"working_directory: {cwd}\n"
                f"success: false\n"
                f"output: {output_str}\n"
                f"error: {current_error_str}"
            )
            return ToolResult(
                output=formatted_output,
                error=current_error_str,
                tool_name=self.name,
                command=command,
            )

    def to_params(self) -> dict:
        logger.debug(f"InterpreterTool.to_params called with api_type: {self.api_type}")
        params = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command or natural language instruction to be executed by open-interpreter.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
        logger.debug(f"InterpreterTool params: {params}")
        return params

# Example of how to use it (for testing purposes, not part of the class):
# async def main():
#     tool = InterpreterTool()
#     # result = await tool(command="list files in current directory")
#     # print("Result for 'list files':")
#     # print(result.output)
#     # result_pip = await tool(command="what is the result of 10 + 5?")
#     # print("\nResult for '10 + 5':")
#     # print(result_pip.output)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
