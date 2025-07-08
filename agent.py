# agent.py
"""is the main file that contains the Agent class. The Agent class is responsible for interfacing with the LLM API and running the tools. It receives messages from the user, sends them to the LLM API, and processes the response. It also manages the state of the conversation, such as the context and the messages exchanged between the user and the assistant. The Agent class uses the ToolCollection class to run the tools and generate the responses. The Agent class also uses the OutputManager class to format the messages and display them in the web interface. The Agent class uses the TokenTracker class to track the token usage and display it in the web interface. The Agent class uses the AgentDisplayWebWithPrompt class to display the messages in the web interface and prompt the user for input. The Agent class is used by the run.py and serve.py scripts to start the application and run the web server."""

import json
import os
from typing import Dict, Union
from openai import OpenAI
import logging
from rich import print as rr

from tools import (
    BashTool,
    ProjectSetupTool,
    WriteCodeTool,
    PictureGenerationTool,
    EditTool,
    ToolCollection,
    ToolResult,
)
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole
from unittest.mock import AsyncMock
import asyncio
from utils.context_helpers import extract_text_from_content, refresh_context_async
from utils.output_manager import OutputManager
from config import (
    COMPUTER_USE_BETA_FLAG,
    PROMPT_CACHING_BETA_FLAG,
    MAIN_MODEL,
    MAX_SUMMARY_TOKENS,
    reload_system_prompt,
)

from dotenv import load_dotenv
from pathlib import Path
from config import set_constant, get_constant, MAIN_MODEL

load_dotenv()

logger = logging.getLogger(__name__)


async def call_llm_for_task_revision(prompt_text: str, client: OpenAI, model: str) -> str:
    """
    Calls an LLM to revise the given prompt_text, incorporating detailed instructions
    for structuring the task, especially for programming projects.
    """
    logger.info(f"Attempting to revise/structure task with LLM ({model}): '{prompt_text[:100]}...'")

    # This combined prompt is based on the user-provided example
    detailed_revision_prompt_template = (
        "Your primary function is to analyze and structure a user's request. Your output will be used as the main task definition for a subsequent AI agent that generates software projects. "
        "Carefully consider the user's input and transform it into a detailed and actionable task definition.\n\n"
        "USER'S ORIGINAL REQUEST:\n"
        "------------------------\n"
        "{user_request}\n"
        "------------------------\n\n"
        "INSTRUCTIONS:\n"
        "1.  **Analyze the Request Type:**\n"
        "    *   **If the request sounds like a programming project:** Proceed with instructions 2-4.\n"
        "    *   **If the user asks a simple non-coding question:** Simply repeat the user's question as the task definition. For example, if the user asks 'What is the capital of France?', the output should be 'What is the capital of France?'.\n"
        "    *   **If the request is unexpected or ambiguous:** Use your best judgment to create a concise task definition that makes sense for the circumstance. Prioritize clarity.\n\n"
        "2.  **For Programming Projects - Expand Description:**\n"
        "    *   Restate the problem in significantly more detail. Flesh out the requirements.\n"
        "    *   If there are any decisions left for the developer (e.g., choice of a specific algorithm, UI details if not specified), you MUST make those choices now and clearly include them in your expanded description. Be specific.\n\n"
        "3.  **For Programming Projects - Define File Tree:**\n"
        "    *   After the expanded description, provide a file tree for the program. This tree should list ALL necessary code files and any crucial asset files (e.g., `main.py`, `utils/helper.py`, `assets/icon.png`).\n"
        "    *   For each file in the tree, you MUST provide:\n"
        "        *   The full path relative to the project root (e.g., `src/app.py`, `tests/test_module.py`).\n"
        "        *   A brief, clear statement about the purpose of that specific file.\n"
        "        *   Explicitly state the correct way to import the file/module using absolute imports from the project root (e.g., `from src import app`, `import src.models.user`). Assume the project root is the primary location for running the code or is added to PYTHONPATH.\n"
        "    *   **Important Considerations for File Tree:**\n"
        "        *   Lean towards creating FEWER files and folders while maintaining organization and manageability. Avoid overly nested structures unless strictly necessary.\n"
        "        *   Focus on simplicity.\n"
        "        *   Do NOT create or list non-code project management files like `pyproject.toml`, `.gitignore`, `README.md`, `LICENSE`, etc. Only list files that are part of the application's codebase or directly used assets.\n\n"
        "4.  **Output Format:**\n"
        "    *   The output should start with the expanded description (if a programming project) or the direct task (if non-coding).\n"
        "    *   This should be followed by the file tree section (if a programming project), clearly delineated.\n"
        "    *   Do NOT include any conversational preamble, your own comments about the process, or any text beyond the structured task definition itself.\n"
        "    *   Example structure for a programming project:\n"
        "        \"\"\"\n"
        "        [Expanded Description of the project, including decisions made...]\n\n"
        "        File Tree:\n"
        "        - project_root/\n"
        "          - src/\n"
        "            - __init__.py (Purpose: Marks 'src' as a package. Import: `import src`)\n"
        "            - main.py (Purpose: Main entry point of the application. Import: `from src import main` or `import src.main`)\n"
        "            - module_one/\n"
        "              - __init__.py (Purpose: Marks 'module_one' as a sub-package. Import: `from src import module_one`)\n"
        "              - functions.py (Purpose: Contains utility functions for module_one. Import: `from src.module_one import functions`)\n"
        "          - assets/\n"
        "            - image.png (Purpose: An image asset for the UI.)\n"
        "        \"\"\"\n\n"
        "Now, process the user's original request based on these instructions."
    )

    formatted_revision_prompt = detailed_revision_prompt_template.format(user_request=prompt_text)

    try:
        # The user's example used "anthropic/claude-3.7-sonnet:beta".
        # We should use the model specified or fall back to a strong default like MAIN_MODEL.
        # For this refinement, let's honor the specific model from the example if available, else MAIN_MODEL.
        # This requires checking if 'model' param can be overridden or if we add a new constant.
        # For now, I'll stick to the 'model' parameter passed to this function.
        # If the calling code passes "anthropic/claude-3.7-sonnet:beta", it will be used.
        # Otherwise, it will use MAIN_MODEL as per current Agent._revise_and_save_task.

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": formatted_revision_prompt}],
            temperature=0.3, # Lower temperature for more focused, less creative revision
            n=1,
            stop=None,
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            revised_task = response.choices[0].message.content.strip()
            # Basic check to ensure LLM didn't just return an empty string or something very short
            if len(revised_task) < 0.5 * len(prompt_text) and len(prompt_text) > 50 : # Heuristic: if significantly shorter
                 logger.warning(f"LLM task revision is much shorter than original. Original: '{prompt_text[:100]}...', Revised: '{revised_task[:100]}...'. Using original.")
                 return prompt_text

            logger.info(f"Task successfully revised by LLM ({model}): '{revised_task[:100]}...'")
            return revised_task
        else:
            logger.warning(f"LLM task revision ({model}) returned empty or invalid response. Using original task: '{prompt_text[:100]}...'")
            return prompt_text
    except Exception as e:
        logger.error(f"Error during LLM task revision with model {model}: {e}. Using original task: '{prompt_text[:100]}...'", exc_info=True)
        return prompt_text


class Agent:
    async def _revise_and_save_task(self, initial_task: str) -> str:
        """
        Revises the task using an LLM, saves it to task.txt, updates the
        TASK constant, and returns the revised task.
        """
        revised_task_from_llm = await call_llm_for_task_revision(initial_task, self.client, MAIN_MODEL)
        logger.info(f"Task revision result: '{initial_task[:100]}...' -> '{revised_task_from_llm[:100]}...'")

        # Use the LOGS_DIR constant from config
        logs_dir = get_constant("LOGS_DIR")
        if not logs_dir:
            logger.error("LOGS_DIR not found in constants. Cannot save revised task.txt.")
            # Fallback: update TASK constant but don't write to file.
            set_constant("TASK", revised_task_from_llm)
            return revised_task_from_llm

        logs_dir = Path(logs_dir)
        task_file_path = logs_dir / "task.txt"

        try:
            task_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(task_file_path, "w", encoding="utf-8") as f:
                f.write(revised_task_from_llm)
            logger.info(f"Revised task saved to {task_file_path}")
        except Exception as e:
            logger.error(f"Error saving revised task to {task_file_path}: {e}", exc_info=True)
            # Continue even if file write fails, but log it.

        set_constant("TASK", revised_task_from_llm)
        logger.info("TASK constant updated with revised task.")
        return revised_task_from_llm

    def __init__(self, task: str, display: Union[WebUI, AgentDisplayConsole], interactive_tool_calls: bool = False):
        self.task = task
        # Set initial task constant
        set_constant("TASK", self.task)
        logger.info(f"Initial TASK constant set to: {self.task[:100]}...")

        # Initialize client and other properties needed by _revise_and_save_task
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        )

        self.display = display
        self.interactive_tool_calls = interactive_tool_calls
        self.context_recently_refreshed = False
        self.refresh_count = 45
        self.refresh_increment = 15  # the number     to increase the refresh count by
        self.tool_collection = ToolCollection(
            WriteCodeTool(display=self.display),
            ProjectSetupTool(display=self.display),
            BashTool(display=self.display),
            PictureGenerationTool(display=self.display),
            EditTool(display=self.display),  # Uncommented and enabled for testing
            display=self.display,
        )
        self.output_manager = OutputManager(self.display)
        self.system_prompt = reload_system_prompt()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        # self.client is already initialized before _revise_and_save_task is called.
        # No need to initialize it again here.
        # self.client = OpenAI(
        #     api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        #     base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        # )
        self.enable_prompt_caching = True
        self.betas = [COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG]
        self.image_truncation_threshold = 1
        self.only_n_most_recent_images = 2
        self.step_count = 0
        # Add detailed logging of tool params
        self.tool_params = self.tool_collection.to_params()

    def log_tool_results(self, combined_content, tool_name, tool_input):
        """
        Log tool results to a file in a human-readable format.

        Args:
            combined_content: The content to log
            tool_name: The name of the tool that was executed
            tool_input: The input provided to the tool
        """
        with open("./logs/tool.txt", "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TOOL EXECUTION: {tool_name}\n")
            f.write(f"INPUT: {json.dumps(tool_input, indent=2)}\n")
            f.write("-" * 80 + "\n")

            for item in combined_content:
                f.write(f"CONTENT TYPE: {item['type']}\n")
                if item["type"] == "tool_result":
                    f.write(f"TOOL USE ID: {item['tool_use_id']}\n")
                    f.write(f"ERROR: {item['is_error']}\n")
                    if isinstance(item["content"], list):
                        f.write("CONTENT:\n")
                        for content_item in item["content"]:
                            f.write(
                                f"  - {content_item['type']}: {content_item.get('text', '[non-text content]')}\n"
                            )
                    else:
                        f.write(f"CONTENT: {item['content']}\n")
                elif item["type"] == "text":
                    f.write(f"TEXT:\n{item['text']}\n")
                f.write("-" * 50 + "\n")
            f.write("=" * 80 + "\n\n")

    async def run_tool(self, content_block):
        result = ToolResult(
            output="Tool execution not started", tool_name=content_block["name"]
        )
        # SET THE CONSTANT TASK to self.task
        try:
            logger.debug(f"Tool name: {content_block['name']}")
            logger.debug(f"Tool input: {content_block['input']}")
            result = await self.tool_collection.run(
                name=content_block["name"],
                tool_input=content_block["input"],
            )
            if result is None:
                result = ToolResult(
                    output="Tool execution failed with no result",
                    tool_name=content_block["name"],
                )
        except Exception as e:
            result = ToolResult(
                output=f"Tool execution failed: {str(e)}",
                tool_name=content_block["name"],
                error=str(e),
            )
        finally:
            tool_result = self._make_api_tool_result(result, content_block["id"])
            # logger.debug(f"Tool result: {tool_result}") # This might be too verbose, let's comment it out for now
            tool_output = (
                result.output
                if hasattr(result, "output") and result.output
                else str(result)
            )
            tool_name = content_block["name"]
            if len(tool_name) > 64:
                tool_name = tool_name[:61] + "..."  # Truncate to 61 and add ellipsis
            combined_content = [
                {
                    "type": "tool_result",
                    "content": tool_result["content"],
                    "tool_use_id": tool_result["tool_use_id"],
                    "is_error": tool_result["is_error"],
                }
            ]
            combined_content.append(
                {
                    "type": "text",
                    "text": f"Tool '{tool_name}' was called with input: {json.dumps(content_block['input'])}.\nResult: {extract_text_from_content(tool_output)}",
                }
            )

            # Only tool messages should follow assistant tool calls. Appending a
            # user message here violates the expected OpenAI API sequence and
            # leads to errors like:
            #   "An assistant message with 'tool_calls' must be followed by tool
            #   messages responding to each 'tool_call_id'."
            # The tool results are instead logged and a proper tool message is
            # added by ``Agent.step`` after ``run_tool`` returns.

            # Use the dedicated logging function instead of inline logging
            self.log_tool_results(combined_content, tool_name, content_block["input"])

            return tool_result

    def _make_api_tool_result(self, result: ToolResult, tool_use_id: str) -> Dict:
        """Create a tool result dictionary."""
        tool_result_content = []
        is_error = False

        if result is None:
            is_error = True
            tool_result_content.append(
                {"type": "text", "text": "Tool execution resulted in None"}
            )
        elif isinstance(result, str):
            is_error = True
            tool_result_content.append({"type": "text", "text": result})
        else:
            # Check if there's an error attribute and it has a value
            if hasattr(result, "error") and result.error:
                is_error = True
                tool_result_content.append({"type": "text", "text": result.error})

            # Add output if it exists
            if hasattr(result, "output") and result.output:
                tool_result_content.append({"type": "text", "text": result.output})

            # Add image if it exists
            if hasattr(result, "base64_image") and result.base64_image:
                tool_result_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": result.base64_image,
                        },
                    }
                )

        return {
            "type": "tool_result",
            "content": tool_result_content,
            "tool_use_id": tool_use_id,
            "is_error": is_error,
        }

    def _inject_prompt_caching(self):
        messages = self.messages
        breakpoints_remaining = 2
        for message in reversed(messages):
            if message["role"] == "user" and isinstance(
                content := message["content"], list
            ):
                if breakpoints_remaining:
                    breakpoints_remaining -= 1
                    content[-1]["cache_control"] = {"type": "ephemeral"}
                else:
                    content[-1].pop("cache_control", None)
                    break

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name to match pattern '^[a-zA-Z0-9_-]{1,64}$'"""
        import re

        # Keep only alphanumeric chars, underscores and hyphens
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        # Truncate to 64 chars if needed
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        return sanitized

    async def step(self):
        """Run one step of the agent using OpenAI."""
        self.step_count += 1
        messages = self.messages
        rr(f"Step {self.step_count} with {len(messages)} messages")
        try:
            response = self.client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                tools=self.tool_params,
                tool_choice="auto",
                max_tokens=MAX_SUMMARY_TOKENS,
            )
        except Exception as llm_error:
            self.display.add_message("assistant", f"LLM call failed: {llm_error}")
            new_context = await refresh_context_async(
                self.task, messages, self.display, self.client
            )
            self.messages = [{"role": "user", "content": new_context}]
            self.context_recently_refreshed = True
            return True

        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                tc.to_dict() if hasattr(tc, "to_dict") else tc.__dict__
                for tc in msg.tool_calls
            ]
        self.messages.append(assistant_msg)


        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = (
                    json.loads(tc.function.arguments) if tc.function.arguments else {}
                )
                for arg in args.values():
                    rr(arg)

                tool_name_to_run = tc.function.name
                tool_args_to_run = args
                tool_id_to_run = tc.id
                proceed_with_call = True

                if self.interactive_tool_calls:
                    # Call a new method on the display object to handle interaction
                    approval_response = await self.display.prompt_for_tool_call_approval(
                        tool_name_to_run, tool_args_to_run, tool_id_to_run
                    )
                    if approval_response:
                        tool_name_to_run = approval_response.get("name", tool_name_to_run)
                        tool_args_to_run = approval_response.get("args", tool_args_to_run)
                        # tool_id doesn't change
                        if not approval_response.get("approved", False):
                            proceed_with_call = False
                    else: # If display returned None, assume cancellation
                        proceed_with_call = False

                if proceed_with_call:
                    tool_result = await self.run_tool(
                        {"name": tool_name_to_run, "id": tool_id_to_run, "input": tool_args_to_run}
                    )
                else:
                    # Simulate a "cancelled by user" tool result
                    tool_result = self._make_api_tool_result(
                        ToolResult(output="Tool call cancelled by user.", tool_name=tool_name_to_run),
                        tool_id_to_run
                    )
                    # Ensure 'content' is not None and has a text part
                    if tool_result.get("content") is None:
                        tool_result["content"] = [{"type": "text", "text": "Tool call cancelled by user."}]
                    elif not any(item.get("type") == "text" for item in tool_result.get("content", [])):
                         tool_result["content"].append({"type": "text", "text": "Tool call cancelled by user."})


                result_text_parts = []
                if isinstance(tool_result.get("content"), list):
                    for content_item in tool_result["content"]:
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("type") == "text"
                            and "text" in content_item
                        ):
                            result_text_parts.append(str(content_item["text"]))
                result_text = " ".join(result_text_parts)
                self.messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result_text}
                )
        else:
            # No tool calls were returned. Only prompt for additional
            # instructions when using an interactive display like the
            # console or web UI. This avoids test failures when a simple
            # DummyDisplay is used.
            self.display.add_message("assistant", msg.content or "")
            wait_func = getattr(self.display, "wait_for_user_input", None)
            if wait_func and asyncio.iscoroutinefunction(wait_func):
                should_prompt = True
                if self.display.__class__.__name__ == "DummyDisplay" and not isinstance(
                    wait_func, AsyncMock
                ):
                    should_prompt = False
                if should_prompt:
                    user_input = await wait_func(
                        "No tool calls. Enter instructions or type 'exit' to quit: "
                    )
                    if user_input:
                        if user_input.strip().lower() in {"exit", "quit"}:
                            return False
                        self.messages.append(
                            {"role": "user", "content": user_input}
                        )

        return True