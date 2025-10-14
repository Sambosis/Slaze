# agent.py
"""is the main file that contains the Agent class. The Agent class is responsible for interfacing with the LLM API and running the tools. It receives messages from the user, sends them to the LLM API, and processes the response. It also manages the state of the conversation, such as the context and the messages exchanged between the user and the assistant. The Agent class uses the ToolCollection class to run the tools and generate the responses. The Agent class also uses the OutputManager class to format the messages and display them in the web interface. The Agent class uses the TokenTracker class to track the token usage and display it in the web interface. The Agent class uses the AgentDisplayWebWithPrompt class to display the messages in the web interface and prompt the user for input. The Agent class is used by the run.py and serve.py scripts to start the application and run the web server."""

import json
import os
import re
import logging
from typing import Dict, Union
from openai import AsyncOpenAI
from rich import print as rr

from tools import (
    # BashTool,

    ProjectSetupTool,
    WriteCodeTool,
    PictureGenerationTool,
    ASTCodeEditorTool,
    ToolCollection,
    ToolResult,
    BashTool
)
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole
from unittest.mock import AsyncMock
import asyncio
from utils.context_helpers import format_messages_to_string
from utils.context_helpers import extract_text_from_content, refresh_context_async
from utils.output_manager import OutputManager
from utils.llm_client import OpenRouterClient
from config import (
    COMPUTER_USE_BETA_FLAG,
    PROMPT_CACHING_BETA_FLAG,
    MAIN_MODEL,
    MAX_SUMMARY_TOKENS,
    reload_system_prompt,
    LOGS_DIR
)

from dotenv import load_dotenv
from pathlib import Path
from config import set_constant, get_constant

load_dotenv()

logger = logging.getLogger(__name__)


async def call_llm_for_task_revision(prompt_text: str, client: AsyncOpenAI, model: str) -> str:
    """
    Calls an LLM to revise the given prompt_text, incorporating detailed instructions
    for structuring the task, especially for programming projects.
    """
    logger.info(f"Attempting to revise/structure task with LLM ({model}): '{prompt_text[:100]}...' ")
    print("Calling LLM for task revision...\n")
    # This combined prompt is based on the user's example
    detailed_revision_prompt_template = """
      Your primary function is to analyze and structure a user's request. Your output will be used as the main task definition for a subsequent AI agent that generates software projects.

      Carefully consider the user's input and transform it into a detailed and actionable task definition.

      ## **USER'S ORIGINAL REQUEST:**

      ## {user_request}

      **INSTRUCTIONS:**

      1.  **Analyze the Request Type:**

            * **If the request sounds like a programming project:** Proceed with instructions 2-5.
            * **If the user asks a simple non-coding question:** Simply repeat the user's question as the task definition. For example, if the user asks 'What is the capital of France?', the output should be 'What is the capital of France?'.
            * **If the request is unexpected or ambiguous:** Use your best judgment to create a concise task definition that makes sense for the circumstance. Prioritize clarity.

      2.  **For Programming Projects - Expand Description:**

            * Restate the problem in significantly more detail. Flesh out the requirements.
            * If there are any decisions left for the developer (e.g., choice of a specific algorithm, UI details if not specified), you MUST make those choices now and clearly include them in your expanded description. Be specific.

      3.  **For Programming Projects - Define Data Dictionary:**

            * After the expanded description, create a **Data Dictionary**. This dictionary serves as a foundational reference to ensure consistency in naming and typing throughout the codebase.
            * Define the key **Classes, functions, methods, and important data structures** that will form the core logic of the application.
            * Present this as a markdown table with the following columns:
            * **Name:** The name of the class, function, or variable (e.g., `User`, `calculate_total_price`).
            * **Type:** The kind of element (e.g., `Class`, `Function`, `Method`, `Variable`).
            * **Data Type / Signature:** For variables, their data type (e.g., `str`, `int`, `list[dict]`). For functions/methods, their signature including parameters and return types (e.g., `def calculate_total(items: list) -> float:`). For classes, list their primary attributes and types.
            * **Description:** A brief, one-sentence explanation of the element's purpose.
            * **Example Data Dictionary Table:**
            | Name | Type | Data Type / Signature | Description |
            | :--- | :--- | :--- | :--- |
            | `Product` | Class | Attributes: `id (int)`, `name (str)`, `price (float)` | Represents a single product in the inventory. |
            | `add_to_cart` | Function | `def add_to_cart(cart: dict, product: Product, quantity: int) -> dict:` | Adds a specified quantity of a product to the shopping cart. |

      4.  **For Programming Projects - Define File Tree:**

            * After the Data Dictionary, provide a file tree for the program. This tree should list ALL necessary code files and any crucial asset files (e.g., `main.py`, `utils/helper.py`, `assets/icon.png`).
            * For each file in the tree, you MUST provide:
            * The filename relative to the project root (e.g., `main.py`, `utils/helper.py`).
            * A brief, clear statement about the purpose of that specific file.
            * Explicitly state the correct way to import the file/module using absolute imports from the project root (e.g., `from src import app`, `import src.models.user`). Assume the project root is added to PYTHONPATH.
            * **Important Considerations for File Tree:**
            * Lean towards creating FEWER files and folders while maintaining organization. Avoid overly nested structures unless strictly necessary.
            * Focus on simplicity.

      5.  **Output Format:**

            * The output must be in markdown format.
            * The output should start with the expanded description (if a programming project) or the direct task (if non-coding).
            * This should be followed by the **Data Dictionary** and then the **File Tree** section, clearly delineated.
            * Do NOT include any conversational preamble, your own comments about the process, or any text beyond the structured task definition itself.
            * **Example structure for a programming project:**
            ```
            [Expanded Description of the project, including decisions made...]

            **Data Dictionary**
            | Name      | Type     | Data Type / Signature        | Description                               |
            |-----------|----------|------------------------------|-------------------------------------------|
            | `User`    | Class    | Attributes: `id`, `username` | Represents a user.                        |
            | `get_user`| Function | `def get_user(id: int) -> User:` | Retrieves a user by their unique ID.      |

            **File Tree**
            - ./
            - main.py (Purpose: Main entry point of the application. Import: `import main`)
            - models/
                  - __init__.py (Purpose: Marks 'models' as a package. Import: `from models import User`)
                  - user.py (Purpose: Contains the User class definition. Import: `from models.user import User`)
            ```

      Now, process the user's original request based on these instructions.
      
      """

    formatted_revision_prompt = detailed_revision_prompt_template.format(user_request=prompt_text)
    print(f"Formatted revision prompt:\n{formatted_revision_prompt}\n")
    model_to_use= MAIN_MODEL

    # First try to use the shared client that the rest of the agent relies on. This client is
    # already configured with any custom base URL, headers, or networking configuration that the
    # user may have provided (for example when routing through a proxy). Reusing it ensures that
    # task revision benefits from the same working configuration as normal LLM calls.

    response = await client.chat.completions.create(
        model=model_to_use,
        messages=[
            {
                "role": "user",
                "content": formatted_revision_prompt,
            }
        ],
    )

    print(f"LLM response for task revision:\n{response}\n")
    print("That was the response from the shared AsyncOpenAI client.\n")
    if (
        response
        and getattr(response, "choices", None)
        and response.choices[0].message
        and response.choices[0].message.content
    ):
        revised_task = response.choices[0].message.content.strip()
        if len(revised_task) < 0.5 * len(prompt_text) and len(prompt_text) > 50:
            logger.warning("LLM task revision is much shorter than original. Using original.")
            return prompt_text
        logger.info(f"Task successfully revised by shared AsyncOpenAI ({model_to_use}): '{revised_task[:100]}...'")
        print("Task successfully revised by shared AsyncOpenAI client.\n")
        return revised_task
    else:
        logger.warning("Shared AsyncOpenAI client returned empty/invalid response. Trying dedicated client.")
        print("Shared AsyncOpenAI client returned empty/invalid response. Trying dedicated client.\n")


class Agent:
    async def _revise_and_save_task(self, initial_task: str) -> str:
        """
        Revises the task using an LLM, saves it to task.txt, updates the
        TASK constant, and returns the revised task.
        """
        revised_task_from_llm = await call_llm_for_task_revision(initial_task, self.client, MAIN_MODEL)
        logger.info(f"Task revision result: '{initial_task[:100]}...' -> '{revised_task_from_llm[:100]}...' ")

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
            if self.display:
                self.display.add_message(msg_type="user", content=revised_task_from_llm or "")
        except Exception as e:
            logger.error(f"Error saving revised task to {task_file_path}: {e}", exc_info=True)
            # Continue even if file write fails, but log it.
        self.task = revised_task_from_llm
        set_constant("TASK", revised_task_from_llm)
        logger.info("TASK constant updated with revised task.")
        return revised_task_from_llm

    def __init__(self, task: str, display: Union[WebUI, AgentDisplayConsole], manual_tool_confirmation: bool = False):
        self.task = task
        # Set initial task constant
        set_constant("TASK", self.task)
        logger.info(f"Initial TASK constant set to: {self.task[:100]}...")

        # Initialize client and other properties needed by _revise_and_save_task
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        )

        self.display = display
        self.manual_tool_confirmation = manual_tool_confirmation
        self.context_recently_refreshed = False
        self.refresh_count = 45
        self.refresh_increment = 15  # the number     to increase the refresh count by
        self.tool_collection = ToolCollection(
            WriteCodeTool(display=self.display),
            ProjectSetupTool(display=self.display),
            BashTool(display=self.display),
            PictureGenerationTool(display=self.display),
            # OpenInterpreterTool(display=self.display),  # Uncommented and enabled for testing
            # EditTool(display=self.display),  # Uncommented and enabled for testing
            ASTCodeEditorTool(display=self.display),
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
            if message["role"] == "user":
                content = message.get("content")
                # Only operate if content is a list of dicts (our structured format)
                if isinstance(content, list) and content:
                    last = content[-1]
                    if isinstance(last, dict):
                        if breakpoints_remaining:
                            breakpoints_remaining -= 1
                            last["cache_control"] = {"type": "ephemeral"}
                        else:
                            last.pop("cache_control", None)
                            break

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name to match pattern '^[a-zA-Z0-9_-]{1,64}$'"""

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
        with open(f"{LOGS_DIR}/messages.md", "w", encoding="utf-8") as f:
            f.write(format_messages_to_string(messages) + "\n"  )
        tool_choice = "auto" if self.step_count > 20 else "auto"
        try:
            response = await self.client.chat.completions.create(
                model=MAIN_MODEL,
                messages=messages,
                tools=self.tool_params,
                tool_choice=tool_choice,
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
        rr(response.choices[0])
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                tc.to_dict() if hasattr(tc, "to_dict") else tc.__dict__
                for tc in msg.tool_calls
            ]
        self.messages.append(assistant_msg)
        tool_messages = self.messages[:-1]  # All messages except the last assistant message
        # prompt creation that gives all but the last message , then a new user message that explains that given the context so far,
        # Give a couple sentences explanation that you are going to do the actions in assistant_msg["tool_calls"]
        tool_prompt = f"""Given the context so far, We will be performing the following actions: {assistant_msg['content'] if 'content' in assistant_msg else ''}
        {assistant_msg['tool_calls'] if 'tool_calls' in assistant_msg else []}
        Please respond with markdown formatted list that contains the actions you will take.
        It should be no more than 2 to 4 items long long and you should use simple statements such as - Create the code for main.py - Run the app - Fix the bugs in the code """
        tool_messages.append({"role": "user", "content": tool_prompt})
        tool_response = await self.client.chat.completions.create(
            model=MAIN_MODEL,
            messages=tool_messages,
            max_tokens=MAX_SUMMARY_TOKENS,
        )
        tool_msg = tool_response.choices[0].message

        # If the assistant returned tool_calls, execute them sequentially and
        # append the tool results into the conversation. Otherwise, run the
        # evaluation step below.
        if assistant_msg.get("tool_calls"):
            for tc in assistant_msg.get("tool_calls", []):
                try:
                    # Prepare a content_block compatible with run_tool
                    content_block = {
                        "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "unknown"),
                        "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown"),
                        "input": tc.get("input") if isinstance(tc, dict) else getattr(tc, "input", {}),
                    }
                    tool_result = await self.run_tool(content_block)

                    # Merge tool_result content into a single text string for the conversation
                    result_text_parts = []
                    if isinstance(tool_result.get("content"), list):
                        for content_item in tool_result["content"]:
                            if isinstance(content_item, dict) and content_item.get("type") == "text" and "text" in content_item:
                                result_text_parts.append(str(content_item["text"]))
                    result_text = " ".join(result_text_parts) if result_text_parts else ""
                    self.messages.append({"role": "tool", "tool_call_id": content_block["id"], "content": result_text})

                except Exception as e:
                    logger.error(f"Error executing tool call {tc}: {e}", exc_info=True)
                    self.messages.append({"role": "tool", "tool_call_id": content_block.get("id", "unknown"), "content": f"Tool execution failed: {e}"})

        else:
            # No tool calls. Time for evaluation.
            evaluation_prompt = f'''
                YouAreAnAutomatedProjectManager.YourJobIsToEvaluateTheProgressOfAnAIAgentTaskedWithDevelopingASoftwareApplication.

                TheInitialTaskWas:
                ---
                {self.task}
                ---

                ReviewTheConversationHistory.TheAgent'sLastMessageWas:
                ---
                {self.messages[-1]['content'] if self.messages[-1]['content'] else 'NoContentInLastMessage.'}
                ---

                HasTheAgentSuccessfullyRunTheApplicationAndProvidedEvidence(e.g.,logs,output)ThatItWorksAsIntendedAndIsFreeOfErrors?

                -IfNO:TheTaskIsNotComplete.GenerateAConciseUserMessageForTheAgent.ThisMessageMustNotBeConversational.ItShouldClearlyStateWhyTheTaskIsIncompleteAndProvideSpecific,ActionableNextStepsForTheAgentToTake(e.g.,"TheApplicationCrashed.AnalyzeTheErrorMessageAndFixTheBugInFileX.","YouHaveWrittenTheCode,NowYouMustRunItToVerifyItWorks.").

                -IfYES:TheTaskAppearsToBeComplete.RespondWithTheSinglePhrase:`TASK_COMPLETE`
                '''
            evaluation_messages = self.messages + [{"role": "user", "content": evaluation_prompt}]

            try:
                response = await self.client.chat.completions.create(
                    model=MAIN_MODEL,
                    messages=evaluation_messages,
                    max_tokens=MAX_SUMMARY_TOKENS,
                )
                evaluation_response = response.choices[0].message.content or ""

                if "TASK_COMPLETE" in evaluation_response:
                    self.display.add_message("assistant", "Evaluation result: Task is complete.")
                    wait_func = getattr(self.display, "wait_for_user_input", None)
                    if wait_func and asyncio.iscoroutinefunction(wait_func):
                        user_input = await wait_func(
                            "Agent believes task is complete. Enter new instructions or type 'exit' to quit: "
                        )
                        if user_input:
                            if user_input.strip().lower() in {"exit", "quit"}:
                                return False
                            self.messages.append({"role": "user", "content": user_input})
                elif evaluation_response:
                    self.messages.append({"role": "user", "content": evaluation_response})
                    self.display.add_message("user", f"Auto-Correction: {evaluation_response}")
                else:
                    # Fallback to old behavior if evaluation fails to produce content
                    wait_func = getattr(self.display, "wait_for_user_input", None)
                    if wait_func and asyncio.iscoroutinefunction(wait_func):
                        user_input = await wait_func(
                            "No tool calls and evaluation failed. Enter instructions or type 'exit' to quit: "
                        )
                        if user_input:
                            if user_input.strip().lower() in {"exit", "quit"}:
                                return False
                            self.messages.append({"role": "user", "content": user_input})

            except Exception as e:
                self.display.add_message("assistant", f"Evaluation LLM call failed: {e}")
                # Fallback to old behavior
                wait_func = getattr(self.display, "wait_for_user_input", None)
                if wait_func and asyncio.iscoroutinefunction(wait_func):
                    user_input = await wait_func(
                        "No tool calls and evaluation failed. Enter instructions or type 'exit' to quit: "
                    )
                    if user_input:
                        if user_input.strip().lower() in {"exit", "quit"}:
                            return False
                        self.messages.append({"role": "user", "content": user_input})

        return True