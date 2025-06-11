# agent.py
"""is the main file that contains the Agent class. The Agent class is responsible for interfacing with the LLM API and running the tools. It receives messages from the user, sends them to the LLM API, and processes the response. It also manages the state of the conversation, such as the context and the messages exchanged between the user and the assistant. The Agent class uses the ToolCollection class to run the tools and generate the responses. The Agent class also uses the OutputManager class to format the messages and display them in the web interface. The Agent class uses the TokenTracker class to track the token usage and display it in the web interface. The Agent class uses the AgentDisplayWebWithPrompt class to display the messages in the web interface and prompt the user for input. The Agent class is used by the run.py and serve.py scripts to start the application and run the web server."""

import asyncio
import json
import os
from re import M
from typing import Dict
from openai import OpenAI
from utils.logger import logger, log_debug as ic, log_info as rr

from tools import (
    BashTool,
    ProjectSetupTool,
    WriteCodeTool,
    PictureGenerationTool,
    EditTool,
    ToolCollection,
    ToolResult,
)
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from utils.context_helpers import *
from utils.output_manager import *
from config import *
# from token_tracker import TokenTracker

from dotenv import load_dotenv

load_dotenv()


class Agent:
    def __init__(self, task: str, display: AgentDisplayWebWithPrompt):
        self.task = task
        # set the task to TASK  in config
        self.display = display
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
        self.messages = []
        self.client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        )
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
            ic(content_block['name'])
            ic(content_block["input"])
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
            # ic(tool_result)
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
        rr(MAIN_MODEL)
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
            assistant_msg["tool_calls"] = [tc.to_dict() for tc in msg.tool_calls]
        self.messages.append(assistant_msg)


        if msg.tool_calls:
            for tc in msg.tool_calls:
                rr(tc)
            for tc in msg.tool_calls:
                args = (
                    json.loads(tc.function.arguments) if tc.function.arguments else {}
                )
                tool_result = await self.run_tool(
                    {"name": tc.function.name, "id": tc.id, "input": args}
                )
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
            # No tool calls were returned; prompt the user for guidance
            self.display.add_message("assistant", msg.content or "")
            user_input = await self.display.wait_for_user_input(
                "No tool calls. Enter instructions or type 'exit' to quit: "
            )
            if user_input:
                if user_input.strip().lower() in {"exit", "quit"}:
                    return False
                self.messages.append({"role": "user", "content": user_input})

        return True
