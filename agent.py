# agent.py 
"""is the main file that contains the Agent class. The Agent class is responsible for interfacing with the LLM API and running the tools. It receives messages from the user, sends them to the LLM API, and processes the response. It also manages the state of the conversation, such as the context and the messages exchanged between the user and the assistant. The Agent class uses the ToolCollection class to run the tools and generate the responses. The Agent class also uses the OutputManager class to format the messages and display them in the web interface. The Agent class uses the TokenTracker class to track the token usage and display it in the web interface. The Agent class uses the AgentDisplayWebWithPrompt class to display the messages in the web interface and prompt the user for input. The Agent class is used by the run.py and serve.py scripts to start the application and run the web server. 
"""
import asyncio
import json
import os
from typing import List, Dict

from anthropic import Anthropic
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaMessageParam,
    BetaToolResultBlockParam,
)
from icecream import ic

from tools import (
    BashTool,
    ProjectSetupTool,
    WriteCodeTool,
    PictureGenerationTool,
    EditTool,
    ToolCollection,
    ToolResult
)
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from utils.context_helpers import *
from utils.output_manager import *
from config import *
from token_tracker import TokenTracker

class Agent:
    def __init__(self, task: str, display: AgentDisplayWebWithPrompt):
        self.task = task
        self.display = display
        self.context_recently_refreshed = False
        self.refresh_count = 24
        self.tool_collection = ToolCollection(
            WriteCodeTool(display=self.display),
            ProjectSetupTool(display=self.display),
            BashTool(display=self.display),
            PictureGenerationTool(display=self.display),
            # EditTool(display=self.display),
            display=self.display
        )
        self.output_manager = OutputManager(self.display)
        self.token_tracker = TokenTracker(self.display)
        self.messages = []
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.enable_prompt_caching = True
        self.betas = [COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG]
        self.image_truncation_threshold = 1
        self.only_n_most_recent_images = 2

    async def run_tool(self, content_block):
        result = ToolResult(output="Tool execution not started", tool_name=content_block["name"])
        try:
            ic(content_block['name'])
            ic(content_block["input"])
            result = await self.tool_collection.run(
                name=content_block["name"],
                tool_input=content_block["input"],
            )
            if result is None:
                result = ToolResult(output="Tool execution failed with no result", tool_name=content_block["name"])
        except Exception as e:
            result = ToolResult(output=f"Tool execution failed: {str(e)}", tool_name=content_block["name"], error=str(e))
        finally:
            tool_result = self._make_api_tool_result(result, content_block["id"])
            ic(tool_result)
            tool_output = result.output if hasattr(result, 'output') and result.output else str(result)
            tool_name = content_block['name']
            if len(tool_name) > 64:
                tool_name = tool_name[:61] + "..."  # Truncate to 61 and add ellipsis
            combined_content = [{
                "type": "tool_result",
                "content": tool_result["content"],
                "tool_use_id": tool_result["tool_use_id"],
                "is_error": tool_result["is_error"]
            }]
            combined_content.append({
                "type": "text",
                "text": f"Tool '{tool_name}' was called with input: {json.dumps(content_block['input'])}.\nResult: {extract_text_from_content(tool_output)}"
            })
            self.messages.append({
                "role": "user",
                "content": combined_content
            })
            with open("./logs/tool.txt", 'a', encoding='utf-8') as f:
                for content in combined_content:
                    for key, value in content.items():
                        f.write(f"{key}: {value}\n")
            return tool_result

    def _make_api_tool_result(self, result: ToolResult, tool_use_id: str) -> Dict:
        """Create a tool result dictionary."""
        tool_result_content = []
        is_error = False
        
        if result is None:
            is_error = True
            tool_result_content.append({"type": "text", "text": "Tool execution resulted in None"})
        elif isinstance(result, str):
            is_error = True
            tool_result_content.append({"type": "text", "text": result})
        else:
            # Check if there's an error attribute and it has a value
            if hasattr(result, 'error') and result.error:
                is_error = True
                tool_result_content.append({"type": "text", "text": result.error})
            
            # Add output if it exists
            if hasattr(result, 'output') and result.output:
                tool_result_content.append({"type": "text", "text": result.output})
            
            # Add image if it exists
            if hasattr(result, 'base64_image') and result.base64_image:
                tool_result_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    }
                })

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
            if message["role"] == "user" and isinstance(content := message["content"], list):
                if breakpoints_remaining:
                    breakpoints_remaining -= 1
                    content[-1]["cache_control"] = BetaCacheControlEphemeralParam({"type": "ephemeral"})
                else:
                    content[-1].pop("cache_control", None)
                    break

    def _maybe_filter_to_n_most_recent_images(self):
        messages = self.messages
        images_to_keep = self.only_n_most_recent_images
        min_removal_threshold = self.image_truncation_threshold
        if images_to_keep is None:
            return messages

        tool_result_blocks = [
            item
            for message in messages
            for item in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ]

        images_to_remove = 0
        images_found = 0
        for tool_result in reversed(tool_result_blocks):
            if isinstance(tool_result.get("content"), list):
                for content in reversed(tool_result.get("content", [])):
                    if isinstance(content, dict) and content.get("type") == "image":
                        images_found += 1

        images_to_remove = max(0, images_found - images_to_keep)

        removed = 0
        for tool_result in tool_result_blocks:
            if isinstance(tool_result.get("content"), list):
                new_content = []
                for content in tool_result.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "image":
                        if removed < images_to_remove:
                            removed += 1
                            continue
                    new_content.append(content)
                tool_result["content"] = new_content

    def _sanitize_tool_name(self, name: str) -> str:
        """Sanitize tool name to match pattern '^[a-zA-Z0-9_-]{1,64}$'"""
        import re
        # Keep only alphanumeric chars, underscores and hyphens
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        # Truncate to 64 chars if needed
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        return sanitized

    async def step(self):
        messages = self.messages
        task = self.task
        if self.enable_prompt_caching:
            self._inject_prompt_caching()
            self.image_truncation_threshold = 1
            system = [{
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}
            }]
        if self.only_n_most_recent_images:
            self._maybe_filter_to_n_most_recent_images()
        try:
            self.tool_collection.to_params()
            truncated_messages = [
                {"role": msg["role"], "content": truncate_message_content(msg["content"])}
                for msg in messages
                ]   
            summary_task = asyncio.create_task(summarize_recent_messages(messages[-4:], self.display))
            ic(f"NUMBER_OF_MESSAGES: {len(messages)}")

            # --- MAIN LLM CALL ---
            response = None  # Initialize response to avoid UnboundLocalError
            try:
                response = self.client.beta.messages.create(
                    max_tokens=MAX_SUMMARY_TOKENS,
                    messages=truncated_messages,
                    model=MAIN_MODEL,
                    tool_choice={"type": "auto"},
                    system=system,
                    tools=self.tool_collection.to_params(),
                    betas=self.betas,
                )
            except Exception as llm_error:
                self.display.add_message("assistant", f"LLM call failed: {str(llm_error)}")
                last_3_messages = messages[-3:] if len(messages) >= 3 else messages
                new_context = await refresh_context_async(task, last_3_messages, self.display)
                self.messages = [{"role": "user", "content": new_context}]
                self.context_recently_refreshed = True

            response_params = []
            if response is not None:
                for block in response.content:
                    if hasattr(block, 'text'):
                        response_params.append({"type": "text", "text": block.text})
                        self.display.add_message("assistant", block.text)
                    elif getattr(block, 'type', None) == "tool_use":
                        sanitized_name = self._sanitize_tool_name(block.name)
                        response_params.append({
                            "type": "tool_use",
                            "name": sanitized_name,
                            "id": block.id,
                            "input": block.input
                        })
            else:
                self.display.add_message("assistant", "LLM response was None.")
                response_params = []

            self.messages.append({"role": "assistant", "content": response_params})
            ic(f"NUNBER_OF_MESSAGES: {len(messages)}")

            with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
                message_string = format_messages_to_string(messages)
                f.write(message_string)

            tool_result_content: List[BetaToolResultBlockParam] = []
            for content_block in response_params:
                self.output_manager.format_content_block(content_block)
                if content_block["type"] == "tool_use":
                    tool_result = await self.run_tool(content_block)
                    tool_result_content.append(tool_result)

            self.display.add_message("user", f"{self.refresh_count+2 - len(messages)} More Messages Until Context Refresh: Currently {len(messages)} of {self.refresh_count}")
            quick_summary = await summary_task  # Now we wait for the summary to complete
            add_summary(quick_summary)
            self.display.add_message("assistant", quick_summary)

            if (not tool_result_content) and (not self.context_recently_refreshed):
                self.display.add_message("assistant", "Awaiting User Input ⌨️ (Type your response in the web interface)")
                user_input = await self.display.wait_for_user_input()
                self.display.add_message("assistant",f"The user has said '{user_input}'")
                if user_input.lower() in ["no", "n"]:# or interupt_counter > 4:
                    return False
                else:
                    self.messages.append({"role": "user", "content": user_input})
                    last_3_messages = messages[-4:]
                    new_context = await refresh_context_async(task, last_3_messages, self.display)
                    self.context_recently_refreshed = True
                    self.messages =[{"role": "user", "content": new_context}]
            else:
                if (len(messages) > self.refresh_count):
                    last_3_messages = messages[-4:]
                    self.display.add_message("user", "refreshing")
                    new_context = await refresh_context_async(task, last_3_messages, self.display)
                    self.context_recently_refreshed = True
                    self.messages =[{"role": "user", "content": new_context}]
                    self.refresh_count += 2

            if self.display.user_interupt:
                last_3_messages = messages[-4:]
                self.display.add_message("assistant", "Awaiting User Input JK!⌨️ (Type your response in the web interface)")
                user_input = await self.display.wait_for_user_input()
                new_context = await refresh_context_async(task, last_3_messages)
                new_context = new_context + user_input
                self.messages =[{"role": "user", "content": new_context}]
                self.refresh_count += 2
                self.context_recently_refreshed = True
                self.display.user_interupt = False
            else:
                self.context_recently_refreshed = False

            # Only update token tracker if response is not None
            if response is not None and hasattr(response, 'usage'):
                self.token_tracker.update(response)
                # self.token_tracker.display(self.display)
            else:
                self.display.add_message("assistant", "No valid LLM response for token usage update.")
            return True

        except Exception as e:
            ic(f"Error in sampling loop: {str(e).encode('ascii', errors='replace').decode('ascii')}")
            ic(f"The error occurred at the following message: {messages[-1]} and line: {e.__traceback__.tb_lineno}")
            ic(e.__traceback__.tb_frame.f_locals)
            self.display.add_message("user", ("Error", str(e)))
            raise
