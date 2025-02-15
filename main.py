import asyncio
import base64
import dis
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from re import U
from typing import Any, Callable, Dict, List, Optional, cast
import webbrowser
from openai import OpenAI
from config import *
write_constants_to_file()

import ftfy
from anthropic import Anthropic, APIResponse
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlock,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)
from dotenv import load_dotenv
from icecream import ic, install
from pyautogui import write
from rich import print as rr
from rich.prompt import Prompt, Confirm
from tools import (
    BashTool,
    EditTool,
    GetExpertOpinionTool,
    WindowsNavigationTool,
    # ToolError,
    WebNavigatorTool,
    ProjectSetupTool,
    WriteCodeTool,
    PictureGenerationTool,
)
from tools import (
    ToolCollection,
    ToolResult
)
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.text import Text
from rich import box
from rich.table import Table
from queue import Queue
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt

from utils.output_manager import OutputManager
import argparse
load_dotenv()
install()

with open(SYSTEM_PROMPT_FILE, 'r', encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
# Global for quick summaries
QUICK_SUMMARIES = []
filename = ""
ic.configureOutput(includeContext=True, outputFunction=write_to_file)
# ──────────────────────────────────────────────────────────────────────────────
def archive_file(file_path):
    """Archive a file by appending moving it to an archive folder with a timestamp."""
    try:
        # Get the filename and extension
        file_path = Path(file_path)
        filename = file_path.stem
        extension = file_path.suffix
        # Create the archive directory if it doesn't exist
        archive_dir = Path(LOGS_DIR, "archive")
        archive_dir.mkdir(parents=True, exist_ok=True)
        # Create the new path with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_path = Path(archive_dir, f"{filename}_{timestamp}{extension}")
        # Move the file to the archive directory
        file_path.rename(new_path)
        return new_path
    except Exception as e:
        return f"Error archiving file: {str(e)}"

archive_file(ICECREAM_OUTPUT_FILE)
archive_file(LOG_FILE)
archive_file(MESSAGES_FILE)
archive_file(CODE_FILE)
archive_file(USER_LOG_FILE)
archive_file(ASSISTANT_LOG_FILE)
archive_file(TOOL_LOG_FILE)

# ──────────────────────────────────────────────────────────────────────────────
# Context Reduction Functions
# ──────────────────────────────────────────────────────────────────────────────

def filter_messages(messages: List[Dict]) -> List[Dict]:
    """
    Keep only messages with role 'user' or 'assistant'.
    Also keep any tool_result messages that contain errors.
    """
    keep_roles = {"user", "assistant"}
    filtered = []
    for msg in messages:
        if msg.get("role") in keep_roles:
            filtered.append(msg)
        elif isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    # Check if any text in the tool result indicates an error
                    text = ""
                    for item in block.get("content", []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            text += item.get("text", "")
                    if "error" in text.lower():
                        filtered.append(msg)
                        break
    return filtered


async def reorganize_context(messages: List[BetaMessageParam], summary: str) -> str:
    """ Reorganize the context by filtering and summarizing messages. """
    conversation_text = ""
    for msg in messages:
        role = msg['role'].upper()
        if isinstance(msg['content'], list):
            for block in msg['content']:
                if isinstance(block, dict):
                    if block.get('type') == 'text':
                        conversation_text += f"\n{role}: {block.get('text', '')}"
                    elif block.get('type') == 'tool_result':
                        for item in block.get('content', []):
                            if item.get('type') == 'text':
                                conversation_text += f"\n{role} (Tool Result): {item.get('text', '')}"
        else:
            conversation_text += f"\n{role}: {msg['content']}"
        
    summary_prompt = f"""I need a 2 part response from you. The first part of the response is to list everything that has been done already. 
    You will be given a lot of context, mucch of which is repetitive, so you are only to list each thing done one time.
    For each thing done (or attempted) list if it worked or not, and if not, why it didn't work.
    You will probably be able to infer from the section labeled <NARATIVE>  </NARATIVE> what has been done. 
    However you will need the info the section labeled <MESSAGES>   </MESSAGES> in order to figure out why it didn't work. 
    You are to response to this needs to be inclosed in XML style tags called <COMPLETED>   </COMPETED>
   
    The second part of the response is to list the steps that need to be taken to complete the task.
    You will need to take the whole context into account in order to    figure out what needs to be done.
    You must list between 0 (you have deemed the task complete) and 4 steps that need to be taken to complete the task.
    You should try to devise a plan that uses the least possible steps to compete the task, while insuring that you complete thetask
    Your response to this needs to be enclosed in XML style tags called <STEPS>   </STEPS>
    Please make sure your steps are clear, concise, and in a logical order and actionable. 
    Number them 1 through 4 in the order they should be done.
    Here is the Narative part:
    <NARATIVE>
    {summary}
    </NARATIVE>
    Here is the messages part:
    <MESSAGES>
    {conversation_text}
    </MESSAGES>

    Remeber to the formats request and to enclose your responses in <COMPLETED>  </COMPLETED> and <STEPS>  </STEPS> tags respectively.
    """
    sum_client = OpenAI()
    model = "o3-mini"
    response = sum_client.chat.completions.create(
        model=model,
        max_completion_tokens=30000,
        messages=[{
            "role": "user",
            "content": summary_prompt
        }]
    )
    summary = response.choices[0].message.content
    start_tag = "<COMPLETED>"
    end_tag = "</COMPLETED>"
    if start_tag in summary and end_tag in summary:
        completed_items = summary[summary.find(start_tag)+len(start_tag):summary.find(end_tag)]
    else:
        completed_items = "No completed items found."
    start_tag = "<STEPS>"
    end_tag = "</STEPS>"
    if start_tag in summary and end_tag in summary:
        steps = summary[summary.find(start_tag)+len(start_tag):summary.find(end_tag)]
    else:
        steps = "No steps found."
    return completed_items, steps

def aggregate_file_states() -> str:
    """
    Aggregate file states. Here we reuse extract_files_content.
    """
    return extract_files_content()

async def refresh_context_async(task: str, messages: List[Dict], display: AgentDisplayWebWithPrompt) -> str:
    """
    Create a combined context string by filtering and (if needed) summarizing messages
    and appending current file contents.
    """
    filtered = filter_messages(messages)
    summary = get_all_summaries()
    last4_messages = format_messages_to_string(messages[-4:])
    completed, steps = await reorganize_context(filtered, summary)

    file_contents = aggregate_file_states()
    combined_content = f"""Original request: {task}
    Current Completed Project Files:
    {file_contents}
    Most Recent activity:
    {last4_messages}
    List if things we've done and what has worked or not:
    {completed}
    ToDo List of tasks in order to complete the task:
    {steps}
    """
    return combined_content

# ──────────────────────────────────────────────────────────────────────────────
def _make_api_tool_result(result: ToolResult, tool_use_id: str) -> Dict:
    """Create a tool result dictionary with proper error handling."""
    tool_result_content = []
    is_error = False

    if result is None:
        is_error = True
        tool_result_content.append({
            "type": "text", 
            "text": "Tool execution resulted in None"
        })
    elif isinstance(result, str):
        is_error = True
        tool_result_content.append({
            "type": "text", 
            "text": result
        })
    elif hasattr(result, 'output') and result.output:
        tool_result_content.append({
            "type": "text", 
            "text": result.output
        })
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

def format_messages_to_restart(messages):
    """
    Format a list of messages into a formatted string.
    """
    try:
        output_pieces = []
        for msg in messages:
            output_pieces.append(f"\n{msg['role'].upper()}:")
            if isinstance(msg["content"], list):
                for content_block in msg["content"]:
                    if isinstance(content_block, dict):
                        if content_block.get("type") == "tool_result":
                            output_pieces.append(
                                f"\nResult:"  
                            )
                            for item in content_block.get("content", []):
                                if item.get("type") == "text":
                                    output_pieces.append(f"\n{item.get('text')}")
                        else:
                            for key, value in content_block.items():
                                output_pieces.append(f"\n{value}")
                    else:
                        output_pieces.append(f"\n{content_block}")
            else:
                output_pieces.append(f"\n{msg['content']}")
            output_pieces.append("\n" + "-" * 80)
        return "".join(output_pieces)
    except Exception as e:
        return f"Error during formatting: {str(e)}"


def format_messages_to_string(messages):
    """
    Format a list of messages into a formatted string.
    """
    try:
        output_pieces = []
        for msg in messages:
            output_pieces.append(f"\n{msg['role'].upper()}:")
            if isinstance(msg["content"], list):
                for content_block in msg["content"]:
                    if isinstance(content_block, dict):
                        if content_block.get("type") == "tool_result":
                            output_pieces.append(
                                f"\nTool Result [ID: {content_block.get('name', 'unknown')}]:"  
                            )
                            for item in content_block.get("content", []):
                                if item.get("type") == "text":
                                    output_pieces.append(f"\nText: {item.get('text')}")
                                elif item.get("type") == "image":
                                    output_pieces.append("\nImage Source: base64 source too big")
                        else:
                            for key, value in content_block.items():
                                output_pieces.append(f"\n{key}: {value}")
                    else:
                        output_pieces.append(f"\n{content_block}")
            else:
                output_pieces.append(f"\n{msg['content']}")
            output_pieces.append("\n" + "-" * 80)
        return "".join(output_pieces)
    except Exception as e:
        return f"Error during formatting: {str(e)}"


class TokenTracker:
    def __init__(self, display: AgentDisplayWebWithPrompt):
        self.total_cache_creation = 0
        self.total_cache_retrieval = 0
        self.total_input = 0
        self.total_output = 0
        self.recent_cache_creation = 0
        self.recent_cache_retrieval = 0
        self.recent_input = 0
        self.recent_output = 0
        self.displayA = display

    def update(self, response):
        self.recent_cache_creation = response.usage.cache_creation_input_tokens
        self.recent_cache_retrieval = response.usage.cache_read_input_tokens
        self.recent_input = response.usage.input_tokens
        self.recent_output = response.usage.output_tokens
        
        self.total_cache_creation += self.recent_cache_creation
        self.total_cache_retrieval += self.recent_cache_retrieval
        self.total_input += self.recent_input
        self.total_output += self.recent_output

    def display(self, displayA: AgentDisplayWebWithPrompt):
        recent_usage = [
            "Recent Token Usage 📊",
            f"Recent Cache Creation: {self.recent_cache_creation:,}",
            f"Recent Cache Retrieval: {self.recent_cache_retrieval:,}",
            f"Recent Input: {self.recent_input:,}",
            f"Recent Output: {self.recent_output:,}",
            f"Recent Total: {self.recent_cache_creation + self.recent_cache_retrieval + self.recent_input + self.recent_output:,}",
        ]
        total_cost = (self.total_cache_creation * 3.75 + self.total_cache_retrieval * 0.30 + self.total_input * 3 + self.total_output * 15) / 1_000_000
        total_usage = [
            "Total Token Usage 📈",
            f"Total Cache Creation: {self.total_cache_creation:,}",
            f"Total Cache Retrieval: {self.total_cache_retrieval:,}",
            f"Total Output: {self.total_output:,}",
            f"Total Tokens: {self.total_cache_creation + self.total_cache_retrieval + self.total_input + self.total_output:,} with a total cost of ${total_cost:.2f} USD.",
        ]
        token_display = f"\n{total_usage}"
        self.displayA.add_message("user", token_display)


def extract_files_content() -> str:
    """ Returns the complete contents of all files with special handling for images """
    LOG_FILE = Path(get_constant('LOG_FILE'))
    # Create the file if it does not exist
    if not LOG_FILE.exists():
        LOG_FILE.touch()
    # Read the log file
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        # Handle case of empty file
        if f.read() == "":
            return "No code yet"
        f.seek(0)  # Reset file pointer to the beginning
        logs = json.loads(f.read())
    
    # Initialize output string
    output = []
    
    # Process each file in the logs
    for filepath in logs.keys():
        try:
            # Convert Windows path to Path object
            path = Path(filepath)
            
            # Skip if file doesn't exist
            if not path.exists():
                continue
            
            # Add file header
            output.append(f"# filepath: {filepath}")
            
            # Check if it's an image file
            if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                output.append("[Image File]")
                output.append(f"Created: {logs[filepath]['created_at']}")
                output.append(f"Last Operation: {logs[filepath]['operations'][-1]['operation']}")
            else:
                # For non-image files, include the content
                content = path.read_text(encoding='utf-8')
                output.append(content)
            
            output.append("\n" + "=" * 80 + "\n")  # Separator between files
            
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue
    
    # Combine all content
    return "\n".join(output)


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "tool_result":
                    for sub_item in item.get("content", []):
                        if sub_item.get("type") == "text":
                            text_parts.append(sub_item.get("text", ""))
        return " ".join(text_parts)
    return ""


def truncate_message_content(content: Any, max_length: int = 900_000) -> Any:
    if isinstance(content, str):
        return content[:max_length]
    elif isinstance(content, list):
        return [truncate_message_content(item, max_length) for item in content]
    elif isinstance(content, dict):
        return {k: truncate_message_content(v, max_length) if k != 'source' else v
                for k, v in content.items()}
    return content


def add_summary(summary: str) -> None:
    """Add a new summary to the global list with timestamp."""
    QUICK_SUMMARIES.append(summary.strip())


def get_all_summaries() -> str:
    """Combine all summaries into a chronological narrative."""
    if not QUICK_SUMMARIES:
        return "No summaries available yet."
    
    combined = "\n"
    for entry in QUICK_SUMMARIES:
        combined += f"{entry}\n"
    return combined



def _inject_prompt_caching(messages: List[BetaMessageParam]):
    breakpoints_remaining = 2
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(content := message["content"], list):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                break


def _maybe_filter_to_n_most_recent_images(
    messages: List[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int
    ):
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        List[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

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


async def summarize_recent_messages(short_messages: List[BetaMessageParam], display: AgentDisplayWebWithPrompt) -> str:    # sum_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    sum_client = OpenAI()
    model = "o3-mini"
    conversation_text = ""
    for msg in short_messages:
        role = msg['role'].upper()
        if isinstance(msg['content'], list):
            for block in msg['content']:
                if isinstance(block, dict):
                    if block.get('type') == 'text':
                        conversation_text += f"\n{role}: {block.get('text', '')}"
                    elif block.get('type') == 'tool_result':
                        for item in block.get('content', []):
                            if item.get('type') == 'text':
                                conversation_text += f"\n{role} (Tool Result): {item.get('text', '')}"
        else:
            conversation_text += f"\n{role}: {msg['content']}"

    summary_prompt = f"""Please provide a concise casual natural language summary of the messages. 
        They are the actual LLM messages log of interaction and you will provide between 3 and 5 conversational style sentences informing someone what was done. 
        Focus on the actions taken and provide the names of any files, functions, directories, or paths mentioned and a basic idea of what was done and why. 
        Your response should be in HTML format that would make it easy for the reader to understand and view the summary.
        Messages to summarize:
        {conversation_text}"""
    messages_prompt = [
    {
          "role": "user",
          "content": [
                    {
                        
                    "type": "text",
                    "text": summary_prompt
                    },
                ]
                }
            ]
    ic(messages_prompt)
    completion = sum_client.chat.completions.create(
                model=model,
                messages=messages_prompt)
    ic(completion)
    # response = sum_client.messages.create(
    #     model=SUMMARY_MODEL,
    #     max_tokens=MAX_SUMMARY_TOKENS,
    #     messages=[{
    #         "role": "user",
    #         "content": summary_prompt
    #     }]
    # )
    summary = completion.choices[0].message.content
    return summary

async def sampling_loop(*, model: str, messages: List[BetaMessageParam], api_key: str, max_tokens: int = 8000, display: AgentDisplayWebWithPrompt) -> List[BetaMessageParam]:
    """Main loop for agentic sampling."""
    task = messages[0]['content']
    context_recently_refreshed = False
    refresh_count = 24

    interupt_counter = 0
    try:
        tool_collection = ToolCollection(
            WriteCodeTool(display=display),
            BashTool(display=display),
            EditTool(display=display),
            GetExpertOpinionTool(),
            WindowsNavigationTool(),
            WebNavigatorTool(),
            ProjectSetupTool(display=display),
            PictureGenerationTool(display=display),
            display=display
        )
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        display.add_message("user", tool_collection.get_tool_names_as_string())
        await asyncio.sleep(0.1) 

        system = BetaTextBlockParam(type="text", text=SYSTEM_PROMPT_FILE)
        output_manager = OutputManager(display)
        client = Anthropic(api_key=api_key)
        i = 0
        running = True
        token_tracker = TokenTracker(display)
        enable_prompt_caching = True
        betas = [COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG]
        image_truncation_threshold = 1
        only_n_most_recent_images = 2
        while running:
            i += 1
            with open(SYSTEM_PROMPT_FILE, 'r', encoding="utf-8") as f:
                SYSTEM_PROMPT = f.read()
            if enable_prompt_caching:
                _inject_prompt_caching(messages)
                image_truncation_threshold = 1
                system = [{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }]

            if only_n_most_recent_images:
                _maybe_filter_to_n_most_recent_images(
                    messages,
                    only_n_most_recent_images,
                    min_removal_threshold=image_truncation_threshold,
                )

            try:
                tool_collection.to_params()

                truncated_messages = [
                    {"role": msg["role"], "content": truncate_message_content(msg["content"])}
                    for msg in messages
                ]
                # --- START ASYNC SUMMARY ---
                summary_task = asyncio.create_task(summarize_recent_messages(messages[-4:], display))  # Create the task, but don't await it yet
                # quick_summary = await summarize_recent_messages(messages[-6:], display)
                # add_summary(quick_summary)
                # display.add_message("assistant", f"<p></p>{quick_summary}<p></p>")
                # await asyncio.sleep(0.1)
                ic(f"NUMBER_OF_MESSAGES: {len(messages)}")

                # --- MAIN LLM CALL ---
                response = client.beta.messages.create(
                    max_tokens=MAX_SUMMARY_TOKENS,
                    
                    messages=truncated_messages,
                    model=MAIN_MODEL,
                    system=system,
                    tools=tool_collection.to_params(),
                    betas=betas,
                    )
                response_params = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        response_params.append({"type": "text", "text": block.text})
                        display.add_message("assistant", block.text)
                    elif getattr(block, 'type', None) == "tool_use":
                        response_params.append({
                            "type": "tool_use",
                            "name": block.name,
                            "id": block.id,
                            "input": block.input
                        })
                messages.append({"role": "assistant", "content": response_params})
                ic(f"NUNBER_OF_MESSAGES: {len(messages)}")

                with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
                    message_string = format_messages_to_string(messages)
                    f.write(message_string)

                tool_result_content: List[BetaToolResultBlockParam] = []
                for content_block in response_params:
                    output_manager.format_content_block(content_block)
                    if content_block["type"] == "tool_use":
                        # display.add_message("user", f"Calling tool: {content_block['name']}")
                        result = ToolResult(output="Tool execution not started")
                        try:
                            ic(content_block['name'])
                            ic(content_block["input"])
                            result = await tool_collection.run(
                                name=content_block["name"],
                                tool_input=content_block["input"],
                            )
                            if result is None:
                                result = ToolResult(output="Tool execution failed with no result")
                        except Exception as e:
                            result = ToolResult(output=f"Tool execution failed: {str(e)}")
                        finally:
                            tool_result = _make_api_tool_result(result, content_block["id"])
                            ic(tool_result)
                            tool_result_content.append(tool_result)
                            tool_output = result.output if hasattr(result, 'output') else str(result)
                            combined_content = [{
                                "type": "tool_result",
                                "content": tool_result["content"],
                                "tool_use_id": tool_result["tool_use_id"],
                                "is_error": tool_result["is_error"]
                            }]
                            combined_content.append({
                                "type": "text",
                                "text": f"Tool '{content_block['name']}' was called with input: {json.dumps(content_block['input'])}.\nResult: {_extract_text_from_content(tool_output)}"
                            })
                            messages.append({
                                "role": "user",
                                "content": combined_content
                            })

                            await asyncio.sleep(0.2)

                            # --- NOW AWAIT AND DISPLAY SUMMARY ---
                ic(f"NUNBER_OF_MESSAGES: {len(messages)}")

                quick_summary = await summary_task  # Now we wait for the summary to complete
                add_summary(quick_summary)
                # display.add_message("assistant", f"<p></p>{quick_summary}")
                await asyncio.sleep(0.1)
                if (not tool_result_content):# and (not context_recently_refreshed):
                    display.add_message("assistant", "Awaiting User Input ⌨️ (Type your response in the web interface)")
                    interupt_counter += 1
                    user_input = await display.wait_for_user_input()
                    display.add_message("assistant",f"The user has said '{user_input}'")
                    if user_input.lower() in ["no", "n"]:# or interupt_counter > 4:
                        running = False
                    else:
                        messages.append({"role": "user", "content": user_input})
                        last_3_messages = messages[-4:]
                        new_context = await refresh_context_async(task, last_3_messages, display)
                        
                        messages =[{"role": "user", "content": new_context}]
                        # maybe extend with the last message. 
                else:
                    if len(messages) > refresh_count:
                        last_3_messages = messages[-3:]
                        display.add_message("assistant", "Awaiting User Input ⌨️ (Type your response in the web interface)")
                        new_context = await refresh_context_async(task, last_3_messages, display)
                        user_input = await display.wait_for_user_input()
                        new_context = f"{new_context}\n The user would like to give you this message:\n{user_input}"
                        display.add_message("assistant",f"The user would like to give you this message: '{user_input}'")

                        messages =[{"role": "user", "content": new_context}]
                        refresh_count += 1
                        context_recently_refreshed = True

                with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
                    message_string = format_messages_to_string(messages)
                    f.write(message_string)
                token_tracker.update(response)
                token_tracker.display(display)
                # messages_to_display = messages[-2:] if len(messages) > 1 else messages[-1:]
            except UnicodeEncodeError as ue:
                ic(f"UnicodeEncodeError: {ue}")
                rr(f"Unicode encoding error: {ue}")
                rr(f"ascii: {ue.args[1].encode('ascii', errors='replace').decode('ascii')}")
                break
            except Exception as e:
                ic(f"Error in sampling loop: {str(e).encode('ascii', errors='replace').decode('ascii')}")
                ic(f"The error occurred at the following message: {messages[-1]} and line: {e.__traceback__.tb_lineno}")
                ic(e.__traceback__.tb_frame.f_locals)
                display.add_message("user", ("Error", str(e)))
                raise
        return messages

    except Exception as e:
        ic(e.__traceback__.tb_lineno)
        ic(e.__traceback__.tb_lasti)
        ic(e.__traceback__.tb_frame.f_code.co_filename)
        ic(e.__traceback__.tb_frame)
        display.add_message("user", ("Initialization Error", str(e)))
        ic(f"Error initializing sampling loop: {str(e)}")
        raise



async def run_sampling_loop(task: str, display: AgentDisplayWebWithPrompt) -> List[BetaMessageParam]:
    """Run the sampling loop with clean output handling."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    messages = []
    if not api_key:
        raise ValueError("API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    messages.append({"role": "user", "content": task})
    # display.add_message("user", task)
    messages = await sampling_loop(
        model=MAIN_MODEL,  # Use MAIN_MODEL from config.py
        messages=messages,
        api_key=api_key,
        display=display
    )
    return messages


async def main_async():
    display = AgentDisplayWebWithPrompt()
    # Set the main event loop in the display.
    display.loop = asyncio.get_running_loop()
    display.start_server()  # Start the Flask/SocketIO server in a background thread.
    webbrowser.open("http://localhost:5000/select_prompt")
    print("Server started. Please use your browser to interact with the application.")
    while True:
        await asyncio.sleep(1)

def main():
    asyncio.run(main_async())
    display.socketio.stop()

if __name__ == "__main__":
    main()
