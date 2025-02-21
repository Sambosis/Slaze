# main.py 
import asyncio
import base64
import os
from datetime import datetime
from pathlib import Path
from typing import List
import webbrowser

import ftfy
from anthropic import Anthropic
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaMessageParam,
    BetaToolResultBlockParam,
)
from dotenv import load_dotenv
from icecream import ic, install

from tools import (
    BashTool,
    ProjectSetupTool,
    WriteCodeTool,
    PictureGenerationTool,
    ToolCollection,
    ToolResult
)

from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from utils.file_logger import *
from utils.context_helpers import *
from utils.output_manager import *
from config import *  # Make sure config.py defines the constants

write_constants_to_file()
load_dotenv()
install()
ic.configureOutput(includeContext=True, outputFunction=write_to_file)

def archive_file(file_path):
    """Archive a file by moving it to an archive folder with a timestamp."""
    try:
        file_path = Path(file_path)
        filename = file_path.stem
        extension = file_path.suffix
        archive_dir = Path(LOGS_DIR, "archive")
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_path = Path(archive_dir, f"{filename}_{timestamp}{extension}")
        file_path.rename(new_path)
        return new_path
    except Exception as e:
        return f"Error archiving file: {str(e)}"
# Global for quick summaries

filename = ""
ic.configureOutput(includeContext=True, outputFunction=write_to_file)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────

def generate_download_link(file_path):
    """Generate a download link for a given file."""
    try:
        file_path = Path(file_path)
        filename = file_path.name
        return f"/download/{filename}"
    except Exception as e:
        return f"Error generating download link: {str(e)}"

with open(SYSTEM_PROMPT_FILE, 'r', encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


def _make_api_tool_result(result: ToolResult, tool_use_id: str) -> Dict:
    """Create a tool result dictionary."""
    tool_result_content = []
    is_error = False
    if result is None:
        is_error = True
        tool_result_content.append({"type": "text", "text": "Tool execution resulted in None"})
    elif isinstance(result, str):
        is_error = True
        tool_result_content.append({"type": "text", "text": result})
    elif hasattr(result, 'output') and result.output:
        tool_result_content.append({"type": "text", "text": result.output})
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


class TokenTracker:
    def __init__(self, display):
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

    def display(self, displayA):
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


def _inject_prompt_caching(messages: List[BetaMessageParam]):
    breakpoints_remaining = 2
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(content := message["content"], list):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam({"type": "ephemeral"})
            else:
                content[-1].pop("cache_control", None)
                break


def _maybe_filter_to_n_most_recent_images(messages: List[BetaMessageParam], images_to_keep: int, min_removal_threshold: int):
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

# ----------------  The main Agent Loop ----------------
async def sampling_loop(
    *,
    model: str,
    messages: List[BetaMessageParam],
    api_key: str,
    max_tokens: int = 8000,
    display: AgentDisplayWebWithPrompt  # Keep this parameter
    ) -> List[BetaMessageParam]:
    """Main loop for agentic sampling."""
    task = messages[0]['content']
    context_recently_refreshed = False
    refresh_count = 10
    try:
        tool_collection = ToolCollection(
            WriteCodeTool(display=display),  # Use the passed-in display
            ProjectSetupTool(display=display), # Use the passed-in display
            BashTool(display=display),        # Use the passed-in display
            PictureGenerationTool(display=display), #Use the passed-in display
            display=display                   # Use the passed-in display
        )
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("")

        display.add_message("user", tool_collection.get_tool_names_as_string())
        await asyncio.sleep(0.1)

        system = BetaTextBlockParam(type="text", text=SYSTEM_PROMPT_FILE)
        output_manager = OutputManager(display) # and here.
        client = Anthropic(api_key=api_key)
        i = 0
        running = True
        token_tracker = TokenTracker(display) # Use passed in Display
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
                # summary_task = asyncio.create_task(summarize_recent_messages(messages[-4:], display))
                ic(f"NUMBER_OF_MESSAGES: {len(messages)}")

                # --- MAIN LLM CALL ---
                try:
                    response = client.beta.messages.create(
                        max_tokens=MAX_SUMMARY_TOKENS,
                        messages=truncated_messages,
                        model=MAIN_MODEL,
                        system=system,
                        tools=tool_collection.to_params(),
                        betas=betas,
                    )
                except Exception as llm_error:
                    display.add_message("assistant", f"LLM call failed, refreshing context: {str(llm_error)}")
                    # Get last few messages for context refresh
                    last_3_messages = messages[-3:] if len(messages) >= 3 else messages
                    new_context = await refresh_context_async(task, last_3_messages, display)
                    messages = [{"role": "user", "content": new_context}]
                    context_recently_refreshed = True
                    continue  # Skip the rest of this iteration and try again with refreshed context

                response_params = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        response_params.append({"type": "text", "text": block.text})
                        display.add_message("assistant", block.text)  # Use passed-in display
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
                                "text": f"Tool '{content_block['name']}' was called with input: {json.dumps(content_block['input'])}.\nResult: {extract_text_from_content(tool_output)}"
                            })
                            messages.append({
                                "role": "user",
                                "content": combined_content
                            })

                            await asyncio.sleep(0.2)

                            # --- NOW AWAIT AND DISPLAY SUMMARY ---
                ic(f"NUNBER_OF_MESSAGES: {len(messages)}")
                display.add_message("user", f"NUNBER_OF_MESSAGES: {len(messages)}")
                # quick_summary = await summary_task  # Now we wait for the summary to complete
                # add_summary(quick_summary)

                await asyncio.sleep(0.1)
                if (not tool_result_content) and (not context_recently_refreshed):
                    display.add_message("assistant", "Awaiting User Input ⌨️ (Type your response in the web interface)")
                    user_input = await display.wait_for_user_input()
                    display.add_message("assistant",f"The user has said '{user_input}'")
                    if user_input.lower() in ["no", "n"]:# or interupt_counter > 4:
                        running = False
                    else:
                        messages.append({"role": "user", "content": user_input})
                        last_3_messages = messages[-4:]
                        new_context = await refresh_context_async(task, last_3_messages, display)
                        context_recently_refreshed = True
                        messages =[{"role": "user", "content": new_context}]
                else:
                    if (len(messages) > refresh_count):
                        last_3_messages = messages[-3:]
                        display.add_message("user", "refreshing")
                        new_context = await refresh_context_async(task, last_3_messages, display)
                        context_recently_refreshed = True
                        messages =[{"role": "user", "content": new_context}]
                        refresh_count += 2

                if display.user_interupt:
                        last_3_messages = messages[-3:]
                        display.add_message("assistant", "Awaiting User Input JK!⌨️ (Type your response in the web interface)")
                        user_input = await display.wait_for_user_input()
                        new_context = await refresh_context_async(task, last_3_messages, display)
                        new_context = new_context + user_input
                        messages =[{"role": "user", "content": new_context}]
                        refresh_count += 2
                        context_recently_refreshed = True
                        display.user_interupt = False
                else:
                    context_recently_refreshed = False
                with open(MESSAGES_FILE, 'w', encoding='utf-8') as f:
                    message_string = format_messages_to_string(messages)
                    f.write(message_string)
                token_tracker.update(response)
                token_tracker.display(display) #pass in display
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
    messages = await sampling_loop(
        model=MAIN_MODEL,
        messages=messages,
        api_key=api_key,
        display=display  # Pass the display instance
    )
    return messages

async def main_async():
    # Create event loop first
    loop = asyncio.get_event_loop()
    
    # Create display with the loop
    display = AgentDisplayWebWithPrompt()
    display.loop = loop
    display.start_server()
    
    # Open browser after server starts
    webbrowser.open("http://localhost:5001/select_prompt")
    print("Server started. Please use your browser to interact with the application.")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")

def main():
    # Set the event loop policy for Windows if needed
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
