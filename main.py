# main.py is the entry point for the application. It starts the server and opens the browser to the correct page.

import asyncio
import base64
import os
import shutil
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
    EditTool,
    PictureGenerationTool,
    ToolCollection,
    ToolResult
)

from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from utils.file_logger import *
from utils.context_helpers import *
from utils.output_manager import *
from config import *  # Make sure config.py defines the constants
from agent import Agent
from min_agent import lmin_agent
write_constants_to_file()
load_dotenv()
install()
ic.configureOutput(includeContext=True, outputFunction=write_to_file)

def archive_logs():
    """Archive all log files in LOGS_DIR by moving them to an archive folder with a timestamp."""
    try:
        # Create timestamp for the archive folder
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archive_dir = Path(LOGS_DIR, "archive", timestamp)
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all files in LOGS_DIR
        log_path = Path(LOGS_DIR)
        log_files = [f for f in log_path.iterdir() if f.is_file()]
        
        # Skip archiving if there are no files
        if not log_files:
            return "No log files to archive"
        
        # Move each file to the archive directory
        for file_path in log_files:
            # Skip archive directory itself
            if "archive" in str(file_path):
                continue
                
            # Create destination path
            dest_path = Path(archive_dir, file_path.name)
            
            # Copy the file if it exists (some might be created later)
            if file_path.exists():
                shutil.copy2(file_path, dest_path)
                
                # Clear the original file but keep it
                with open(file_path, 'w') as f:
                    f.write('')
        
        return f"Archived {len(log_files)} log files to {archive_dir}"
    except Exception as e:
        return f"Error archiving files: {str(e)}"

# Global for quick summaries
quick_summary= []
filename = ""
ic.configureOutput(includeContext=True, outputFunction=write_to_file)
# ──────────────────────────────────────────────────────────────────────────────

# Archive all existing logs
archive_logs()

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
    agent: Agent,
    max_tokens: int = 180000,
    ) -> List[BetaMessageParam]:
    """Main loop for agentic sampling."""
    running = True
    while running:
        try:
            running = await agent.step()
            # response = await lmin_agent()
        except UnicodeEncodeError as ue:
            ic(f"UnicodeEncodeError: {ue}")
            rr(f"Unicode encoding error: {ue}")
            rr(f"ascii: {ue.args[1].encode('ascii', errors='replace').decode('ascii')}")
            break
        except Exception as e:
            ic(f"Error in sampling loop: {str(e).encode('ascii', errors='replace').decode('ascii')}")
            ic(f"The error occurred at the following message: {agent.messages[-1]} and line: {e.__traceback__.tb_lineno}")
            ic(e.__traceback__.tb_frame.f_locals)
            agent.display.add_message("user", ("Error", str(e)))
            raise
    return agent.messages

async def run_sampling_loop(task: str, display: AgentDisplayWebWithPrompt) -> List[BetaMessageParam]:
    """Run the sampling loop with clean output handling."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    # set the task as a constant in config
    agent = Agent(task=task, display=display)
    agent.messages.append({"role": "user", "content": task})
    messages = await sampling_loop(
        agent=agent,
        max_tokens=28000,
    )
    return messages

async def main_async():
    # Create event loop first
    loop = asyncio.get_event_loop()
    
    # Create display with the loop
    app, socketio = AgentDisplayWebWithPrompt.create_app(loop=loop) # MODIFY THIS LINE
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
