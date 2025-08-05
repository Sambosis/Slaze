from typing import Any, Dict, List, Union
from pathlib import Path
from datetime import datetime

import os
from utils.web_ui import WebUI
from utils.agent_display_console import AgentDisplayConsole
# from config import write_to_file # Removed as it was for ic
# Removed: from load_constants import *
from config import MAIN_MODEL, get_constant, googlepro # Import get_constant
from utils.file_logger import aggregate_file_states
from openai import OpenAI
import logging
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
# from icecream import ic # Removed
# from rich import print as rr # Removed

# ic.configureOutput(includeContext=True, outputFunction=write_to_file) # Removed

logger = logging.getLogger(__name__)

QUICK_SUMMARIES = []





def format_messages_to_string(messages):
    """Return a human readable string for a list of messages."""

    def _val(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    try:
        output_pieces = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            output_pieces.append(f"\n{role}:")

            if "tool_call_id" in msg:
                output_pieces.append(f"\nTool Call ID: {msg['tool_call_id']}")

            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = _val(_val(tc, "function"), "name")
                    args = _val(_val(tc, "function"), "arguments")
                    tc_id = _val(tc, "id")
                    output_pieces.append(
                        f"\nTool Call -> {name or 'unknown'} (ID: {tc_id or 'n/a'})"
                    )
                    if args:
                        try:
                            parsed = json.loads(args) if isinstance(args, str) else args
                            formatted = json.dumps(parsed, indent=2)
                        except Exception:
                            formatted = str(args)
                        output_pieces.append(f"\nArguments: {formatted}")

            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "text":
                            output_pieces.append(f"\n{block.get('text', '')}")
                        elif btype == "image":
                            output_pieces.append("\n[Image content omitted]")
                        elif btype == "tool_use":
                            output_pieces.append(f"\nTool Call: {block.get('name')}")
                            if "input" in block:
                                inp = block["input"]
                                if isinstance(inp, (dict, list)):
                                    output_pieces.append(
                                        f"\nInput: {json.dumps(inp, indent=2)}"
                                    )
                                else:
                                    output_pieces.append(f"\nInput: {inp}")
                        elif btype == "tool_result":
                            output_pieces.append(
                                f"\nTool Result [ID: {block.get('tool_use_id', 'unknown')}]"
                            )
                            if block.get("is_error"):
                                output_pieces.append("\nError: True")
                            for item in block.get("content", []):
                                if item.get("type") == "text":
                                    output_pieces.append(f"\n{item.get('text', '')}")
                                elif item.get("type") == "image":
                                    output_pieces.append("\n[Image content omitted]")
                        else:
                            for key, value in block.items():
                                if key == "cache_control":
                                    continue
                                output_pieces.append(f"\n{key}: {value}")
                    else:
                        output_pieces.append(f"\n{block}")
            elif content is not None:
                output_pieces.append(f"\n{content}")

            output_pieces.append("\n" + "-" * 80)

        return "".join(output_pieces)
    except Exception as e:
        return f"Error during formatting: {str(e)}"





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


def extract_text_from_content(content: Any) -> str:
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








def get_all_summaries() -> str:
    """Combine all summaries into a chronological narrative."""
    if not QUICK_SUMMARIES:
        return "No summaries available yet."

    combined = "\n"
    for entry in QUICK_SUMMARIES:
        combined += f"{entry}\n"
    return combined


@retry(
    stop=stop_after_attempt(max_attempt_number=5),
    wait=wait_random_exponential(multiplier=2, min=4, max=10),
)
async def refresh_context_async(
    task: str, messages: List[Dict], display: Union[WebUI, AgentDisplayConsole], client
) -> str:
    """
    Create a combined context string by filtering and (if needed) summarizing messages
    and appending current file contents.
    
    Improvements:
    - Better error handling and recovery
    - More efficient content processing
    - Intelligent content prioritization
    - Single LLM call for better performance
    """
    try:
        # Step 1: Gather all context data
        logger.info("Starting context refresh - gathering data...")
        
        # Filter messages and get summaries
        filtered = filter_messages(messages)
        summary = get_all_summaries()
        
        # Get file and code information
        file_contents = aggregate_file_states()
        
        # Get code skeletons with fallback logic
        from utils.file_logger import get_all_current_skeleton, get_all_current_code
        code_skeletons = get_all_current_code() or get_all_current_skeleton()
        if not code_skeletons or code_skeletons == "No Python files have been tracked yet.":
            code_skeletons = "No code skeletons available."
        
        # Extract image information
        images_info = _extract_images_info(file_contents)
        
        # Step 2: Intelligent content prioritization and truncation
        prioritized_content = _prioritize_and_truncate_content(
            file_contents, code_skeletons, images_info
        )
        
        # Step 3: Single comprehensive LLM call
        logger.info("Making single LLM call for context reorganization and task update...")
        combined_prompt = _build_comprehensive_prompt(
            task, filtered, summary, prioritized_content, images_info
        )
        
        response = await _make_llm_call_with_retry(client, combined_prompt)
        
        # Step 4: Parse response and build final context
        parsed_response = _parse_llm_response(response)
        final_context = _build_final_context(
            task, parsed_response, prioritized_content
        )
        
        logger.info(f"Context refresh completed successfully. Final context length: {len(final_context)} chars")
        return final_context
        
    except Exception as e:
        logger.error(f"Error in refresh_context_async: {str(e)}", exc_info=True)
        # Fallback: return a minimal context to keep the system running
        return _build_fallback_context(task, messages, file_contents if 'file_contents' in locals() else "")


def _extract_images_info(file_contents: str) -> str:
    """Extract image information from file contents."""
    if "## Generated Images:" not in file_contents:
        return ""
    
    images_section = file_contents.split("## Generated Images:")[1]
    if "##" in images_section:
        images_section = images_section.split("##")[0]
    return "## Generated Images:\n" + images_section.strip()


def _prioritize_and_truncate_content(file_contents: str, code_skeletons: str, images_info: str) -> Dict[str, str]:
    """
    Intelligently prioritize and truncate content based on importance.
    Returns a dictionary with prioritized content sections.
    """
    MAX_TOTAL_CHARS = 150000  # Reduced from 200k for better performance
    MAX_FILE_CONTENT_CHARS = 100000
    MAX_CODE_SKELETON_CHARS = 30000
    MAX_IMAGES_CHARS = 20000
    
    # Prioritize recent files and important content
    truncated_files = file_contents
    if len(file_contents) > MAX_FILE_CONTENT_CHARS:
        # Smart truncation: keep beginning and end, indicate truncation
        keep_start = MAX_FILE_CONTENT_CHARS // 2
        keep_end = MAX_FILE_CONTENT_CHARS // 2
        truncated_files = (
            file_contents[:keep_start] + 
            f"\n\n... [TRUNCATED {len(file_contents) - MAX_FILE_CONTENT_CHARS} characters] ...\n\n" + 
            file_contents[-keep_end:]
        )
    
    # Truncate code skeletons if needed
    truncated_code = code_skeletons
    if len(code_skeletons) > MAX_CODE_SKELETON_CHARS:
        truncated_code = code_skeletons[:MAX_CODE_SKELETON_CHARS] + "\n... [CODE TRUNCATED] ..."
    
    # Truncate images info if needed
    truncated_images = images_info
    if len(images_info) > MAX_IMAGES_CHARS:
        truncated_images = images_info[:MAX_IMAGES_CHARS] + "\n... [IMAGES TRUNCATED] ..."
    
    return {
        "files": truncated_files,
        "code": truncated_code,
        "images": truncated_images
    }


def _build_comprehensive_prompt(task: str, filtered_messages: List[Dict], summary: str, 
                               content: Dict[str, str], images_info: str) -> str:
    """Build a single comprehensive prompt for both context reorganization and task update."""
    
    # Build conversation text from filtered messages
    conversation_text = ""
    for msg in filtered_messages:
        role = msg["role"].upper()
        if isinstance(msg["content"], list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    conversation_text += f"\n{role}: {block.get('text', '')}"
        else:
            conversation_text += f"\n{role}: {msg['content']}"
    
    return f"""You are helping to refresh the context for an ongoing project. Your task is to:
1. Analyze what has been completed so far
2. Identify the next steps needed
3. Provide an updated, comprehensive task description

VERY IMPORTANT INSTRUCTIONS:
- This is a CONTINUING PROJECT with existing files and completed work
- ALL files mentioned as completed ARE ALREADY CREATED AND FUNCTIONAL
- Do NOT suggest recreating existing files or redoing completed steps
- Focus on what needs to be done next to complete the project

ORIGINAL TASK:
{task}

CONVERSATION HISTORY:
{conversation_text}

PREVIOUS SUMMARIES:
{summary}

CURRENT PROJECT STATE:
{content['files']}

CODE STRUCTURE:
{content['code']}

{content['images']}

Please provide your response in this exact format:

<COMPLETED>
[List all tasks/steps that have been successfully completed - these are DONE and exist]
</COMPLETED>

<NEXT_STEPS>
[Numbered list of 1-4 specific next steps to complete the project]
</NEXT_STEPS>

<UPDATED_TASK>
[Updated task description that incorporates lessons learned, current state, and clear guidance on imports and file organization]
</UPDATED_TASK>

Make sure your updated task includes:
- What has been accomplished
- What still needs to be done
- Tips and lessons learned
- Clear guidance on how files should work together
- Import instructions and dependencies
"""


async def _make_llm_call_with_retry(client, prompt: str, max_retries: int = 3) -> str:
    """Make LLM call with built-in retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MAIN_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=get_constant("MAX_SUMMARY_TOKENS", 20000)
            )
            
            if not response or not response.choices:
                raise ValueError("No response received from LLM")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from LLM")
            
            return content
            
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            # Wait before retry
            import asyncio
            await asyncio.sleep(2 ** attempt)


def _parse_llm_response(response: str) -> Dict[str, str]:
    """Parse the LLM response into structured components."""
    def extract_section(text: str, start_tag: str, end_tag: str) -> str:
        if start_tag in text and end_tag in text:
            return text[text.find(start_tag) + len(start_tag):text.find(end_tag)].strip()
        return ""
    
    return {
        "completed": extract_section(response, "<COMPLETED>", "</COMPLETED>") or "No completed items found.",
        "next_steps": extract_section(response, "<NEXT_STEPS>", "</NEXT_STEPS>") or "No next steps found.",
        "updated_task": extract_section(response, "<UPDATED_TASK>", "</UPDATED_TASK>") or "Task update not available."
    }


def _build_final_context(task: str, parsed_response: Dict[str, str], content: Dict[str, str]) -> str:
    """Build the final context string."""
    return f"""Original request: 
{task}

IMPORTANT: This is a CONTINUING PROJECT. All files listed below ALREADY EXIST and are FULLY FUNCTIONAL.
DO NOT recreate any existing files or redo completed steps. Continue the work from where it left off.

Current Project Files and Assets:
{content['files']}

Code Skeletons (Structure of Python files):
{content['code']}

COMPLETED STEPS (These have ALL been successfully completed - DO NOT redo these):
{parsed_response['completed']}

NEXT STEPS (Continue the project by completing these):
{parsed_response['next_steps']}

{content['images']}

Updated Request:
{parsed_response['updated_task']}

NOTES: 
- All files mentioned in completed steps ALREADY EXIST in the locations specified.
- All completed steps have ALREADY BEEN DONE successfully.
- Continue the project by implementing the next steps, building on the existing work.
"""


def _build_fallback_context(task: str, messages: List[Dict], file_contents: str) -> str:
    """Build a minimal fallback context when the main process fails."""
    recent_messages = messages[-5:] if len(messages) > 5 else messages
    message_text = ""
    for msg in recent_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = extract_text_from_content(content)
        message_text += f"{role}: {content}\n"
    
    truncated_files = file_contents[:50000] if len(file_contents) > 50000 else file_contents
    
    return f"""FALLBACK CONTEXT - Limited information available due to processing error.

Original Task: {task}

Recent Messages:
{message_text}

Current Files (truncated):
{truncated_files}

Note: This is a fallback context. Some information may be incomplete.
Please continue with the available context or request a context refresh.
"""