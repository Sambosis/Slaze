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


async def summarize_context_with_llm(messages: List[Dict[str, Any]], file_contents: str, code_skeletons: str) -> str:
    """Summarize the context using an LLM to produce a summary of completed steps and next steps."""
    conversation_text = ""

    # Look for tool results related to image generation
    image_generation_results = []

    for msg in messages:
        role = msg["role"].upper()
        if isinstance(msg["content"], list):
            for block in msg["content"]:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        conversation_text += f"\n{role}: {block.get('text', '')}"
                    elif block.get("type") == "tool_result":
                        # Track image generation results
                        if any(
                            "picture_generation" in str(item)
                            for item in block.get("content", [])
                        ):
                            for item in block.get("content", []):
                                if item.get(
                                    "type"
                                ) == "text" and "Generated image" in item.get(
                                    "text", ""
                                ):
                                    image_generation_results.append(
                                        item.get("text", "")
                                    )

                        for item in block.get("content", []):
                            if item.get("type") == "text":
                                conversation_text += (
                                    f"\n{role} (Tool Result): {item.get('text', '')}"
                                )
        else:
            conversation_text += f"\n{role}: {msg['content']}"

    # Add special section for image generation if we found any
    if image_generation_results:
        conversation_text += "\n\nIMAGE GENERATION RESULTS:\n" + "\n".join(
            image_generation_results
        )
    logger.debug(f"Conversation text for summarize_context_with_llm: {conversation_text[:500]}...") # Log snippet
    summary_prompt = f"""Your task is to analyze the state of an ongoing software development project and provide a concise summary.
Based on the conversation history, file contents, and code structure, generate two lists: one of completed tasks and one of the next immediate actions.

**INSTRUCTIONS:**
1.  **Analyze the provided context thoroughly.** The context includes the conversation history, the full contents of the project files, and the structure of the code.
2.  **Assume all files and completed steps mentioned are already done and exist.** Do not suggest re-doing them.
3.  **Your output MUST be in two distinct, clearly labeled sections:** `<COMPLETED>` and `<NEXT_STEPS>`.
4.  **Completed Section:** List all tangible achievements, such as created files, implemented features, and fixed bugs.
5.  **Next Steps Section:** List 1-4 specific, actionable steps required to move the project forward. These should be immediate, not long-term goals.
6.  **Be concise and factual.** Avoid conversational language.

**CONTEXT:**

**Project Files:**
```
{file_contents}
```

**Code Structure:**
```
{code_skeletons}
```

**Conversation History:**
```
{conversation_text}
```

**OUTPUT FORMAT:**

<COMPLETED>
[A bulleted list of all completed steps and created files.]
</COMPLETED>

<NEXT_STEPS>
[A numbered list of 1-4 concrete next steps.]
</NEXT_STEPS>
"""

    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        sum_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        model = MAIN_MODEL
        response = sum_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": summary_prompt}]
        )
        logger.debug(f"Reorganize context API response: {response}")
        if not response or not response.choices:
            raise ValueError("No response received from OpenRouter API")

        summary = response.choices[0].message.content
        logger.debug(f"Reorganized context summary: {summary[:500]}...") # Log snippet
        if not summary:
            raise ValueError("Empty response content from OpenRouter API")

        start_tag = "<COMPLETED>"
        end_tag = "</COMPLETED>"
        if start_tag in summary and end_tag in summary:
            completed_items = summary[
                summary.find(start_tag) + len(start_tag) : summary.find(end_tag)
            ]
        else:
            completed_items = "No completed items found."

        start_tag = "<NEXT_STEPS>"
        end_tag = "</NEXT_STEPS>"
        if start_tag in summary and end_tag in summary:
            steps = summary[
                summary.find(start_tag) + len(start_tag) : summary.find(end_tag)
            ]
        else:
            steps = "No steps found."

        return completed_items, steps

    except Exception as e:
        logger.error(f"Error in reorganize_context: {str(e)}", exc_info=True)
        # Return default values in case of error
        return (
            "Error processing context. Please try again.",
            "Error processing steps. Please try again.",
        )

@retry(
    stop=stop_after_attempt(max_attempt_number=5),
    wait=wait_random_exponential(multiplier=2, min=4, max=10),
)
async def refresh_context_async(
    task: str, messages: List[Dict], display: Union[WebUI, AgentDisplayConsole], client
) -> str:
    """
    Create a combined context string by filtering and summarizing messages
    and appending current file contents. This version is streamlined to use a single LLM call.
    """
    filtered_messages = filter_messages(messages)

    # 1. Aggregate current state
    file_contents = aggregate_file_states()
    if len(file_contents) > 200000:
        file_contents = (
            file_contents[:70000] + " ... [TRUNCATED] ... " + file_contents[-70000:]
        )

    from utils.file_logger import get_all_current_skeleton, get_all_current_code
    code_skeletons = get_all_current_skeleton()
    current_code = get_all_current_code()
    if current_code:
        code_skeletons = current_code
    elif not code_skeletons or code_skeletons == "No Python files have been tracked yet.":
        code_skeletons = "No code skeletons available."

    # 2. Summarize context with a single LLM call
    completed, next_steps = await summarize_context_with_llm(
        filtered_messages, file_contents, code_skeletons
    )

    # 3. Construct the final context string
    # No second LLM call is needed. We directly use the summary.
    # The "updated task" is now implicitly the "next steps".

    combined_content = f"""Original request:
{task}

IMPORTANT: This is a CONTINUING PROJECT. All files listed below ALREADY EXIST and are FULLY FUNCTIONAL.
DO NOT recreate any existing files or redo completed steps. Continue the work from where it left off.

Current Project Files and Assets:
{file_contents}

Code Skeletons (Structure of Python files):
{code_skeletons}

COMPLETED STEPS (These have ALL been successfully completed - DO NOT redo these):
{completed}

NEXT STEPS (Continue the project by completing these):
{next_steps}

NOTES:
- All files mentioned in completed steps ALREADY EXIST in the locations specified.
- All completed steps have ALREADY BEEN DONE successfully.
- Your task is to continue the project by implementing the NEXT STEPS, building on the existing work.
"""
    logger.info(f"Refreshed context combined_content (first 500 chars): {combined_content[:500]}...")
    return combined_content