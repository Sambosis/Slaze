from typing import Any, Dict, List
from anthropic.types.beta import (
    BetaMessageParam,
)
import os
from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from load_constants import *
from config import write_to_file
from utils.file_logger import aggregate_file_states
from openai import OpenAI
from icecream import ic
from rich import print as rr

ic.configureOutput(includeContext=True, outputFunction=write_to_file)

QUICK_SUMMARIES = []


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
                            output_pieces.append("\nResult:")
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
                                    output_pieces.append(
                                        "\nImage Source: base64 source too big"
                                    )
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


async def summarize_recent_messages(
    short_messages: List[BetaMessageParam], display: AgentDisplayWebWithPrompt
) -> str:
    """
    Summarize the most recent messages.
    """
    gflash = "google/gemini-2.5-flash-preview"
    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        sum_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        all_summaries = get_all_summaries()
        model = gflash
        conversation_text = ""
        for msg in short_messages:
            role = msg["role"].upper()
            if isinstance(msg["content"], list):
                for block in msg["content"]:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content = block.get("text", "")
                            if len(content) > 150000:
                                content = (
                                    content[:70000]
                                    + " ... [TRUNCATED] ... "
                                    + content[-70000:]
                                )
                            conversation_text += f"\n{role}: {content}"
                        elif block.get("type") == "tool_result":
                            for item in block.get("content", []):
                                if item.get("type") == "text":
                                    content = item.get("text", "")
                                    if len(content) > 150000:
                                        content = (
                                            content[:70000]
                                            + " ... [TRUNCATED] ... "
                                            + content[-70000:]
                                        )
                                    conversation_text += (
                                        f"\n{role} (Tool Result): {content}"
                                    )
            else:
                content = msg["content"]
                if len(content) > 150000:
                    content = (
                        content[:70000] + " ... [TRUNCATED] ... " + content[-70000:]
                    )
                conversation_text += f"\n{role}: {content}"
        ic(f"conversation_text: {conversation_text}")
        summary_prompt = f"""Please provide your response in a concise markdown format with short statements that document what happened. Structure your response as a list with clear labels for each step, such as:

            - **Action:** [brief description of what was done]
            - **Result:** [outcome of the action]

            
            - Here are the actions that have been logged so far.  You should not repeat these, they are only to give you context to what is going on. 
            Previous Actions:
            {all_summaries}
            Please be specific but concise, focusing on documenting the sequence of events in this structured format.
            Messages to summarize:
            {conversation_text}"""
        response = sum_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": summary_prompt}]
        )
        ic(f"response: {response}")

        # Add error handling for response
        if not response or not response.choices or len(response.choices) == 0:
            error_msg = "Error: No valid response received from API"
            print(response)
            ic(error_msg)
            return "Error generating summary: No valid response received from API"

        summary = response.choices[0].message.content

        # Check if summary is None or empty
        if not summary:
            error_msg = "Error: Empty summary received from API"
            ic(error_msg)
            return "Error generating summary: Empty summary received from API"

        ic(f"summary: {summary}")
        return summary
    except Exception as e:
        error_msg = f"Error generating summary: {str(e)}"
        ic(error_msg)
        return error_msg


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


def truncate_message_content(content: Any, max_length: int = 150_000) -> Any:
    if isinstance(content, str):
        if len(content) > max_length:
            return content[:70000] + " ... [TRUNCATED] ... " + content[-70000:]
        return content
    elif isinstance(content, list):
        return [truncate_message_content(item, max_length) for item in content]
    elif isinstance(content, dict):
        return {
            k: truncate_message_content(v, max_length) if k != "source" else v
            for k, v in content.items()
        }
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


async def reorganize_context(messages: List[BetaMessageParam], summary: str) -> str:
    """Reorganize the context by filtering and summarizing messages."""
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
    ic(f"conversation_text: {conversation_text}")
    summary_prompt = f"""I need a summary of completed steps and next steps for a project that is ALREADY IN PROGRESS. 
    This is NOT a new project - you are continuing work on an existing codebase.

    VERY IMPORTANT INSTRUCTIONS:
    1. ALL FILES mentioned as completed or created ARE ALREADY CREATED AND FULLY FUNCTIONAL.
       - Do NOT suggest recreating these files.
       - Do NOT suggest checking if these files exist.
       - Assume all files mentioned in completed steps exist exactly where they are described.
    
    2. ALL STEPS listed as completed HAVE ALREADY BEEN SUCCESSFULLY DONE.
       - Do NOT suggest redoing any completed steps.
    
    3. Your summary should be in TWO clearly separated parts:
       a. COMPLETED: List all tasks/steps that have been completed so far
       b. NEXT STEPS: List 1-4 specific, actionable steps that should be taken next to complete the project
    
    4. List each completed item and next step ONLY ONCE, even if it appears multiple times in the context.
    
    5. If any images were generated, mention each image, its purpose, and its location in the COMPLETED section.
    
    Please format your response with:
    <COMPLETED>
    [List of ALL completed steps and created files - these are DONE and exist]
    </COMPLETED>

    <NEXT_STEPS>
    [Numbered list of 1-4 next steps to complete the project]
    </NEXT_STEPS>

    Here is the Summary part:
    {summary}
    
    Here is the messages part:
    <MESSAGES>
    {conversation_text}
    </MESSAGES>
    """

    try:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        sum_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        model = "meta-llama/llama-3.3-70b-instruct:nitro"
        response = sum_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": summary_prompt}]
        )
        ic(f"response: {response}")
        if not response or not response.choices:
            raise ValueError("No response received from OpenRouter API")

        summary = response.choices[0].message.content
        ic(f"summary: {summary}")
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
        ic(f"Error in reorganize_context: {str(e)}")
        # Return default values in case of error
        return (
            "Error processing context. Please try again.",
            "Error processing steps. Please try again.",
        )


async def refresh_context_async(
    task: str, messages: List[Dict], display: AgentDisplayWebWithPrompt, client
) -> str:
    """
    Create a combined context string by filtering and (if needed) summarizing messages
    and appending current file contents.
    """
    filtered = filter_messages(messages)
    summary = get_all_summaries()
    completed, next_steps = await reorganize_context(filtered, summary)

    file_contents = aggregate_file_states()
    if len(file_contents) > 200000:
        file_contents = (
            file_contents[:70000] + " ... [TRUNCATED] ... " + file_contents[-70000:]
        )

    # Get code skeletons
    from utils.file_logger import get_all_current_skeleton

    code_skeletons = get_all_current_skeleton()
    if not code_skeletons or code_skeletons == "No Python files have been tracked yet.":
        code_skeletons = "No code skeletons available."

    # Extract information about images generated
    images_info = ""
    if "## Generated Images:" in file_contents:
        images_section = file_contents.split("## Generated Images:")[1]
        if "##" in images_section:
            images_section = images_section.split("##")[0]
        images_info = "## Generated Images:\n" + images_section.strip()

    # call the LLM and pass it all current messages then the task and ask it to give an updated version of the task
    prompt = f""" Your job is to update the task based on the current state of the project.
    The task is: {task}
    The current state of the project is:
    {file_contents}
    {code_skeletons}
    {completed}
    {next_steps}
    {images_info}

    Once again, here is the task that I need you to give an updated version of.  
    Make sure that you give any tips, lessons learned,  what has been done, and what needs to be done.
    Make sure you give clear guidance on how to import various files and in general how they should work together.
    """

    messages = [{"role": "user", "content": prompt}]
    response = client.beta.messages.create(
        model=MAIN_MODEL,
        messages=messages,
        max_tokens=MAX_SUMMARY_TOKENS,
    )
    new_task = response.content[0].text

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


    Updated Request:
    {new_task}
    NOTES: 
    - All files mentioned in completed steps ALREADY EXIST in the locations specified.
    - All completed steps have ALREADY BEEN DONE successfully.
    - Continue the project by implementing the next steps, building on the existing work.
    """
    rr(f"combined_content: {combined_content}")
    return combined_content
