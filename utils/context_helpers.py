from typing import Any, Callable, Dict, List, Optional, cast
from anthropic import Anthropic, APIResponse
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlock,
    BetaMessageParam,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
)

from utils.agent_display_web_with_prompt import AgentDisplayWebWithPrompt
from load_constants import write_to_file, ICECREAM_OUTPUT_FILE
from utils.file_logger import aggregate_file_states
from openai import OpenAI
from icecream import ic, install
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


async def refresh_context_async(task: str, messages: List[Dict], display: AgentDisplayWebWithPrompt) -> str:
    """
    Create a combined context string by filtering and (if needed) summarizing messages
    and appending current file contents.
    """
    filtered = filter_messages(messages)
    summary = ""#get_all_summaries()
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

