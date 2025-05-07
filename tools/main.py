import openai
import json
import os
from edit import edit_file, edit_file_function
from bash import bash_command, bash_command_function

# from function_schema import windows_navigate_function, edit_file_function, bash_command_function
from dotenv import load_dotenv
from rich import print as rr

# Load environment variables from .env file
load_dotenv()
from tenacity import retry, stop_after_attempt, wait_fixed

# Set your OpenAI API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

model = "gpt-4o"


def process_tool_calls(tool_calls):
    """Process all tool calls in parallel and return their results."""
    results = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "windows_navigate":
            rr("windows_navigate: ", arguments)
            result = windows_navigate(**arguments)
        elif function_name == "edit_file":
            rr("edit_file: ", arguments)
            result = edit_file(**arguments)
        elif function_name == "bash_command":
            rr("bash_command: ", arguments)
            result = bash_command(**arguments)
            rr("result: ", result)
        else:
            result = f"Function '{function_name}' is not supported."

        results.append(
            {
                "role": "tool",
                "content": json.dumps(result),  # Ensure the result is JSON-serializable
                "tool_call_id": tool_call.id,
            }
        )
    return results


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def dodo(messages, tools):
    rr("messages:", messages)
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
    return response


def main():
    tools = [
        {"type": "function", "function": windows_navigate_function},
        {"type": "function", "function": edit_file_function},
        {"type": "function", "function": bash_command_function},
    ]

    # Initialize conversation with a system message
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that can perform call functions to operate a computer.",
        }
    ]
    first = True
    i = 0
    while True:
        i += 1
        if first:
            with open(
                r"C:\mygit\compuse\computer_use_demo\prompts\prompt copy.md", "r"
            ) as file:
                prompt = file.read()
            user_input = prompt
            # Add user input to the conversation
            messages.append({"role": "user", "content": user_input})
            first = False
        elif i % 9 == 0:
            # Get user input
            user_input = input("User: ")
            # Add user input to the conversation
            messages.append({"role": "user", "content": user_input})
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break
        else:
            # Add user input to the conversation
            messages.append(
                {"role": "user", "content": "Continue working towards your goal."}
            )
        try:
            # Make the first API call
            response = dodo(messages, tools)

            # Extract the assistant's message
            assistant_message = response.choices[0].message
            rr("assistant_message: ", assistant_message)
            messages.append(assistant_message.to_dict())
            if assistant_message.tool_calls:
                # Process all tool calls and get their results
                tool_responses = process_tool_calls(assistant_message.tool_calls)
                rr("tool response: ", tool_responses)
                # Append tool call results to the messages
                messages.extend(tool_responses)
                # rr("messages", messages)
                # After processing all tool calls, make a follow-up API call
                follow_up_response = openai.chat.completions.create(
                    model=model, messages=messages
                )

                # Extract and display the assistant's follow-up response
                follow_up_message = follow_up_response.choices[0].message
                messages.append(follow_up_message.to_dict())
                print(f"Assistant: {follow_up_message.content}")

            else:
                # Handle responses without tool calls
                messages.append(assistant_message.to_dict())
                print(f"Assistant: {assistant_message.content}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
