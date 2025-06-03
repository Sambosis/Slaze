import json  # Added for json.dumps
from typing import List, Dict, Any, Optional

# from pydantic import BaseModel, Field # Removed Pydantic imports
from anthropic import Anthropic  # Assuming this is the correct import
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
)  # Assuming this is correct


# Assuming these are defined elsewhere, providing stubs for completeness
class AgentDisplayWebWithPrompt:
    pass


class ToolCollection:
    def __init__(self, *args, **kwargs):
        pass

    def to_params(self):
        return []  # Placeholder


class OutputManager:
    def __init__(self, display):
        pass


class TokenTracker:
    def __init__(self, display):
        pass


# Placeholder for your tool classes
class WriteCodeTool:
    def __init__(self, display):
        pass


class ProjectSetupTool:
    def __init__(self, display):
        pass


class BashTool:
    def __init__(self, display):
        pass


class PictureGenerationTool:
    def __init__(self, display):
        pass


class DockerEditTool:
    def __init__(self, display):
        pass


class AgentState:  # No longer inherits from BaseModel
    """
    Class to store and manage the current state of the Agent.
    """

    def __init__(
        self,
        task: str,
        context_recently_refreshed: bool = False,
        refresh_count: int = 45,
        refresh_increment: int = 15,
        messages: Optional[List[Dict[str, Any]]] = None,
        enable_prompt_caching: bool = True,
        image_truncation_threshold: int = 1,
        only_n_most_recent_images: int = 2,
        step_count: int = 0,
        tool_params: Optional[List[Any]] = None,
    ):
        self.task: str = task
        self.context_recently_refreshed: bool = context_recently_refreshed
        self.refresh_count: int = refresh_count
        self.refresh_increment: int = refresh_increment
        self.messages: List[Dict[str, Any]] = messages if messages is not None else []
        self.enable_prompt_caching: bool = enable_prompt_caching
        self.image_truncation_threshold: int = image_truncation_threshold
        self.only_n_most_recent_images: int = only_n_most_recent_images
        self.step_count: int = step_count
        self.tool_params: List[Any] = tool_params if tool_params is not None else []
        # self.betas: List[str] = [] # Assuming COMPUTER_USE_BETA_FLAG and PROMPT_CACHING_BETA_FLAG are strings

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the state to a dictionary."""
        return {
            "task": self.task,
            "context_recently_refreshed": self.context_recently_refreshed,
            "refresh_count": self.refresh_count,
            "refresh_increment": self.refresh_increment,
            "messages": self.messages,
            "enable_prompt_caching": self.enable_prompt_caching,
            "image_truncation_threshold": self.image_truncation_threshold,
            "only_n_most_recent_images": self.only_n_most_recent_images,
            "step_count": self.step_count,
            "tool_params": self.tool_params,
            # "betas": self.betas,
        }


class Agent:
    def __init__(self, task: str, display: AgentDisplayWebWithPrompt):
        # Initialize state using the regular AgentState class
        self.state = AgentState(
            task=task, tool_params=[]
        )  # tool_params will be updated below

        self.display = display  # External dependency

        self.tool_collection = ToolCollection(
            WriteCodeTool(display=self.display),
            ProjectSetupTool(display=self.display),
            BashTool(display=self.display),
            PictureGenerationTool(display=self.display),
            DockerEditTool(display=self.display),
            display=self.display,
        )
        self.output_manager = OutputManager(self.display)
        self.token_tracker = TokenTracker(self.display)
        # Ensure ANTHROPIC_API_KEY is handled, e.g., via os.getenv
        # For this example, assuming it's handled elsewhere or not strictly needed for compilation
        try:
            self.client = Anthropic()
        except Exception as e:
            print(f"Warning: Could not initialize Anthropic client: {e}")
            self.client = None

        # Update state based on initialized components
        self.state.tool_params = self.tool_collection.to_params()
        # self.state.betas = [COMPUTER_USE_BETA_FLAG, PROMPT_CACHING_BETA_FLAG] # If these are constants

    def log_tool_results(self, combined_content, tool_name, tool_input):
        # ... (your existing method, can remain as is)
        # Ensure logging to console or string if file I/O is restricted
        log_message = f"\n{'=' * 80}\n"
        log_message += f"TOOL EXECUTION: {tool_name}\n"
        log_message += f"INPUT: {json.dumps(tool_input, indent=2)}\n"
        log_message += f"{'-' * 80}\n"

        for item in combined_content:
            log_message += f"CONTENT TYPE: {item['type']}\n"
            if item["type"] == "tool_result":
                log_message += f"TOOL USE ID: {item['tool_use_id']}\n"
                log_message += f"ERROR: {item['is_error']}\n"
                if isinstance(item["content"], list):
                    log_message += "CONTENT:\n"
                    for content_item in item["content"]:
                        log_message += f"  - {content_item['type']}: {content_item.get('text', '[non-text content]')}\n"
                else:
                    log_message += f"CONTENT: {item['content']}\n"
            elif item["type"] == "text":
                log_message += f"TEXT:\n{item['text']}\n"
            log_message += f"{'-' * 50}\n"
        log_message += f"{'=' * 80}\n\n"
        print(log_message)  # Print to console instead of file
        pass

    async def run_tool(self, content_block: Dict[str, Any]):
        # ... (your existing method)
        # Update self.state.messages as needed
        # For example:
        # self.state.messages.append({"role": "user", "content": combined_content})
        # This method should return a Dict as per _make_api_tool_result usage
        # For now, returning a placeholder
        class ToolResultPlaceholder:  # Placeholder for ToolResult if it's not defined
            def __init__(self, output, tool_name, error=None, base64_image=None):
                self.output = output
                self.tool_name = tool_name
                self.error = error
                self.base64_image = base64_image

        # Simplified run_tool logic for demonstration
        tool_name = content_block.get("name", "unknown_tool")
        tool_input = content_block.get("input", {})
        tool_id = content_block.get("id", "unknown_id")

        print(f"Simulating run_tool: {tool_name} with input {tool_input}")

        # Simulate a tool result
        simulated_result = ToolResultPlaceholder(
            output=f"Successfully executed {tool_name}", tool_name=tool_name
        )

        api_tool_result = self._make_api_tool_result(simulated_result, tool_id)

        combined_content = [
            {
                "type": "tool_result",
                "content": api_tool_result["content"],
                "tool_use_id": api_tool_result["tool_use_id"],
                "is_error": api_tool_result["is_error"],
            }
        ]
        combined_content.append(
            {
                "type": "text",
                # Assuming extract_text_from_content is defined elsewhere
                "text": f"Tool '{tool_name}' was called with input: {json.dumps(tool_input)}.\nResult: {simulated_result.output}",
            }
        )
        self.state.messages.append({"role": "user", "content": combined_content})
        self.log_tool_results(combined_content, tool_name, tool_input)
        return api_tool_result

    def _make_api_tool_result(self, result: Any, tool_use_id: str) -> Dict:
        # ... (your existing method, ensure result has expected attributes like .error, .output, .base64_image)
        tool_result_content = []
        is_error = False

        if result is None:
            is_error = True
            tool_result_content.append(
                {"type": "text", "text": "Tool execution resulted in None"}
            )
        elif isinstance(result, str):  # Should ideally be a ToolResult-like object
            is_error = True
            tool_result_content.append({"type": "text", "text": result})
        else:
            if hasattr(result, "error") and result.error:
                is_error = True
                tool_result_content.append({"type": "text", "text": str(result.error)})

            if hasattr(result, "output") and result.output:
                tool_result_content.append({"type": "text", "text": str(result.output)})

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

        if (
            not tool_result_content and not is_error
        ):  # Ensure there's always some content if not an error
            tool_result_content.append(
                {
                    "type": "text",
                    "text": "Tool executed successfully with no specific output.",
                }
            )

        return {
            "type": "tool_result",
            "content": tool_result_content,
            "tool_use_id": tool_use_id,
            "is_error": is_error,
        }

    def _inject_prompt_caching(self):
        messages = self.state.messages  # Use state's messages
        breakpoints_remaining = 2
        for message in reversed(messages):
            if message["role"] == "user" and isinstance(
                (content := message["content"]), list  # Python 3.8+ walrus operator
            ):
                if breakpoints_remaining > 0:  # Check should be > 0
                    breakpoints_remaining -= 1
                    # Ensure content is not empty and the last item is a dict
                    if content and isinstance(content[-1], dict):
                        content[-1]["cache_control"] = BetaCacheControlEphemeralParam(
                            {"type": "ephemeral"}
                        )
                    elif content and not isinstance(content[-1], dict):
                        # If last item is not a dict, append a new dict for cache_control
                        # This case might need more specific handling based on expected message structure
                        print(
                            "Warning: Last content item is not a dict, cannot inject cache_control directly."
                        )
                    elif not content:
                        print(
                            "Warning: Content list is empty, cannot inject cache_control."
                        )

                else:  # breakpoints_remaining is 0
                    if content and isinstance(content[-1], dict):
                        content[-1].pop("cache_control", None)
                    break
        # No return needed as it modifies self.state.messages in place

    def _sanitize_tool_name(self, name: str) -> str:
        import re  # Import re locally if not already at module level

        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        return sanitized

    async def step(self):
        self.state.step_count += 1
        # #laminar.set_session(session_id=f"step_{self.state.step_count}") # If #laminar is used

        messages = self.state.messages
        task = self.state.task

        if self.state.enable_prompt_caching:
            self._inject_prompt_caching()
            # self.state.image_truncation_threshold = 1 # This seems to be a constant assignment

        # --- Placeholder for LLM call and further logic ---
        print(f"--- Agent Step {self.state.step_count} ---")
        print(f"Current Task: {task}")
        print(f"Number of messages: {len(messages)}")

        # Simulate LLM response and tool usage for demonstration
        if (
            not messages or messages[-1]["role"] == "assistant"
        ):  # If no messages or last was assistant, simulate user input
            self.state.messages.append({"role": "user", "content": "Tell me a joke."})
            print("Simulated user input: Tell me a joke.")

        # Simulate an LLM response that might call a tool or just text
        # For this example, let's assume it calls a tool if step_count is odd, else text response
        response_params = []
        if self.state.step_count % 2 != 0:  # Odd step, simulate tool call
            tool_to_call = "BashTool"  # Example
            tool_id = f"tool_{self.state.step_count}"
            response_params.append(
                {
                    "type": "tool_use",
                    "name": self._sanitize_tool_name(tool_to_call),
                    "id": tool_id,
                    "input": {"command": "ls -l"},
                }
            )
            response_params.append(
                {"type": "text", "text": f"Okay, I will run the {tool_to_call}."}
            )
        else:  # Even step, simulate text response
            response_params.append(
                {
                    "type": "text",
                    "text": "Why did the scarecrow win an award? Because he was outstanding in his field!",
                }
            )

        self.state.messages.append({"role": "assistant", "content": response_params})
        print(f"Simulated LLM Response: {response_params}")

        # Simulate tool execution if a tool_use block is present
        tool_result_content_list = []
        for content_block in response_params:
            if content_block.get("type") == "tool_use":
                # In a real scenario, OutputManager might format here
                # self.output_manager.format_content_block(content_block)
                print(f"Executing tool: {content_block.get('name')}")
                tool_result = await self.run_tool(
                    content_block
                )  # run_tool appends to self.state.messages
                if tool_result:
                    tool_result_content_list.append(
                        tool_result
                    )  # Collect results if needed for further logic

        # Simulate context refresh logic
        if len(self.state.messages) > self.state.refresh_count:
            print("Simulating context refresh...")
            # Simplified refresh: keep only the last few messages + new context
            # new_context_text = f"Context refreshed based on last {min(3, len(self.state.messages))} messages."
            # self.state.messages = self.state.messages[-3:] if len(self.state.messages) > 3 else self.state.messages
            # self.state.messages.insert(0, {"role": "user", "content": new_context_text}) # Prepend new context
            self.state.messages = [
                {"role": "user", "content": "System: Context has been refreshed."}
            ]  # Simplified
            self.state.context_recently_refreshed = True
            self.state.refresh_count += self.state.refresh_increment
            print(f"Context refreshed. New refresh count: {self.state.refresh_count}")
        else:
            self.state.context_recently_refreshed = False

        # Simulate waiting for user input if no tool was called
        if not tool_result_content_list and not self.state.context_recently_refreshed:
            print("Awaiting User Input (simulated)")
            # In a real app, this would involve self.display.wait_for_user_input()
            # For simulation, we might just end the step or add a placeholder user message
            # self.state.messages.append({"role": "user", "content": "Okay, I understand."})

        # Simulate token tracking
        # if response is not None and hasattr(response, "usage"):
        #     self.token_tracker.update(response) # Assuming token_tracker.update exists
        print("Token usage update (simulated).")

        return True  # Indicate step was successful

    def save_state(self, filepath: str):
        """Saves the current agent state to a JSON file."""
        # Using to_dict and json.dumps as model_dump_json is Pydantic-specific
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.state.to_dict(), f, indent=4)
            print(f"Agent state saved to {filepath}")
        except IOError as e:
            print(f"Error saving state to {filepath}: {e}")

    @classmethod
    def load_state(cls, filepath: str, display: AgentDisplayWebWithPrompt):
        """Loads agent state from a JSON file and creates a new Agent instance."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                state_data = json.load(f)
        except FileNotFoundError:
            print(
                f"Error: State file {filepath} not found. Creating agent with default state."
            )
            # Fallback to creating a new agent with a default task if file not found
            agent = cls(task="Default task (state file not found)", display=display)
            return agent
        except json.JSONDecodeError as e:
            print(
                f"Error decoding JSON from {filepath}: {e}. Creating agent with default state."
            )
            agent = cls(task="Default task (JSON error)", display=display)
            return agent

        # Create AgentState instance manually
        loaded_state = AgentState(
            task=state_data.get("task", "Default Task"),  # Provide defaults
            context_recently_refreshed=state_data.get(
                "context_recently_refreshed", False
            ),
            refresh_count=state_data.get("refresh_count", 45),
            refresh_increment=state_data.get("refresh_increment", 15),
            messages=state_data.get("messages", []),
            enable_prompt_caching=state_data.get("enable_prompt_caching", True),
            image_truncation_threshold=state_data.get("image_truncation_threshold", 1),
            only_n_most_recent_images=state_data.get("only_n_most_recent_images", 2),
            step_count=state_data.get("step_count", 0),
            tool_params=state_data.get("tool_params", []),
        )

        agent = cls(task=loaded_state.task, display=display)
        agent.state = loaded_state

        print(f"Agent state loaded from {filepath}")
        return agent


if __name__ == "__main__":
    mock_display = AgentDisplayWebWithPrompt()
    agent_state_file = "agent_current_state.json"

    # Create or load agent
    # Check if state file exists to decide whether to load or create new
    try:
        # Attempt to load to see if file is valid, even if we create a new one first for demo
        # In a real app, you'd typically load if exists, else create new.
        open(agent_state_file, "r").close()  # Check if file exists and is readable
        print(f"Attempting to load agent from {agent_state_file}")
        my_agent = Agent.load_state(agent_state_file, mock_display)
    except FileNotFoundError:
        print(f"State file {agent_state_file} not found. Creating new agent.")
        my_agent = Agent(task="Develop a web application.", display=mock_display)
    except Exception as e:  # Catch other potential errors during initial load check
        print(
            f"Error during initial load check of {agent_state_file}: {e}. Creating new agent."
        )
        my_agent = Agent(task="Develop a web application.", display=mock_display)

    # Simulate some steps or changes in state
    print(
        f"\nInitial state: Step count {my_agent.state.step_count}, Messages: {len(my_agent.state.messages)}"
    )

    # Run a few steps (requires async context for step method)
    import asyncio

    async def run_agent_steps(agent, num_steps=2):
        for i in range(num_steps):
            print(f"\n--- Running Agent Step {i+1}/{num_steps} ---")
            await agent.step()
            print(
                f"State after step: Step count {agent.state.step_count}, Messages: {len(agent.state.messages)}"
            )
            # Small delay for readability if needed
            # await asyncio.sleep(0.1)

    asyncio.run(run_agent_steps(my_agent, 2))

    # Save the state
    my_agent.save_state(agent_state_file)

    # Example of loading again (optional, just to show it works)
    print("\n--- Reloading Agent ---")
    loaded_agent = Agent.load_state(agent_state_file, mock_display)
    print(f"Loaded agent task: {loaded_agent.state.task}")
    print(f"Loaded agent step count: {loaded_agent.state.step_count}")
    # print(f"Loaded agent messages: {loaded_agent.state.messages}") # Can be verbose
    print(f"Number of loaded messages: {len(loaded_agent.state.messages)}")

    print("\n--- Example run completed ---")
