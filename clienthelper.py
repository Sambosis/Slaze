import os
from typing import List, Dict, Optional, Any, Union
from rich import print as rr  # Using rich for better console output
# Ensure you have the necessary libraries installed:
# pip install openai anthropic python-dotenv
# Create a .env file in your project root for API keys if you prefer:
# OPENAI_API_KEY="your_openai_key"
# ANTHROPIC_API_KEY="your_anthropic_key"
# OPENROUTER_API_KEY="your_openrouter_key"

from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

class MinimalLLMClient:
    """
    A minimal LLM client for OpenAI, Anthropic, and OpenAI-compatible (OpenRouter) services,
    with model and default parameters defined at initialization.
    """

    def __init__(self,
                 provider: str,
                 model: str,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 1024,
                 system_prompt: Optional[str] = "You are a helpful assistant.",
                 openrouter_site_url: Optional[str] = None,
                 openrouter_app_name: Optional[str] = None):
        """
        Initializes the LLM client.

        Args:
            provider (str): The LLM provider. Supported: "openai", "anthropic", "openrouter".
            model (str): The default model name to use for this client instance.
            api_key (Optional[str]): The API key for the provider.
                                     If None, it will try to read from environment variables:
                                     - OPENAI_API_KEY for "openai"
                                     - ANTHROPIC_API_KEY for "anthropic"
                                     - OPENROUTER_API_KEY for "openrouter" (falls back to OPENAI_API_KEY)
            base_url (Optional[str]): The base URL for OpenAI-compatible services.
                                      Required for "openrouter" if not using the default.
                                      Default for "openrouter": "https://openrouter.ai/api/v1".
                                      Ignored for "openai" and "anthropic" official APIs.
            temperature (float): Default sampling temperature.
            max_tokens (int): Default maximum number of tokens to generate.
            system_prompt (Optional[str]): Default system prompt.
            openrouter_site_url (Optional[str]): Your site URL, for OpenRouter's HTTP-Referer header.
            openrouter_app_name (Optional[str]): Your app name, for OpenRouter's X-Title header.
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.openrouter_site_url = openrouter_site_url
        self.openrouter_app_name = openrouter_app_name

        if api_key is None:
            if self.provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY")
                if not api_key:
                    api_key = os.getenv("OPENAI_API_KEY") # Fallback for OpenRouter

        if not api_key:
            raise ValueError(f"API key for {provider} not provided or found in environment variables.")
        self.api_key = api_key

        if self.provider == "openai":
            from openai import OpenAI as OpenAIClient
            self.client = OpenAIClient(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic as AnthropicClient
            self.client = AnthropicClient(api_key=self.api_key)
        elif self.provider == "openrouter":
            from openai import OpenAI as OpenAIClient # OpenRouter uses OpenAI SDK
            if base_url is None:
                base_url = "https://openrouter.ai/api/v1"
            self.base_url = base_url
            self.client = OpenAIClient(api_key=self.api_key, base_url=self.base_url)
            self.extra_headers = {}
            if self.openrouter_site_url:
                self.extra_headers["HTTP-Referer"] = self.openrouter_site_url
            if self.openrouter_app_name:
                self.extra_headers["X-Title"] = self.openrouter_app_name
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers are 'openai', 'anthropic', 'openrouter'.")

    def set_model(self, model: str) -> 'MinimalLLMClient':
        """Sets the default model for this client."""
        self.model = model
        return self

    def set_system_prompt(self, system_prompt: Optional[str]) -> 'MinimalLLMClient':
        """Sets the default system prompt for this client."""
        self.system_prompt = system_prompt
        return self

    def set_max_tokens(self, max_tokens: int) -> 'MinimalLLMClient':
        """Sets the default maximum tokens for this client."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        self.max_tokens = max_tokens
        return self

    def set_temperature(self, temperature: float) -> 'MinimalLLMClient':
        """Sets the default temperature for this client."""
        if not (0.0 <= temperature <= 2.0): # OpenAI typical range
            # Anthropic range is 0.0 to 1.0, but we'll use OpenAI's broader range for consistency here
            print(f"Warning: Temperature {temperature} might be outside the typical optimal range for some models.")
        self.temperature = temperature
        return self

    def generate(self,
                 user_prompt: str,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 system_prompt: Optional[str] = None,
                 model_override: Optional[str] = None,
                 messages_override: Optional[List[Dict[str, str]]] = None
                 ) -> Any: # Returns the raw response object from the SDK
        """
        Calls the LLM with the given user prompt and other optional parameters.

        Args:
            user_prompt (str): The user's prompt.
            temperature (Optional[float]): Override instance's default temperature.
            max_tokens (Optional[int]): Override instance's default max_tokens.
            system_prompt (Optional[str]): Override instance's default system_prompt.
                                           Set to empty string "" to explicitly have no system prompt
                                           if instance default is set.
            model_override (Optional[str]): Override the instance's default model for this call.
            messages_override (Optional[List[Dict[str, str]]]):
                If provided, this list of messages will be used directly,
                ignoring user_prompt and system_prompt arguments.

        Returns:
            Any: The raw response object from the underlying SDK (e.g., OpenAI's ChatCompletion, Anthropic's Message).

        Raises:
            Exception: If the API call fails.
        """
        current_model = model_override if model_override is not None else self.model
        current_temp = temperature if temperature is not None else self.temperature
        current_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # Handle system_prompt override: if explicitly passed as None, use self.system_prompt.
        # If passed as a string (even empty ""), use that.
        current_system_prompt = system_prompt if system_prompt is not None else self.system_prompt

        actual_messages: List[Dict[str, str]]
        if messages_override:
            actual_messages = messages_override
        else:
            actual_messages = []
            if current_system_prompt and self.provider != "anthropic": # Anthropic handles system via a dedicated param
                actual_messages.append({"role": "system", "content": current_system_prompt})
            actual_messages.append({"role": "user", "content": user_prompt})

        try:
            if self.provider == "openai" or self.provider == "openrouter":
                api_params = {
                    "model": current_model,
                    "messages": actual_messages,
                    "temperature": current_temp,
                    "max_tokens": current_max_tokens,
                }
                if self.provider == "openrouter" and self.extra_headers:
                    api_params["extra_headers"] = self.extra_headers

                # For OpenAI, if system prompt is handled outside messages_override and
                # was intended as the first message in actual_messages.
                # If messages_override is used, it should contain the system prompt if needed.
                # If not using messages_override, and system prompt is set, and provider is openai/openrouter
                # ensure it's part of actual_messages (which it is by default construction above)

                return self.client.chat.completions.create(**api_params)

            elif self.provider == "anthropic":
                # Anthropic's Messages API uses a top-level `system` parameter.
                # `actual_messages` here should not contain the system message if it was built from user_prompt.
                anthropic_messages = [msg for msg in actual_messages if msg.get("role") != "system"]

                # If messages_override was used, extract system prompt from it if present.
                final_system_prompt_for_anthropic = current_system_prompt
                if messages_override:
                    for msg in messages_override:
                        if msg.get("role") == "system":
                            final_system_prompt_for_anthropic = msg.get("content")
                            anthropic_messages = [m for m in messages_override if m.get("role") != "system"]
                            break

                api_params = {
                    "model": current_model,
                    "messages": anthropic_messages,
                    "temperature": current_temp,
                    "max_tokens": current_max_tokens,
                }
                if final_system_prompt_for_anthropic: # Only add if it's not None or empty
                    api_params["system"] = final_system_prompt_for_anthropic

                return self.client.messages.create(**api_params)

        except Exception as e:
            print(f"Error calling LLM provider {self.provider} with model {current_model}: {e}")
            raise

    def get_text(self, response: Any) -> str:
        """
        Extracts the primary text content from the LLM's response object.

        Args:
            response (Any): The raw response object from the generate() method.

        Returns:
            str: The generated text content.
        """
        if self.provider == "openai" or self.provider == "openrouter":
            if response and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if message and message.content:
                    return message.content.strip()
            return ""
        elif self.provider == "anthropic":
            if response and response.content:
                generated_text = []
                for block in response.content:
                    if block.type == "text":
                        generated_text.append(block.text)
                return "".join(generated_text).strip()
            return ""
        return "Unsupported provider for get_text or invalid response."

    def get_meta(self, response: Any) -> Dict[str, Any]:
        """
        Extracts metadata from the LLM's response object into a standardized dictionary.

        Args:
            response (Any): The raw response object from the generate() method.

        Returns:
            Dict[str, Any]: A dictionary containing metadata.
        """
        meta: Dict[str, Any] = {
            "model_used": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "finish_reason": None,
            "raw_response_id": None,
            "provider_specific": {}
        }

        try:
            if self.provider == "openai" or self.provider == "openrouter":
                if response:
                    meta["model_used"] = response.model
                    meta["raw_response_id"] = response.id
                    if response.usage:
                        meta["input_tokens"] = response.usage.prompt_tokens
                        meta["output_tokens"] = response.usage.completion_tokens
                        meta["total_tokens"] = response.usage.total_tokens
                    if response.choices and len(response.choices) > 0:
                        meta["finish_reason"] = response.choices[0].finish_reason
                    meta["provider_specific"] = {
                        "system_fingerprint": getattr(response, 'system_fingerprint', None)
                    }
            elif self.provider == "anthropic":
                if response:
                    meta["model_used"] = response.model
                    meta["raw_response_id"] = response.id
                    meta["finish_reason"] = response.stop_reason # or stop_sequence
                    if response.usage:
                        meta["input_tokens"] = response.usage.input_tokens
                        meta["output_tokens"] = response.usage.output_tokens
                        meta["total_tokens"] = response.usage.input_tokens + response.usage.output_tokens
                    meta["provider_specific"] = {
                        "role": response.role,
                        "stop_sequence": getattr(response, 'stop_sequence', None)
                    }
        except AttributeError as e:
            print(f"Could not extract some metadata, attribute missing: {e}")
        except Exception as e:
            print(f"An error occurred while extracting metadata: {e}")

        return meta

#
# --- Example Usage ---
if __name__ == "__main__":

    # --- OpenRouter Example (Model defined at init) ---
    # print the app name in the color red using the print from rich library
    rr("\033[91mMyGreatLLMApp\033[0m")
    # now in blue
    rr("\033[94mMyGreatLLMApp\033[0m")
    
    rr("\n--- OpenRouter (Google Gemini Flash) Example ---")
    try:
        or_gem_client = MinimalLLMClient(
            provider="openrouter",
            model="google/gemini-flash-1.5",  # Model defined here
            system_prompt="You are a concise and factual assistant.",
            openrouter_site_url="http://my-app.com",  # Optional
            openrouter_app_name="MyGreatLLMApp",  # Optional
            # API key will be loaded from .env or environment variables
        )
        prompt = "Tell me three interesting facts about Mars."
        rr(f"Prompt: {prompt}")
        response_obj = or_gem_client.generate(prompt)
        text_response = or_gem_client.get_text(response_obj)
        meta_response = or_gem_client.get_meta(response_obj)
        rr(f"Text Response:")
        rr(text_response)
        rr(f"Metadata:\n")
        rr(meta_response)

        # Change system prompt and max_tokens for subsequent calls
        or_gem_client.set_system_prompt(
            "You are a poet writing about planets."
        ).set_max_tokens(50)
        rr(
            f"\nClient's new system prompt: '{or_gem_client.system_prompt}', max_tokens: {or_gem_client.max_tokens}"
        )

        response_obj_2 = or_gem_client.generate("A short verse on Venus.")
        rr(f"Poetic Response:\n{or_gem_client.get_text(response_obj_2)}")
        rr(f"Metadata:\n")
        rr(or_gem_client.get_meta(response_obj_2))

        # Override parameters for a single call
        rr("\nOverriding temperature and model for a single call:\n")
        response_obj_3 = or_gem_client.generate(
            user_prompt="What is OpenRouter?",
            temperature=0.2,
        )
        rr(f"Text Response (NOT Mistral):\n{or_gem_client.get_text(response_obj_3)}")
        rr(f"Metadata (NOT Mistral):\n")
        rr(or_gem_client.get_meta(response_obj_3))

    except Exception as e:
        rr(f"OpenRouter Example Failed: {e}")

    # --- OpenAI Example ---
    rr("\n--- OpenAI Example ---")
    try:
        openai_gpt_client = MinimalLLMClient(
            provider="openai", model="gpt-4o-mini", temperature=0.8, max_tokens=150
        )
        openai_gpt_client.set_system_prompt("You are an expert storyteller.")
        response_obj = openai_gpt_client.generate(
            "Tell a very short story about a brave mouse."
        )
        rr(f"Story:\n{openai_gpt_client.get_text(response_obj)}")
        rr(f"Metadata:\n")
        rr(openai_gpt_client.get_meta(response_obj))
    except Exception as e:
        rr(f"OpenAI Example Failed: {e}")

    # --- Anthropic Example ---
    rr("\n--- Anthropic Example ---\n\n")
    try:
        anthropic_claude_client = MinimalLLMClient(
            provider="anthropic",
            model="claude-3-haiku-20240307",
            system_prompt="You explain complex topics simply.",
        )
        response_obj = anthropic_claude_client.generate(
            user_prompt="What is the main idea behind quantum entanglement?",
            max_tokens=200,
        )
        rr(f"Explanation:\n{anthropic_claude_client.get_text(response_obj)}")
        rr(f"Metadata:\n")
        rr(anthropic_claude_client.get_meta(response_obj))

        # Example using messages_override for a more complex interaction
        rr("\nAnthropic example with messages_override:")
        messages = [
            {
                "role": "user",
                "content": "Hello Claude, can you write a haiku about seasons?",
            },
            {
                "role": "assistant",
                "content": "Green leaves softly fall,\nWinter's chill then sun's warm kiss,\nNature's gentle spin.",
            },
            {
                "role": "user",
                "content": "That was lovely! Can you write another about the ocean?",
            },
        ]
        rr(messages)
        response_obj_msg_override = anthropic_claude_client.generate(
            user_prompt="",  # Ignored when messages_override is used
            messages_override=messages,
            system_prompt="You are a Haiku master.",  # This will be used by Anthropic if not in messages_override
        )
        rr(
            f"Ocean Haiku:\n{anthropic_claude_client.get_text(response_obj_msg_override)}"
        )
        rr(f"Metadata:\n")
        rr(anthropic_claude_client.get_meta(response_obj_msg_override)  )

    except Exception as e:
        rr(f"Anthropic Example Failed: {e}")
