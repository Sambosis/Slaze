import os

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

openai41mini : str = "openai/gpt-4.1-mini"
gemma3n4b : str = "google/gemma-3n-e4b-it"
sonnet4 : str = "anthropic/claude-sonnet-4"
openai41 : str = "openai/gpt-4.1"
openaio3 : str = "openai/o3"
openaio3pro : str = "openai/o3-pro"
googlepro : str = "google/gemini-2.5-pro-preview"
googleflash : str = "google/gemini-2.5-flash-preview"
googleflashlite : str = "google/gemini-2.5-flash-lite-preview-06-17"
grok4 : str = "x-ai/grok-4"
SUMMARY_MODEL : str = googleflashlite  # Model for summaries
MAIN_MODEL : str = f"{googleflashlite}"  # Primary model for main agent operations
CODE_MODEL : str = f"{googleflashlite}:web"  # Model for code generation tasks
BASE_URL : str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL : str = MAIN_MODEL

BASE_SYSTEM_PROMPT : str = "You are a helpful AI assistant. "


def get_llm(
    base_url : str = BASE_URL,
    api_key : str = str(os.getenv("OPENROUTER_API_KEY")),
    is_async : bool = False,
) -> OpenAI | AsyncOpenAI:
    """
    Get the appropriate LLM instance based on the async flag.
    """
    if is_async:
        return AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        )
    else:
        return OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        )


def chat(prompt_str, model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)):
    """
    Send a chat message to the LLM.
    
    """
    llm = get_llm()
    messages = [{"role": "user", "content": prompt_str}]
    if isinstance(llm, AsyncOpenAI):

        return llm.chat.completions.create(messages=messages, model=model).choices[0].message.content
    else:
        return llm.chat.completions.create(messages=messages, model=model).choices[0].message.content


if __name__ == "__main__":
    # Example usage
    response = chat(prompt_str="Hello, how can you assist me today?")
    print(response)
