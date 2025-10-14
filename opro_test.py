from utils.llm_client import OpenRouterClient
from config import MAIN_MODEL
from dotenv import load_dotenv
import os
import asyncio
from openai import AsyncOpenAI
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")



async def main():
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    prepared_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! How can you assist me today?"},
    ]

    completion = await client.chat.completions.create(model=MAIN_MODEL, messages=prepared_messages)
    print(completion)

if __name__ == "__main__":
    asyncio.run(main())
