"""
Example integration of LLM logging into the existing agent architecture.
This demonstrates how to use the logged OpenAI client in place of the regular client.
"""

import asyncio
import os
import uuid
from pathlib import Path

from utils.llm_logger import (
    init_db, 
    get_logged_client, 
    get_logged_async_client, 
    query_interactions
)
from config import MAIN_MODEL


async def example_basic_logging():
    """Basic example of using the logged OpenAI client."""
    
    # Initialize database
    await init_db()
    print("Database initialized!")
    
    # Get a logged client (async version)
    client = get_logged_async_client(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    
    # Create a session ID for this conversation
    session_id = str(uuid.uuid4())
    print(f"Session ID: {session_id}")
    
    # Make a request (this will be logged automatically)
    try:
        response = await client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            session_id=session_id,  # Pass session_id to group related messages
            max_tokens=100
        )
        
        print("Response:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error making request: {e}")
        return
    
    # Query the logged interactions
    print("\n--- Querying logged interactions ---")
    interactions = await query_interactions(session_id=session_id)
    
    for interaction in interactions:
        print(f"[{interaction.created_at}] {interaction.role}: {interaction.content[:100]}...")
        if interaction.prompt_tokens or interaction.completion_tokens:
            print(f"  Tokens: {interaction.prompt_tokens} prompt, {interaction.completion_tokens} completion")


async def example_agent_integration():
    """Example of how to modify the Agent class to use logging."""
    
    # This is a simplified example showing how you would modify agent.py
    # to use the logged client instead of the regular OpenAI client
    
    await init_db()
    
    # Instead of:
    # self.client = OpenAI(...)
    
    # Use:
    logged_client = get_logged_client(
        api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    
    # The logged client can be used exactly like the regular client
    session_id = str(uuid.uuid4())
    
    try:
        response = logged_client.chat.completions.create(
            model=MAIN_MODEL,
            messages=[
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write a simple Python function to calculate fibonacci numbers."}
            ],
            session_id=session_id,
            max_tokens=500
        )
        
        print("Generated code:", response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error: {e}")


async def example_query_usage_stats():
    """Example of querying usage statistics from logged interactions."""
    
    # Get all interactions from the last session
    all_interactions = await query_interactions(limit=50)
    
    if not all_interactions:
        print("No interactions found in database")
        return
    
    # Calculate token usage stats
    total_prompt_tokens = sum(i.prompt_tokens for i in all_interactions)
    total_completion_tokens = sum(i.completion_tokens for i in all_interactions)
    
    print(f"Total interactions: {len(all_interactions)}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total tokens: {total_prompt_tokens + total_completion_tokens}")
    
    # Show breakdown by role
    role_counts = {}
    for interaction in all_interactions:
        role_counts[interaction.role] = role_counts.get(interaction.role, 0) + 1
    
    print("\nBreakdown by role:")
    for role, count in role_counts.items():
        print(f"  {role}: {count} messages")
    
    # Show recent sessions
    sessions = set(i.session_id for i in all_interactions)
    print(f"\nRecent sessions: {len(sessions)}")
    for session_id in list(sessions)[:5]:  # Show first 5
        session_interactions = [i for i in all_interactions if i.session_id == session_id]
        print(f"  {session_id}: {len(session_interactions)} interactions")


if __name__ == "__main__":
    print("=== LLM Logging Examples ===")
    
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run examples
    print("\n1. Basic logging example:")
    asyncio.run(example_basic_logging())
    
    print("\n\n2. Agent integration example:")
    asyncio.run(example_agent_integration())
    
    print("\n\n3. Usage statistics example:")
    asyncio.run(example_query_usage_stats())