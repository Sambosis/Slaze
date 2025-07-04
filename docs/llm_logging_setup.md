# LLM Logging System

A minimal, production-ready logging system for OpenAI API interactions using SQLAlchemy 2.0 and async operations.

## Features

- **Async everywhere** – matches OpenAI async client + high-throughput logging
- **Single transactional write** – prevents mismatched request/response rows
- **Plain SQLAlchemy models** – easy to query, migrate, or stream to analytics  
- **No framework lock-in** – drop-in replacement for OpenAI clients
- **Flexible database support** – PostgreSQL for production, SQLite for development/testing

## Database Schema

| Column | Type | Purpose |
|--------|------|---------|
| `id` | BIGSERIAL PK | Row ID |
| `session_id` | UUID | Logical chat session |
| `role` | TEXT | "user" / "assistant" / "system" |
| `content` | TEXT | Raw message content |
| `model` | TEXT | Model ID used for the response |
| `prompt_tokens` | INT | Request tokens |
| `completion_tokens` | INT | Response tokens |
| `created_at` | TIMESTAMPTZ | DB-default NOW() |

## Installation

1. **Install dependencies** (already added to requirements.txt and pyproject.toml):
   ```bash
   pip install sqlalchemy>=2.0 asyncpg aiosqlite psycopg2-binary
   ```

2. **Initialize database**:
   ```bash
   python utils/init_llm_db.py
   ```

## Configuration

Set the database connection string via environment variable:

```bash
# For PostgreSQL (production)
export DB_DSN="postgresql+asyncpg://user:password@localhost/llm_logs"

# For SQLite (development/testing) - default
export DB_DSN="sqlite+aiosqlite:///./logs/llm_logs.db"
```

## Usage

### Basic Usage

```python
from utils.llm_logger import get_logged_client, init_db
import asyncio

async def main():
    # Initialize database (run once)
    await init_db()
    
    # Get logged client (drop-in replacement for OpenAI client)
    client = get_logged_client(
        api_key="your-api-key",
        base_url="https://openrouter.ai/api/v1"  # optional
    )
    
    # Use exactly like regular OpenAI client
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        session_id="optional-session-id"  # Groups related messages
    )
    
    print(response.choices[0].message.content)

asyncio.run(main())
```

### Async Usage

```python
from utils.llm_logger import get_logged_async_client, init_db

async def main():
    await init_db()
    
    client = get_logged_async_client(
        api_key="your-api-key",
        base_url="https://openrouter.ai/api/v1"
    )
    
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"}
        ],
        session_id="my-session-123"
    )
    
    print(response.choices[0].message.content)
```

### Integration with Existing Agent

To integrate with your existing `Agent` class in `agent.py`:

```python
# In agent.py
from utils.llm_logger import get_logged_client, init_db

class Agent:
    def __init__(self, task: str, display):
        # ... existing code ...
        
        # Replace this:
        # self.client = OpenAI(...)
        
        # With this:
        self.client = get_logged_client(
            api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        
        # Initialize database on first use
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(init_db())
        except RuntimeError:
            asyncio.run(init_db())
```

### Querying Logged Data

```python
from utils.llm_logger import query_interactions

# Get all interactions for a session
interactions = await query_interactions(session_id="my-session-123")

# Get recent assistant messages
assistant_messages = await query_interactions(role="assistant", limit=10)

# Calculate token usage
total_prompt_tokens = sum(i.prompt_tokens for i in interactions)
total_completion_tokens = sum(i.completion_tokens for i in interactions)

print(f"Total tokens used: {total_prompt_tokens + total_completion_tokens}")
```

### Direct Database Queries

```python
from utils.llm_logger import SessionLocal
from sqlalchemy import select

async with SessionLocal() as db:
    # Get conversation history
    query = select(Interaction).where(
        Interaction.session_id == "my-session-123"
    ).order_by(Interaction.created_at)
    
    result = await db.execute(query)
    interactions = result.scalars().all()
    
    for interaction in interactions:
        print(f"{interaction.role}: {interaction.content}")
```

## Environment Variables

- `DB_DSN`: Database connection string
- `OPENAI_API_KEY` or `OPENROUTER_API_KEY`: API key for OpenAI/OpenRouter
- `OPENAI_BASE_URL`: Base URL for API (optional, defaults to OpenRouter)

## File Structure

```
utils/
├── llm_logger.py          # Main logging module
├── init_llm_db.py         # Database initialization script
examples/
├── llm_logging_example.py # Usage examples
docs/
├── llm_logging_setup.md   # This documentation
```

## Database Migration

To create/update database tables:

```bash
# Using the initialization script
python utils/init_llm_db.py

# Or programmatically
python -c "import asyncio; from utils.llm_logger import init_db; asyncio.run(init_db())"
```

## Production Considerations

1. **Use PostgreSQL** for production environments
2. **Set up proper indexes** for performance:
   ```sql
   CREATE INDEX idx_interactions_session_id ON interactions(session_id);
   CREATE INDEX idx_interactions_created_at ON interactions(created_at);
   ```
3. **Monitor database size** and set up log rotation if needed
4. **Use connection pooling** for high-throughput applications
5. **Consider partitioning** by date for large datasets

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure SQLAlchemy dependencies are installed
2. **Database connection**: Check DB_DSN environment variable
3. **Permissions**: Ensure database user has CREATE/INSERT permissions
4. **Event loop**: The sync client falls back to no-logging if in async context

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Examples

See `examples/llm_logging_example.py` for comprehensive usage examples including:
- Basic logging
- Agent integration
- Usage statistics
- Database queries

## Why This Design?

- **Async everywhere** – matches OpenAI async client + high-throughput logging
- **Single transactional write** – prevents mismatched request/response rows  
- **Plain SQLAlchemy models** – easy to query, migrate, or stream to analytics
- **No framework lock-in** – you can swap in LangSmith, OpenTelemetry, or a Kafka producer inside the wrapper without touching calling code

Drop this module into your service and every LLM call is durably logged, queryable (`SELECT * FROM interactions WHERE session_id = ... ORDER BY id;`) and ready for later analysis or replay.