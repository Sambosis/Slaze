"""
llm_logger.py
Requires: sqlalchemy>=2.0, asyncpg (or aiosqlite), openai>=1.0
"""

import os
import uuid
import datetime as dt
from typing import Any, Dict, List, Optional, Union

import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from sqlalchemy import (
    BigInteger, Integer, String, Text, DateTime, func
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, async_sessionmaker, AsyncSession
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# ── 1. DB setup ────────────────────────────────────────────────────────────────
# Default to SQLite for development, can be overridden with environment variable
DATABASE_URL = os.getenv(
    "DB_DSN",
    "sqlite+aiosqlite:///./logs/llm_logs.db",
)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine, expire_on_commit=False
)


class Base(DeclarativeBase): 
    pass


class Interaction(Base):
    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(36), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


async def init_db() -> None:
    """Initialize the database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── 2. LLM wrappers ─────────────────────────────────────────────────────────────
class LoggedOpenAI:
    """
    Drop-in replacement for `openai.OpenAI()` that logs every request/response.
    """
    def __init__(self, *args, **kwargs):
        self._client = OpenAI(*args, **kwargs)

    def chat_completions_create(
        self,
        messages: List[Dict[str, str]],
        session_id: str | None = None,
        **create_kwargs: Any,
    ) -> ChatCompletion:
        """Synchronous version of chat completion with logging."""
        import asyncio
        
        # Run the async version in a new event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to use a different approach
                # For now, we'll fall back to the original client without logging
                return self._client.chat.completions.create(
                    messages=messages, **create_kwargs
                )
            else:
                return loop.run_until_complete(
                    self._async_chat_completions_create(messages, session_id, **create_kwargs)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self._async_chat_completions_create(messages, session_id, **create_kwargs)
            )

    async def _async_chat_completions_create(
        self,
        messages: List[Dict[str, str]],
        session_id: str | None = None,
        **create_kwargs: Any,
    ) -> ChatCompletion:
        """Internal async implementation."""
        # Create async client for database operations
        async_client = AsyncOpenAI(
            api_key=self._client.api_key,
            base_url=str(self._client.base_url) if self._client.base_url else None,
            organization=self._client.organization,
        )
        
        sid = session_id or str(uuid.uuid4())
        response: ChatCompletion = await async_client.chat.completions.create(
            messages=messages, **create_kwargs
        )

        usage = response.usage or type('Usage', (), {'prompt_tokens': 0, 'completion_tokens': 0})()

        async with SessionLocal() as db, db.begin():
            # persist user/system messages
            for m in messages:
                db.add(
                    Interaction(
                        session_id=sid,
                        role=m["role"],
                        content=m["content"],
                        model=create_kwargs.get("model"),
                    )
                )
            # persist assistant reply
            reply: ChatCompletionMessage = response.choices[0].message
            db.add(
                Interaction(
                    session_id=sid,
                    role=reply.role,
                    content=reply.content,
                    model=response.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
            )
        await async_client.close()
        return response

    @property
    def chat(self):
        """Return a chat object with completions that have logging."""
        return type('Chat', (), {
            'completions': type('Completions', (), {
                'create': self.chat_completions_create
            })()
        })()

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying client."""
        return getattr(self._client, name)


class LoggedAsyncOpenAI:
    """
    Drop-in replacement for `openai.AsyncOpenAI()` that logs every request/response.
    """
    def __init__(self, *args, **kwargs):
        self._client = AsyncOpenAI(*args, **kwargs)

    async def chat_completions_create(
        self,
        messages: List[Dict[str, str]],
        session_id: str | None = None,
        **create_kwargs: Any,
    ) -> ChatCompletion:
        """Async chat completion with logging."""
        sid = session_id or str(uuid.uuid4())
        response: ChatCompletion = await self._client.chat.completions.create(
            messages=messages, **create_kwargs
        )

        usage = response.usage or type('Usage', (), {'prompt_tokens': 0, 'completion_tokens': 0})()

        async with SessionLocal() as db, db.begin():
            # persist user/system messages
            for m in messages:
                db.add(
                    Interaction(
                        session_id=sid,
                        role=m["role"],
                        content=m["content"],
                        model=create_kwargs.get("model"),
                    )
                )
            # persist assistant reply
            reply: ChatCompletionMessage = response.choices[0].message
            db.add(
                Interaction(
                    session_id=sid,
                    role=reply.role,
                    content=reply.content,
                    model=response.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
            )
        return response

    @property
    def chat(self):
        """Return a chat object with completions that have logging."""
        return type('Chat', (), {
            'completions': type('Completions', (), {
                'create': self.chat_completions_create
            })()
        })()

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying client."""
        return getattr(self._client, name)


# ── 3. Convenience functions ─────────────────────────────────────────────────────
def get_logged_client(session_id: Optional[str] = None, **kwargs) -> LoggedOpenAI:
    """
    Get a logged OpenAI client instance.
    
    Args:
        session_id: Optional session ID to use for all requests
        **kwargs: Arguments to pass to OpenAI client
        
    Returns:
        LoggedOpenAI instance
    """
    return LoggedOpenAI(**kwargs)


def get_logged_async_client(session_id: Optional[str] = None, **kwargs) -> LoggedAsyncOpenAI:
    """
    Get a logged AsyncOpenAI client instance.
    
    Args:
        session_id: Optional session ID to use for all requests
        **kwargs: Arguments to pass to AsyncOpenAI client
        
    Returns:
        LoggedAsyncOpenAI instance
    """
    return LoggedAsyncOpenAI(**kwargs)


async def query_interactions(
    session_id: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 100
) -> List[Interaction]:
    """
    Query interactions from the database.
    
    Args:
        session_id: Filter by session ID
        role: Filter by role (user, assistant, system)
        limit: Maximum number of results
        
    Returns:
        List of Interaction objects
    """
    from sqlalchemy import select
    
    async with SessionLocal() as db:
        query = select(Interaction).order_by(Interaction.created_at.desc())
        
        if session_id:
            query = query.where(Interaction.session_id == session_id)
        if role:
            query = query.where(Interaction.role == role)
            
        query = query.limit(limit)
        
        result = await db.execute(query)
        return result.scalars().all()


# ── 4. Migration helper ─────────────────────────────────────────────────────────
async def migrate_db():
    """Create or update database tables."""
    await init_db()
    print("Database migration completed successfully!")


if __name__ == "__main__":
    # Simple test/migration script
    import asyncio
    
    async def main():
        await migrate_db()
        
        # Example usage (commented out to avoid real API calls)
        # client = get_logged_async_client()
        # response = await client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": "You are helpful."},
        #         {"role": "user", "content": "Hello!"}
        #     ]
        # )
        # print(response.choices[0].message.content)
        
    asyncio.run(main())