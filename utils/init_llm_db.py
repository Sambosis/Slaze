#!/usr/bin/env python3
"""
Database initialization script for LLM logging system.
Run this script to create the database tables.
"""

import asyncio
import os
from pathlib import Path

from llm_logger import init_db, DATABASE_URL


async def main():
    """Initialize the LLM logging database."""
    print("Initializing LLM logging database...")
    print(f"Database URL: {DATABASE_URL}")
    
    # Create logs directory if it doesn't exist
    if DATABASE_URL.startswith("sqlite"):
        db_path = DATABASE_URL.replace("sqlite+aiosqlite:///", "")
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        print(f"Ensured database directory exists: {db_dir}")
    
    try:
        await init_db()
        print("✅ Database initialized successfully!")
        print("\nYou can now use the logged OpenAI clients in your application.")
        print("\nExample usage:")
        print("  from utils.llm_logger import get_logged_client")
        print("  client = get_logged_client(api_key='...')")
        print("  response = client.chat.completions.create(...)")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        print("\nPlease check:")
        print("1. Database connection string (DB_DSN environment variable)")
        print("2. Database server is running (for PostgreSQL)")
        print("3. Required Python packages are installed:")
        print("   pip install sqlalchemy>=2.0 asyncpg aiosqlite")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)