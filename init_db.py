import asyncio
from sqlalchemy import text
from src.database.vector_store import engine
from src.database.models import Base

async def init_database():
    print("Connecting to database and initializing components...")
    
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("pgvector extension enabled!")
        await conn.run_sync(Base.metadata.create_all)
        print("Database tables created successfully!")

if __name__ == "__main__":
    asyncio.run(init_database())