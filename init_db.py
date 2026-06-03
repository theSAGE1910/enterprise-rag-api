import asyncio
from database import engine, Base

import models 
from sqlalchemy import text

async def init_database():
    print("Connecting to database and initializing components...")
    
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("pgvector extension enabled!")

        await conn.run_sync(Base.metadata.create_all)
        print("Database tables created successfully!")

if __name__ == "__main__":
    asyncio.run(init_database())