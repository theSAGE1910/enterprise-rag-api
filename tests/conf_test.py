import os
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text

# Import your FastAPI app and database components
from main import app
from src.database.vector_store import get_db
from src.database.models import Base

# --- Test Database Configuration ---
# Use a specific test database URL. If not provided, it attempts to manipulate a local test db.
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/test_rag_db")

# Create a specialized engine and session for testing
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def override_get_db():

    """Dependency override to use the test database."""

    async with TestingSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# Apply the dependency override to the FastAPI app
app.dependency_overrides[get_db] = override_get_db

@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_database():

    """
    Sets up the test database schema before any tests run and tears it down after.
    Requires the test database to exist and have pgvector installed.
    """

    async with test_engine.begin() as conn:
        # Ensure the vector extension is available in the test db
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Create all tables defined in your models
        await conn.run_sync(Base.metadata.create_all)
    
    yield # Run the tests
    
    async with test_engine.begin() as conn:
        # Drop all tables after tests are done
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def db_session() -> AsyncSession:

    """Fixture providing a test database session."""

    async with TestingSessionLocal() as session:
        yield session
        # Optionally, you could rollback here to keep tests completely isolated
        # await session.rollback()

@pytest_asyncio.fixture
async def client():

    """Fixture providing an async HTTP client for the FastAPI app."""
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac