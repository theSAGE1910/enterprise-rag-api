import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_read_root(client: AsyncClient):

    """Test the root endpoint."""

    response = await client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Enterprise RAG API!"}

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):

    """Test the health check endpoint."""

    response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "database_connected": True}

@pytest.mark.asyncio
async def test_upload_document(client: AsyncClient):

    """Test the document upload and ingestion endpoint."""

    # Create a dummy text file in memory for testing
    file_content = b"This is a test policy document. The secret code is 12345."
    files = {"file": ("test_doc.txt", file_content, "text/plain")}
    
    # We might need to mock the actual Gemini API calls here in a real production 
    # environment to avoid hitting rate limits or paying for test tokens.
    # For now, we assume the API key is present in the testing environment.
    
    response = await client.post("/api/v1/upload", files=files)
    
    assert response.status_code == 200
    assert response.json()["filename"] == "test_doc.txt"
    assert "successfully" in response.json()["message"]

@pytest.mark.asyncio
async def test_query_knowledge_base(client: AsyncClient):

    """Test the RAG query endpoint."""
    
    # Note: This test relies on the upload test having run previously or 
    # test data being seeded. In a strictly isolated test suite, you would 
    # seed the database directly via the db_session fixture before querying.
    
    payload = {
        "question": "What is the secret code?",
        "filename": "test_doc.txt"
    }
    
    response = await client.post("/api/v1/query", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["question"]
    assert "retrieved_context" in data
    assert "answer" in data
    # The LLM should extract the code from the context
    assert "12345" in data["answer"]