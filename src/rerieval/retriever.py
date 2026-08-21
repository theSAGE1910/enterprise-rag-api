from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from src.database.models import DocumentChunk
from src.embeddings.embedder import get_embeddings_model

async def retrieve_context(query: str, db: AsyncSession, filename: str | None = None, limit: int = 2) -> list[DocumentChunk]:

    """Retrieves relevant document chunks from the vector database based on a query."""

    embeddings_engine = get_embeddings_model()
    query_vector = await embeddings_engine.aembed_query(query)

    stmt = select(DocumentChunk)

    if filename:
        stmt = stmt.filter(DocumentChunk.filename == filename)

    stmt = stmt.order_by(DocumentChunk.embedding.cosine_distance(query_vector)).limit(limit)

    result = await db.execute(stmt)
    return result.scalars().all()