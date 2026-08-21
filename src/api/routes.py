import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from src.database.vector_store import get_db, AsyncSessionLocal
from src.database.models import DocumentChunk
from src.ingestion.loader import load_document
from src.chunking.chunker import chunk_text
from src.embeddings.embedder import get_embeddings_model
from src.retrieval.retriever import retrieve_context
from src.prompts.prompt_templates import get_rag_system_prompt
from src.llm.llm_client import get_llm_client

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    filename: str | None = None

class QueryResponse(BaseModel):
    query: str
    retrieved_context: str
    answer: str

@router.get("/")
def read_root():
    return {"message": "Welcome to the Enterprise RAG API!"}

@router.get("/health")
def health_check():
    return {"status": "healthy", "database_connected": True}

@router.post("/api/v1/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Ingestion Pipeline
        raw_text = load_document(file_path)
        chunks = chunk_text(raw_text)
        
        embeddings_engine = get_embeddings_model()
        vectors = embeddings_engine.embed_documents(chunks)
        
        async with AsyncSessionLocal() as session:
            async with session.begin():
                for chunk_text_content, vector in zip(chunks, vectors):
                    db_chunk = DocumentChunk(
                        filename=file.filename,
                        page_number=1,
                        content=chunk_text_content,
                        embedding=vector
                    )
                    session.add(db_chunk)
                await session.commit()
                
        return {"filename": file.filename, "message": "Document vectorized and saved successfully!"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(payload: QueryRequest, db: AsyncSession = Depends(get_db)):
    try:
        matched_chunks = await retrieve_context(payload.question, db, payload.filename)
        
        if not matched_chunks:
            raise HTTPException(status_code=404, detail="No relevant context found in database.")
            
        context_text = "\n\n".join([chunk.content for chunk in matched_chunks])
        
        system_prompt = get_rag_system_prompt(context_text, payload.question)
        
        llm = get_llm_client()
        ai_response = await llm.ainvoke(system_prompt)
        
        return QueryResponse(
            query=payload.question,
            retrieved_context=context_text,
            answer=ai_response.content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))