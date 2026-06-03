import asyncio
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

from database import get_db
from models import DocumentChunk

load_dotenv()

app = FastAPI(
    title="Enterprise RAG API",
    description="Knowledge base search API using pgvector and LangChain",
    version="0.1.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    query: str
    retrieved_context: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Enterprise RAG API!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "database_connected": True}

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_knowledge_base(payload: QueryRequest, db: AsyncSession = Depends(get_db)):
    try:
        embeddings_engine = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        query_vector = await embeddings_engine.aembed_query(payload.question)

        stmt = (
            select(DocumentChunk)
            .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
            .limit(2)
        )
        result = await db.execute(stmt)
        matched_chunks = result.scalars().all()

        if not matched_chunks:
            raise HTTPException(status_code=404, detail="No relevant context found in database.")
        
        context_text = "\n\n".join([chunk.content for chunk in matched_chunks])

        system_prompt = (
            "You are a secure corporate assistant. Use ONLY the following retrieved business context to answer "
            "the employee's question. If the answer cannot be found in the context, state clearly that you do "
            "not possess that information.\n\n"
            f"--- CONTEXT ---\n{context_text}\n----------------\n\n"
            f"QUESTION: {payload.question}"
        )

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
        ai_response = await llm.ainvoke(system_prompt)

        return QueryResponse(
            query=payload.question,
            retrieved_context=context_text,
            answer=ai_response.content
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))