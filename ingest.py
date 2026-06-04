import asyncio
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from sqlalchemy.future import select

from database import AsyncSessionLocal
from models import DocumentChunk

load_dotenv()

async def ingest_document(file_path: str):
    print(f"Reading document: {file_path}...")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    if file_path.lower().endswith('.pdf'):
        print("PDF detected. Initializing PyPDFLoader...")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        raw_text = "\n\n".join([page.page_content for page in pages])
    else:
        print("Text file detected. Reading directly...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len
    )
    
    chunks = text_splitter.split_text(raw_text)
    print(f"Split document into {len(chunks)} individual chunks.")

    embeddings_engine = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    print("Generating embeddings via Gemini API...")
    vectors = embeddings_engine.embed_documents(chunks)

    filename = os.path.basename(file_path)
    print("Saving chunks and vectors to pgvector database...")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for i, (chunk_text, vector) in enumerate(zip(chunks, vectors)):
                db_chunk = DocumentChunk(
                    filename=filename,
                    page_number=1,
                    content=chunk_text,
                    embedding=vector
                )
                session.add(db_chunk)
        
        await session.commit()
    
    print("Ingestion pipeline complete! Data successfully vectorized and stored.")

if __name__ == "__main__":
    target_file = os.path.join("data", "company_policy.txt")
    asyncio.run(ingest_document(target_file))