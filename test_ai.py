import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

def test_gemini_pipeline():
    print("--- Testing Gemini Embeddings ---")

    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    sample_text = "FastAPI is a modern web framework for building APIs with Python."
    vector = embeddings_model.embed_query(sample_text)
    
    print(f"Text: '{sample_text}'")
    print(f"Vector Length (Dimensions: {len(vector)})")
    print(f"First 5 numbers in the vector: {vector[:5]}")

    print("--- Testing Gemini LLM ---")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    response = llm.invoke("Explain Retrieval-Augmented Generation (RAG) in one short sentence.")
    print(f"AI Response: {response.content}")



if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Please add your API_KEY to the .env file first!")
    else:
        test_gemini_pipeline()