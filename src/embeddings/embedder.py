from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.utils.helpers import load_config

def get_embeddings_model() -> GoogleGenerativeAIEmbeddings:

    """Initializes and returns the embeddings model."""

    config = load_config()
    model_name = config['models']['embeddings']
    return GoogleGenerativeAIEmbeddings(model=model_name)