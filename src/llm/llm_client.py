from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.helpers import load_config

def get_llm_client() -> ChatGoogleGenerativeAI:

    """Initializes and returns the LLM client."""

    config = load_config()
    model_name = config['models']['llm']
    temperature = config['models']['llm_temperature']
    
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)