from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.helpers import load_config

def chunk_text(raw_text: str) -> list[str]:
    
    """Splits raw text into smaller chunks based on config."""

    config = load_config()
    chunk_size = config['chunking']['chunk_size']
    chunk_overlap = config['chunking']['chunk_overlap']

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    return text_splitter.split_text(raw_text)