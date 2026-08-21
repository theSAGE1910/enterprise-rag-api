import os
from langchain_community.document_loaders import PyPDFLoader

def load_document(file_path: str) -> str:

    """Loads a document and returns its raw text."""
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        return "\n\n".join([page.page_content for page in pages])
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()