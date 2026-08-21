import pytest
from src.chunking.chunker import chunk_text
from src.utils.helpers import load_config

def test_chunk_text():

    """Test the text splitting logic."""

    # Given a long text
    long_text = "Word " * 200 # Creates a string with 200 words
    
    # When chunked
    chunks = chunk_text(long_text)
    
    # Then verify it was split based on config
    config = load_config()
    
    assert len(chunks) > 1
    # Check that no chunk exceeds the configured size (plus a small buffer for Langchain formatting)
    for chunk in chunks:
        assert len(chunk) <= config['chunking']['chunk_size'] + 50 

def test_load_config():
    """Test loading the configuration file."""
    config = load_config()
    assert "models" in config
    assert "chunking" in config
    assert "database" in config
    assert config["chunking"]["chunk_size"] > 0