import os
import chromadb
from pathlib import Path
from .gemini_embeddings import GeminiEmbeddingFunction

# Default to local project directory if env var not set
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"
CHROMA_PATH = os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_CHROMA_PATH))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "legal_knowledge_base")

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    print(f">>> INITIALIZING CHROMA AT: {CHROMA_PATH}")
    emb_fn = GeminiEmbeddingFunction()
    _client = chromadb.PersistentClient(path=CHROMA_PATH)

    # IMPORTANT: embedding_function must match ingest time + query time
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection
