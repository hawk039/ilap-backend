import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from collections import defaultdict

# Configuration
BASE_DIR = Path(__file__).parent.parent
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
CHUNKS_FILE_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns_chunks.json"
COLLECTION_NAME = "legal_knowledge_base"

def ingest_data():
    # Initialize Chroma Client
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    # Using default embedding function (all-MiniLM-L6-v2)
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # Rebuild from scratch (MVP friendly)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' to start fresh.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    if not CHUNKS_FILE_PATH.exists():
        print(f"Error: Chunks file not found at {CHUNKS_FILE_PATH}")
        return

    print(f"Reading chunks from {CHUNKS_FILE_PATH}...")
    chunks = json.loads(CHUNKS_FILE_PATH.read_text(encoding="utf-8"))
    
    if not chunks:
        print("No data to ingest.")
        return

    print(f"Found {len(chunks)} chunks. Ingesting...")

    docs, metas, ids = [], [], []
    section_counts = defaultdict(int)

    for c in chunks:
        section = c["section"]
        # Handle duplicate sections by appending an index
        count = section_counts[section]
        if count == 0:
            unique_id = f'BNS_2024_{section}'
        else:
            unique_id = f'BNS_2024_{section}_{count}'
        
        section_counts[section] += 1
        
        ids.append(unique_id)
        docs.append(c["text"])
        
        # Metadata for filtering and citation
        metas.append({
            "law": c["act"], # Mapping 'act' to 'law' to match retrieval service expectation
            "act": c["act"],
            "section": c["section"],
            "effective_from": c["effective_from"],
            "version": "v2024",
            "type": "bare_act" # Adding type for confidence scoring
        })

    # Add to collection
    collection.add(documents=docs, metadatas=metas, ids=ids)

    print(f"Successfully ingested {len(docs)} chunks into collection '{COLLECTION_NAME}' at {CHROMA_DB_PATH}")

if __name__ == "__main__":
    ingest_data()
