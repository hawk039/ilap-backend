import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path to allow imports from app
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from app.chroma_store import get_collection

CHUNKS_FILE_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns_chunks.json"

def ingest_data():
    # Get collection from centralized store (uses Gemini embeddings)
    collection = get_collection()
    
    # Optional: Clear existing data if you want a fresh start
    # collection.delete() # Be careful with this in production!
    # For now, we'll just add/upsert. If you want to clear, you might need to access the client directly
    # or just delete the chroma_db folder manually.
    
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
            "law": c["act"],
            "act": c["act"],
            "section": c["section"],
            "effective_from": c["effective_from"],
            "version": "v2024",
            "type": "bare_act"
        })

    # Add to collection
    collection.add(documents=docs, metadatas=metas, ids=ids)

    print(f"Successfully ingested {len(docs)} chunks into collection.")

if __name__ == "__main__":
    ingest_data()
