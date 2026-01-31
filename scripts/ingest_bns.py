import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to sys.path to allow imports from app
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from app.chroma_store import get_collection

CHUNKS_FILE_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns_chunks.json"
BATCH_SIZE = 100  # Gemini limit per embed batch (keep <= 100)

def ingest_data():
    # Get collection from centralized store (uses Gemini embeddings)
    collection = get_collection()
    
    if not CHUNKS_FILE_PATH.exists():
        print(f"Error: Chunks file not found at {CHUNKS_FILE_PATH}")
        return

    print(f"Reading chunks from {CHUNKS_FILE_PATH}...")
    chunks = json.loads(CHUNKS_FILE_PATH.read_text(encoding="utf-8"))
    
    if not chunks:
        print("No data to ingest.")
        return

    print(f"Found {len(chunks)} chunks. Ingesting in batches of {BATCH_SIZE}...")

    section_counts = defaultdict(int)

    batch_docs, batch_metas, batch_ids = [], [], []
    total_ingested = 0

    def flush_batch():
        nonlocal total_ingested, batch_docs, batch_metas, batch_ids
        if not batch_docs:
            return

        # If you re-run ingestion often, prefer upsert instead of add
        # (add will error if IDs already exist)
        try:
            collection.upsert(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)
        except AttributeError:
            # fallback if your chroma version doesn't have upsert
            collection.add(documents=batch_docs, metadatas=batch_metas, ids=batch_ids)

        total_ingested += len(batch_docs)
        print(f"Ingested {total_ingested} / {len(chunks)}")

        batch_docs = []
        batch_metas = []
        batch_ids = []

    for c in chunks:
        section = c["section"]

        count = section_counts[section]
        unique_id = f'BNS_2024_{section}' if count == 0 else f'BNS_2024_{section}_{count}'
        section_counts[section] += 1

        batch_ids.append(unique_id)
        batch_docs.append(c["text"])
        batch_metas.append({
            "law": c["act"],
            "act": c["act"],
            "section": c["section"],
            "effective_from": c["effective_from"],
            "version": "v2024",
            "type": "bare_act"
        })

        if len(batch_docs) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    print(f"✅ Successfully ingested {total_ingested} chunks.")

if __name__ == "__main__":
    ingest_data()
