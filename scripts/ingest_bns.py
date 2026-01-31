import json
import sys
import chromadb
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to sys.path to allow imports from app
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))

from app.chroma_store import get_collection, CHROMA_PATH, COLLECTION_NAME

CHUNKS_FILE_PATH = BASE_DIR / "knowledge_base" / "BNS" / "v2024" / "bns_chunks.json"
BATCH_SIZE = 100  # Gemini limit per embed batch (keep <= 100)

def ingest_data():
    # 1. FORCE RESET: Delete existing collection to resolve embedding mismatch
    print(f"Resetting collection '{COLLECTION_NAME}' at {CHROMA_PATH}...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        client.delete_collection(COLLECTION_NAME)
        print("✅ Deleted old collection.")
    except Exception as e:
        print(f"ℹ️ Collection delete skipped (might not exist): {e}")

    # 2. Get (re-create) collection with NEW embedding function
    try:
        collection = get_collection()
    except RuntimeError as e:
        print(f"Error initializing collection: {e}")
        print("Make sure GEMINI_API_KEY is set in your .env file.")
        return

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
