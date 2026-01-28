import yaml
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Any

# Configuration (should match ingestion script)
CHROMA_DB_PATH = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION_NAME = "legal_knowledge_base"

def get_legal_sources():
    config_path = Path(__file__).parent.parent.parent / "config" / "legal_sources.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def retrieve_sections(query: str) -> List[Dict[str, Any]]:
    legal_sources = get_legal_sources()
    
    # Initialize Chroma Client
    # Note: In a production app, you'd want to initialize this once (e.g., in a startup event)
    # and pass it around, rather than creating it on every request.
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_func
        )
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return []

    # Query the collection
    results = collection.query(
        query_texts=[query],
        n_results=3 # Retrieve top 3 matches
    )

    # Transform results to the expected format
    matches = []
    if results["documents"]:
        for i, doc_text in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            matches.append({
                "act": metadata.get("law", "Unknown Act"), # Updated key
                "section": metadata.get("section", "Unknown Section"),
                "title": "Unknown Title", 
                "text": doc_text,
                "effective_from": metadata.get("version", "Unknown Date"),
                "type": metadata.get("type", "bare_act"), # Updated key usage
                "exact_match": False, 
                "relevance_score": results["distances"][0][i] if results["distances"] else 0.0
            })

    return matches
