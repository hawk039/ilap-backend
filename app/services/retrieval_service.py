import yaml
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Any

# Configuration
CHROMA_DB_PATH = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION_NAME = "legal_knowledge_base"
SIMILARITY_THRESHOLD = 0.5

def get_legal_sources():
    config_path = Path(__file__).parent.parent.parent / "config" / "legal_sources.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def retrieve_sections(query: str) -> List[Dict[str, Any]]:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return []

    results = collection.query(query_texts=[query], n_results=3)

    matches = []
    if results["documents"]:
        for i, doc_text in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 1.0
            similarity = 1 - distance
            
            if similarity < SIMILARITY_THRESHOLD:
                continue

            metadata = results["metadatas"][0][i]
            matches.append({
                "act": metadata.get("law", "Unknown Act"),
                "section": metadata.get("section", "Unknown Section"),
                "title": "Unknown Title", 
                "text": doc_text,
                "effective_from": metadata.get("version", "Unknown Date"),
                "type": metadata.get("type", "bare_act"),
                "exact_match": False, 
                "relevance_score": similarity
            })

    return matches
