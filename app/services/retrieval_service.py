import re
import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Any, Optional

# Configuration
# Use environment variable or default to local path relative to project root
DEFAULT_CHROMA_PATH = Path(__file__).parent.parent.parent / "chroma_db"
CHROMA_DB_PATH = os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_CHROMA_PATH))
COLLECTION_NAME = "legal_knowledge_base"

# Retrieval tuning
CANDIDATES_K = 25          # high recall
FINAL_K = 5                # what you return
SIMILARITY_THRESHOLD = 0.35  # lower than before; we rerank + gate later

PUNISHMENT_ANCHORS = [
    "shall be punished", "punished with", "imprisonment", "fine", "death",
    "rigorous imprisonment", "simple imprisonment", "liable to fine"
]

STOPWORDS = {
    "what","is","the","a","an","of","for","in","indian","india","law",
    "under","section","bns","ipc","bnss","bsa","act","please","explain",
    "punishment","penalty"
}

# --- GLOBAL INITIALIZATION (Load once at startup) ---
print(f">>> CHROMA PATH USED: {CHROMA_DB_PATH}")
try:
    _CLIENT = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    _EMBEDDING_FUNC = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    # Use get_or_create to avoid startup crash if collection doesn't exist yet
    _COLLECTION = _CLIENT.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_EMBEDDING_FUNC)
    print("ChromaDB and Embedding Model initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR initializing ChromaDB: {e}")
    _COLLECTION = None
# ----------------------------------------------------

def _normalize_meta(meta: dict) -> dict:
    # Support both old and new metadata keys
    act = meta.get("act") or meta.get("law") or "Unknown"
    section = str(meta.get("section") or "Unknown")
    effective_from = meta.get("effective_from") or meta.get("version") or "Unknown"
    act_name = meta.get("act_name") or act
    version = meta.get("version") or meta.get("ver") or ""

    return {
        "act": act,
        "act_name": act_name,
        "section": section,
        "effective_from": effective_from,
        "version": version,
        "type": meta.get("type", "bare_act"),
    }

def _intent_type(query: str) -> str:
    q = query.lower()
    if re.search(r"\bsection\s+\d{1,4}\b", q):
        return "section_lookup"
    if any(k in q for k in ["punishment", "penalty", "sentence", "imprisonment", "fine", "death"]):
        return "punishment"
    return "general"

def _extract_section_number(query: str) -> Optional[str]:
    m = re.search(r"\bsection\s+(\d{1,4})\b", query.lower())
    return m.group(1) if m else None

def _extract_target_keywords(query: str) -> List[str]:
    # crude but effective: tokenize, remove stopwords, keep a few keywords
    tokens = re.findall(r"[a-zA-Z]+", query.lower())
    keywords = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
    return keywords[:6]

def _anchor_score(text: str, anchors: List[str]) -> float:
    t = text.lower()
    hits = sum(1 for a in anchors if a in t)
    return hits / max(1, len(anchors))

def _keyword_score(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    t = text.lower()
    hits = sum(1 for k in keywords if k in t)
    return hits / len(keywords)

def _final_score(similarity: float, text: str, intent: str, keywords: List[str]) -> float:
    # lexical rerank weights
    if intent == "punishment":
        return 0.55 * similarity + 0.30 * _anchor_score(text, PUNISHMENT_ANCHORS) + 0.15 * _keyword_score(text, keywords)
    return 0.80 * similarity + 0.20 * _keyword_score(text, keywords)

def _format_matches(docs: List[str], metas: List[dict], sims: List[float]) -> List[Dict[str, Any]]:
    matches = []
    for doc_text, meta, sim in zip(docs, metas, sims):
        nm = _normalize_meta(meta or {})
        matches.append({
            "act": nm["act"],
            "section": nm["section"],
            "title": nm.get("act_name", "Unknown"),
            "text": doc_text,
            "effective_from": nm["effective_from"],
            "type": nm["type"],
            "exact_match": False,
            "relevance_score": float(sim),
        })
    return matches

def retrieve_sections(query: str) -> List[Dict[str, Any]]:
    if _COLLECTION is None:
        print("Error: Collection not initialized.")
        return []

    intent = _intent_type(query)

    # 1) Deterministic section lookup (no embeddings)
    if intent == "section_lookup":
        sec = _extract_section_number(query)
        if sec:
            # If query mentions BNS/IPC, you can add act filters; for now keep it simple:
            res = _COLLECTION.get(
                where={"section": str(sec)},
                include=["documents", "metadatas"]
            )
            if res and res.get("documents"):
                # Build matches with high confidence-like score
                docs = res["documents"]
                metas = res["metadatas"]
                sims = [0.99] * len(docs)
                out = _format_matches(docs, metas, sims)
                # mark exact_match True
                for m in out:
                    m["exact_match"] = True
                return out[:FINAL_K]

    # 2) High-recall semantic retrieval
    results = _COLLECTION.query(query_texts=[query], n_results=CANDIDATES_K)

    if not results or not results.get("documents") or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
    distances = results["distances"][0] if results.get("distances") else [1.0] * len(docs)

    # Convert distance->similarity (works for cosine where distance ~ 1 - cosine_sim)
    sims = [max(0.0, 1.0 - float(d)) for d in distances]

    # Basic threshold to discard total junk (keep low because we rerank)
    candidates = []
    keywords = _extract_target_keywords(query)

    for doc_text, meta, sim in zip(docs, metas, sims):
        if sim < SIMILARITY_THRESHOLD:
            continue
        score = _final_score(sim, doc_text, intent, keywords)
        candidates.append((score, sim, doc_text, meta))

    if not candidates:
        return []

    # 3) Rerank and answerability gate (prevents random punishment sections)
    candidates.sort(key=lambda x: x[0], reverse=True)

    filtered = []
    if intent == "punishment":
        # Must have at least one punishment anchor in at least one top chunk
        for score, sim, doc_text, meta in candidates:
            if _anchor_score(doc_text, PUNISHMENT_ANCHORS) > 0:
                filtered.append((score, sim, doc_text, meta))
        # If none contain any punishment language, refuse
        if not filtered:
            return []
    else:
        filtered = candidates

    top = filtered[:FINAL_K]
    out = _format_matches(
        [t[2] for t in top],
        [t[3] for t in top],
        [t[1] for t in top],  # keep raw similarity in relevance_score
    )
    return out
