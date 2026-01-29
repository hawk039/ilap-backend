from pathlib import Path
from app.services.retrieval_service import retrieve_sections
from schemas.response import AskResponse, Citation, Proof, ProofSource
from app.responses.refusals import NO_LAW_FOUND, NON_LEGAL_QUERY, UNDERSPECIFIED_QUERY, MODEL_EMPTY_RESPONSE
from app.llm.factory import get_llm
from typing import List, Dict, Any

# Helper to robustly access document text
def doc_text(doc: Dict[str, Any]) -> str:
    return (doc.get("text") or doc.get("text_snippet") or "").strip()

LEGAL_INTENT_KEYWORDS = [
    "crime", "offence", "punishment", "section", "law", "act", "ipc", "bns", 
    "illegal", "imprisonment", "fine", "penalty"
]

NON_LEGAL_KEYWORDS = [
    "relationship", "love", "affair", "marriage problem", "cheating partner"
]

MAX_CONTEXT_CHARS = 6000

def classify_intent(query: str) -> str:
    """
    Classifies the query intent into one of three states.
    """
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in NON_LEGAL_KEYWORDS):
        return "non_legal"
    if any(keyword in query_lower for keyword in LEGAL_INTENT_KEYWORDS):
        return "legal"
    return "underspecified_legal"

def calculate_confidence(docs: List[Dict[str, Any]]) -> float:
    """
    Calculates confidence score based on heuristics.
    """
    if not docs:
        return 0.0
    
    base_score = max([doc.get("relevance_score", 0.0) for doc in docs])
    score = base_score
    
    punishment_keywords = ["punishment", "imprisonment", "fine", "death", "forfeiture", "rigorous", "simple"]
    # Use the robust doc_text helper
    has_punishment = any(any(k in doc_text(doc).lower() for k in punishment_keywords) for doc in docs)
    if has_punishment:
        score += 0.2
        
    is_fresh = any("2023" in str(doc.get("effective_from", "")) or "2024" in str(doc.get("effective_from", "")) for doc in docs)
    if is_fresh:
        score += 0.1
        
    return min(score, 0.9)

def get_answer(query: str) -> AskResponse:
    # 0. Intent Classification Gate
    intent = classify_intent(query)
    if intent == "non_legal":
        return AskResponse(answer=NON_LEGAL_QUERY, citations=[], confidence=0.0, proof=None)
    if intent == "underspecified_legal":
        return AskResponse(answer=UNDERSPECIFIED_QUERY, citations=[], confidence=0.0, proof=None)

    # 1. Retrieve Docs
    retrieved_docs = retrieve_sections(query)
    if not retrieved_docs:
        return AskResponse(answer=NO_LAW_FOUND, citations=[], confidence=0.0, proof=None)

    # 2. Calculate Confidence & Check Thresholds
    confidence = calculate_confidence(retrieved_docs)
    if confidence < 0.3:
        return AskResponse(answer=NO_LAW_FOUND, citations=[], confidence=confidence, proof=None)

    # 3. Construct Citations and Proof (BEFORE LLM call)
    # Deduplicate citations
    seen_citations = set()
    citations = []
    for doc in retrieved_docs:
        key = (doc.get("act"), doc.get("section"), doc.get("effective_from"))
        if key in seen_citations:
            continue
        seen_citations.add(key)
        citations.append(Citation(
            act=doc.get("act", "Unknown"),
            section=doc.get("section", "Unknown"),
            effective_from=doc.get("effective_from", "Unknown")
        ))
    
    proof = Proof(
        sources=[
            ProofSource(
                act=doc.get("act", "Unknown"),
                section=doc.get("section", "Unknown"),
                text_snippet=doc_text(doc),
                relevance_score=doc.get("relevance_score", 0.0)
            ) for doc in retrieved_docs
        ],
        reasoning="The answer is synthesized from the retrieved legal provisions."
    )

    # 4. Synthesize Answer
    context = "\n\n---\n\n".join([doc_text(doc) for doc in retrieved_docs])
    
    # Cap context length
    context = context[:MAX_CONTEXT_CHARS]

    # Don't call LLM if context is empty
    if not context.strip():
        return AskResponse(answer=NO_LAW_FOUND, citations=[], confidence=0.0, proof=None)
    
    prompt_template_path = Path(__file__).parent.parent / "prompts" / "legal_synthesis.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{{context}}", context).replace("{{query}}", query)
    
    llm = get_llm()
    answer = llm.generate(prompt)

    # Handle LLM failure but preserve proof
    if not answer or not answer.strip():
        return AskResponse(
            answer=MODEL_EMPTY_RESPONSE, 
            citations=citations, 
            confidence=confidence, 
            proof=proof
        )
    
    return AskResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        proof=proof
    )
