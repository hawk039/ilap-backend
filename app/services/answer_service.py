from pathlib import Path
from app.services.retrieval_service import retrieve_sections
from schemas.response import AskResponse, Citation, Proof, ProofSource
from app.responses.refusals import NO_LAW_FOUND, NON_LEGAL_QUERY, UNDERSPECIFIED_QUERY
from app.llm.local_llm import LocalLLM
from typing import List, Dict, Any

LEGAL_INTENT_KEYWORDS = [
    "crime", "offence", "punishment", "section", "law", "act", "ipc", "bns", 
    "illegal", "imprisonment", "fine", "penalty"
]

NON_LEGAL_KEYWORDS = [
    "relationship", "love", "affair", "marriage problem", "cheating partner"
]

def classify_intent(query: str) -> str:
    """
    Classifies the query intent into one of three states:
    - "non_legal": Contains non-legal domain keywords.
    - "legal": Contains explicit legal keywords.
    - "underspecified_legal": Neither of the above.
    """
    query_lower = query.lower()
    
    # 1. Check for Non-Legal Intent
    if any(keyword in query_lower for keyword in NON_LEGAL_KEYWORDS):
        return "non_legal"

    # 2. Check for Explicit Legal Intent
    if any(keyword in query_lower for keyword in LEGAL_INTENT_KEYWORDS):
        return "legal"

    # 3. Default to Underspecified
    return "underspecified_legal"

def calculate_confidence(docs: List[Dict[str, Any]]) -> float:
    """
    Calculates confidence score based on heuristics.
    Base confidence = highest similarity score
    +0.2 if punishment keyword detected
    +0.1 if source freshness confirmed
    Cap at 0.9
    """
    if not docs:
        return 0.0
    
    base_score = max([doc.get("relevance_score", 0.0) for doc in docs])
    score = base_score
    
    punishment_keywords = ["punishment", "imprisonment", "fine", "death", "forfeiture", "rigorous", "simple"]
    has_punishment = any(any(k in doc.get("text", "").lower() for k in punishment_keywords) for doc in docs)
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

    # 1. Retrieve Docs (Only if intent is "legal")
    retrieved_docs = retrieve_sections(query)

    if not retrieved_docs:
        return AskResponse(answer=NO_LAW_FOUND, citations=[], confidence=0.0, proof=None)

    # 2. Calculate Confidence & Check Thresholds
    confidence = calculate_confidence(retrieved_docs)
    
    if confidence < 0.3:
        return AskResponse(answer=NO_LAW_FOUND, citations=[], confidence=confidence, proof=None)

    # 3. Synthesize Answer (if confidence is sufficient)
    # Construct context from retrieved documents
    context = "\n\n---\n\n".join([doc['text'] for doc in retrieved_docs])
    
    # Load and format prompt
    prompt_template_path = Path(__file__).parent.parent / "prompts" / "legal_synthesis.txt"
    with open(prompt_template_path, "r") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.replace("{{context}}", context).replace("{{query}}", query)
    
    # Generate answer using LLM
    llm = LocalLLM()
    answer = llm.generate(prompt)
    
    # 4. Construct Citations and Proof
    citations = [
        Citation(act=doc.get("act", "Unknown"), section=doc.get("section", "Unknown"), effective_from=doc.get("effective_from", "Unknown"))
        for doc in retrieved_docs
    ]
    
    proof = Proof(
        sources=[
            ProofSource(
                act=doc.get("act", "Unknown"),
                section=doc.get("section", "Unknown"),
                text_snippet=doc.get("text", ""),
                relevance_score=doc.get("relevance_score", 0.0)
            ) for doc in retrieved_docs
        ],
        reasoning="The answer is synthesized from the retrieved legal provisions."
    )

    return AskResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        proof=proof
    )
