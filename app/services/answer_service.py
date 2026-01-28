from app.services.retrieval_service import retrieve_sections
from schemas.response import AskResponse, Citation, Proof, ProofSource
from app.responses.refusals import NO_LAW_FOUND, NON_LEGAL_QUERY
from typing import List, Dict, Any

LEGAL_INTENT_KEYWORDS = [
    "crime", "offence", "punishment", "section", "law", "act", "ipc", "bns", 
    "illegal", "imprisonment", "fine", "penalty"
]

NON_LEGAL_KEYWORDS = [
    "relationship", "love", "affair", "marriage problem", "cheating partner"
]

def validate_legal_intent(query: str) -> bool:
    """
    Checks if the query contains at least one legal intent keyword
    AND does not contain any non-legal domain keywords.
    """
    query_lower = query.lower()
    
    # Negative Gate: Reject if non-legal keywords are present
    if any(keyword in query_lower for keyword in NON_LEGAL_KEYWORDS):
        return False

    # Positive Gate: Must contain at least one legal keyword
    return any(keyword in query_lower for keyword in LEGAL_INTENT_KEYWORDS)

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
    
    # Base confidence = highest similarity score
    # Docs are typically sorted by relevance, but we take max to be sure
    base_score = max([doc.get("relevance_score", 0.0) for doc in docs])
    
    score = base_score
    
    # +0.2 if punishment keyword detected
    punishment_keywords = ["punishment", "imprisonment", "fine", "death", "forfeiture", "rigorous", "simple"]
    has_punishment = any(any(k in doc.get("text", "").lower() for k in punishment_keywords) for doc in docs)
    if has_punishment:
        score += 0.2
        
    # +0.1 if source freshness confirmed (e.g., 2023, 2024)
    is_fresh = any("2023" in str(doc.get("effective_from", "")) or "2024" in str(doc.get("effective_from", "")) for doc in docs)
    if is_fresh:
        score += 0.1
        
    # Cap at 0.9
    return min(score, 0.9)

def get_answer(query: str) -> AskResponse:
    # 0. Legal Intent Gate
    if not validate_legal_intent(query):
        return AskResponse(
            answer=NON_LEGAL_QUERY,
            citations=[],
            confidence=0.0,
            proof=None
        )

    # 1. Retrieve Docs
    retrieved_docs = retrieve_sections(query)

    # 2. Check relevance
    # retrieve_sections already filters by threshold, so we just check if empty
    relevant_docs = retrieved_docs 

    if not relevant_docs:
        return AskResponse(
            answer=NO_LAW_FOUND,
            citations=[],
            confidence=0.0,
            proof=None
        )

    # 3. Calculate Confidence
    confidence = calculate_confidence(relevant_docs)
    
    # Refusal based on confidence
    if confidence < 0.3:
        return AskResponse(
            answer=NO_LAW_FOUND,
            citations=[],
            confidence=confidence,
            proof=None
        )

    # 4. Generate Answer (Dummy for now, but with confidence)
    # In a real system, this is where the LLM generation happens
    answer = "This is a generated answer based on the retrieved documents."
    
    citations = [
        Citation(act=doc.get("act", "Unknown"), section=doc.get("section", "Unknown"), effective_from=doc.get("effective_from", "Unknown"))
        for doc in relevant_docs
    ]
    
    # 5. Attach proof object
    proof = Proof(
        sources=[
            ProofSource(
                act=doc.get("act", "Unknown"),
                section=doc.get("section", "Unknown"),
                text_snippet=doc.get("text", ""),
                relevance_score=doc.get("relevance_score", 0.0)
            ) for doc in relevant_docs
        ],
        reasoning="This answer is based on the retrieved legal documents."
    )

    response = AskResponse(
        answer=answer,
        citations=citations,
        confidence=confidence,
        proof=proof
    )

    return response
