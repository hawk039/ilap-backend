from app.services.retrieval_service import retrieve_sections
from schemas.response import AskResponse, Citation, Proof, ProofSource
from typing import List, Dict, Any

def calculate_confidence(docs: List[Dict[str, Any]]) -> float:
    """
    Calculates confidence score based on the type of documents retrieved.
    High (0.9) -> Bare Act + exact section
    Medium (0.6) -> Act + interpretation
    Low (0.3) -> Case-law dependent
    """
    if not docs:
        return 0.0
    
    has_bare_act = any(doc.get("type") == "bare_act" for doc in docs)
    has_exact_section = any(doc.get("exact_match", False) for doc in docs)
    has_interpretation = any(doc.get("type") == "interpretation" for doc in docs)
    has_case_law = any(doc.get("type") == "case_law" for doc in docs)

    if has_bare_act and has_exact_section:
        return 0.9
    elif has_bare_act or has_interpretation:
        return 0.6
    elif has_case_law:
        return 0.3
    
    return 0.1 # Default low confidence

def get_answer(query: str) -> AskResponse:
    # 1. Retrieve Docs
    retrieved_docs = retrieve_sections(query)

    # 2. Check relevance (dummy implementation)
    # In a real scenario, we would filter retrieved_docs based on relevance scores
    relevant_docs = retrieved_docs 

    # 3. Generate answer (dummy implementation)
    if not relevant_docs:
        answer = "Could not find a relevant legal provision."
        confidence = 0.0
        citations = []
        proof = None
    else:
        answer = "This is a generated answer based on the retrieved documents."
        confidence = calculate_confidence(relevant_docs)
        
        citations = [
            Citation(act=doc.get("act", "Unknown"), section=doc.get("section", "Unknown"), effective_from=doc.get("effective_from", "Unknown"))
            for doc in relevant_docs
        ]
        
        # 4. Attach proof object
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

    # 5. Validate schema (handled by FastAPI's response_model)
    # 6. Return OR refuse
    return response
