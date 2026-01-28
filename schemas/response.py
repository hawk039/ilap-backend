from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class Citation(BaseModel):
    act: str
    section: str
    effective_from: str

class ProofSource(BaseModel):
    act: str
    section: str
    text_snippet: str
    relevance_score: float

class Proof(BaseModel):
    sources: List[ProofSource]
    reasoning: str

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence: float
    disclaimer: str = "This response is informational and not legal advice."
    proof: Optional[Proof] = None
