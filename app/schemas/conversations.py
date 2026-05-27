from datetime import datetime
from typing import Any

from pydantic import Field

from app.schemas.common import APIModel, ConversationStatus


class CreateConversationRequest(APIModel):
    law_type: str = Field(min_length=2, max_length=120)
    title: str | None = Field(default=None, max_length=200)


class UpdateConversationRequest(APIModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    status: ConversationStatus | None = None


class AskConversationRequest(APIModel):
    query: str = Field(min_length=1, max_length=4000)
    idempotency_key: str | None = Field(default=None, max_length=128)


class Citation(APIModel):
    act: str
    section: str
    effective_from: str | None = None
    law_type: str | None = None


class ProofSource(APIModel):
    act: str
    section: str
    text_snippet: str
    relevance_score: float


class ProofPayload(APIModel):
    sources: list[ProofSource] = Field(default_factory=list)
    reasoning: str | None = None


class NormalizedAnswerPayload(APIModel):
    answer: str
    disclaimer: str | None = None
    category_note: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    proof: ProofPayload | None = None


class ConversationSummary(APIModel):
    id: str
    title: str
    law_type: str
    ai_session_id: str | None = None
    latest_context_turn_id: str | None = None
    last_message_preview: str | None = None
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime | None = None
    last_message_at: datetime | None = None


class ConversationMessage(APIModel):
    id: str
    role: str
    text: str
    disclaimer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime


class AskConversationResponse(APIModel):
    conversation_id: str
    message_id: str
    answer: str
    disclaimer: str | None = None
    category_note: str | None = None
    returned_turn_id: str | None = None
    created_at: datetime


class AIServiceRequest(APIModel):
    query: str
    law_type: str | None = None
    session_id: str | None = None
    context_turn_id: str | None = None


class AIServiceResponse(APIModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    disclaimer: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    category_note: str | None = None
    proof: ProofPayload | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
