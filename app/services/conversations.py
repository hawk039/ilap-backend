from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import conflict, not_found
from app.integrations.ai_client import ask_ai_service
from app.models import Conversation, Message, User
from app.schemas.conversations import (
    AIServiceRequest,
    AskConversationResponse,
    ConversationMessage,
    ConversationSummary,
    CreateConversationRequest,
    NormalizedAnswerPayload,
    UpdateConversationRequest,
)
from app.services.audit import write_audit_log
from app.services.categories import get_category_by_law_type


def _now() -> datetime:
    return datetime.now(UTC)


def _resolve_ai_law_type(conversation: Conversation) -> str | None:
    messages = sorted(conversation.messages, key=lambda item: item.created_at, reverse=True)
    for message in messages:
        normalized = message.normalized_answer_payload or {}
        citations = normalized.get("citations", [])
        if not citations:
            continue
        first_citation = citations[0] or {}
        law_type = first_citation.get("law_type") or first_citation.get("lawType")
        if isinstance(law_type, str) and law_type.strip():
            return law_type.strip()
    return None


def to_summary(conversation: Conversation) -> ConversationSummary:
    preview = None
    if conversation.messages:
        latest = max(conversation.messages, key=lambda item: item.created_at)
        preview = (latest.assistant_answer or latest.user_query or "")[:160]
    return ConversationSummary(
        id=conversation.id,
        title=conversation.title,
        law_type=conversation.law_type,
        ai_session_id=conversation.ai_session_id,
        latest_context_turn_id=conversation.latest_context_turn_id,
        last_message_preview=preview,
        status=conversation.status,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        last_message_at=conversation.last_message_at,
    )


def get_conversation_for_user(db: Session, *, user_id: str, conversation_id: str) -> Conversation:
    conversation = db.scalar(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    )
    if conversation is None:
        raise not_found("Conversation not found.", "conversation_not_found")
    return conversation


def create_conversation(db: Session, *, user: User, payload: CreateConversationRequest) -> Conversation:
    category = get_category_by_law_type(db, payload.law_type)
    if category is None:
        raise not_found("Legal category not found.", "category_not_found")
    now = _now()
    conversation = Conversation(
        user_id=user.id,
        law_type=payload.law_type,
        title=payload.title or f"{payload.law_type} conversation",
        status="active",
        created_at=now,
        updated_at=now,
    )
    db.add(conversation)
    db.flush()
    write_audit_log(
        db,
        actor_type="user",
        actor_id=user.id,
        action="conversation.created",
        entity_type="conversation",
        entity_id=conversation.id,
        metadata={"lawType": payload.law_type},
    )
    db.commit()
    db.refresh(conversation)
    return conversation


def list_conversations(db: Session, *, user_id: str, limit: int, offset: int) -> list[Conversation]:
    return list(
        db.scalars(
            select(Conversation)
            .where(Conversation.user_id == user_id, Conversation.status != "deleted")
            .order_by(desc(Conversation.last_message_at), desc(Conversation.created_at))
            .offset(offset)
            .limit(limit)
        ).all()
    )


def update_conversation(db: Session, *, conversation: Conversation, payload: UpdateConversationRequest, actor_id: str) -> Conversation:
    updates = payload.model_dump(exclude_none=True)
    for field, value in updates.items():
        setattr(conversation, field, value)
    conversation.updated_at = _now()
    write_audit_log(
        db,
        actor_type="user",
        actor_id=actor_id,
        action="conversation.updated",
        entity_type="conversation",
        entity_id=conversation.id,
        metadata=updates,
    )
    db.commit()
    db.refresh(conversation)
    return conversation


async def ask_conversation(
    db: Session,
    *,
    conversation: Conversation,
    user: User,
    query: str,
    idempotency_key: str | None,
    settings: Settings,
    request_id: str,
) -> AskConversationResponse:
    if idempotency_key:
        existing = db.scalar(
            select(Message).where(
                Message.conversation_id == conversation.id,
                Message.request_idempotency_key == idempotency_key,
            )
        )
        if existing:
            normalized = existing.normalized_answer_payload or {}
            return AskConversationResponse(
                conversation_id=conversation.id,
                message_id=existing.id,
                answer=existing.assistant_answer or "",
                disclaimer=normalized.get("disclaimer"),
                category_note=normalized.get("category_note"),
                returned_turn_id=existing.returned_turn_id,
                created_at=existing.created_at,
            )
    ai_response = await ask_ai_service(
        settings,
        AIServiceRequest(
            query=query,
            law_type=_resolve_ai_law_type(conversation),
            session_id=conversation.ai_session_id,
            context_turn_id=conversation.latest_context_turn_id,
        ),
        request_id,
    )
    if idempotency_key and db.scalar(
        select(Message).where(
            Message.conversation_id == conversation.id,
            Message.request_idempotency_key == idempotency_key,
        )
    ):
        raise conflict("Duplicate idempotency key.", "duplicate_idempotency_key")
    normalized = NormalizedAnswerPayload(
        answer=ai_response.answer,
        disclaimer=ai_response.disclaimer,
        category_note=ai_response.category_note,
        citations=ai_response.citations,
        confidence=ai_response.confidence,
        proof=ai_response.proof,
    )
    now = _now()
    message = Message(
        conversation_id=conversation.id,
        user_query=query,
        assistant_answer=ai_response.answer,
        raw_ai_payload=ai_response.raw_payload,
        normalized_answer_payload=normalized.model_dump(mode="json", by_alias=True),
        request_law_type=conversation.law_type,
        context_turn_id_used=conversation.latest_context_turn_id,
        returned_turn_id=ai_response.turn_id,
        request_idempotency_key=idempotency_key,
        created_at=now,
    )
    db.add(message)
    conversation.ai_session_id = ai_response.session_id or conversation.ai_session_id
    conversation.latest_context_turn_id = ai_response.turn_id or conversation.latest_context_turn_id
    conversation.last_message_at = now
    conversation.updated_at = now
    write_audit_log(
        db,
        actor_type="user",
        actor_id=user.id,
        action="conversation.ask",
        entity_type="conversation",
        entity_id=conversation.id,
        metadata={"messageId": message.id, "requestId": request_id},
    )
    db.commit()
    db.refresh(message)
    return AskConversationResponse(
        conversation_id=conversation.id,
        message_id=message.id,
        answer=ai_response.answer,
        disclaimer=ai_response.disclaimer,
        category_note=ai_response.category_note,
        returned_turn_id=ai_response.turn_id,
        created_at=message.created_at,
    )


def get_messages(db: Session, *, conversation_id: str, limit: int, offset: int) -> list[ConversationMessage]:
    rows = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .offset(offset)
            .limit(limit)
        ).all()
    )
    results: list[ConversationMessage] = []
    for row in rows:
        results.append(
            ConversationMessage(
                id=f"{row.id}_user",
                role="user",
                text=row.user_query,
                created_at=row.created_at,
            )
        )
        if row.assistant_answer:
            normalized = row.normalized_answer_payload or {}
            results.append(
                ConversationMessage(
                    id=f"{row.id}_assistant",
                    role="assistant",
                    text=row.assistant_answer,
                    disclaimer=normalized.get("disclaimer"),
                    citations=normalized.get("citations", []),
                    created_at=row.created_at,
                )
            )
    return results
