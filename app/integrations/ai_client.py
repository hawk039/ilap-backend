from __future__ import annotations

from uuid import uuid4

import httpx

from app.core.config import Settings
from app.core.errors import upstream_failure
from app.schemas.conversations import AIServiceRequest, AIServiceResponse


async def ask_ai_service(settings: Settings, payload: AIServiceRequest, request_id: str) -> AIServiceResponse:
    try:
        async with httpx.AsyncClient(timeout=settings.ai_service_timeout_seconds) as client:
            response = await client.post(
                f"{settings.ai_service_base_url.rstrip('/')}/ask",
                json=payload.model_dump(mode="json", by_alias=True, exclude_none=True),
                headers={"X-Request-Id": request_id},
            )
            response.raise_for_status()
            data = response.json()
    except Exception:
        if settings.ai_stub_mode and settings.environment != "production":
            data = {
                "answer": (
                    f"Development AI fallback for {payload.law_type or 'the detected legal category'}. "
                    "Configure the standalone AI service and disable AI_STUB_MODE for production."
                ),
                "disclaimer": "This response is informational and not legal advice.",
                "sessionId": payload.session_id or f"ai_session_{uuid4().hex}",
                "turnId": f"turn_{uuid4().hex}",
                "category_note": f"Generated under {payload.law_type or 'an inferred legal category'}.",
            }
        else:
            raise upstream_failure()
    return AIServiceResponse(
        answer=data.get("answer", ""),
        citations=data.get("citations", []),
        confidence=data.get("confidence"),
        disclaimer=data.get("disclaimer"),
        session_id=data.get("sessionId") or data.get("session_id"),
        turn_id=data.get("turnId") or data.get("turn_id"),
        category_note=data.get("categoryNote") or data.get("category_note"),
        proof=data.get("proof"),
        raw_payload=data,
    )
