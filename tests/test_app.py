from pathlib import Path
from contextlib import contextmanager

from fastapi.testclient import TestClient
from sqlalchemy import select


@contextmanager
def build_client(tmp_path: Path):
    env = {
        "DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}",
        "SYNC_DATABASE_URL": f"sqlite:///{tmp_path / 'test.db'}",
        "AUTO_CREATE_SCHEMA": "true",
        "AI_STUB_MODE": "true",
        "JWT_SECRET_KEY": "test-secret-key-that-is-long-enough",
        "ADMIN_BOOTSTRAP_EMAIL": "admin@ilap.local",
        "ADMIN_BOOTSTRAP_PASSWORD": "AdminPass123!",
        "REQUIRE_VERIFIED_EMAIL": "false",
    }
    import os

    os.environ.update(env)
    from app.core.config import get_settings
    from app.db.session import get_engine, get_session_local

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_session_local.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as client:
        yield client


def test_healthcheck(tmp_path: Path):
    with build_client(tmp_path) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_legal_categories_expose_ai_law_type_mapping(tmp_path: Path):
    with build_client(tmp_path) as client:
        response = client.get("/api/v1/legal-categories")
        assert response.status_code == 200, response.text

        categories = response.json()
        assert len(categories) == 6

        consumer_rights = next(item for item in categories if item["lawType"] == "Consumer Rights")
        assert consumer_rights["aiLawTypes"] == [
            "consumer_rights",
            "contract_law",
            "insurance_law",
        ]


def test_register_login_and_conversation_flow(tmp_path: Path):
    with build_client(tmp_path) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"fullName": "Test User", "email": "user@example.com", "password": "StrongPass123!"},
        )
        assert register.status_code == 201, register.text
        auth = register.json()
        access_token = auth["session"]["accessToken"]

        categories = client.get("/api/v1/legal-categories")
        assert categories.status_code == 200
        assert len(categories.json()) >= 1

        conversation = client.post(
            "/api/v1/conversations",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"lawType": "Criminal Law"},
        )
        assert conversation.status_code == 201, conversation.text
        conversation_id = conversation.json()["id"]

        ask = client.post(
            f"/api/v1/conversations/{conversation_id}/ask",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"query": "What should I do if police call me for questioning?"},
        )
        assert ask.status_code == 200, ask.text
        assert ask.json()["conversationId"] == conversation_id

        history = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert history.status_code == 200
        assert len(history.json()) == 2


def test_register_creates_email_verification_token(tmp_path: Path):
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/auth/register",
            json={"fullName": "Verify User", "email": "verify@example.com", "password": "StrongPass123!"},
        )
        assert response.status_code == 201, response.text

        from app.db.session import get_session_local
        from app.models import EmailVerificationToken, User

        with get_session_local()() as db:
            user = db.scalar(select(User).where(User.email == "verify@example.com"))
            token = db.scalar(select(EmailVerificationToken).where(EmailVerificationToken.user_id == user.id))
            assert user is not None
            assert token is not None


def test_login_requires_verified_email_when_enabled(tmp_path: Path):
    with build_client(tmp_path) as client:
        client.post(
            "/api/v1/auth/register",
            json={"fullName": "Locked User", "email": "locked@example.com", "password": "StrongPass123!"},
        )

        from app.core.config import get_settings
        get_settings.cache_clear()

        import os
        os.environ["REQUIRE_VERIFIED_EMAIL"] = "true"

        try:
            response = client.post(
                "/api/v1/auth/login",
                json={"email": "locked@example.com", "password": "StrongPass123!"},
            )
            assert response.status_code == 403, response.text
        finally:
            os.environ["REQUIRE_VERIFIED_EMAIL"] = "false"


def test_ai_law_type_is_inferred_from_first_citation_for_follow_up_turns(tmp_path: Path, monkeypatch):
    captured_law_types: list[str | None] = []

    async def fake_ask_ai_service(settings, payload, request_id):
        captured_law_types.append(payload.law_type)
        from app.schemas.conversations import AIServiceResponse

        if len(captured_law_types) == 1:
            return AIServiceResponse(
                answer="First answer",
                citations=[
                    {
                        "act": "ICA",
                        "section": "56",
                        "effectiveFrom": "1872-09-01",
                        "lawType": "contract_law",
                    }
                ],
                session_id="ai-session-1",
                turn_id="turn-1",
                raw_payload={},
            )
        return AIServiceResponse(
            answer="Second answer",
            citations=[],
            session_id="ai-session-1",
            turn_id="turn-2",
            raw_payload={},
        )

    monkeypatch.setattr("app.services.conversations.ask_ai_service", fake_ask_ai_service)

    with build_client(tmp_path) as client:
        register = client.post(
            "/api/v1/auth/register",
            json={"fullName": "Test User", "email": "user@example.com", "password": "StrongPass123!"},
        )
        assert register.status_code == 201, register.text
        access_token = register.json()["session"]["accessToken"]

        conversation = client.post(
            "/api/v1/conversations",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"lawType": "Consumer Rights"},
        )
        assert conversation.status_code == 201, conversation.text
        conversation_id = conversation.json()["id"]

        first_ask = client.post(
            f"/api/v1/conversations/{conversation_id}/ask",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"query": "First question"},
        )
        assert first_ask.status_code == 200, first_ask.text

        second_ask = client.post(
            f"/api/v1/conversations/{conversation_id}/ask",
            headers={"Authorization": f"Bearer {access_token}"},
            json={"query": "Second question"},
        )
        assert second_ask.status_code == 200, second_ask.text

    assert captured_law_types == [None, "contract_law"]
