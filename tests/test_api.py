"""The HTTP API surface."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from t2_assistant import api, store
from t2_assistant.api import app
from t2_assistant.conversation import TurnResult

client = TestClient(app)
# Every route but /, /health and /auth/* now requires a signed-in session.
# Create one directly (skipping the OTP/email round trip - these tests are
# about chat/API behaviour, not sign-in) and attach it to the shared client.
client.cookies.set("t2_session", store.create_session("test@t2.sa"))


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_web_page_loads() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "T2 Assistant" in response.text


def test_chat_shape_and_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    """/chat returns the turn and the conversation is saved (agent call faked)."""

    def fake_run_turn(
        message: str,
        conversation_id: str | None = None,
        *,
        model: str | None = None,
        user_email: str = "test@t2.sa",
    ) -> TurnResult:
        cid = conversation_id or store.create_conversation(message, user_email)
        store.add_message(cid, "user", message)
        store.add_message(cid, "assistant", "hello!")
        run_id = store.save_run(
            conversation_id=cid,
            user_message=message,
            route="greeting",
            route_reason="a greeting",
            reply="hello!",
            model=model or "test-model",
            trace_id=None,
            duration_ms=1,
        )
        return TurnResult(
            cid, run_id, None, "hello!", "greeting", "a greeting", [], model or "test-model"
        )

    monkeypatch.setattr(api, "run_turn", fake_run_turn)

    first = client.post("/chat", json={"message": "hi"}).json()
    assert first["route"] == "greeting"
    assert first["reply"] == "hello!"
    assert first["run_id"]
    cid = first["conversation_id"]

    client.post("/chat", json={"message": "and again", "conversation_id": cid})

    conversation = client.get(f"/conversations/{cid}").json()
    roles = [m["role"] for m in conversation["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]

    # the runs endpoint sees both requests
    runs = client.get("/runs").json()
    assert len([r for r in runs if r["conversation_id"] == cid]) == 2

    # feedback on the first run
    ok = client.post(f"/runs/{first['run_id']}/feedback", json={"helpful": True})
    assert ok.status_code == 200
    assert client.get(f"/runs/{first['run_id']}").json()["feedback"] == "helpful"


def test_chat_rejects_an_unknown_model() -> None:
    response = client.post("/chat", json={"message": "hi", "model": "not-a-real-model"})
    assert response.status_code == 422


def test_chat_echoes_a_valid_chosen_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_turn(
        message: str,
        conversation_id: str | None = None,
        *,
        model: str | None = None,
        user_email: str = "test@t2.sa",
    ) -> TurnResult:
        cid = conversation_id or store.create_conversation(message, user_email)
        resolved = model or "openai/gpt-oss-120b"
        run_id = store.save_run(
            conversation_id=cid,
            user_message=message,
            route="greeting",
            route_reason="a greeting",
            reply="hello!",
            model=resolved,
            trace_id=None,
            duration_ms=1,
        )
        return TurnResult(cid, run_id, None, "hello!", "greeting", "a greeting", [], resolved)

    monkeypatch.setattr(api, "run_turn", fake_run_turn)

    response = client.post("/chat", json={"message": "hi", "model": "qwen/qwen3.8-27b"})
    assert response.status_code == 200
    assert response.json()["model"] == "qwen/qwen3.8-27b"


def test_unknown_conversation_is_404() -> None:
    assert client.get("/conversations/nope").status_code == 404


def test_unknown_run_feedback_is_404() -> None:
    assert client.post("/runs/nope/feedback", json={"helpful": False}).status_code == 404


def test_empty_message_is_rejected() -> None:
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "   "}).status_code == 422


def test_message_over_the_length_limit_is_rejected() -> None:
    response = client.post("/chat", json={"message": "x" * 5000})
    assert response.status_code == 422


def test_provider_failure_returns_a_clean_503(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Groq outage must not crash the request - NFR-07."""
    import httpx
    from groq import APIConnectionError

    def failing_run_turn(
        message: str,
        conversation_id: str | None = None,
        *,
        model: str | None = None,
        user_email: str | None = None,
    ) -> TurnResult:
        raise APIConnectionError(request=httpx.Request("POST", "https://groq.example"))

    monkeypatch.setattr(api, "run_turn", failing_run_turn)

    response = client.post("/chat", json={"message": "how many leave days?"})

    assert response.status_code == 503
    assert "temporarily unavailable" in response.json()["detail"]


def test_unexpected_error_returns_500_not_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_run_turn(
        message: str,
        conversation_id: str | None = None,
        *,
        model: str | None = None,
        user_email: str | None = None,
    ) -> TurnResult:
        raise ValueError("something unrelated broke")

    monkeypatch.setattr(api, "run_turn", broken_run_turn)

    # the real server never raises into its own process; ask the test client to
    # behave the same way (assert on the response, not on a propagated exception)
    lenient_client = TestClient(app, raise_server_exceptions=False)
    lenient_client.cookies.set("t2_session", store.create_session("test@t2.sa"))
    response = lenient_client.post("/chat", json={"message": "how many leave days?"})

    assert response.status_code == 500
    assert "detail" in response.json()


@pytest.mark.integration
def test_chat_greeting_end_to_end() -> None:
    """Real call to Groq: a greeting is routed to 'greeting'."""
    response = client.post("/chat", json={"message": "hello there!"})
    assert response.status_code == 200
    assert response.json()["route"] == "greeting"


def test_protected_routes_need_a_session() -> None:
    """A fresh client, with no session cookie at all, must be turned away -
    the shared `client` above is pre-authenticated, so this uses its own."""
    anon = TestClient(app)
    assert anon.get("/conversations").status_code == 401
    assert anon.post("/chat", json={"message": "hi"}).status_code == 401


def test_public_routes_work_without_a_session() -> None:
    anon = TestClient(app)
    assert anon.get("/").status_code == 200
    assert anon.get("/health").status_code == 200
    assert anon.get("/auth/me").json() == {"email": None}


def test_sign_in_flow_end_to_end() -> None:
    anon = TestClient(app)
    email = f"flow-test-{uuid.uuid4().hex[:10]}@t2.sa"

    registered = anon.post("/auth/login", json={"email": email, "password": "correct-password"})
    assert registered.status_code == 200
    assert registered.json() == {"status": "ok", "email": email, "created": True}
    assert anon.get("/auth/me").json() == {"email": email}
    assert anon.get("/conversations").status_code == 200  # now allowed in

    anon.post("/auth/logout")
    assert anon.get("/auth/me").json() == {"email": None}
    assert anon.get("/conversations").status_code == 401

    wrong = anon.post("/auth/login", json={"email": email, "password": "a-wrong-guess"})
    assert wrong.status_code == 401
    assert anon.get("/auth/me").json() == {"email": None}  # still signed out

    right = anon.post("/auth/login", json={"email": email, "password": "correct-password"})
    assert right.status_code == 200
    assert right.json()["created"] is False  # already existed - not a fresh registration
    assert anon.get("/auth/me").json() == {"email": email}


def test_disallowed_email_cannot_sign_in() -> None:
    anon = TestClient(app)
    response = anon.post("/auth/login", json={"email": "someone@gmail.com", "password": "whatever"})
    assert response.status_code == 403


def _authed_client(email: str) -> TestClient:
    c = TestClient(app)
    c.cookies.set("t2_session", store.create_session(email))
    return c


def test_user_cannot_see_or_touch_another_users_conversation() -> None:
    email_a = f"a-{uuid.uuid4().hex[:8]}@t2.sa"
    email_b = f"b-{uuid.uuid4().hex[:8]}@t2.sa"
    client_a, client_b = _authed_client(email_a), _authed_client(email_b)
    cid = store.create_conversation("A's private chat", email_a)

    assert client_b.get(f"/conversations/{cid}").status_code == 404
    assert client_b.patch(f"/conversations/{cid}", json={"title": "hijacked"}).status_code == 404
    assert client_b.put(f"/conversations/{cid}/folder", json={"folder_id": None}).status_code == 404
    assert client_b.delete(f"/conversations/{cid}").status_code == 404
    assert cid not in {c["id"] for c in client_b.get("/conversations").json()}

    assert client_a.get(f"/conversations/{cid}").status_code == 200  # still A's, untouched


def test_user_cannot_see_or_touch_another_users_folder() -> None:
    email_a = f"a-{uuid.uuid4().hex[:8]}@t2.sa"
    email_b = f"b-{uuid.uuid4().hex[:8]}@t2.sa"
    client_a, client_b = _authed_client(email_a), _authed_client(email_b)
    fid = store.create_folder("A's folder", email_a)

    assert client_b.patch(f"/folders/{fid}", json={"name": "hijacked"}).status_code == 404
    assert client_b.delete(f"/folders/{fid}").status_code == 404
    assert fid not in {f["id"] for f in client_b.get("/folders").json()}
    assert fid in {f["id"] for f in client_a.get("/folders").json()}


def test_user_cannot_move_a_conversation_into_another_users_folder() -> None:
    email_a = f"a-{uuid.uuid4().hex[:8]}@t2.sa"
    email_b = f"b-{uuid.uuid4().hex[:8]}@t2.sa"
    client_a = _authed_client(email_a)
    cid = store.create_conversation("A's chat", email_a)
    other_folder = store.create_folder("B's folder", email_b)

    response = client_a.put(f"/conversations/{cid}/folder", json={"folder_id": other_folder})
    assert response.status_code == 404


def test_user_cannot_see_or_touch_another_users_run() -> None:
    email_a = f"a-{uuid.uuid4().hex[:8]}@t2.sa"
    email_b = f"b-{uuid.uuid4().hex[:8]}@t2.sa"
    client_a, client_b = _authed_client(email_a), _authed_client(email_b)
    cid = store.create_conversation("A's chat", email_a)
    run_id = store.save_run(
        conversation_id=cid,
        user_message="q",
        route="answer",
        route_reason="r",
        reply="a",
        model="openai/gpt-oss-120b",
        trace_id=None,
        duration_ms=10,
    )

    assert client_b.get(f"/runs/{run_id}").status_code == 404
    assert client_b.post(f"/runs/{run_id}/feedback", json={"helpful": True}).status_code == 404
    assert run_id not in {r["id"] for r in client_b.get("/runs").json()}

    assert client_a.get(f"/runs/{run_id}").status_code == 200
