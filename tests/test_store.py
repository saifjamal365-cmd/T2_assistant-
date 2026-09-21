"""The SQLite store."""

from __future__ import annotations

import uuid

from t2_assistant import store


def test_create_and_read_a_conversation() -> None:
    cid = store.create_conversation("Leave questions")

    assert store.conversation_exists(cid)
    assert not store.conversation_exists("does-not-exist")

    store.add_message(cid, "user", "how many leave days?")
    store.add_message(cid, "assistant", "25 days. Sources: HR-0001")

    conversation = store.get_conversation(cid)
    assert conversation is not None
    assert [m.role for m in conversation.messages] == ["user", "assistant"]
    assert conversation.messages[1].content.startswith("25 days")


def test_list_is_newest_first_with_counts() -> None:
    first = store.create_conversation("older")
    store.add_message(first, "user", "hi")
    second = store.create_conversation("newer")

    listed = store.list_conversations()
    ids = [c.id for c in listed]

    assert ids.index(second) < ids.index(first)  # newer before older
    assert next(c for c in listed if c.id == first).message_count == 1


def test_create_user_and_read_the_password_hash() -> None:
    email = f"store-test-{uuid.uuid4().hex[:10]}@t2.sa"

    assert store.get_password_hash(email) is None  # unknown email

    store.create_user(email, "scrypt$abc$def")
    assert store.get_password_hash(email) == "scrypt$abc$def"


def test_save_run() -> None:
    cid = store.create_conversation("run test")
    run_id = store.save_run(
        conversation_id=cid,
        user_message="how many leave days?",
        route="answer",
        route_reason="a policy question",
        reply="25 days.",
        model="openai/gpt-oss-120b",
        trace_id="trace-123",
        duration_ms=1400,
    )
    saved = store.get_run(run_id)
    assert saved is not None
    assert saved.model == "openai/gpt-oss-120b"
