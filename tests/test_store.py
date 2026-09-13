"""The SQLite store."""

from __future__ import annotations

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


def test_save_run() -> None:
    cid = store.create_conversation("run test")
    store.save_run(
        conversation_id=cid,
        user_message="how many leave days?",
        route="answer",
        route_reason="a policy question",
        reply="25 days.",
        trace_id="trace-123",
        duration_ms=1400,
    )  # just needs to not raise
