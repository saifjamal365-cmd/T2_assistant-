"""The SQLite store."""

from __future__ import annotations

import uuid

from t2_assistant import store


def test_create_and_read_a_conversation() -> None:
    email = f"store-test-{uuid.uuid4().hex[:10]}@t2.sa"
    cid = store.create_conversation("Leave questions", email)

    assert store.conversation_exists(cid, email)
    assert not store.conversation_exists("does-not-exist", email)

    store.add_message(cid, "user", "how many leave days?")
    store.add_message(cid, "assistant", "25 days. Sources: HR-0001")

    conversation = store.get_conversation(cid, email)
    assert conversation is not None
    assert [m.role for m in conversation.messages] == ["user", "assistant"]
    assert conversation.messages[1].content.startswith("25 days")


def test_list_is_newest_first_with_counts() -> None:
    email = f"store-test-{uuid.uuid4().hex[:10]}@t2.sa"
    first = store.create_conversation("older", email)
    store.add_message(first, "user", "hi")
    second = store.create_conversation("newer", email)

    listed = store.list_conversations(email)
    ids = [c.id for c in listed]

    assert ids.index(second) < ids.index(first)  # newer before older
    assert next(c for c in listed if c.id == first).message_count == 1


def test_create_user_and_read_the_password_hash() -> None:
    email = f"store-test-{uuid.uuid4().hex[:10]}@t2.sa"

    assert store.get_password_hash(email) is None  # unknown email

    store.create_user(email, "scrypt$abc$def")
    assert store.get_password_hash(email) == "scrypt$abc$def"


def test_save_run() -> None:
    email = f"store-test-{uuid.uuid4().hex[:10]}@t2.sa"
    cid = store.create_conversation("run test", email)
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
    saved = store.get_run(run_id, email)
    assert saved is not None
    assert saved.model == "openai/gpt-oss-120b"


def _email() -> str:
    return f"store-test-{uuid.uuid4().hex[:10]}@t2.sa"


def test_conversations_are_isolated_per_owner() -> None:
    email_a, email_b = _email(), _email()
    cid = store.create_conversation("A's chat", email_a)

    assert store.get_conversation(cid, email_b) is None
    assert cid not in {c.id for c in store.list_conversations(email_b)}
    assert not store.rename_conversation(cid, "hijacked", email_b)
    assert not store.set_conversation_folder(cid, None, email_b)
    assert not store.delete_conversation(cid, email_b)

    still_there = store.get_conversation(cid, email_a)
    assert still_there is not None
    assert still_there.title == "A's chat"  # untouched by any of the above


def test_folders_are_isolated_per_owner() -> None:
    email_a, email_b = _email(), _email()
    fid = store.create_folder("A's folder", email_a)

    assert fid not in {f.id for f in store.list_folders(email_b)}
    assert not store.rename_folder(fid, "hijacked", email_b)
    assert not store.delete_folder(fid, email_b)
    assert any(f.id == fid for f in store.list_folders(email_a))  # untouched


def test_cannot_move_a_conversation_into_someone_elses_folder() -> None:
    email_a, email_b = _email(), _email()
    cid = store.create_conversation("A's chat", email_a)
    other_folder = store.create_folder("B's folder", email_b)

    assert not store.set_conversation_folder(cid, other_folder, email_a)
    conversation = store.get_conversation(cid, email_a)
    assert conversation is not None
    assert conversation.folder_id is None  # move was refused, not partially applied


def test_runs_are_isolated_per_owner() -> None:
    email_a, email_b = _email(), _email()
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

    assert store.get_run(run_id, email_b) is None
    assert run_id not in {r.id for r in store.list_runs(email_b)}
    assert store.get_run(run_id, email_a) is not None
