"""run_turn(): memory across turns."""

from __future__ import annotations

import pytest

from t2_assistant import conversation, store
from t2_assistant.conversation import ChatResult


def test_run_turn_saves_both_turns_and_passes_history(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_history: list[list[object]] = []

    def fake_chat(message: str, history: list[object] | None = None) -> ChatResult:
        seen_history.append(list(history or []))
        return ChatResult(
            reply=f"reply to: {message}", route="answer", route_reason="q", sources=[]
        )

    monkeypatch.setattr(conversation, "chat", fake_chat)

    first = conversation.run_turn("how many leave days?")
    assert seen_history[0] == []  # new conversation, no history

    assert first.run_id  # a run was recorded

    second = conversation.run_turn("what about part-time?", first.conversation_id)
    assert second.conversation_id == first.conversation_id
    # the second turn was given the first turn's two messages as history
    assert len(seen_history[1]) == 2

    saved = store.get_conversation(first.conversation_id)
    assert saved is not None
    assert [m.content for m in saved.messages] == [
        "how many leave days?",
        "reply to: how many leave days?",
        "what about part-time?",
        "reply to: what about part-time?",
    ]


def test_run_turn_starts_fresh_for_unknown_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        conversation,
        "chat",
        lambda message, history=None: ChatResult("ok", "greeting", "hi", []),
    )
    result = conversation.run_turn("hello", conversation_id="not-a-real-id")
    assert store.conversation_exists(result.conversation_id)
