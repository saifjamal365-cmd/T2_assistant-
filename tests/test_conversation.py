"""run_turn(): memory across turns."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from t2_assistant import conversation, store
from t2_assistant.agents import answer, graph, greeting
from t2_assistant.agents.router import RouterDecision
from t2_assistant.agents.state import Route
from t2_assistant.config import settings
from t2_assistant.conversation import ChatResult


def test_run_turn_saves_both_turns_and_passes_history(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_history: list[list[object]] = []

    def fake_chat(
        message: str,
        history: list[object] | None = None,
        *,
        model: str | None = None,
        user_email: str | None = None,
    ) -> ChatResult:
        seen_history.append(list(history or []))
        return ChatResult(
            reply=f"reply to: {message}",
            route="answer",
            route_reason="q",
            sources=[],
            model=model or settings.llm_model,
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
        lambda message, history=None, **kw: ChatResult("ok", "greeting", "hi", [], "test-model"),
    )
    result = conversation.run_turn("hello", conversation_id="not-a-real-id")
    assert store.conversation_exists(result.conversation_id)


def test_run_turn_threads_and_persists_the_chosen_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_models: list[str | None] = []

    def fake_chat(
        message: str,
        history: list[object] | None = None,
        *,
        model: str | None = None,
        user_email: str | None = None,
    ) -> ChatResult:
        seen_models.append(model)
        resolved = model or settings.llm_model
        return ChatResult(reply="ok", route="answer", route_reason="q", sources=[], model=resolved)

    monkeypatch.setattr(conversation, "chat", fake_chat)

    with_model = conversation.run_turn("a question", model="qwen/qwen3.8-27b")
    assert seen_models[-1] == "qwen/qwen3.8-27b"
    saved = store.get_run(with_model.run_id)
    assert saved is not None
    assert saved.model == "qwen/qwen3.8-27b"

    without_model = conversation.run_turn("another question")
    assert seen_models[-1] is None
    assert without_model.model == settings.llm_model


def _route_to(monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    # graph.py imported decide_route by name, so patch it on the graph module -
    # same pattern as test_graph.py's own _route_to
    monkeypatch.setattr(
        graph,
        "decide_route",
        lambda _messages, _model: RouterDecision(route=route, reason=f"forced {route}"),
    )


def test_chat_reads_steps_from_the_specialists_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    _route_to(monkeypatch, "answer")
    monkeypatch.setattr(
        answer, "respond", lambda _messages, _model: AIMessage("x", additional_kwargs={"steps": 3})
    )

    result = conversation.chat("a policy question")

    assert result.steps == 3


def test_chat_defaults_steps_to_one_when_the_specialist_omits_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _route_to(monkeypatch, "greeting")
    # a real greeting reply has no additional_kwargs at all - only answer.py sets "steps"
    monkeypatch.setattr(greeting, "respond", lambda _messages, _model: AIMessage("hi"))

    result = conversation.chat("hey")

    assert result.steps == 1
