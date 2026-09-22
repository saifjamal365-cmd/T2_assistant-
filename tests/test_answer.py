"""The answer agent.

Unit tests replace the knowledge-base search and the LLM with fakes, so they
run fast and offline. The `integration` test uses the real index and Groq.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from t2_assistant.agents import answer
from t2_assistant.config import settings
from t2_assistant.knowledge.search import Passage


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    def invoke(self, _prompt: object) -> AIMessage:
        return AIMessage(self._reply)


class _ScriptedLLM:
    """Returns replies in call order - for tests exercising several distinct
    LLM call sites in one respond() run (spelling, write, rephrase, judge)."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)

    def invoke(self, _prompt: object) -> AIMessage:
        return AIMessage(next(self._replies))


def _passage(text: str) -> Passage:
    return Passage(
        text=text,
        doc_id="HR-0001",
        title="Annual Leave Policy",
        department="Human Resources",
        doc_type="Policy",
        version="3.2",
        score=0.9,
    )


def test_no_passages_means_dont_know(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer, "search", lambda *a, **k: [])
    monkeypatch.setattr(answer, "get_llm", lambda _model: _FakeLLM("leave days"))

    reply = answer.respond([HumanMessage("what is the parental leave policy?")], "test-model")

    assert "don't know" in str(reply.content).lower()


def test_answer_is_built_from_passages(monkeypatch: pytest.MonkeyPatch) -> None:
    passages = [_passage("Full-time staff get 25 days.")]
    monkeypatch.setattr(answer, "search", lambda *a, **k: passages)
    monkeypatch.setattr(
        answer, "get_llm", lambda _model: _FakeLLM("You get 25 days.\nSources: HR-0001")
    )

    reply = answer.respond([HumanMessage("how many annual leave days?")], "test-model")

    assert "25 days" in str(reply.content)
    assert "HR-0001" in str(reply.content)
    # fresh question, first write succeeds: spelling-fix + one write, nothing wasted
    assert reply.additional_kwargs["steps"] == 2


def test_follow_up_is_rewritten_as_a_standalone_question(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_queries: list[str] = []

    def fake_search(query: str, *a: object, **k: object) -> list[Passage]:
        seen_queries.append(query)
        return [_passage("Part-time staff get 12 days.")]

    monkeypatch.setattr(answer, "search", fake_search)
    monkeypatch.setattr(
        answer,
        "get_llm",
        lambda _model: _FakeLLM("How many annual leave days do part-time employees get?"),
    )

    reply = answer.respond(
        [
            HumanMessage("How many annual leave days do full-time employees get?"),
            AIMessage("25 days."),
            HumanMessage("and for part-time staff?"),
        ],
        "test-model",
    )

    # the search query is the rewritten standalone question, not the bare follow-up
    assert seen_queries[0] == "How many annual leave days do part-time employees get?"
    # history present: spelling-fix + standalone-rewrite + one write
    assert reply.additional_kwargs["steps"] == 3


def test_steps_count_one_retry_then_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    passages = [_passage("Full-time staff get 25 days.")]
    monkeypatch.setattr(answer, "search", lambda *a, **k: passages)
    scripted = _ScriptedLLM(
        [
            "how many annual leave days?",  # spelling-fix
            answer._DONT_KNOW_EN,  # first write: declines
            "better search query",  # rephrase
            "You get 25 days.\nSources: HR-0001",  # second write: succeeds
        ]
    )
    # one shared instance, reused across every get_llm() call in this run - a
    # fresh instance per call would reset the reply queue each time
    monkeypatch.setattr(answer, "get_llm", lambda _model: scripted)

    reply = answer.respond([HumanMessage("how many annual leave days?")], "test-model")

    assert "25 days" in str(reply.content)
    # spelling-fix + write (declines) + rephrase + write (succeeds) - no judge needed
    assert reply.additional_kwargs["steps"] == 4


def test_steps_count_full_decline_through_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    passages = [_passage("Full-time staff get 25 days.")]
    monkeypatch.setattr(answer, "search", lambda *a, **k: passages)
    scripted = _ScriptedLLM(
        [
            "what is the pet dragon policy?",  # spelling-fix
            answer._DONT_KNOW_EN,  # first write: declines
            "better search query",  # rephrase
            answer._DONT_KNOW_EN,  # second write: declines again
            answer._DONT_KNOW_EN,  # judge: confirms the decline
        ]
    )
    monkeypatch.setattr(answer, "get_llm", lambda _model: scripted)

    reply = answer.respond([HumanMessage("what is the pet dragon policy?")], "test-model")

    assert "don't know" in str(reply.content).lower()
    # spelling-fix + write + rephrase + write + judge - every attempt declined
    assert reply.additional_kwargs["steps"] == 5


@pytest.mark.integration
def test_answer_end_to_end() -> None:
    """Real index + real Groq: a known question gets a sourced answer."""
    reply = answer.respond(
        [HumanMessage("How many annual leave days do full-time employees get?")],
        settings.llm_model,
    )
    text = str(reply.content)
    assert "25" in text
    assert "HR-" in text  # cites an HR document
