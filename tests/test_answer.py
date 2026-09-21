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

    answer.respond(
        [
            HumanMessage("How many annual leave days do full-time employees get?"),
            AIMessage("25 days."),
            HumanMessage("and for part-time staff?"),
        ],
        "test-model",
    )

    # the search query is the rewritten standalone question, not the bare follow-up
    assert seen_queries[0] == "How many annual leave days do part-time employees get?"


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
