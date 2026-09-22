"""The graph wiring: the router's choice must decide which specialist runs.

These tests replace the LLM-backed pieces with fakes, so they are fast and need
no network. They check the *plumbing*, not the model's judgement.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from t2_assistant.agents import answer, clarify, graph, greeting, summarise
from t2_assistant.agents.graph import build_graph
from t2_assistant.agents.router import RouterDecision
from t2_assistant.agents.state import Route


@pytest.fixture
def fake_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace each specialist with one that returns a recognisable reply."""
    for name, module in {
        "greeting": greeting,
        "answer": answer,
        "summarise": summarise,
        "clarify": clarify,
    }.items():
        monkeypatch.setattr(
            module,
            "respond",
            lambda _messages, _model, _n=name: AIMessage(f"[{_n}] handled it"),
        )


def _route_to(monkeypatch: pytest.MonkeyPatch, route: Route) -> None:
    # graph.py imported decide_route by name, so patch it on the graph module
    monkeypatch.setattr(
        graph,
        "decide_route",
        lambda _messages, _model: RouterDecision(route=route, reason=f"forced {route}"),
    )


@pytest.mark.parametrize("route", ["greeting", "answer", "summarise", "clarify"])
def test_router_choice_selects_matching_specialist(
    monkeypatch: pytest.MonkeyPatch, fake_agents: None, route: Route
) -> None:
    _route_to(monkeypatch, route)
    graph = build_graph()

    final = graph.invoke(
        {
            "messages": [HumanMessage("anything")],
            "route": None,
            "route_reason": None,
            "answer_model": "test-model",
        }
    )

    assert final["route"] == route
    assert final["messages"][-1].content == f"[{route}] handled it"


def test_history_is_preserved(monkeypatch: pytest.MonkeyPatch, fake_agents: None) -> None:
    _route_to(monkeypatch, "greeting")
    graph = build_graph()

    final = graph.invoke(
        {
            "messages": [HumanMessage("hi"), AIMessage("hello"), HumanMessage("hi again")],
            "route": None,
            "route_reason": None,
            "answer_model": "test-model",
        }
    )

    # 3 original + 1 reply
    assert len(final["messages"]) == 4
