"""The agent graph.

This is where the team is wired together:

        START
          |
          v
      [ router ]              decides: greeting | answer | summarise | clarify
       /  |   |  \\
      v   v   v   v
 greeting answer summarise clarify
      \\   |   |   /
          v
         END

Each box is a "node" - a small function that takes the shared state, does its
part, and returns the fields it changed. The arrows after the router are
"conditional edges": which one is taken depends on `state["route"]`.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from t2_assistant.agents import answer, clarify, greeting, summarise
from t2_assistant.agents.router import decide_route
from t2_assistant.agents.state import AgentState
from t2_assistant.config import settings

# ---- nodes -------------------------------------------------------------


def _router_node(state: AgentState) -> dict[str, object]:
    # Always the fixed default model, never the user's per-turn choice: the
    # router needs tool-calling for its structured output, and not every
    # offered model supports that (confirmed: allam-2-7b flatly rejects tool
    # calls). The user's choice still drives whichever specialist actually
    # writes the visible reply - only this internal classification step is
    # pinned.
    decision = decide_route(state["messages"], settings.llm_model)
    return {"route": decision.route, "route_reason": decision.reason}


def _greeting_node(state: AgentState) -> dict[str, object]:
    return {"messages": [greeting.respond(state["messages"], state["model"])]}


def _answer_node(state: AgentState) -> dict[str, object]:
    return {"messages": [answer.respond(state["messages"], state["model"])]}


def _summarise_node(state: AgentState) -> dict[str, object]:
    return {"messages": [summarise.respond(state["messages"], state["model"])]}


def _clarify_node(state: AgentState) -> dict[str, object]:
    return {"messages": [clarify.respond(state["messages"], state["model"])]}


# ---- the branch after the router -------------------------------------


def _pick_specialist(state: AgentState) -> str:
    """Return the name of the node to run next (from the router's choice)."""
    return state["route"] or "clarify"


# ---- build & compile ------------------------------------------------


def build_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    graph = StateGraph(AgentState)

    graph.add_node("router", _router_node)
    graph.add_node("greeting", _greeting_node)
    graph.add_node("answer", _answer_node)
    graph.add_node("summarise", _summarise_node)
    graph.add_node("clarify", _clarify_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _pick_specialist,
        {
            "greeting": "greeting",
            "answer": "answer",
            "summarise": "summarise",
            "clarify": "clarify",
        },
    )
    for specialist in ("greeting", "answer", "summarise", "clarify"):
        graph.add_edge(specialist, END)

    return graph.compile()


# Compiled once at import; reused for every request.
compiled_graph = build_graph()
