"""The one entry point the rest of the app calls.

`chat()` is pure: a message (plus the earlier turns) in, a reply out, as one
MLflow trace.

`run_turn()` wraps it with memory: it loads the conversation's history from the
store, calls `chat()`, then saves both turns and a run record. This is what the
API uses.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlflow
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from mlflow.entities import SpanType

from t2_assistant import store
from t2_assistant.agents.graph import compiled_graph
from t2_assistant.agents.state import Route
from t2_assistant.config import settings
from t2_assistant.tracing import init_tracing

# Configure MLflow tracing as soon as this module is imported.
init_tracing()


@dataclass
class ChatResult:
    """What `chat()` returns."""

    reply: str
    route: Route
    route_reason: str
    sources: list[dict[str, object]]
    model: str
    # LLM calls the chosen specialist needed (router excluded - a fixed cost,
    # see graph.py). Only the answer route ever varies from 1; the default
    # covers greeting/summarise/clarify, none of which set it explicitly -
    # see answer.py's respond() for what counts as a step.
    steps: int = 1


@dataclass
class TurnResult:
    """What `run_turn()` returns - a ChatResult plus the conversation and run it belongs to."""

    conversation_id: str
    run_id: str
    trace_id: str | None
    reply: str
    route: Route
    route_reason: str
    sources: list[dict[str, object]]
    model: str
    steps: int = 1


@mlflow.trace(span_type=SpanType.AGENT)
def chat(
    message: str,
    history: list[AnyMessage] | None = None,
    *,
    model: str | None = None,
    user_email: str | None = None,
) -> ChatResult:
    """Answer one user message.

    `history` is the earlier conversation (list of messages), or None for a new
    conversation. `model` picks the Groq model for whichever specialist writes
    the visible reply; omitted, it falls back to settings.llm_model. Routing
    itself always uses that same fixed default, never the caller's choice -
    see graph.py's _router_node - so the trace is tagged with both, under
    separate keys, rather than leaving the router's model implicit.
    `user_email`, when known, tags the trace with who asked.
    """
    resolved_model = model or settings.llm_model
    messages: list[AnyMessage] = [*(history or []), HumanMessage(message)]

    # Tag the trace before the graph runs, so even a mid-turn failure is
    # still attributable to a model and a user.
    mlflow.update_current_trace(
        tags={"model": resolved_model, "router_model": settings.llm_model},
        metadata={"mlflow.trace.user": user_email} if user_email else None,
    )

    final_state = compiled_graph.invoke(
        {
            "messages": messages,
            "route": None,
            "route_reason": None,
            "answer_model": resolved_model,
        }
    )

    last = final_state["messages"][-1]
    assert isinstance(last, AIMessage)  # the specialist always adds an AI reply
    steps = last.additional_kwargs.get("steps", 1)

    # A second tag, after the graph runs: unlike model/router_model above,
    # this isn't known until the specialist finishes, so it can't be set
    # up front. update_current_trace merges into the existing tags rather
    # than replacing them, so model/router_model/user stay intact.
    mlflow.update_current_trace(tags={"steps": str(steps)})

    return ChatResult(
        reply=str(last.content),
        route=final_state["route"] or "clarify",
        route_reason=final_state["route_reason"] or "",
        sources=last.additional_kwargs.get("passages", []),
        model=resolved_model,
        steps=steps,
    )


def _to_messages(stored: list[store.Message]) -> list[AnyMessage]:
    out: list[AnyMessage] = []
    for message in stored:
        if message.role == "user":
            out.append(HumanMessage(message.content))
        else:
            out.append(AIMessage(message.content))
    return out


def _title_from(message: str) -> str:
    text = " ".join(message.split())
    return text[:60] + ("..." if len(text) > 60 else "")


def run_turn(
    message: str,
    conversation_id: str | None = None,
    *,
    model: str | None = None,
    user_email: str,
) -> TurnResult:
    """Answer a message inside a conversation, saving the turn and a run record.

    Starts a new conversation when `conversation_id` is None - also when it
    names a conversation that exists but belongs to someone else: the same
    "unknown id" behaviour, on purpose, since a non-owner should never learn
    a conversation exists at all.
    """
    if conversation_id is None or not store.conversation_exists(conversation_id, user_email):
        conversation_id = store.create_conversation(_title_from(message), user_email)

    history = _to_messages(store.get_messages(conversation_id, user_email))

    started = time.perf_counter()
    result = chat(message, history=history, model=model, user_email=user_email)
    duration_ms = int((time.perf_counter() - started) * 1000)

    trace_id = mlflow.get_last_active_trace_id()

    store.add_message(conversation_id, "user", message)
    store.add_message(conversation_id, "assistant", result.reply)
    run_id = store.save_run(
        conversation_id=conversation_id,
        user_message=message,
        route=result.route,
        route_reason=result.route_reason,
        reply=result.reply,
        model=result.model,
        trace_id=trace_id,
        duration_ms=duration_ms,
    )
    if result.sources:
        store.save_sources(run_id, result.sources)

    return TurnResult(
        conversation_id=conversation_id,
        run_id=run_id,
        trace_id=trace_id,
        reply=result.reply,
        route=result.route,
        route_reason=result.route_reason,
        sources=result.sources,
        model=result.model,
        steps=result.steps,
    )
