"""The router agent.

It reads the conversation and decides which specialist should handle the latest
message. The decision is made by the LLM reasoning about meaning - there is no
list of keywords. The model must return a structured answer (route + reason),
which we enforce with `with_structured_output`.
"""

from __future__ import annotations

from langchain_core.messages import AnyMessage, SystemMessage
from pydantic import BaseModel, Field

from t2_assistant.agents.llm import get_llm
from t2_assistant.agents.state import Route

ROUTER_SYSTEM = """You are the router for T2 Assistant, an internal company assistant \
that answers policy questions and summarises meeting notes, in English and Arabic.

Read the conversation and decide who should handle the user's latest message. \
Choose exactly one route:

- "greeting": greetings, thanks, or small talk ("hi", "how are you", "شكرا"). \
There is no task to do.
- "answer": a question that should be answered from company documents - HR, IT, \
finance, facilities, legal, travel, expenses, onboarding, and so on.
- "summarise": the user has given meeting notes, a transcript, or a report and \
wants a summary and/or a list of action items.
- "clarify": the message is too vague or incomplete to act on, and a single \
short follow-up question is needed first.

Decide from the meaning of the message, not from specific words. The message may \
be in English or Arabic. Give a short reason of one sentence."""


class RouterDecision(BaseModel):
    """The router's structured output."""

    route: Route = Field(description="Which specialist should handle the message.")
    reason: str = Field(description="One short sentence explaining the choice.")


def decide_route(messages: list[AnyMessage]) -> RouterDecision:
    """Ask the LLM which specialist should handle the latest message."""
    decider = get_llm().with_structured_output(RouterDecision)
    prompt: list[AnyMessage] = [SystemMessage(ROUTER_SYSTEM), *messages]
    result = decider.invoke(prompt)
    assert isinstance(result, RouterDecision)  # narrow the type for mypy
    return result
