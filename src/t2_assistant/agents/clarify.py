"""The clarify agent.

Used when the request is too vague to act on. It asks exactly one short
follow-up question, in the user's language, so the next message can be routed
properly.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage

from t2_assistant.agents.llm import get_llm

CLARIFY_SYSTEM = """You are T2 Assistant, an internal company assistant.

The user's message is too vague or incomplete to act on. Ask ONE short, specific
follow-up question that would let you help - for example which policy area they
mean, or what they want done with the text they sent.

Reply with just the question, in the SAME language as the user (English or
Arabic). Do not guess an answer."""


def respond(messages: list[AnyMessage]) -> AIMessage:
    """Return one short clarifying question."""
    reply = get_llm().invoke([SystemMessage(CLARIFY_SYSTEM), *messages])
    assert isinstance(reply, AIMessage)  # narrow the type for mypy
    return reply
