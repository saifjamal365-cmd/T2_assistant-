"""The clarify agent.

Used when the router judges the request too vague to act on. It asks exactly
one short follow-up question, in the user's language, so the next message can
be routed properly.

The router occasionally sends a question about the conversation itself here by
mistake (for example, some Arabic phrasings of "what did I ask first?" are
misread as unclear) - so this agent can also answer that kind of question
directly from the visible history, as a safety net, instead of always asking a
follow-up.

This agent never searches the knowledge base, so its follow-up question must
not state a specific number, date, or other policy detail - it has no way to
know if one is correct, and doing so would be a guess dressed up as a question.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage

from t2_assistant.agents.llm import get_llm

CLARIFY_SYSTEM = """You are T2 Assistant, an internal company assistant.

The user's message was sent here because it looked too vague or incomplete to
act on. You have NOT looked at any company document - you cannot see whether
any number, date, or policy detail below is correct.

- If it is actually a clear question about the conversation itself (for example
  "what did I ask you first?" or "what did you just say?"), answer it briefly
  using the conversation shown to you - do not treat it as vague.
- Otherwise, ask ONE short, general follow-up question that would let you help -
  for example which policy area they mean, or what they want done with the text
  they sent. Do NOT name a specific number, duration, amount, or other policy
  detail in your question, even as an example or to sound helpful - you have not
  verified it and it may be wrong. Ask only about what the user means, never
  about a specific figure.

Reply in the SAME language as the user (English or Arabic)."""


def respond(messages: list[AnyMessage]) -> AIMessage:
    """Return one short clarifying question."""
    reply = get_llm().invoke([SystemMessage(CLARIFY_SYSTEM), *messages])
    assert isinstance(reply, AIMessage)  # narrow the type for mypy
    return reply
