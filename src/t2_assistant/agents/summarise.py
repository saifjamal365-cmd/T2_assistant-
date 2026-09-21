"""The summary agent.

Turns meeting notes, a transcript or a report into a short summary plus a list
of action items. It uses only the text the user sent - it does not look anything
up and does not invent details.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage

from t2_assistant.agents.llm import get_llm

SUMMARISE_SYSTEM = """You are T2 Assistant, an internal company assistant.

The user has given you meeting notes, a transcript or a report. Reply in the
SAME language as the user (English or Arabic), with exactly two parts:

Summary: 2-4 sentences covering what was discussed and decided.

Action items: a bullet list, each as "owner - task - due date" when the date is
known. If the text names no actions, write "No action items."

Use only what is in the user's text. Do not add anything that is not there."""


def respond(messages: list[AnyMessage], model: str) -> AIMessage:
    """Return a summary and action items for the user's text."""
    reply = get_llm(model).invoke([SystemMessage(SUMMARISE_SYSTEM), *messages])
    assert isinstance(reply, AIMessage)  # narrow the type for mypy
    return reply
