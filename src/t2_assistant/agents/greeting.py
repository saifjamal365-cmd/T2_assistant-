"""The greeting agent.

Handles greetings, thanks and small talk. It does not answer policy questions.
It replies briefly, in the user's language, and reminds the user what the
assistant is for.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage

from t2_assistant.agents.llm import get_llm

GREETING_SYSTEM = """You are T2 Assistant, an internal company assistant.

The user has greeted you or made small talk. Reply in ONE or TWO short, warm,
professional sentences, in the SAME language as the user (English or Arabic).

Briefly mention that you can answer questions about company policy and summarise
meeting notes.

You have NOT been given any policy documents here, so you cannot state any
policy fact, number, date, procedure, or contact detail - not even one you think
you remember. If the message is not plainly a greeting or thanks (for example it
looks like a real question, even a short or oddly-phrased one), do not try to
answer it. Just say briefly that you can help with that, and ask them to ask it
as a question."""


def respond(messages: list[AnyMessage], model: str) -> AIMessage:
    """Return a short greeting reply."""
    reply = get_llm(model).invoke([SystemMessage(GREETING_SYSTEM), *messages])
    assert isinstance(reply, AIMessage)  # narrow the type for mypy
    return reply
