"""The router agent.

It reads the conversation and decides which specialist should handle the latest
message. The decision is made by the LLM reasoning about meaning - there is no
list of keywords. The model must return a structured answer (route + reason),
which we enforce with `with_structured_output`.

The model occasionally returns output that fails to parse into that structure
(a Groq "output_parse_failed" error). Retrying alone does not fix it: testing
found this failure is deterministic for a given prompt, not random - it is
Groq's structured-output feature becoming unreliable as the prompt gets
longer. A 4-message conversation failed 6/6 times; trimming it to 3 messages
(same content, just shorter) made it succeed every time. The router only
needs recent context to classify the latest message, not the full transcript
(the specialist agents still get the full history separately), so only the
last few messages are sent here to stay well clear of that failure mode. A
single retry is kept as a safety net for the rare, genuinely transient case.
"""

from __future__ import annotations

from groq import APIError
from langchain_core.messages import AnyMessage, SystemMessage
from pydantic import BaseModel, Field

from t2_assistant.agents.llm import get_llm
from t2_assistant.agents.state import Route

_MAX_ROUTE_ATTEMPTS = 2
_MAX_HISTORY_MESSAGES = 4  # verified safe: the failing 5-message prompt above
# succeeded every time once trimmed to the last 4 - see the module docstring

ROUTER_SYSTEM = """You are the router for T2 Assistant, an internal company assistant \
that answers policy questions and summarises meeting notes, in English and Arabic.

Read the conversation and decide who should handle the user's latest message. \
Choose exactly one route:

- "greeting": greetings, thanks, or small talk ("hi", "how are you", "شكرا"). \
There is no task to do.
- "answer": a question to answer - either a company-policy question (HR, IT, \
finance, facilities, legal, travel, expenses, onboarding, and so on), or a \
question about the conversation itself ("what did I ask you first?", "what did \
you just say?"). Both are handled by the same agent, which can see the chat so far.
- "summarise": the user has given meeting notes, a transcript, or a report and \
wants a summary and/or a list of action items.
- "clarify": the message is too vague or incomplete to act on, and a single \
short follow-up question is needed first. Also use this for a short message \
about something being done TO the employee (by "them"/the company/someone else) \
using a term that has more than one real company meaning - check who is doing \
the action to whom before picking a meaning. For example a short Arabic HR \
message using "إحالة" about the employee ("هل يحق لهم إحالتي") is NOT \
automatically about the employee-referral bonus program (which is the employee \
referring a candidate, the opposite direction) - it is at least as likely to \
mean the company transferring or letting the employee go. When the direction or \
meaning is genuinely unclear like this, ask rather than pick the more common \
association of the word. Do not use "clarify" just because the answer needs \
more context beyond that.

Decide from the meaning of the message, not from specific words. The message may \
be in English or Arabic. Give a short reason of one sentence."""


class RouterDecision(BaseModel):
    """The router's structured output."""

    route: Route = Field(description="Which specialist should handle the message.")
    reason: str = Field(description="One short sentence explaining the choice.")


def decide_route(messages: list[AnyMessage], model: str) -> RouterDecision:
    """Ask the LLM which specialist should handle the latest message."""
    decider = get_llm(model).with_structured_output(RouterDecision)
    recent = messages[-_MAX_HISTORY_MESSAGES:]
    prompt: list[AnyMessage] = [SystemMessage(ROUTER_SYSTEM), *recent]
    last_error: Exception | None = None
    for _attempt in range(_MAX_ROUTE_ATTEMPTS):
        try:
            result = decider.invoke(prompt)
            if isinstance(result, RouterDecision):
                return result
            # a malformed reply (e.g. None) without a raised error - treat it
            # the same as a failed attempt instead of crashing the request
            last_error = TypeError(f"router returned {type(result).__name__}, not a RouterDecision")
        except APIError as exc:
            last_error = exc
    assert last_error is not None  # the loop always sets it before falling through
    raise last_error
