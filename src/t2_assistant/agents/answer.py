"""The answer agent.

For a policy question it:
  1. turns the latest message into a standalone question (so a follow-up like
     "and for part-time staff?" carries the earlier context into the search),
  2. searches the knowledge base for the closest passages,
  3. writes a short answer using ONLY those passages, with the source document
     ids, or says it does not know,
  4. if the first answer is "I don't know" and it still has tries left, asks the
     model for a better search query and tries once more.

It never uses knowledge from outside the retrieved passages (FR-03, FR-05).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from t2_assistant.agents.llm import get_llm
from t2_assistant.config import settings
from t2_assistant.knowledge.search import Passage, search

_DONT_KNOW = "I don't know based on the current company documents."

ANSWER_SYSTEM = f"""You are T2 Assistant, an internal company assistant.

Answer the user's question using ONLY the passages below. Each passage starts
with its document code, like "HR-0003". The passages come from T2's current
policy documents.

Rules:
- If the passages contain the answer: give a short, direct answer (2-4 sentences).
  Then, on a new line, write "Sources: " followed by the document codes you used
  (for example: "Sources: HR-0003, HR-0005").
- If some passages are for a specific office (the title says so) and the question
  does not mention an office, use the general (Head Office) figure and add one
  line that some offices differ.
- If the passages do NOT contain the answer: reply with exactly this sentence and
  nothing else: "{_DONT_KNOW}"
- Never add facts that are not in the passages.
- Reply in the same language as the question.
"""

_REPHRASE_SYSTEM = """The search of the company knowledge base did not find an
answer. Rewrite the user's question as a short search query (a few keywords or a
plain sentence) that might match the documents better. Reply with the query only."""

_STANDALONE_SYSTEM = """Rewrite the user's last message as one standalone question
that makes sense on its own, filling in anything it refers to from the earlier
conversation. Keep the user's language. Reply with the question only."""


def _last_user_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _standalone_question(messages: list[AnyMessage]) -> str:
    """The latest question, with earlier context folded in (for follow-ups)."""
    latest = _last_user_text(messages)
    earlier = [m for m in messages if isinstance(m, HumanMessage | AIMessage)][:-1]
    if not earlier:
        return latest

    transcript = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in messages
    )
    reply = get_llm().invoke([SystemMessage(_STANDALONE_SYSTEM), HumanMessage(transcript)])
    return str(reply.content).strip() or latest


def _format_passages(passages: list[Passage]) -> str:
    blocks = []
    for passage in passages:
        header = f"{passage.doc_id} - {passage.title} (v{passage.version})"
        blocks.append(f"--- {header} ---\n{passage.text}")
    return "\n\n".join(blocks)


def _write_answer(question: str, passages: list[Passage]) -> str:
    if not passages:
        return _DONT_KNOW
    prompt = [
        SystemMessage(ANSWER_SYSTEM + "\n\nPassages:\n" + _format_passages(passages)),
        HumanMessage(question),
    ]
    reply = get_llm().invoke(prompt)
    return str(reply.content).strip()


def _better_query(question: str) -> str:
    reply = get_llm().invoke([SystemMessage(_REPHRASE_SYSTEM), HumanMessage(question)])
    return str(reply.content).strip() or question


def respond(messages: list[AnyMessage]) -> AIMessage:
    """Answer the latest question from the knowledge base."""
    question = _standalone_question(messages)

    query = question
    answer = _DONT_KNOW
    for attempt in range(settings.max_retrieval_tries):
        passages = search(query)
        answer = _write_answer(question, passages)
        if not answer.startswith(_DONT_KNOW[:15]):
            break
        if attempt + 1 < settings.max_retrieval_tries:
            query = _better_query(question)

    return AIMessage(answer)
