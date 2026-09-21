"""The answer agent.

For a policy question it:
  1. fixes spelling/typing mistakes in the latest message, as its own focused
     step (kept separate from standalone-question rewriting below, which
     already has several other jobs and was proving unreliable at fixing
     typos as a side effect of the rest of its work),
  2. turns the latest message into a standalone question (so a follow-up like
     "and for part-time staff?" carries the earlier context into the search),
  3. searches the knowledge base for the closest passages,
  4. writes a short answer using ONLY those passages, with the source document
     ids, or says it does not know,
  5. if the first answer is "I don't know" and it still has tries left, asks the
     model for a better search query and tries once more,
  6. if it is still about to decline after that, one last "judge" pass looks
     again at every passage seen across all attempts combined, before giving
     up - a wrong "I don't know" is worse than the extra cost of this one
     additional check, but only paid when a decline is about to happen, not
     on every question.

It never invents a policy fact from outside the retrieved passages (FR-03,
FR-05). It is also given the recent conversation, so a question about the
conversation itself ("what did I ask you first?") is answered from that
history instead of being searched for in the documents.

The exact passages behind a non-decline answer - only the ones actually named
in its "Sources:" line - are attached to the returned message's
`additional_kwargs["passages"]`, so a caller can show the user precisely what
grounded the answer, not just the document id. Each passage also carries
`highlights`: the specific sentence(s) in it closest in meaning to the answer,
found with the same local embedding model used for search - no extra model
call, so this costs nothing beyond a few short CPU encodes.
"""

from __future__ import annotations

import re
from dataclasses import asdict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from t2_assistant.agents.llm import get_llm
from t2_assistant.config import settings
from t2_assistant.knowledge.embeddings import embed_passages, embed_query
from t2_assistant.knowledge.search import Passage, search

_DOC_ID = re.compile(r"\b[A-Z]{2,4}-\d{3,4}\b")
_SOURCES_LINE = re.compile(r"\n?(?:Sources?|المصدر|المصادر)\s*:.*$", re.IGNORECASE | re.DOTALL)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?؟])\s+")
_MIN_HIGHLIGHT_CHARS = 15  # skips short section headers like "3. Policy"


def _cited_doc_ids(answer: str) -> set[str]:
    """The document codes the answer actually named, anywhere in its text -
    not just after "Sources:", since that label differs by language."""
    return set(_DOC_ID.findall(answer))


def _answer_body(answer: str) -> str:
    """The answer's substance, with the trailing "Sources: ..." line removed -
    that label isn't meaningful content to compare against a passage."""
    return _SOURCES_LINE.sub("", answer).strip()


def _candidate_sentences(passage_text: str) -> list[str]:
    """Passage text broken into short, highlightable pieces: split on blank
    lines/bullets first (this knowledge base is bullet-heavy), then on
    sentence endings within a line. Section headers and other short lines
    are dropped - nothing worth highlighting is that short."""
    candidates: list[str] = []
    for line in passage_text.split("\n"):
        line = line.strip().lstrip("-•").strip()
        if not line:
            continue
        for piece in _SENTENCE_SPLIT.split(line):
            piece = piece.strip()
            if len(piece) >= _MIN_HIGHLIGHT_CHARS:
                candidates.append(piece)
    return candidates


def _best_highlights(answer: str, passage_text: str) -> list[str]:
    """The sentence(s) in `passage_text` closest in meaning to `answer` - the
    local embedding model already used for search, not a model call, so this
    is free and fast. Ties close to the top score are included too, since the
    answer may be supported by more than one sentence."""
    candidates = _candidate_sentences(passage_text)
    if not candidates:
        return []
    answer_vec = embed_query(_answer_body(answer))
    sentence_vecs = embed_passages(candidates)
    scored = sorted(
        zip(candidates, sentence_vecs, strict=True),
        key=lambda pair: sum(a * b for a, b in zip(answer_vec, pair[1], strict=True)),
        reverse=True,
    )
    top_score = sum(a * b for a, b in zip(answer_vec, scored[0][1], strict=True))
    return [
        sentence
        for sentence, vec in scored[:3]
        if top_score - sum(a * b for a, b in zip(answer_vec, vec, strict=True)) < 0.02
    ]


_DONT_KNOW_EN = (
    "I don't know based on the current company documents. "
    "Could you rephrase your question or add a bit more detail?"
)
_DONT_KNOW_AR = "لا أملك إجابة لذلك بناءً على مستندات الشركة الحالية. هل يمكنك توضيح سؤالك أكثر؟"
_DONT_KNOW = _DONT_KNOW_EN  # default / fallback only - the model is told to pick EN or AR


def _is_arabic(text: str) -> bool:
    return any("؀" <= ch <= "ۿ" for ch in text)


def _dont_know_for(question: str) -> str:
    return _DONT_KNOW_AR if _is_arabic(question) else _DONT_KNOW_EN


def _is_decline(answer: str) -> bool:
    return answer.startswith(_DONT_KNOW_EN[:15]) or answer.startswith(_DONT_KNOW_AR[:15])


ANSWER_SYSTEM = f"""You are T2 Assistant, an internal company assistant.

You are given the recent conversation and passages retrieved from T2's current
policy documents. Each passage starts with its document code, like "HR-0003".

Rules:
- If the question is about company policy: answer using ONLY the passages below.
  If the passages contain the answer, give a short, direct answer (2-4 sentences),
  then on a new line write "Sources: " followed by the document codes you used
  (for example: "Sources: HR-0003, HR-0005"). If the passages do NOT contain the
  answer, reply with exactly this sentence and nothing else, matching the
  question's language: "{_DONT_KNOW_EN}" for an English question, or
  "{_DONT_KNOW_AR}" for an Arabic question.
- The question may get a real, named policy's detail wrong (a number, a frequency,
  a date) - for example "since it happens every 6 months, when's the next one?"
  when it is actually yearly. If the passages describe that SAME policy, correct
  the wrong detail and give the real answer - do not decline just because a
  detail was wrong.
- The question may instead ask about a specific item, scenario, or category that
  the passages never name at all (for example a particular kind of purchase,
  expense, or theme) - even if a passage about a related general policy was
  retrieved. A general policy does not automatically cover every specific case a
  question can invent. Decline unless the passages specifically name that item,
  or plainly state the rule applies to all cases of that kind.
- If some passages are for a specific office (the title says so) and the question
  does not mention an office, use the general (Head Office) figure and add one
  line that some offices differ.
- If the question is instead about the conversation itself - recalling it (for
  example "what did I ask you first?"), or judging or classifying something in
  it (for example "was that a financial question or a policy question?", "did
  you just contradict yourself?") - answer briefly using the conversation shown
  below - do not use the passages for this, and do not decline.
- The passages may cover two or more CLEARLY DIFFERENT policies that could each
  match a word in the question (for example, one passage about an employee
  referring a job candidate for a bonus, and a separate passage about the
  company ending an employee's own contract - both can match a word like
  "referral" even though they mean different things). If the question's wording
  does not make clear which one is meant, say in one short sentence what the two
  possible meanings are and ask which one they meant - do not silently pick one
  and answer as if the question were unambiguous.
- Never add a policy fact that is not in the passages.
- Reply in the same language as the question.
"""

_REPHRASE_SYSTEM = """The search of the company knowledge base did not find an
answer. The user's own words may not match the policy document's words for the
same idea - company documents tend to use more formal, specific HR/policy terms.
Think about what the underlying policy topic is actually likely to be called, not
just a paraphrase of the question. For example, a question about running out of
annual leave and not being paid is really about "unpaid leave"; a question about
withdrawing salary early is really about a "salary advance" or "payroll
exception". Rewrite the question as a short search query (a few keywords or a
plain sentence) using that more likely policy terminology. Reply with the query
only."""

_STANDALONE_SYSTEM = """Rewrite the user's last message as one standalone question
that makes sense on its own, with no earlier conversation needed to understand it.
Fill in not just pronouns but the specific policy or topic being discussed - a
search engine will use only your rewritten question, not the conversation, so it
must name the actual subject. For example, if the conversation is about an early
salary withdrawal exception and the last message is "should I contact Finance?",
write "should I contact Finance about an early salary withdrawal exception?", not
just "should I contact Finance?". Keep the user's language. Reply with the
question only."""

_SPELLCHECK_SYSTEM = """Fix spelling and typing mistakes in the message below, so
it reads as a normal, correctly-written sentence. This is the ONLY thing to do -
do not answer the message, do not translate it, do not add or remove information,
do not change what is being asked. Keep the same language. A word can be missing
a letter, have an extra letter, or not be a real word at all - use the rest of
the sentence to work out the word that was actually meant. If the message already
has no mistakes, return it completely unchanged. Reply with the corrected message
only, nothing else."""


def _last_user_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _correct_spelling(text: str, model: str) -> str:
    """Fix typos before anything else sees the message - a single, focused
    step, kept separate from standalone-question rewriting (which already has
    several other jobs to do) so it isn't competing for the model's attention."""
    if not text.strip():
        return text
    reply = get_llm(model).invoke([SystemMessage(_SPELLCHECK_SYSTEM), HumanMessage(text)])
    return str(reply.content).strip() or text


def _transcript(messages: list[AnyMessage]) -> str:
    return "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
        if isinstance(m, HumanMessage | AIMessage)
    )


def _standalone_question(messages: list[AnyMessage], transcript: str, model: str) -> str:
    """The latest question, with earlier context folded in (for follow-ups)."""
    latest = _last_user_text(messages)
    earlier = [m for m in messages if isinstance(m, HumanMessage | AIMessage)][:-1]
    if not earlier:
        return latest

    reply = get_llm(model).invoke([SystemMessage(_STANDALONE_SYSTEM), HumanMessage(transcript)])
    return str(reply.content).strip() or latest


def _format_passages(passages: list[Passage]) -> str:
    blocks = []
    for passage in passages:
        header = f"{passage.doc_id} - {passage.title} (v{passage.version})"
        blocks.append(f"--- {header} ---\n{passage.text}")
    return "\n\n".join(blocks)


def _write_answer(question: str, passages: list[Passage], transcript: str, model: str) -> str:
    if not passages:
        return _dont_know_for(question)
    prompt = [
        SystemMessage(
            ANSWER_SYSTEM
            + "\n\nRecent conversation:\n"
            + transcript
            + "\n\nPassages:\n"
            + _format_passages(passages)
        ),
        HumanMessage(question),
    ]
    reply = get_llm(model).invoke(prompt)
    return str(reply.content).strip()


def _better_query(question: str, model: str) -> str:
    reply = get_llm(model).invoke([SystemMessage(_REPHRASE_SYSTEM), HumanMessage(question)])
    return str(reply.content).strip() or question


_JUDGE_SYSTEM = f"""You are double-checking a decision to tell the user "I don't
know" to a company policy question - before it is sent, take one more careful
look. Below are ALL the passages retrieved across every search attempt so far,
combined - there may be more here than were shown in any single attempt.

Read the question and these passages again, carefully. If, on this closer look,
the passages actually DO answer the question - even if it takes connecting two
passages, or the answer is a general rule that covers this specific case - write
that answer now: a short, direct answer (2-4 sentences), then on a new line
"Sources: " followed by the document codes used.

Only if the passages genuinely do not cover this question, even after this
second look, confirm the decline: reply with exactly this sentence and nothing
else, matching the question's language: "{_DONT_KNOW_EN}" for an English
question, or "{_DONT_KNOW_AR}" for an Arabic question.

Never add a fact that is not in the passages. Reply in the same language as the
question."""


def _judge_before_declining(
    question: str, passages: list[Passage], transcript: str, model: str
) -> str:
    """A last, careful look before giving up - the one place a second LLM
    pass is worth its extra cost, since a wrong "I don't know" is the failure
    that matters most for a policy assistant."""
    if not passages:
        return _dont_know_for(question)
    prompt = [
        SystemMessage(
            _JUDGE_SYSTEM
            + "\n\nRecent conversation:\n"
            + transcript
            + "\n\nAll passages seen so far:\n"
            + _format_passages(passages)
        ),
        HumanMessage(question),
    ]
    reply = get_llm(model).invoke(prompt)
    return str(reply.content).strip()


def respond(messages: list[AnyMessage], model: str) -> AIMessage:
    """Answer the latest question from the knowledge base."""
    corrected = _correct_spelling(_last_user_text(messages), model)
    fixed_messages = [*messages[:-1], HumanMessage(corrected)]

    transcript = _transcript(fixed_messages)
    question = _standalone_question(fixed_messages, transcript, model)

    query = question
    answer = _dont_know_for(question)
    seen_passages: dict[str, Passage] = {}
    last_passages: list[Passage] = []
    for attempt in range(settings.max_retrieval_tries):
        passages = search(query)
        last_passages = passages
        for passage in passages:
            existing = seen_passages.get(passage.doc_id)
            if existing is None or passage.score > existing.score:
                seen_passages[passage.doc_id] = passage
        answer = _write_answer(question, passages, transcript, model)
        if not _is_decline(answer):
            break
        if attempt + 1 < settings.max_retrieval_tries:
            query = _better_query(question, model)

    if _is_decline(answer):
        last_passages = list(seen_passages.values())
        answer = _judge_before_declining(question, last_passages, transcript, model)

    if _is_decline(answer):
        # The model is asked to match the question's language for this exact
        # sentence, but doesn't always get it right - pick it in code instead
        # of trusting that, so a decline is never in the wrong language.
        answer = _dont_know_for(question)
        sources: list[Passage] = []
    else:
        cited = _cited_doc_ids(answer)
        sources = [p for p in last_passages if p.doc_id in cited] or last_passages

    passages_out = [
        {**asdict(p), "highlights": _best_highlights(answer, p.text)} for p in sources
    ]
    kwargs = {"passages": passages_out} if passages_out else {}
    return AIMessage(content=answer, additional_kwargs=kwargs)
