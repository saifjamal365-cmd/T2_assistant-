"""Full-pipeline evaluation: run every question through the real assistant.

Unlike retrieval.py, this calls chat() - router, retrieval, and the language
model - so it is slower and costs real API calls. It measures what a user
actually sees:

  - for an answerable question: did it cite the right document, and not
    say "I don't know"?
  - for an unanswerable one: did it correctly say "I don't know" (or ask to
    clarify) instead of inventing something? (NFR-03, groundedness)

Runs with a small thread pool, since each question is an independent network
call to Groq. A question that errors (timeout, API error) is recorded as a
failure, not a crash - one bad question should not stop a 500-question run.

Groq's free tier caps how many tokens per minute the whole run may use. With
500 questions that limit is hit constantly - not a real failure, just "wait
and try again" - so a rate-limit response is retried with backoff and does
NOT count against the assistant. Only a genuine error (after retrying) does.

The free tier also caps tokens PER DAY (much larger than the per-minute
window), and that one cannot be waited out mid-run - once it is hit, every
remaining question fails until the quota resets. load_previous_results()
lets a later run pick up only the questions that failed, instead of
re-spending tokens on ones that already succeeded.
"""

from __future__ import annotations

import contextvars
import json
import random
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from groq import RateLimitError

from t2_assistant.agents.answer import _is_decline
from t2_assistant.conversation import ChatResult, chat
from t2_assistant.evaluation.dataset import EvalItem

_MAX_RATE_LIMIT_RETRIES = 8
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

# Fewest LLM calls (router excluded) a route could ever take for a fresh,
# standalone question - every eval item is one, since EvalItem carries no
# conversation history. See answer.py's respond() for what a "step" is.
_OPTIMAL_STEPS_BY_ROUTE: dict[str, int] = {
    "answer": 2,  # spelling-fix + one successful write-answer pass
    "greeting": 1,
    "clarify": 1,
    "summarise": 1,
}


def _optimal_steps(route: str) -> int:
    return _OPTIMAL_STEPS_BY_ROUTE.get(route, 1)


@dataclass
class AnswerResult:
    item_id: str
    route: str
    reply: str
    said_dont_know: bool
    cited_expected_doc: bool
    correct: bool
    duration_ms: int
    error: str | None
    retries: int = 0
    # None on an errored item - it never reached a ChatResult, so there is
    # nothing to measure. See report.py's _convergence_pairs.
    steps: int | None = None
    optimal_steps: int | None = None


def _chat_with_rate_limit_retry(question: str) -> tuple[ChatResult, int, int]:
    """Call chat(), waiting out Groq's "too many tokens this minute" response.

    Returns the result, how many times a rate limit made us wait, and the
    duration of the call that actually succeeded - not counting time spent
    waiting out the rate limit, so latency numbers reflect the assistant
    itself, not this evaluation harness's free-tier quota.
    """
    for attempt in range(_MAX_RATE_LIMIT_RETRIES):
        try:
            started = time.perf_counter()
            result = chat(question)
            return result, attempt, int((time.perf_counter() - started) * 1000)
        except RateLimitError as exc:
            match = _RETRY_AFTER.search(str(exc))
            wait = float(match.group(1)) if match else 2**attempt
            time.sleep(wait + random.uniform(0.1, 1.0))
    started = time.perf_counter()
    result = chat(question)  # let a final failure raise normally
    return result, _MAX_RATE_LIMIT_RETRIES, int((time.perf_counter() - started) * 1000)


def _evaluate_one(item: EvalItem) -> AnswerResult:
    started = time.perf_counter()
    try:
        result, retries, duration_ms = _chat_with_rate_limit_retry(item.question)

        said_dont_know = _is_decline(result.reply)
        cited = any(doc in result.reply for doc in item.expected_docs)

        if item.answerable:
            correct = cited and not said_dont_know
        else:
            correct = said_dont_know or result.route == "clarify"

        return AnswerResult(
            item_id=item.id,
            route=result.route,
            reply=result.reply,
            said_dont_know=said_dont_know,
            cited_expected_doc=cited,
            correct=correct,
            duration_ms=duration_ms,
            error=None,
            retries=retries,
            steps=result.steps,
            optimal_steps=_optimal_steps(result.route),
        )
    except Exception as exc:  # noqa: BLE001 - one bad question must not stop the run
        duration_ms = int((time.perf_counter() - started) * 1000)
        return AnswerResult(
            item_id=item.id,
            route="error",
            reply="",
            said_dont_know=False,
            cited_expected_doc=False,
            correct=False,
            duration_ms=duration_ms,
            error=str(exc),
        )


def evaluate_answers(
    items: list[EvalItem],
    workers: int = 2,
    on_result: Callable[[AnswerResult, int, int], None] | None = None,
) -> list[AnswerResult]:
    results: list[AnswerResult] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Copy the tracing context into each worker thread - MLflow tracks
        # the active run/trace via contextvars, which a plain pool.submit()
        # does not inherit, so without this every question's trace would be
        # orphaned instead of attached to the run started around this call.
        # A fresh copy per item, not one shared context reused for all of
        # them: a single Context object can only be entered by one thread
        # at a time, and reusing it across concurrent submissions crashes
        # with "cannot enter context: ... already entered" the moment two
        # workers genuinely overlap - confirmed live, workers=2 hits this on
        # nearly every run, not just in theory.
        futures = {
            pool.submit(contextvars.copy_context().run, _evaluate_one, item): item
            for item in items
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if on_result:
                on_result(result, done, len(items))
    return results


def load_previous_results(path: Path) -> dict[str, AnswerResult]:
    """Load a prior answers.jsonl, keyed by item id - empty dict if it's missing."""
    if not path.exists():
        return {}
    results = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results[row["item_id"]] = AnswerResult(**row)
    return results
