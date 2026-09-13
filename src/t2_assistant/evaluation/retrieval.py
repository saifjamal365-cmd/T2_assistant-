"""Retrieval-only evaluation.

For every answerable question: does search() return a passage from the right
topic - any of its documents, not only the one the question text came from
(see build_eval_set.py)? This needs only the embedding model - no LLM calls -
so it runs against all 384 answerable questions in a couple of minutes. It
measures NFR-02 (retrieval accuracy) directly, separate from how well the
language model then writes the answer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from t2_assistant.evaluation.dataset import EvalItem
from t2_assistant.knowledge.search import search


@dataclass
class RetrievalResult:
    item_id: str
    hit: bool  # the expected document was somewhere in the results
    rank: int | None  # 1-based position, or None if it was not returned at all
    top_doc: str | None  # the document search ranked first


def evaluate_retrieval(
    items: list[EvalItem],
    k: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[RetrievalResult]:
    answerable = [item for item in items if item.answerable and item.expected_docs]
    results: list[RetrievalResult] = []

    for index, item in enumerate(answerable, start=1):
        passages = search(item.question, k=k)
        doc_ids = [passage.doc_id for passage in passages]
        expected = set(item.expected_docs)
        rank = next((i for i, doc in enumerate(doc_ids, start=1) if doc in expected), None)
        top_doc = doc_ids[0] if doc_ids else None
        results.append(
            RetrievalResult(item_id=item.id, hit=rank is not None, rank=rank, top_doc=top_doc)
        )
        if on_progress:
            on_progress(index, len(answerable))

    return results
