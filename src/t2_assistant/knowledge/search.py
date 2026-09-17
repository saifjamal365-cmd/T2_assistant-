"""Search the knowledge base.

Given a question, return the passages whose meaning is closest to it, each with
the document it came from. This is what the answer agent calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow.entities import SpanType

from t2_assistant.config import settings
from t2_assistant.knowledge.embeddings import embed_query
from t2_assistant.knowledge.vector_store import get_collection


@dataclass
class Passage:
    """One retrieved passage and where it came from."""

    text: str
    doc_id: str
    title: str
    department: str
    doc_type: str
    version: str
    score: float  # 1.0 = identical meaning, 0.0 = unrelated

    def citation(self) -> str:
        return f"{self.doc_id} - {self.title}"


def _filter(active_only: bool, head_office_only: bool) -> dict[str, object] | None:
    clauses: list[dict[str, object]] = []
    if active_only:
        clauses.append({"status": "Active"})
    if head_office_only:
        # office_code ("HO"), not the "office" display name: that name is
        # localized per document language (English "Head Office" vs Arabic
        # "المكتب الرئيسي"), so filtering on it would silently exclude every
        # Arabic head-office document from every default search.
        clauses.append({"office_code": "HO"})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@mlflow.trace(span_type=SpanType.RETRIEVER)
def search(
    query: str,
    k: int | None = None,
    *,
    active_only: bool = True,
    head_office_only: bool = True,
) -> list[Passage]:
    """Return up to `k` passages closest in meaning to `query`.

    - `active_only` keeps out documents marked "Superseded", so the answer comes
      from the current version of a policy.
    - `head_office_only` keeps out the per-office variants, so a question with no
      office named is answered from the general policy.
    - At most one passage per document is returned (the best-matching one), so
      the answer agent sees `k` different documents, not `k` slices of one.
    """
    k = k or settings.retrieval_k
    query_vector = embed_query(query)

    result = get_collection().query(
        query_embeddings=[query_vector],  # type: ignore[arg-type]  # chromadb hint too narrow
        n_results=k * 4,  # over-fetch, then keep the best passage per document
        where=_filter(active_only, head_office_only),  # type: ignore[arg-type]
    )

    documents = result["documents"][0] if result["documents"] else []
    metadatas = result["metadatas"][0] if result["metadatas"] else []
    distances = result["distances"][0] if result["distances"] else []

    best_by_doc: dict[str, Passage] = {}
    for text, meta, distance in zip(documents, metadatas, distances, strict=False):
        doc_id = str(meta.get("doc_id", ""))
        passage = Passage(
            text=text,
            doc_id=doc_id,
            title=str(meta.get("title", "")),
            department=str(meta.get("department", "")),
            doc_type=str(meta.get("type", "")),
            version=str(meta.get("version", "")),
            score=round(1.0 - float(distance), 3),
        )
        if doc_id not in best_by_doc or passage.score > best_by_doc[doc_id].score:
            best_by_doc[doc_id] = passage

    ranked = sorted(best_by_doc.values(), key=lambda p: p.score, reverse=True)
    return ranked[:k]
