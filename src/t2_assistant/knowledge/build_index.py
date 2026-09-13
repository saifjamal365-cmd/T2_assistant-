"""Build (or rebuild) the search index.

This is an offline job, not part of answering a request. It:

  1. reads every document listed in data/knowledge_base/manifest.csv
  2. splits each one into passages (chunking.py)
  3. turns each passage into a vector (embeddings.py)
  4. stores the passages, their vectors and their document details in Chroma

Run it whenever the documents change:

    python -m t2_assistant.knowledge.build_index

It always rebuilds from scratch, so removing a document from the manifest also
removes it from the index.
"""

from __future__ import annotations

import csv
import time
from collections import Counter
from collections.abc import Iterator
from itertools import islice

from t2_assistant.config import settings
from t2_assistant.knowledge.chunking import split_text
from t2_assistant.knowledge.embeddings import embed_passages
from t2_assistant.knowledge.vector_store import get_collection, reset_index

# Chroma accepts str / int / float / bool metadata values only.
_META_FIELDS = (
    "doc_id",
    "title",
    "department",
    "type",
    "language",
    "office",
    "version",
    "status",
    "effective_date",
    "superseded_by",
)

_BATCH = 256


def _manifest_rows() -> list[dict[str, str]]:
    with (settings.knowledge_base_dir / "manifest.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _passages() -> Iterator[tuple[str, str, dict[str, str]]]:
    """Yield (id, passage_text, metadata) for every passage in the corpus."""
    for row in _manifest_rows():
        text = (settings.knowledge_base_dir / row["file"]).read_text(encoding="utf-8")
        metadata = {field: row.get(field, "") or "" for field in _META_FIELDS}
        for index, passage in enumerate(split_text(text)):
            yield f"{row['doc_id']}::{index}", passage, metadata


def _batched(
    items: Iterator[tuple[str, str, dict[str, str]]], size: int
) -> Iterator[list[tuple[str, str, dict[str, str]]]]:
    while batch := list(islice(items, size)):
        yield batch


def main() -> None:
    started = time.perf_counter()
    print(f"Rebuilding the index at {settings.chroma_dir} ...")
    reset_index()
    collection = get_collection()

    total_passages = 0
    languages: Counter[str] = Counter()
    documents: set[str] = set()

    for batch in _batched(_passages(), _BATCH):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metadatas = [item[2] for item in batch]
        # chromadb's type hints are narrower than what it accepts at runtime.
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,  # type: ignore[arg-type]
            embeddings=embed_passages(texts),  # type: ignore[arg-type]
        )

        total_passages += len(batch)
        for meta in metadatas:
            languages[meta["language"]] += 1
            documents.add(meta["doc_id"])
        print(f"  {total_passages} passages", end="\r")

    seconds = time.perf_counter() - started
    print(f"\nDone in {seconds:.0f}s.")
    print(f"  documents indexed : {len(documents)}")
    print(f"  passages          : {total_passages}")
    print(f"  by language       : {dict(languages)}")


if __name__ == "__main__":
    main()
