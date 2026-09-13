"""The embedding model.

An "embedding" is a list of numbers that captures the meaning of a piece of
text. Two texts that mean the same thing get similar numbers - even if they use
different words, or different languages. That is how the assistant searches the
knowledge base by meaning instead of by exact words.

We use a multilingual E5 model (Arabic + English), which is trained for search:
questions and documents are embedded a little differently, so a short question
still matches a longer passage that answers it. E5 needs a small prefix on each
text - "query: " for a question, "passage: " for a document - which this module
adds for you.

The model runs locally on the CPU: no text leaves the machine, no per-call cost.
The files download once on first use and are cached.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from t2_assistant.config import settings

_QUERY_PREFIX = "query: "
_PASSAGE_PREFIX = "passage: "


@lru_cache
def _model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(settings.embedding_model)


def _encode(texts: list[str]) -> list[list[float]]:
    vectors = _model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed knowledge-base passages (for indexing)."""
    return _encode([_PASSAGE_PREFIX + t for t in texts])


def embed_query(text: str) -> list[float]:
    """Embed a user's question (for searching)."""
    return _encode([_QUERY_PREFIX + text])[0]
