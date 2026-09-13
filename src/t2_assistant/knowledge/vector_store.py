"""The Chroma vector index.

Chroma keeps every passage together with its embedding (the list of numbers from
embeddings.py) in a folder on disk (settings.chroma_dir). Given a new piece of
text, it quickly finds the stored passages whose meaning is closest.

Two jobs share the same collection:
  - build_index.py  writes passages into it
  - search.py       reads the closest passages out of it
"""

from __future__ import annotations

import shutil
from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from t2_assistant.config import settings


@lru_cache
def get_collection() -> Collection:
    """Open (or create) the collection stored under settings.chroma_dir."""
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},  # match the normalised embeddings
    )


def reset_index() -> None:
    """Delete the whole index folder. Call before a full rebuild.

    Safe to call when nothing has opened the collection yet (as in build_index.py).
    """
    get_collection.cache_clear()
    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
