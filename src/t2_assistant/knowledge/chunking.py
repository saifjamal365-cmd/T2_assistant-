"""Splitting a document into passages.

A whole policy document is too big to send to the language model for every
question, and most questions are answered by one part of it. So each document is
cut into overlapping "passages" of roughly `settings.chunk_size` characters.

The split follows blank lines (paragraph boundaries) where it can, so a passage
is usually a few whole paragraphs rather than a sentence cut in half. Each new
passage repeats the last `settings.chunk_overlap` characters of the previous one,
so a fact that sits on a boundary is not lost.
"""

from __future__ import annotations

from t2_assistant.config import settings


def split_text(text: str) -> list[str]:
    """Cut `text` into overlapping passages."""
    size = settings.chunk_size
    overlap = settings.chunk_overlap

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    passages: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > size:
            passages.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        passages.append(current)

    # Hard-split any passage that is still much bigger than the target
    # (a single very long paragraph).
    result: list[str] = []
    step = max(size - overlap, 1)
    for passage in passages:
        if len(passage) <= size * 1.5:
            result.append(passage)
        else:
            for start in range(0, len(passage), step):
                result.append(passage[start : start + size])
    return result
