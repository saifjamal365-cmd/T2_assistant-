"""Splitting documents into passages."""

from __future__ import annotations

from t2_assistant.config import settings
from t2_assistant.knowledge.chunking import split_text


def test_short_text_is_one_passage() -> None:
    assert split_text("one short paragraph") == ["one short paragraph"]


def test_empty_text_gives_no_passages() -> None:
    assert split_text("") == []
    assert split_text("\n\n   \n\n") == []


def test_long_text_is_split_with_overlap() -> None:
    para = "This sentence is about the annual leave policy and how to request days off. "
    text = "\n\n".join(para * 4 for _ in range(12))  # well over chunk_size

    passages = split_text(text)

    assert len(passages) > 1
    # every passage respects the size budget (allow the hard-split slack)
    assert all(len(p) <= settings.chunk_size * 1.5 for p in passages)
    # neighbours share some text
    assert passages[0][-settings.chunk_overlap :] in passages[1]
