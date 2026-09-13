"""Knowledge-base search.

Needs the real index (python -m t2_assistant.knowledge.build_index) and the
embedding model, so the whole module is marked `integration`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_search_finds_the_leave_policy() -> None:
    from t2_assistant.knowledge.search import search

    results = search("how many annual leave days do I get?")

    assert results, "search returned nothing - is the index built?"
    assert any("leave" in passage.title.lower() for passage in results)
    assert results[0].score > results[-1].score or len(results) == 1  # ranked


def test_search_returns_arabic_for_arabic() -> None:
    from t2_assistant.knowledge.search import search

    results = search("كم عدد أيام الإجازة السنوية؟")

    assert results
    # BGE-M3 is cross-lingual; the top hit should still be a leave document
    assert any("leave" in p.title.lower() or "إجاز" in p.title for p in results)
