"""Load the evaluation set built by scripts/build_eval_set.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from t2_assistant.config import PROJECT_ROOT

DATASET_PATH = PROJECT_ROOT / "eval" / "dataset.json"


@dataclass
class EvalItem:
    id: str
    family: str  # "faq_exact" | "faq_paraphrased" | "unanswerable"
    question: str
    language: str  # "en" | "ar"
    department: str | None
    topic: str | None
    expected_doc: str | None  # the document the question's text itself came from
    expected_docs: list[str]  # any of these counts as a correctly sourced citation
    expected_fact: str | None  # the real, sourced answer text (for reference)
    answerable: bool


def load_dataset(path: Path = DATASET_PATH, limit: int | None = None) -> list[EvalItem]:
    """Read eval/dataset.json. `limit` takes the first N items (for a quick run)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = [EvalItem(**row) for row in raw]
    return items[:limit] if limit else items
