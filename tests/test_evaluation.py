"""The evaluation package: dataset loading, scoring, and the report."""

from __future__ import annotations

import pytest

from t2_assistant.conversation import ChatResult
from t2_assistant.evaluation import quality
from t2_assistant.evaluation.dataset import EvalItem, load_dataset
from t2_assistant.evaluation.quality import AnswerResult, _evaluate_one, _optimal_steps
from t2_assistant.evaluation.report import build_report
from t2_assistant.evaluation.retrieval import RetrievalResult


def test_dataset_has_balanced_questions() -> None:
    """The exact count grows with the knowledge base (more topics get a full
    FAQ family as the corpus scales up) - what matters is that it stays
    internally consistent and evenly split, not a fixed number."""
    items = load_dataset()

    assert len(items) > 0
    assert len({i.id for i in items}) == len(items)  # every id is unique

    answerable = [i for i in items if i.answerable]
    assert len(answerable) > 0
    assert all(i.expected_doc for i in answerable)  # every answerable item is gradeable
    assert all(i.expected_doc in i.expected_docs for i in answerable)

    unanswerable = [i for i in items if not i.answerable]
    assert len(unanswerable) > 0
    assert all(i.expected_doc is None and not i.expected_docs for i in unanswerable)

    en_count = sum(1 for i in items if i.language == "en")
    ar_count = sum(1 for i in items if i.language == "ar")
    assert en_count == ar_count  # the generator balances languages exactly


def test_dataset_limit() -> None:
    assert len(load_dataset(limit=10)) == 10


def _item(**overrides: object) -> EvalItem:
    base = dict(
        id="X0001",
        family="faq_exact",
        question="How many annual leave days do I get?",
        language="en",
        department="HR",
        topic="annual_leave",
        expected_doc="HR-0001",
        expected_docs=["HR-0001", "HR-0005"],
        expected_fact="25 days.",
        answerable=True,
    )
    base.update(overrides)
    return EvalItem(**base)  # type: ignore[arg-type]  # dict[str, object] vs. the real field types


def test_report_scores_a_correct_and_an_incorrect_answer() -> None:
    items = [
        _item(id="A1"),
        _item(id="A2", answerable=False, expected_doc=None, expected_docs=[], question="pets?"),
    ]
    retrieval = [RetrievalResult(item_id="A1", hit=True, rank=1, top_doc="HR-0001")]
    answers = [
        AnswerResult(
            item_id="A1",
            route="answer",
            reply="25 days. Sources: HR-0001",
            said_dont_know=False,
            cited_expected_doc=True,
            correct=True,
            duration_ms=500,
            error=None,
        ),
        AnswerResult(
            item_id="A2",
            route="answer",
            reply="I don't know based on the current company documents.",
            said_dont_know=True,
            cited_expected_doc=False,
            correct=True,
            duration_ms=400,
            error=None,
        ),
    ]

    report = build_report(items, retrieval, answers)
    m = report["metrics"]

    assert m["retrieval_recall_at_k"] == 100.0
    assert m["answer_accuracy_answerable"] == 100.0
    assert m["honesty_rate_unanswerable"] == 100.0
    assert m["false_answer_rate_unanswerable"] == 0.0
    assert m["overall_accuracy"] == 100.0


def test_report_catches_a_hallucinated_answer() -> None:
    items = [
        _item(id="A3", answerable=False, expected_doc=None, expected_docs=[], question="pets?")
    ]
    answers = [
        AnswerResult(
            item_id="A3",
            route="answer",
            reply="Yes, pets are welcome under HR-0001.",
            said_dont_know=False,
            cited_expected_doc=False,
            correct=False,
            duration_ms=300,
            error=None,
        )
    ]

    report = build_report(items, [], answers)

    assert report["metrics"]["honesty_rate_unanswerable"] == 0.0
    assert report["metrics"]["false_answer_rate_unanswerable"] == 100.0


def test_optimal_steps_is_two_for_answer_and_one_for_the_trivial_routes() -> None:
    assert _optimal_steps("answer") == 2
    assert _optimal_steps("greeting") == 1
    assert _optimal_steps("clarify") == 1
    assert _optimal_steps("summarise") == 1


def test_evaluate_one_captures_steps_from_the_chat_result(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(question: str) -> ChatResult:
        return ChatResult(
            reply="25 days. Sources: HR-0001",
            route="answer",
            route_reason="q",
            sources=[],
            model="test-model",
            steps=4,
        )

    monkeypatch.setattr(quality, "chat", fake_chat)

    item = _item(id="S1")
    result = _evaluate_one(item)

    assert result.steps == 4
    assert result.optimal_steps == 2  # the route's minimum, not what this run took


def test_convergence_score_and_exclusion_of_trivial_routes() -> None:
    items = [_item(id=f"C{i}") for i in range(3)]
    answers = [
        # answer route, first write succeeds: no wasted work, ratio 1.0
        AnswerResult(
            item_id="C0",
            route="answer",
            reply="25 days. Sources: HR-0001",
            said_dont_know=False,
            cited_expected_doc=True,
            correct=True,
            duration_ms=100,
            error=None,
            steps=2,
            optimal_steps=2,
        ),
        # answer route, needed a retry: ratio 0.5 - the interesting signal
        AnswerResult(
            item_id="C1",
            route="answer",
            reply="25 days. Sources: HR-0001",
            said_dont_know=False,
            cited_expected_doc=True,
            correct=True,
            duration_ms=200,
            error=None,
            steps=4,
            optimal_steps=2,
        ),
        # misrouted to clarify - always steps=1/optimal=1 by construction;
        # must not dilute or crash the answer-route-only convergence math
        AnswerResult(
            item_id="C2",
            route="clarify",
            reply="Could you clarify?",
            said_dont_know=False,
            cited_expected_doc=False,
            correct=False,
            duration_ms=50,
            error=None,
            steps=1,
            optimal_steps=1,
        ),
    ]

    report = build_report(items, [], answers)
    m = report["metrics"]

    assert m["convergence_sample_size"] == 2  # only the two "answer" rows
    assert m["avg_steps"] == 3.0  # mean(2, 4)
    assert m["avg_optimal_steps"] == 2.0
    assert m["convergence_score"] == round((2 / 2 + 2 / 4) / 2, 3)  # mean of per-item ratios
