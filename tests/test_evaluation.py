"""The evaluation package: dataset loading, scoring, and the report."""

from __future__ import annotations

from t2_assistant.evaluation.dataset import EvalItem, load_dataset
from t2_assistant.evaluation.quality import AnswerResult
from t2_assistant.evaluation.report import build_report
from t2_assistant.evaluation.retrieval import RetrievalResult


def test_dataset_has_500_balanced_questions() -> None:
    items = load_dataset()

    assert len(items) == 500
    assert len({i.id for i in items}) == 500  # every id is unique

    answerable = [i for i in items if i.answerable]
    assert len(answerable) == 384
    assert all(i.expected_doc for i in answerable)  # every answerable item is gradeable
    assert all(i.expected_doc in i.expected_docs for i in answerable)

    unanswerable = [i for i in items if not i.answerable]
    assert len(unanswerable) == 116
    assert all(i.expected_doc is None and not i.expected_docs for i in unanswerable)

    assert sum(1 for i in items if i.language == "en") == 250
    assert sum(1 for i in items if i.language == "ar") == 250


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
