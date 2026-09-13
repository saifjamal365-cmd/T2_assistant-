"""Turn the two evaluation passes into numbers, a report, and an MLflow run."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow

from t2_assistant.config import PROJECT_ROOT, settings
from t2_assistant.evaluation.dataset import EvalItem
from t2_assistant.evaluation.quality import AnswerResult
from t2_assistant.evaluation.retrieval import RetrievalResult

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


def _pct(numerator: int, denominator: int) -> float:
    return round(100 * numerator / denominator, 1) if denominator else 0.0


def build_report(
    items: list[EvalItem],
    retrieval: list[RetrievalResult],
    answers: list[AnswerResult],
) -> dict[str, Any]:
    by_id = {item.id: item for item in items}
    answerable_items = [item for item in items if item.answerable]
    unanswerable_items = [item for item in items if not item.answerable]

    retrieval_hits = sum(1 for r in retrieval if r.hit)
    answer_by_id = {a.item_id: a for a in answers}

    answerable_correct = sum(
        1 for item in answerable_items if (a := answer_by_id.get(item.id)) and a.correct
    )
    unanswerable_correct = sum(
        1 for item in unanswerable_items if (a := answer_by_id.get(item.id)) and a.correct
    )
    # hallucinated instead of declining
    false_answers = len(unanswerable_items) - unanswerable_correct
    errors = sum(1 for a in answers if a.error)

    durations = [a.duration_ms for a in answers if a.error is None]

    # breakdown by family (exact-wording vs paraphrased vs unanswerable)
    by_family: dict[str, dict[str, int]] = {}
    for item in items:
        family = by_family.setdefault(item.family, {"total": 0, "correct": 0})
        family["total"] += 1
        result = answer_by_id.get(item.id)
        if result and result.correct:
            family["correct"] += 1

    # breakdown by department (answerable items only)
    by_department: dict[str, dict[str, int]] = {}
    for item in answerable_items:
        dept = item.department or "unknown"
        row = by_department.setdefault(dept, {"total": 0, "correct": 0})
        row["total"] += 1
        result = answer_by_id.get(item.id)
        if result and result.correct:
            row["correct"] += 1

    # breakdown by language
    by_language: dict[str, dict[str, int]] = {}
    for item in items:
        row = by_language.setdefault(item.language, {"total": 0, "correct": 0})
        row["total"] += 1
        result = answer_by_id.get(item.id)
        if result and result.correct:
            row["correct"] += 1

    route_counts = Counter(a.route for a in answers)

    metrics = {
        "dataset_size": len(items),
        "answerable_count": len(answerable_items),
        "unanswerable_count": len(unanswerable_items),
        "retrieval_recall_at_k": _pct(retrieval_hits, len(retrieval)),
        "answer_accuracy_answerable": _pct(answerable_correct, len(answerable_items)),
        "honesty_rate_unanswerable": _pct(unanswerable_correct, len(unanswerable_items)),
        "false_answer_rate_unanswerable": _pct(false_answers, len(unanswerable_items)),
        "overall_accuracy": _pct(answerable_correct + unanswerable_correct, len(items)),
        "error_rate": _pct(errors, len(answers)),
        "avg_duration_ms": round(statistics.mean(durations)) if durations else None,
        "p95_duration_ms": (
            round(statistics.quantiles(durations, n=20)[18]) if len(durations) >= 20 else None
        ),
    }

    return {
        "metrics": metrics,
        "by_family": by_family,
        "by_department": by_department,
        "by_language": by_language,
        "route_counts": dict(route_counts),
        "_by_id": by_id,  # not written to the report file, used by callers
    }


def write_report(report: dict[str, Any], answers: list[AnswerResult]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / "answers.jsonl").write_text(
        "\n".join(json.dumps(asdict(a), ensure_ascii=False) for a in answers), encoding="utf-8"
    )

    to_save = {k: v for k, v in report.items() if not k.startswith("_")}
    (RESULTS_DIR / "report.json").write_text(
        json.dumps(to_save, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def counts_table(heading: str, header: str, rows: dict[str, dict[str, int]]) -> list[str]:
        out = [
            "",
            f"## {heading}",
            "",
            f"| {header} | Total | Correct | Accuracy |",
            "|---|---|---|---|",
        ]
        for name, row in rows.items():
            accuracy = _pct(row["correct"], row["total"])
            out.append(f"| {name} | {row['total']} | {row['correct']} | {accuracy}% |")
        return out

    m = report["metrics"]
    dont_know_label = 'Honesty rate (says "I don\'t know" when it should)'
    lines = [
        "# T2 Assistant - Evaluation Report",
        "",
        f"Dataset: {m['dataset_size']} questions ({m['answerable_count']} answerable, "
        f"{m['unanswerable_count']} deliberately unanswerable).",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value | Requirement |",
        "|---|---|---|",
        f"| Retrieval recall (top-5) | {m['retrieval_recall_at_k']}% | NFR-02 |",
        f"| Answer accuracy (answerable) | {m['answer_accuracy_answerable']}% | TO-7 |",
        f"| {dont_know_label} | {m['honesty_rate_unanswerable']}% | NFR-03 |",
        f"| False-answer rate (should decline, didn't) | {m['false_answer_rate_unanswerable']}%"
        " | NFR-03 |",
        f"| Overall accuracy | {m['overall_accuracy']}% | TO-7 |",
        f"| Error rate | {m['error_rate']}% | NFR-07 |",
        f"| Average answer time | {m['avg_duration_ms']} ms | NFR-01 |",
        f"| 95th percentile answer time | {m['p95_duration_ms']} ms | NFR-01 |",
    ]
    by_department = dict(sorted(report["by_department"].items()))
    by_language = dict(sorted(report["by_language"].items()))
    lines += counts_table("By question type", "Family", report["by_family"])
    lines += counts_table("By department (answerable questions)", "Department", by_department)
    lines += counts_table("By language (NFR-05, bilingual quality)", "Language", by_language)

    lines += ["", "## Routes chosen", "", "| Route | Count |", "|---|---|"]
    for route, count in sorted(report["route_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {route} | {count} |")

    report_path = RESULTS_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def log_to_mlflow(report: dict[str, Any]) -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_name="evaluation"):
        for key, value in report["metrics"].items():
            if isinstance(value, int | float):
                mlflow.log_metric(key, value)
        mlflow.log_artifact(str(RESULTS_DIR / "report.md"))
        mlflow.log_artifact(str(RESULTS_DIR / "report.json"))
