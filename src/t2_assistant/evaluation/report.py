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


def _convergence_pairs(answers: list[AnswerResult]) -> list[tuple[int, int]]:
    """(steps, optimal_steps) for runs that actually exercised the answer
    agent's retry/judge machinery. greeting/clarify/summarise always score
    1.0 by construction (no loops, no branches - see answer.py's respond())
    and would dilute the one signal this is meant to expose; an errored run
    has no step count at all."""
    pairs: list[tuple[int, int]] = []
    for a in answers:
        steps, optimal = a.steps, a.optimal_steps
        if a.error is None and a.route == "answer" and steps and optimal:
            pairs.append((steps, optimal))
    return pairs


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

    # breakdown by family, for convergence specifically - same answer-route-
    # only eligibility as _convergence_pairs, just grouped instead of flat
    family_pairs: dict[str, list[tuple[int, int]]] = {}
    for item in items:
        result = answer_by_id.get(item.id)
        if result is None or result.error is not None or result.route != "answer":
            continue
        steps, optimal = result.steps, result.optimal_steps
        if not steps or not optimal:
            continue
        family_pairs.setdefault(item.family, []).append((steps, optimal))
    convergence_by_family = {
        family: {
            "n": len(pairs),
            "avg_steps": round(statistics.mean(s for s, _ in pairs), 2),
            "avg_optimal_steps": round(statistics.mean(o for _, o in pairs), 2),
            "convergence_score": round(statistics.mean(o / s for s, o in pairs), 3),
        }
        for family, pairs in family_pairs.items()
    }

    route_counts = Counter(a.route for a in answers)
    convergence_pairs = _convergence_pairs(answers)

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
        "avg_steps": (
            round(statistics.mean(s for s, _ in convergence_pairs), 2)
            if convergence_pairs
            else None
        ),
        "avg_optimal_steps": (
            round(statistics.mean(o for _, o in convergence_pairs), 2)
            if convergence_pairs
            else None
        ),
        "convergence_score": (
            round(statistics.mean(o / s for s, o in convergence_pairs), 3)
            if convergence_pairs
            else None
        ),
        "convergence_sample_size": len(convergence_pairs),
    }

    return {
        "metrics": metrics,
        "by_family": by_family,
        "by_department": by_department,
        "by_language": by_language,
        "convergence_by_family": convergence_by_family,
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
        f"| Convergence score (answer route) | {m['convergence_score']} | — |",
    ]
    by_department = dict(sorted(report["by_department"].items()))
    by_language = dict(sorted(report["by_language"].items()))
    lines += counts_table("By question type", "Family", report["by_family"])
    lines += counts_table("By department (answerable questions)", "Department", by_department)
    lines += counts_table("By language (NFR-05, bilingual quality)", "Language", by_language)

    lines += [
        "",
        "## Convergence (answer-agent efficiency)",
        "",
        "How many LLM calls the answer agent actually needed versus the fewest "
        "it could ever need for a fresh question - the router's own fixed call "
        "is excluded, since it never varies. 1.0 means no wasted retries or "
        "judge passes; lower means the retry/judge machinery is triggering "
        "more than strictly necessary.",
        "",
        "| Avg steps | Avg optimal steps | Convergence score | Sample size |",
        "|---|---|---|---|",
        f"| {m['avg_steps']} | {m['avg_optimal_steps']} | {m['convergence_score']} "
        f"| {m['convergence_sample_size']} |",
        "",
        "| Family | N | Avg steps | Avg optimal | Convergence |",
        "|---|---|---|---|---|",
    ]
    for family, row in report["convergence_by_family"].items():
        lines.append(
            f"| {family} | {row['n']} | {row['avg_steps']} | {row['avg_optimal_steps']} "
            f"| {row['convergence_score']} |"
        )

    lines += ["", "## Routes chosen", "", "| Route | Count |", "|---|---|"]
    for route, count in sorted(report["route_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {route} | {count} |")

    report_path = RESULTS_DIR / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def log_to_mlflow(report: dict[str, Any], run_id: str | None = None) -> None:
    """Log the summary metrics and report files to MLflow.

    `run_id`, when given, logs into that already-open run instead of
    starting a new one - run.py passes the id of the run it opened around
    the answer pass itself, so each question's trace (see quality.py's
    evaluate_answers) and this run's summary numbers end up in the same
    place, and the run can be expanded to see the individual questions
    behind it. Left as None, this creates its own fresh run, exactly as
    before - unchanged for any other caller.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)
    with mlflow.start_run(run_id=run_id, run_name="evaluation" if run_id is None else None):
        for key, value in report["metrics"].items():
            if isinstance(value, int | float):
                mlflow.log_metric(key, value)
        mlflow.log_artifact(str(RESULTS_DIR / "report.md"))
        mlflow.log_artifact(str(RESULTS_DIR / "report.json"))
