"""Run the evaluation and print/report the results.

    python -m t2_assistant.evaluation.run                # the full 500 questions
    python -m t2_assistant.evaluation.run --limit 20      # a quick smoke run
    python -m t2_assistant.evaluation.run --workers 8     # more parallel answer calls
    python -m t2_assistant.evaluation.run --resume        # keep prior successes, retry failures

Writes eval/results/report.md (read this), report.json and answers.jsonl, and
logs the metrics as an MLflow run named "evaluation".

Groq's free tier has a daily token cap on top of the per-minute one, and it
cannot be waited out mid-run. --resume re-reads the previous answers.jsonl,
keeps every question that already succeeded, and only spends new tokens on
the ones that errored - so a run interrupted by the daily cap can be
completed over several days without re-asking questions it already got
right.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlflow

from t2_assistant.config import settings
from t2_assistant.evaluation.dataset import DATASET_PATH, load_dataset
from t2_assistant.evaluation.quality import AnswerResult, evaluate_answers, load_previous_results
from t2_assistant.evaluation.report import RESULTS_DIR, build_report, log_to_mlflow, write_report
from t2_assistant.evaluation.retrieval import evaluate_retrieval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--workers", type=int, default=2, help="parallel workers, answer pass")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-answers", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep previously successful answers, only retry ones that errored",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DATASET_PATH),
        help="path to an eval dataset json file (default: the full 500-question set)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    items = load_dataset(path=Path(args.dataset), limit=args.limit)
    print(f"Loaded {len(items)} evaluation questions.")

    retrieval_results = []
    if not args.skip_retrieval:
        print("\n--- Retrieval pass (embeddings only) ---")
        started = time.perf_counter()

        def _retrieval_progress(done: int, total: int) -> None:
            print(f"  {done}/{total}", end="\r", file=sys.stderr)

        retrieval_results = evaluate_retrieval(items, on_progress=_retrieval_progress)
        print(f"  done in {time.perf_counter() - started:.0f}s" + " " * 10)

    answer_results = []
    run_id: str | None = None
    if not args.skip_answers:
        to_run = items
        reused: list[AnswerResult] = []
        if args.resume:
            previous = load_previous_results(RESULTS_DIR / "answers.jsonl")
            item_ids = {item.id for item in items}
            reused = [r for r in previous.values() if not r.error and r.item_id in item_ids]
            reused_ids = {r.item_id for r in reused}
            to_run = [item for item in items if item.id not in reused_ids]
            print(f"Resuming: {len(reused)} already succeeded, {len(to_run)} left to (re)try.")

        print(f"\n--- Answer pass (full pipeline, {args.workers} workers) ---")
        started = time.perf_counter()

        def _answer_progress(_result: object, done: int, total: int) -> None:
            print(f"  {done}/{total}", end="\r", file=sys.stderr)

        # Opened here, not just around the later metric-logging step, so
        # each question's own trace (chat() is @mlflow.trace'd) attaches to
        # this run while it's actually being asked - not after the fact,
        # when there would be nothing left to attach it to. Reused (resumed)
        # results don't call chat() again, so they have no fresh trace to
        # attach either way - only newly-run questions show up under it.
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment)
        with mlflow.start_run(run_name="evaluation") as run:
            run_id = run.info.run_id
            new_results = evaluate_answers(to_run, workers=args.workers, on_result=_answer_progress)
        answer_results = reused + new_results
        print(f"  done in {time.perf_counter() - started:.0f}s" + " " * 10)

    if not retrieval_results and not answer_results:
        print("Nothing to report (both passes skipped).")
        return

    report = build_report(items, retrieval_results, answer_results)
    report_path = write_report(report, answer_results)

    print("\n--- Results ---")
    for key, value in report["metrics"].items():
        print(f"  {key:32s} {value}")
    print(f"\nFull report: {report_path}")

    if answer_results:
        log_to_mlflow(report, run_id=run_id)
        print("Metrics logged to MLflow (run: evaluation).")


if __name__ == "__main__":
    main()
