"""Looking at what happened.

Every request is already traced in MLflow (tracing.py). This module ties the
saved run records to those traces:

  - `trace_url()` builds a link that opens a run's trace in the MLflow UI.
  - `record_feedback()` stores a viewer's thumbs-up / thumbs-down on a run and
    also attaches it to the trace, so answer quality shows up in MLflow.
"""

from __future__ import annotations

from functools import lru_cache

import mlflow
from mlflow.entities import AssessmentSource

from t2_assistant import store
from t2_assistant.config import settings


@lru_cache
def _experiment_id() -> str | None:
    try:
        client = mlflow.MlflowClient(settings.mlflow_tracking_uri)
        experiment = client.get_experiment_by_name(settings.mlflow_experiment)
        return experiment.experiment_id if experiment else None
    except Exception:
        return None


def trace_url(trace_id: str | None) -> str | None:
    """A link that opens this trace in the MLflow UI, or None if there is no trace."""
    if not trace_id:
        return None
    experiment_id = _experiment_id()
    base = settings.mlflow_ui_url.rstrip("/")
    if experiment_id is None:
        return f"{base}/#/traces/{trace_id}"
    return f"{base}/#/experiments/{experiment_id}/traces?selectedEvaluationId={trace_id}"


def record_feedback(run_id: str, helpful: bool, comment: str | None) -> bool:
    """Save a rating for a run and attach it to its trace. False if the run is unknown."""
    run = store.get_run(run_id)
    if run is None:
        return False

    store.set_feedback(run_id, "helpful" if helpful else "not_helpful", comment)

    if run.trace_id:
        try:
            mlflow.log_feedback(
                trace_id=run.trace_id,
                name="helpful",
                value=helpful,
                rationale=comment or None,
                source=AssessmentSource(source_type="HUMAN", source_id="web-ui"),
            )
        except Exception:
            pass  # the rating is still saved in our store

    return True
