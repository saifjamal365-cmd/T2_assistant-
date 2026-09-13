"""Tracing / observability.

A "trace" is a step-by-step recording of one request: which agent ran, what it
sent to the LLM, what came back, and how long each part took.

We do not write any of that by hand. MLflow's `langchain.autolog()` watches
LangGraph and the Groq client and records every step automatically. This module
just switches it on, once, at startup.

After running the app, look at the traces with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db

run from the project folder, then open http://localhost:5000 and click "Traces".
"""

from __future__ import annotations

import mlflow

from t2_assistant.config import settings

_initialised = False


def init_tracing() -> None:
    """Point MLflow at the local store and enable automatic tracing.

    Safe to call more than once; it only does the work the first time.
    """
    global _initialised
    if _initialised:
        return

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    # One line: trace every LangGraph node and every Groq chat call.
    mlflow.langchain.autolog()

    _initialised = True
