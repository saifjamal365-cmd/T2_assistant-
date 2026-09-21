"""Trace links and feedback."""

from __future__ import annotations

from t2_assistant import observability, store


def test_trace_url_is_none_without_a_trace() -> None:
    assert observability.trace_url(None) is None


def test_trace_url_points_at_the_mlflow_ui() -> None:
    url = observability.trace_url("tr-abc123")
    assert url is not None
    assert url.startswith("http")
    assert "tr-abc123" in url


def test_record_feedback_updates_the_store() -> None:
    cid = store.create_conversation("feedback test")
    run_id = store.save_run(
        conversation_id=cid,
        user_message="q",
        route="answer",
        route_reason="r",
        reply="a",
        model="openai/gpt-oss-120b",
        trace_id=None,
        duration_ms=1,
    )

    assert observability.record_feedback(run_id, helpful=False, comment="not clear")
    assert not observability.record_feedback("no-such-run", helpful=True, comment=None)

    run = store.get_run(run_id)
    assert run is not None
    assert run.feedback == "not_helpful"
    assert run.feedback_comment == "not clear"
