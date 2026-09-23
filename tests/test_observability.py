"""Trace links and feedback."""

from __future__ import annotations

import uuid

from t2_assistant import observability, store


def test_trace_url_is_none_without_a_trace() -> None:
    assert observability.trace_url(None) is None


def test_trace_url_points_at_the_mlflow_ui() -> None:
    url = observability.trace_url("tr-abc123")
    assert url is not None
    assert url.startswith("http")
    assert "tr-abc123" in url


def test_record_feedback_updates_the_store() -> None:
    email = f"obs-test-{uuid.uuid4().hex[:10]}@t2.sa"
    cid = store.create_conversation("feedback test", email)
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

    assert observability.record_feedback(
        run_id, helpful=False, comment="not clear", user_email=email
    )
    assert not observability.record_feedback(
        "no-such-run", helpful=True, comment=None, user_email=email
    )

    run = store.get_run(run_id, email)
    assert run is not None
    assert run.feedback == "not_helpful"
    assert run.feedback_comment == "not clear"


def test_record_feedback_rejects_a_non_owner() -> None:
    email_a = f"obs-test-{uuid.uuid4().hex[:10]}@t2.sa"
    email_b = f"obs-test-{uuid.uuid4().hex[:10]}@t2.sa"
    cid = store.create_conversation("feedback test", email_a)
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

    assert not observability.record_feedback(run_id, helpful=True, comment=None, user_email=email_b)
    run = store.get_run(run_id, email_a)
    assert run is not None
    assert run.feedback is None  # the rejected attempt left nothing behind
