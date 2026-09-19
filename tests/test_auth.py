"""Sign-in: allowed emails, one-time codes, and sessions."""

from __future__ import annotations

import pytest

from t2_assistant import auth, store
from t2_assistant.config import settings


@pytest.fixture(autouse=True)
def _no_real_email(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this file must be able to run with no network access -
    replace the real Resend call with a no-op."""
    monkeypatch.setattr(auth, "_send_email", lambda to, code: None)


def test_domain_email_is_allowed() -> None:
    assert auth.is_allowed_email(f"someone@{settings.auth_email_domain}")
    upper_domain = settings.auth_email_domain.upper()
    assert auth.is_allowed_email(f"Someone@{upper_domain}")  # case-insensitive


def test_unrelated_email_is_rejected() -> None:
    assert not auth.is_allowed_email("someone@gmail.com")
    assert not auth.is_allowed_email("not-an-email")


def test_listed_test_email_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "auth_allowed_test_emails", ["test.user@gmail.com"])
    assert auth.is_allowed_email("Test.User@gmail.com")  # case-insensitive
    assert not auth.is_allowed_email("someone-else@gmail.com")


def test_request_code_rejects_a_disallowed_email() -> None:
    with pytest.raises(ValueError):
        auth.request_code("someone@gmail.com")


def test_full_round_trip_with_the_real_generated_code(monkeypatch: pytest.MonkeyPatch) -> None:
    email = f"person2@{settings.auth_email_domain}"
    seen: dict[str, str] = {}
    monkeypatch.setattr(auth, "_send_email", lambda to, code: seen.__setitem__("code", code))

    auth.request_code(email)
    real_code = seen["code"]

    assert auth.verify_code(email, "000000") is None  # wrong code
    token = auth.verify_code(email, real_code)
    assert token is not None
    assert store.session_email(token) == email

    # the same code cannot be used again
    assert auth.verify_code(email, real_code) is None


def test_too_many_wrong_attempts_locks_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    email = f"person3@{settings.auth_email_domain}"
    seen: dict[str, str] = {}
    monkeypatch.setattr(auth, "_send_email", lambda to, code: seen.__setitem__("code", code))
    auth.request_code(email)

    for _ in range(settings.otp_max_attempts):
        assert auth.verify_code(email, "000000") is None
    # even the real code is now refused - the attempt budget is spent
    assert auth.verify_code(email, seen["code"]) is None


def test_logout_ends_the_session() -> None:
    email = f"person4@{settings.auth_email_domain}"
    token = store.create_session(email)
    assert store.session_email(token) == email
    store.delete_session(token)
    assert store.session_email(token) is None
