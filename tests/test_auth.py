"""Sign-in: allowed emails, password accounts, and sessions."""

from __future__ import annotations

import uuid

import pytest

from t2_assistant import auth, store
from t2_assistant.config import settings


def _unique_email() -> str:
    """A fresh, never-before-seen email each call - `users` is keyed on
    email, and tests share the on-disk store.db, so a fixed literal would
    collide (IntegrityError) on a second test run."""
    return f"test-{uuid.uuid4().hex[:10]}@{settings.auth_email_domain}"


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


def test_disallowed_email_is_rejected() -> None:
    with pytest.raises(ValueError):
        auth.register_or_sign_in("someone@gmail.com", "whatever-password")


def test_new_email_registers_and_signs_in() -> None:
    email = _unique_email()
    result = auth.register_or_sign_in(email, "correct horse battery staple")
    assert result is not None
    token, created = result
    assert created is True
    assert store.session_email(token) == email


def test_existing_email_signs_in_with_the_right_password() -> None:
    email = _unique_email()
    first = auth.register_or_sign_in(email, "my-real-password")
    assert first is not None
    first_token, _ = first

    second = auth.register_or_sign_in(email, "my-real-password")
    assert second is not None
    second_token, created = second
    assert created is False  # already existed - not a fresh registration
    assert second_token != first_token  # each sign-in gets its own session
    assert store.session_email(second_token) == email


def test_wrong_password_is_rejected_and_the_real_one_still_works() -> None:
    email = _unique_email()
    auth.register_or_sign_in(email, "the-real-password")

    assert auth.register_or_sign_in(email, "a-guess") is None

    retry = auth.register_or_sign_in(email, "the-real-password")
    assert retry is not None


def test_password_is_hashed_and_salted() -> None:
    email_a, email_b = _unique_email(), _unique_email()
    auth.register_or_sign_in(email_a, "same-password")
    auth.register_or_sign_in(email_b, "same-password")

    hash_a = store.get_password_hash(email_a)
    hash_b = store.get_password_hash(email_b)
    assert hash_a is not None
    assert hash_b is not None
    assert hash_a.startswith("scrypt$")
    assert "same-password" not in hash_a
    assert hash_a != hash_b  # different salts, even for the same password


def test_logout_ends_the_session() -> None:
    email = f"person4@{settings.auth_email_domain}"
    token = store.create_session(email)
    assert store.session_email(token) == email
    store.delete_session(token)
    assert store.session_email(token) is None
