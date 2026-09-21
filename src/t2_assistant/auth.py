"""Sign-in: a company email and a password, checked against a stored hash.

    1. the user gives an email - it must end in the company domain
       (settings.auth_email_domain), or be one of a short list of real test
       inboxes (settings.auth_allowed_test_emails) used before every real
       company inbox exists,
    2. an email seen for the first time is registered on the spot with the
       password given; an email already on file must match its stored
       password,
    3. on success a session is created, the same as before.

Passwords are hashed with scrypt (Python's own standard library - no extra
dependency) and a random salt per account; the raw password is never stored.
This has no email-verification step, unlike the one-time-code flow it
replaces: `is_allowed_email` is the only check that someone claiming a given
address is really allowed to use it.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

from t2_assistant import store
from t2_assistant.config import settings

_SCRYPT_N = 2**14  # cost factor - hashlib's own recommended baseline
_SCRYPT_R = 8  # (a few tens of ms and ~16 MB per hash - fine for a login form)
_SCRYPT_P = 1
_SALT_BYTES = 16
_KEY_LEN = 32


def is_allowed_email(email: str) -> bool:
    email = email.strip().lower()
    if "@" not in email:
        return False
    if email in {e.lower() for e in settings.auth_allowed_test_emails}:
        return True
    return email.endswith("@" + settings.auth_email_domain.lower())


def _hash_password(password: str) -> str:
    """Hash with scrypt and a fresh random salt. Stored as a self-describing
    "scrypt$<salt_hex>$<hash_hex>" string, so the scheme could change later
    without breaking existing hashes."""
    salt = secrets.token_bytes(_SALT_BYTES)
    digest = hashlib.scrypt(
        password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=_KEY_LEN
    )
    return f"scrypt${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Check a password against a hash produced by `_hash_password`."""
    try:
        scheme, salt_hex, hash_hex = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(
        password.encode(),
        salt=bytes.fromhex(salt_hex),
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_KEY_LEN,
    )
    return hmac.compare_digest(digest.hex(), hash_hex)


def register_or_sign_in(email: str, password: str) -> tuple[str, bool] | None:
    """Single entry point for email+password sign-in.

    A new (allowed) email is registered with `password` on the spot; an
    existing email must match its stored password. Returns
    `(session_token, created)` on success - `created` is True only when this
    call just registered the account - or None if the account exists and the
    password was wrong. Raises ValueError if the email is not allowed to
    sign in here.
    """
    email = email.strip().lower()
    if not is_allowed_email(email):
        raise ValueError("email not allowed")

    existing_hash = store.get_password_hash(email)
    if existing_hash is None:
        store.create_user(email, _hash_password(password))
        created = True
    elif _verify_password(password, existing_hash):
        created = False
    else:
        return None

    return store.create_session(email), created
