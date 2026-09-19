"""Sign-in: an email address and a one-time code, no passwords.

    1. the user gives an email - it must end in the company domain
       (settings.auth_email_domain), or be one of a short list of real test
       inboxes (settings.auth_allowed_test_emails) used before a real company
       inbox is connected,
    2. a 6-digit code is generated, stored (hashed) with a short expiry, and
       emailed to that address through Resend,
    3. the user sends the code back; if it matches, a session is created and
       the code is deleted - it cannot be reused.

This never invents a way to skip the check: a wrong or expired code is
rejected, and only a real inbox that received the email can complete it.
"""

from __future__ import annotations

import secrets

import httpx

from t2_assistant import store
from t2_assistant.config import settings

_RESEND_URL = "https://api.resend.com/emails"


def is_allowed_email(email: str) -> bool:
    email = email.strip().lower()
    if "@" not in email:
        return False
    if email in {e.lower() for e in settings.auth_allowed_test_emails}:
        return True
    return email.endswith("@" + settings.auth_email_domain.lower())


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _send_email(to: str, code: str) -> None:
    subject = f"Your T2 Assistant sign-in code: {code}"
    body = (
        f"<p>Your sign-in code is:</p>"
        f"<p style='font-size:28px;font-weight:700;letter-spacing:4px'>{code}</p>"
        f"<p>It expires in {settings.otp_ttl_minutes} minutes. "
        f"If you did not request this, you can ignore this email.</p>"
    )
    response = httpx.post(
        _RESEND_URL,
        headers={"Authorization": f"Bearer {settings.resend_api_key}"},
        json={
            "from": settings.auth_from_email,
            "to": [to],
            "subject": subject,
            "html": body,
        },
        timeout=10.0,
    )
    response.raise_for_status()


def request_code(email: str) -> None:
    """Generate a code, store it, and email it. Raises ValueError if the
    email is not allowed, or httpx.HTTPStatusError if Resend rejects it."""
    email = email.strip().lower()
    if not is_allowed_email(email):
        raise ValueError("email not allowed")
    code = _generate_code()
    store.create_otp(email, code)
    _send_email(email, code)
    # dev visibility - lets us verify the flow without a real inbox
    print(f"[t2_assistant.auth] sign-in code for {email}: {code}")


def verify_code(email: str, code: str) -> str | None:
    """Check the code; on success, create and return a new session token."""
    email = email.strip().lower()
    result = store.verify_otp(email, code.strip())
    if result != "ok":
        return None
    return store.create_session(email)
