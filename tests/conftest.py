"""Shared test setup.

- Make sure the app can build its settings even on a machine with no .env file
  (a CI runner): if GROQ_API_KEY is not set, use a dummy value. Tests that would
  really call Groq are marked `integration` and skipped by default.
- Point the SQLite store at a throwaway file so tests never touch the real
  store.db.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_env_file = Path(__file__).resolve().parents[1] / ".env"
if not _env_file.exists() and "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = "test-key-not-used"

from t2_assistant.config import settings  # noqa: E402  (must follow the env setup)

settings.store_db = Path(tempfile.mkdtemp(prefix="t2-test-")) / "store.db"
