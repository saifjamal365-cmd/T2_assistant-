"""Application settings.

Everything the app needs to know that changes between machines or must stay
secret lives in a `.env` file (never committed). This module loads that file
once into a typed `Settings` object, so the rest of the code just does:

    from t2_assistant.config import settings
    settings.groq_api_key

If a required value is missing, the app fails immediately at startup with a
clear message, instead of halfway through a request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# The project root = two levels up from this file (src/t2_assistant/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed view of the .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Groq) ---------------------------------------------------------
    groq_api_key: str = Field(..., description="Groq API key. Required.")
    llm_model: str = Field(
        "openai/gpt-oss-120b",
        description="Groq model id used for routing and the specialist agents.",
    )
    llm_temperature: float = Field(0.2, description="Lower = more consistent answers. 0-1.")

    # --- Tracing (MLflow) --------------------------------------------------
    # MLflow 3 stores traces in a database. We use a local SQLite file.
    # A relative URI (resolved against the current folder) is used on purpose:
    # the MLflow server (`mlflow ui`) mishandles Windows paths that contain
    # spaces, and this project's path does. So always run commands from the
    # project folder; both the app and `mlflow ui` then use the same file.
    mlflow_tracking_uri: str = Field(
        "sqlite:///mlflow.db",
        description="Where MLflow stores traces (run from the project folder).",
    )
    mlflow_experiment: str = Field("t2-assistant", description="Name the traces are grouped under.")
    mlflow_ui_url: str = Field(
        "http://localhost:5000",
        description="Where `mlflow ui` is served, for building links to a trace.",
    )

    # --- Paths -----------------------------------------------------------
    knowledge_base_dir: Path = Field(
        PROJECT_ROOT / "data" / "knowledge_base",
        description="Folder holding the policy documents.",
    )
    chroma_dir: Path = Field(
        PROJECT_ROOT / "chroma",
        description="Folder where the Chroma vector index is stored.",
    )
    store_db: Path = Field(
        PROJECT_ROOT / "store.db",
        description="SQLite file holding conversations, messages and run records.",
    )

    # --- Retrieval (Phase 4) -------------------------------------------
    # Multilingual E5 (Arabic + English), trained for search, fast enough on the
    # CPU. Swap for e5-small (faster) or e5-large (better) via .env - all use the
    # same "query:" / "passage:" prefixes. BGE-M3 is higher quality but needs a
    # GPU to index the corpus in reasonable time (and a code change - no prefix).
    embedding_model: str = Field(
        "intfloat/multilingual-e5-base",
        description="Sentence-transformers model that turns text into vectors (Arabic + English).",
    )
    chroma_collection: str = Field(
        "policies", description="Name of the collection inside the Chroma index."
    )
    chunk_size: int = Field(
        900, description="Passage length in characters when a document is split."
    )
    chunk_overlap: int = Field(150, description="Characters shared between neighbouring passages.")
    retrieval_k: int = Field(
        5, description="How many passages the answer agent retrieves per search."
    )

    # --- Agent behaviour -------------------------------------------------
    max_retrieval_tries: int = Field(
        2, description="How many times the answer agent may re-search before replying."
    )

    # --- Sign-in (email + password) ----------------------------------------
    auth_email_domain: str = Field(
        "t2.sa", description="Only emails ending in this domain may sign in."
    )
    auth_allowed_test_emails: list[str] = Field(
        default_factory=list,
        description="Exact email addresses allowed to sign in even outside auth_email_domain - "
        "for testing before a real company inbox is connected.",
    )
    session_ttl_days: int = Field(30, description="How long a signed-in session lasts.")


@lru_cache
def get_settings() -> Settings:
    """Load settings once and reuse them (cached)."""
    return Settings()  # type: ignore[call-arg]  # values come from the .env file


# Convenience: import this directly.
settings = get_settings()
