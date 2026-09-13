"""One place that builds the connection to the Groq LLM.

Every agent calls `get_llm()` instead of creating its own client, so the model
name, temperature and API key are configured once (in config.py / .env).
"""

from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from t2_assistant.config import settings


@lru_cache
def get_llm() -> ChatGroq:
    """Return a shared ChatGroq client (created once, then reused)."""
    return ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,  # type: ignore[arg-type]  # str is accepted
    )
