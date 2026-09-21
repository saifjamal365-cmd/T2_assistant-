"""One place that builds the connection to the Groq LLM.

Every agent calls `get_llm(model)` instead of creating its own client, so the
temperature and API key are configured once (in config.py / .env), and each
distinct model gets exactly one shared client.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, get_args

from langchain_groq import ChatGroq

from t2_assistant.config import settings

# The models offered to a user - not just listed on Groq's docs, but each run
# through the same real test battery in tests/test_models.py (an English and
# an Arabic factual question, a harder nuanced one, and one that should be
# declined) against this project's own API key and its actual answer-writing
# prompt. Several candidates that looked fine in isolation failed this: two
# Llama models are Enterprise-only and 404 on a normal key despite being
# listed on Groq's docs page; allam-2-7b cannot do the tool-calling the
# router needs and, on harder questions, degrades into repeating its own
# input; groq/compound-mini silently ignored an Arabic question and answered
# in English; openai/gpt-oss-safeguard-20b fabricated an entirely unrelated
# policy in Ukrainian. Used as the type of ChatIn.model in api.py - the one
# place an unvalidated model string enters the system - so an unknown id is
# rejected with a 422 before it reaches Groq.
ModelId = Literal[
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
AVAILABLE_MODELS: tuple[str, ...] = get_args(ModelId)


@lru_cache
def get_llm(model: str) -> ChatGroq:
    """Return a shared ChatGroq client for `model` - one client per distinct
    model string, created once and reused."""
    return ChatGroq(
        model=model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,  # type: ignore[arg-type]  # str is accepted
    )
