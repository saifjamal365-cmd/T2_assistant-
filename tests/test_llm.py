"""get_llm(): one shared client per model."""

from __future__ import annotations

from t2_assistant.agents.llm import get_llm


def test_get_llm_caches_per_model_not_globally() -> None:
    a = get_llm("openai/gpt-oss-120b")
    b = get_llm("openai/gpt-oss-120b")
    c = get_llm("qwen/qwen3.8-27b")

    assert a is b  # same model - the cached client is reused
    assert a is not c  # different model - a separate client
