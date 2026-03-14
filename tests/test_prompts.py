"""
tests/test_prompts.py — Unit tests for agent prompt templates.
"""

from __future__ import annotations

import pytest

from src.agents.prompts import (
    build_gemini_prompt,
    build_groq_prompts,
    build_openai_prompts,
    build_openrouter_prompts,
)

_LANGSMITH_PROMPT_ENV_VARS = (
    "LANGSMITH_PROMPT_OPENAI_SYSTEM",
    "LANGSMITH_PROMPT_OPENAI_USER",
    "LANGSMITH_PROMPT_GEMINI",
    "LANGSMITH_PROMPT_GROQ_SYSTEM",
    "LANGSMITH_PROMPT_GROQ_USER",
    "LANGSMITH_PROMPT_OPENROUTER_SYSTEM",
    "LANGSMITH_PROMPT_OPENROUTER_USER",
)


@pytest.fixture(autouse=True)
def _clear_langsmith_prompt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _LANGSMITH_PROMPT_ENV_VARS:
        monkeypatch.delenv(key, raising=False)


class TestPromptTemplates:
    def test_openai_prompts_contain_expected_content(self) -> None:
        system, user = build_openai_prompts("market-data")
        assert "BUY|SELL|HOLD" in system
        assert "market-data" in user
        assert "$market_data" not in user

    def test_gemini_prompt_contains_expected_content(self) -> None:
        prompt = build_gemini_prompt("gemini-market")
        assert "gemini-market" in prompt
        assert "$market_data" not in prompt
        assert "multi-timeframe" in prompt.lower()
        assert "1 m, 5 m, and 1 H" in prompt
        assert "Default to HOLD on divergence" in prompt
        assert "timeframe_divergence" in prompt

    def test_groq_prompts_contain_expected_content(self) -> None:
        system, user = build_groq_prompts("snapshot-json")
        assert "BUY|SELL|HOLD" in system
        assert "decisive and fast" in system.lower()
        assert "<30 words" in system
        assert "snapshot-json" in user

    def test_openrouter_prompts_contain_expected_content(self) -> None:
        system, user = build_openrouter_prompts("router-market")
        assert "consensus" in system.lower()
        assert "default to hold" in system.lower()
        assert "most conservative" in system.lower()
        assert "router-market" in user
