"""
tests/test_prompts.py — Unit tests for agent prompt templates.
"""

from __future__ import annotations

from src.agents.prompts import (
    build_gemini_prompt,
    build_groq_prompts,
    build_openai_prompts,
    build_openrouter_prompts,
)


class TestPromptTemplates:
    def test_openai_prompts_include_market_data(self) -> None:
        system, user = build_openai_prompts("market-data")
        assert "BUY|SELL|HOLD" in system
        assert "market-data" in user
        assert "$market_data" not in user

    def test_gemini_prompt_includes_market_data(self) -> None:
        prompt = build_gemini_prompt("gemini-market")
        assert "gemini-market" in prompt
        assert "multi-timeframe" in prompt.lower()
        assert "1M, 5M, and 1H" in prompt
        assert "timeframe_divergence" in prompt

    def test_groq_prompts_include_snapshot(self) -> None:
        system, user = build_groq_prompts("snapshot-json")
        assert "BUY|SELL|HOLD" in system
        assert "decisive and fast" in system.lower()
        assert "<30 words" in system
        assert "snapshot-json" in user

    def test_openrouter_prompts_include_market_data(self) -> None:
        system, user = build_openrouter_prompts("router-market")
        assert "consensus" in system.lower()
        assert "default to hold" in system.lower()
        assert "most conservative" in system.lower()
        assert "router-market" in user
