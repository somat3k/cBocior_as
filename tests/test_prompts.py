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

    def test_gemini_prompt_includes_market_data(self) -> None:
        prompt = build_gemini_prompt("gemini-market")
        assert "gemini-market" in prompt
        assert "timeframe" in prompt.lower()

    def test_groq_prompts_include_snapshot(self) -> None:
        system, user = build_groq_prompts("snapshot-json")
        assert "BUY|SELL|HOLD" in system
        assert "snapshot-json" in user

    def test_openrouter_prompts_include_market_data(self) -> None:
        system, user = build_openrouter_prompts("router-market")
        assert "consensus" in system.lower()
        assert "router-market" in user
