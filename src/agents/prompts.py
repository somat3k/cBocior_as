"""
src/agents/prompts.py — Prompt templates + LangSmith prompt hub integration.

Provides default prompt templates for each agent role and optional overrides
via LangSmith prompt hub. Set LANGSMITH_PROMPT_* environment variables to the
prompt IDs if you want to pull templates from LangSmith.
"""

from __future__ import annotations

import inspect
import os
from functools import lru_cache
from string import Template
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_ENV_KEYS: dict[str, str] = {
    "openai_system": "LANGSMITH_PROMPT_OPENAI_SYSTEM",
    "openai_user": "LANGSMITH_PROMPT_OPENAI_USER",
    "gemini": "LANGSMITH_PROMPT_GEMINI",
    "groq_system": "LANGSMITH_PROMPT_GROQ_SYSTEM",
    "groq_user": "LANGSMITH_PROMPT_GROQ_USER",
    "openrouter_system": "LANGSMITH_PROMPT_OPENROUTER_SYSTEM",
    "openrouter_user": "LANGSMITH_PROMPT_OPENROUTER_USER",
}

# ---------------------------------------------------------------------------
# Default templates (fallbacks)
# ---------------------------------------------------------------------------

_OPENAI_SYSTEM = (
    "You are an expert algorithmic FX trading analyst with deep "
    "knowledge of technical analysis and market microstructure. "
    "Your role is to provide the final trade justification.\n\n"
    "Given a market data JSON payload, output ONLY a valid JSON "
    "object with these exact fields:\n"
    '{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
    '"reasoning": "concise justification max 200 words"}\n\n'
    "Rules:\n"
    "- Cite at least one indicator and one model signal.\n"
    "- Be conservative: default to HOLD when uncertain.\n"
    "- Confidence < 0.55 must result in HOLD."
)

_OPENAI_USER = (
    "Analyse the following market data and produce a trading signal:\n\n"
    "$market_data"
)

_GEMINI_PROMPT = (
    "You are an expert algorithmic FX trading analyst specialising in "
    "multi-timeframe technical analysis.\n\n"
    "Analyse the indicators from 1M, 5M, and 1H timeframes in the "
    "provided market data payload. Identify pattern confluences and "
    "divergences across timeframes.\n\n"
    "Output ONLY a valid JSON object:\n"
    '{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
    '"reasoning": "timeframe analysis max 200 words"}\n\n'
    "Rules:\n"
    "- Flag timeframe_divergence in reasoning if signals differ across "
    "timeframes.\n"
    "- Default to HOLD on divergence.\n"
    "- Confidence range: 0.0 (no signal) to 1.0 (strong confluence).\n\n"
    "Market data:\n$market_data"
)

_GROQ_SYSTEM = (
    "FX trading signal generator. Respond with ONLY a JSON object: "
    '{"action":"BUY|SELL|HOLD","confidence":0.0-1.0,'
    '"reasoning":"<30 words"}. Be decisive and fast.'
)

_GROQ_USER = "Market snapshot: $snapshot"

_OPENROUTER_SYSTEM = (
    "You are the final consensus arbiter for a multi-agent algorithmic "
    "FX trading system. You receive a payload containing market data, "
    "indicator snapshots, model signals, and preliminary agent analyses "
    "from OpenAI, Gemini, and Groq.\n\n"
    "Your task is to evaluate all available information and produce a "
    "single authoritative trading decision.\n\n"
    "Output ONLY a valid JSON object:\n"
    '{"action": "BUY|SELL|HOLD", "confidence": 0.0-1.0, '
    '"reasoning": "consensus reasoning max 150 words"}\n\n'
    "Rules:\n"
    "- If agent signals conflict, default to HOLD.\n"
    "- Risk flags must reduce confidence proportionally.\n"
    "- Be the most conservative of all agents."
)

_OPENROUTER_USER = (
    "Produce the final consensus trading signal for this payload:\n\n"
    "$market_data"
)


# ---------------------------------------------------------------------------
# LangSmith prompt hub helper
# ---------------------------------------------------------------------------

class PromptHub:
    """Lazy wrapper around LangSmith prompt hub."""

    def __init__(self) -> None:
        self._client: Any | None = None
        try:
            from langsmith import Client
        except ImportError:
            logger.debug("LangSmith client unavailable, using default prompts")
            return
        try:
            self._client = Client()
        except (ValueError, RuntimeError, ConnectionError) as exc:
            logger.warning("LangSmith client init failed", error=str(exc))
            self._client = None

    def pull(self, prompt_id: str) -> str | None:
        if not prompt_id or self._client is None:
            return None
        puller = getattr(self._client, "pull_prompt", None)
        if not callable(puller):
            return None
        try:
            prompt_obj = puller(prompt_id)
        except (ValueError, RuntimeError, ConnectionError) as exc:
            logger.warning(
                "LangSmith prompt pull failed",
                prompt_id=prompt_id,
                error=str(exc),
            )
            return None
        return _coerce_prompt(prompt_obj)


@lru_cache(maxsize=1)
def _prompt_hub() -> PromptHub:
    return PromptHub()


def _coerce_prompt(prompt_obj: Any) -> str | None:
    if prompt_obj is None:
        return None
    if isinstance(prompt_obj, str):
        return prompt_obj
    template = getattr(prompt_obj, "template", None)
    if isinstance(template, str):
        return template
    formatter = getattr(prompt_obj, "format", None)
    if callable(formatter):
        try:
            signature = inspect.signature(formatter)
        except (TypeError, ValueError):
            signature = None
        if signature and signature.parameters:
            logger.debug(
                "Prompt formatter requires arguments; skipping",
                params=list(signature.parameters),
            )
        else:
            try:
                formatted = formatter()
                if isinstance(formatted, str):
                    return formatted
            except (TypeError, ValueError):
                logger.debug("Prompt formatter failed without args")
    return str(prompt_obj)


def _prompt_id(key: str) -> str | None:
    env_key = _PROMPT_ENV_KEYS.get(key)
    if not env_key:
        return None
    value = os.getenv(env_key, "").strip()
    return value or None


def _resolve_template(key: str, fallback: str) -> str:
    prompt_id = _prompt_id(key)
    if prompt_id:
        template = _prompt_hub().pull(prompt_id)
        if template:
            return template
        logger.warning(
            "LangSmith prompt fallback used",
            key=key,
            prompt_id=prompt_id,
        )
    return fallback


def _inject_context(template: str, context: dict[str, Any]) -> str:
    return Template(template).safe_substitute(context)


# ---------------------------------------------------------------------------
# Public prompt builders
# ---------------------------------------------------------------------------

def build_openai_prompts(market_data: str) -> tuple[str, str]:
    system = _resolve_template("openai_system", _OPENAI_SYSTEM)
    user = _resolve_template("openai_user", _OPENAI_USER)
    return system, _inject_context(user, {"market_data": market_data})


def build_gemini_prompt(market_data: str) -> str:
    template = _resolve_template("gemini", _GEMINI_PROMPT)
    return _inject_context(template, {"market_data": market_data})


def build_groq_prompts(snapshot: str) -> tuple[str, str]:
    system = _resolve_template("groq_system", _GROQ_SYSTEM)
    user = _resolve_template("groq_user", _GROQ_USER)
    return system, _inject_context(user, {"snapshot": snapshot})


def build_openrouter_prompts(market_data: str) -> tuple[str, str]:
    system = _resolve_template("openrouter_system", _OPENROUTER_SYSTEM)
    user = _resolve_template("openrouter_user", _OPENROUTER_USER)
    return system, _inject_context(user, {"market_data": market_data})
