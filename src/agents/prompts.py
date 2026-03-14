"""
src/agents/prompts.py — Prompt templates + LangSmith prompt hub integration.

Provides the default Groq prompt template and optional overrides via the
LangSmith prompt hub. Set LANGSMITH_PROMPT_GROQ_* environment variables to
the prompt IDs if you want to pull templates from LangSmith.
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
    "groq_system": "LANGSMITH_PROMPT_GROQ_SYSTEM",
    "groq_user": "LANGSMITH_PROMPT_GROQ_USER",
}
LANGSMITH_PROMPT_ENV_VARS: tuple[str, ...] = tuple(_PROMPT_ENV_KEYS.values())
_MAX_TEMPLATE_SNIPPET_LENGTH = 160

# ---------------------------------------------------------------------------
# Default templates (fallbacks)
# ---------------------------------------------------------------------------

_GROQ_SYSTEM = (
    "FX trading signal generator. Respond with ONLY a JSON object: "
    '{"action":"BUY|SELL|HOLD","confidence":0.0-1.0,'
    '"reasoning":"<30 words"}. Be decisive and fast.'
)

_GROQ_USER = "Market snapshot: $snapshot"



# ---------------------------------------------------------------------------
# LangSmith prompt hub helper
# ---------------------------------------------------------------------------

class PromptHub:
    """Lazy wrapper around LangSmith prompt hub."""

    def __init__(self) -> None:
        self._client: Any | None = None
        client_cls: type | None
        try:
            from langsmith import Client
        except ImportError:
            logger.debug("LangSmith client unavailable, using default prompts")
            client_cls = None
        else:
            client_cls = Client

        if client_cls is not None:
            try:
                self._client = client_cls()
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
        if signature:
            if _has_required_params(signature):
                logger.debug(
                    "Prompt formatter requires arguments; skipping",
                    params=list(signature.parameters),
                )
                return str(prompt_obj)
        try:
            formatted = formatter()
            if isinstance(formatted, str):
                return formatted
        except (TypeError, ValueError):
            logger.debug("Prompt formatter failed without args")
    return str(prompt_obj)


def _has_required_params(signature: inspect.Signature) -> bool:
    return any(
        param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        for param in signature.parameters.values()
    )


def _prompt_id(key: str) -> str | None:
    env_key = _PROMPT_ENV_KEYS.get(key)
    if not env_key:
        return None
    value = os.getenv(env_key, "").strip()
    return value or None


@lru_cache(maxsize=None)
def _pull_langsmith_prompt(key: str, prompt_id: str | None) -> str | None:
    """Cache prompt lookups per (key, prompt_id) pair."""
    if not prompt_id:
        return None
    logger.debug("Pulling LangSmith prompt", key=key, prompt_id=prompt_id)
    try:
        template = _prompt_hub().pull(prompt_id)
    except (ConnectionError, RuntimeError, ValueError) as exc:
        logger.warning(
            "LangSmith prompt pull raised error",
            key=key,
            prompt_id=prompt_id,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return None
    if template is None:
        return None
    if not isinstance(template, str):
        logger.warning(
            "LangSmith prompt returned non-string",
            key=key,
            prompt_id=prompt_id,
            value_type=type(template).__name__,
        )
        return None
    return template


def _resolve_template(key: str, fallback: str) -> str:
    prompt_id = _prompt_id(key)
    if prompt_id:
        template = _pull_langsmith_prompt(key, prompt_id)
        if template:
            return template
        logger.warning(
            "LangSmith prompt fallback used",
            key=key,
            prompt_id=prompt_id,
        )
    return fallback


def _inject_context(template: str, context: dict[str, Any]) -> str:
    try:
        return Template(template).substitute(context)
    except KeyError as exc:
        missing = exc.args[0] if exc.args else "unknown"
        snippet = template[:_MAX_TEMPLATE_SNIPPET_LENGTH].replace("\n", "\\n")
        raise ValueError(
            f"Missing prompt template variable '{missing}' in template: "
            f"'{snippet}'"
        ) from exc
    except ValueError as exc:
        snippet = template[:_MAX_TEMPLATE_SNIPPET_LENGTH].replace("\n", "\\n")
        raise ValueError(
            "Invalid prompt template placeholder; ensure placeholders use "
            "$name syntax and escape literal '$' as '$$'. "
            f"Template: '{snippet}'"
        ) from exc


# ---------------------------------------------------------------------------
# Public prompt builder
# ---------------------------------------------------------------------------


def build_groq_prompts(snapshot: str) -> tuple[str, str]:
    system = _resolve_template("groq_system", _GROQ_SYSTEM)
    user = _resolve_template("groq_user", _GROQ_USER)
    return system, _inject_context(user, {"snapshot": snapshot})
