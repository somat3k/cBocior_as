"""
src/agents/base_agent.py — Abstract base class for all AI agents.

Provides:
- Retry logic with exponential back-off + jitter
- Per-call timeout enforcement
- LangSmith tracing
- Standardised payload input/output contract
"""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from typing import Any

from langsmith import traceable

from constants import BOT_MAX_CONCURRENT_AGENTS
from src.utils.logger import get_logger
from src.utils.payload import TradingPayload

logger = get_logger(__name__)

# Per-agent call timeout (seconds)
AGENT_TIMEOUT: int = 30
# Retry parameters
MAX_RETRIES: int = 5
BACKOFF_BASE: float = 2.0
BACKOFF_MAX: float = 60.0


class AgentError(Exception):
    """Raised when an agent call fails permanently."""


class BaseAgent(ABC):
    """
    Abstract base for all trading AI agents.

    Subclasses must implement ``_call(payload) -> TradingPayload``.
    """

    agent_id: str = "base"
    degraded: bool = False

    def __init__(self, timeout: int = AGENT_TIMEOUT) -> None:
        self.timeout = timeout
        self._consecutive_failures: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def analyse(self, payload: TradingPayload) -> TradingPayload:
        """
        Analyse a TradingPayload and return an enriched response.

        Retries on transient failures with exponential back-off.
        Marks the agent as degraded after MAX_RETRIES failures.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = await asyncio.wait_for(
                    self._call(payload), timeout=self.timeout
                )
                self._consecutive_failures = 0
                self.degraded = False
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "Agent timeout",
                    agent=self.agent_id,
                    attempt=attempt,
                    timeout=self.timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Agent error",
                    agent=self.agent_id,
                    attempt=attempt,
                    error=str(exc),
                )

            if attempt < MAX_RETRIES:
                wait = min(
                    BACKOFF_MAX,
                    BACKOFF_BASE ** attempt + random.uniform(0, 1),
                )
                logger.debug(
                    "Backing off before retry",
                    agent=self.agent_id,
                    wait_s=round(wait, 2),
                )
                await asyncio.sleep(wait)

        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_RETRIES:
            self.degraded = True
            logger.error(
                "Agent marked DEGRADED",
                agent=self.agent_id,
                failures=self._consecutive_failures,
            )
        raise AgentError(
            f"Agent '{self.agent_id}' failed after {MAX_RETRIES} attempts."
        )

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    async def _call(self, payload: TradingPayload) -> TradingPayload:
        """Perform the actual LLM / API call."""

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Return the system prompt for this agent role."""
        return (
            "You are an expert algorithmic trading analyst. "
            "Analyse the provided market data payload and return a structured "
            "JSON response with fields: action (BUY/SELL/HOLD), "
            "confidence (0.0-1.0), and reasoning (string)."
        )

    def _parse_llm_response(
        self,
        raw: str,
        base_payload: TradingPayload,
    ) -> TradingPayload:
        """
        Parse LLM JSON response into a TradingPayload.

        Falls back to HOLD if parsing fails.
        """
        import json as _json

        try:
            # Extract JSON block (handle markdown code fences)
            text = raw.strip()
            if "```" in text:
                text = text.split("```")[1].strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            data = _json.loads(text)
            action_str = str(data.get("action", "HOLD")).upper()
            confidence = float(data.get("confidence", 0.5))
            reasoning = str(data.get("reasoning", ""))

            from src.utils.payload import TradingAction
            try:
                action = TradingAction(action_str)
            except ValueError:
                action = TradingAction.HOLD

            return base_payload.model_copy(update={
                "source": self.agent_id,
                "action": action,
                "confidence": max(0.0, min(1.0, confidence)),
                "reasoning": reasoning,
            })
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse LLM response, defaulting to HOLD",
                agent=self.agent_id,
                error=str(exc),
                raw=raw[:200],
            )
            from src.utils.payload import TradingAction
            return base_payload.model_copy(update={
                "source": self.agent_id,
                "action": TradingAction.HOLD,
                "confidence": 0.0,
                "reasoning": f"Parse error: {exc}",
            })

    def _format_payload_for_prompt(self, payload: TradingPayload) -> str:
        """Serialize the payload to a compact JSON string for the prompt."""
        try:
            import ujson as json
        except ImportError:
            import json  # type: ignore[no-redef]
        return json.dumps(payload.model_dump(), indent=2)
