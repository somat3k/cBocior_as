"""
src/agents/groq_agent.py — Groq OSS 120B agent (single decision path).

Role: Primary signal generation (configured via GROQ_MODEL).
Target response time: < 10 seconds.
"""

from __future__ import annotations

from groq import AsyncGroq, GroqError

from constants import GROQ_API_KEY, GROQ_MODEL
from src.agents.base_agent import BaseAgent
from src.agents.prompts import build_groq_prompts
from src.utils.logger import get_logger
from src.utils.payload import TradingPayload

logger = get_logger(__name__)

# Groq agent has a shorter timeout (speed-focused)
GROQ_TIMEOUT: int = 10


class GroqAgent(BaseAgent):
    """OSS 120B powered primary signal agent (via GROQ_MODEL)."""

    agent_id: str = "groq"

    def __init__(self) -> None:
        super().__init__(timeout=GROQ_TIMEOUT)
        self._client = AsyncGroq(api_key=GROQ_API_KEY)
        self._models = self._parse_model_list(GROQ_MODEL)

    @staticmethod
    def _parse_model_list(raw: str) -> list[str]:
        return [model.strip() for model in raw.split(",") if model.strip()]

    async def _call(self, payload: TradingPayload) -> TradingPayload:
        # Compact prompt to minimise tokens (< 500 tokens total)
        last_indicators = payload.indicators[-1] if payload.indicators else None
        last_signals = payload.model_signals[-1] if payload.model_signals else None

        quick_data = {
            "symbol": payload.symbol,
            "latest_indicator": last_indicators.model_dump() if last_indicators else {},
            "latest_model_signal": last_signals.model_dump() if last_signals else {},
        }

        try:
            import ujson as json
        except ImportError:
            import json  # type: ignore[no-redef]

        compact = json.dumps(quick_data)

        system_prompt, user_prompt = build_groq_prompts(compact)

        if not self._models:
            raise RuntimeError("No Groq models configured")

        fallback_error = RuntimeError(
            "All Groq model attempts failed (tried: "
            f"{', '.join(self._models)}), but no error was captured"
        )
        last_error: Exception = fallback_error
        for model in self._models:
            try:
                logger.debug("Calling Groq", model=model)
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=128,
                )
            except GroqError as exc:
                last_error = exc
                logger.warning("Groq model call failed", model=model, error=str(exc))
                continue

            raw = response.choices[0].message.content or "{}"
            result = self._parse_llm_response(raw, payload)

            # Groq confidence capped at 0.6 (speed-only mode)
            return result.model_copy(
                update={"confidence": min(result.confidence, 0.6)}
            )

        raise last_error
