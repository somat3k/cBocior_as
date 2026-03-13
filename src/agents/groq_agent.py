"""
src/agents/groq_agent.py — Groq LLaMA-3 70B agent (high-speed path).

Role: Rapid preliminary signal scoring.
Target response time: < 10 seconds.
"""

from __future__ import annotations

from groq import AsyncGroq

from constants import GROQ_API_KEY, GROQ_MODEL
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.utils.payload import TradingPayload

logger = get_logger(__name__)

# Groq agent has a shorter timeout (speed-focused)
GROQ_TIMEOUT: int = 10


class GroqAgent(BaseAgent):
    """LLaMA-3 70B powered speed-optimised preliminary signal agent."""

    agent_id: str = "groq"

    def __init__(self) -> None:
        super().__init__(timeout=GROQ_TIMEOUT)
        self._client = AsyncGroq(api_key=GROQ_API_KEY)

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

        compact = json.dumps(quick_data, indent=None)

        system_prompt = (
            "FX trading signal generator. Respond with ONLY a JSON object: "
            '{"action":"BUY|SELL|HOLD","confidence":0.0-1.0,'
            '"reasoning":"<30 words"}. Be decisive and fast.'
        )

        logger.debug("Calling Groq", model=GROQ_MODEL)
        response = await self._client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Market snapshot: {compact}"},
            ],
            temperature=0.0,
            max_tokens=128,
        )

        raw = response.choices[0].message.content or "{}"
        result = self._parse_llm_response(raw, payload)

        # Groq confidence capped at 0.6 (speed-only mode)
        return result.model_copy(
            update={"confidence": min(result.confidence, 0.6)}
        )
