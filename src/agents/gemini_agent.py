"""
src/agents/gemini_agent.py — Google Gemini agent.

Role: Multi-timeframe pattern synthesis and divergence detection.
"""

from __future__ import annotations

import asyncio

from google import genai
from google.genai import types

from constants import GEMINI_API_KEY, GEMINI_MODEL
from src.agents.base_agent import BaseAgent
from src.utils.logger import get_logger
from src.utils.payload import TradingPayload

logger = get_logger(__name__)


class GeminiAgent(BaseAgent):
    """Gemini powered multi-timeframe analysis agent."""

    agent_id: str = "gemini"

    def __init__(self) -> None:
        super().__init__()
        self._client = genai.Client(api_key=GEMINI_API_KEY)

    async def _call(self, payload: TradingPayload) -> TradingPayload:
        market_data = self._format_payload_for_prompt(payload)

        prompt = (
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
            f"Market data:\n{market_data}"
        )

        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
            response_mime_type="application/json",
        )

        logger.debug("Calling Gemini", model=GEMINI_MODEL)
        # google-genai is synchronous; run in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config,
            ),
        )

        raw = response.text or "{}"
        return self._parse_llm_response(raw, payload)
