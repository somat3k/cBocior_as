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
from src.agents.prompts import build_gemini_prompt
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

        prompt = build_gemini_prompt(market_data)

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
