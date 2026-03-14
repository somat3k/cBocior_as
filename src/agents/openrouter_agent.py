"""
src/agents/openrouter_agent.py — OpenRouter consensus arbiter agent.

Role: Receive outputs from all other agents and emit final consensus decision.
Uses Claude 3.5 Sonnet by default via OpenRouter API.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from constants import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL
from src.agents.base_agent import BaseAgent
from src.agents.prompts import build_openrouter_prompts
from src.utils.logger import get_logger
from src.utils.payload import TradingPayload

logger = get_logger(__name__)


class OpenRouterAgent(BaseAgent):
    """
    Claude 3.5 Sonnet (via OpenRouter) consensus arbitration agent.

    Receives aggregated outputs from all other agents and produces the
    final signed-off trading payload.
    """

    agent_id: str = "openrouter"

    def __init__(self) -> None:
        super().__init__()
        # OpenRouter uses an OpenAI-compatible API
        self._client = AsyncOpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )

    async def _call(self, payload: TradingPayload) -> TradingPayload:
        market_data = self._format_payload_for_prompt(payload)

        system_prompt, user_prompt = build_openrouter_prompts(market_data)

        logger.debug("Calling OpenRouter", model=OPENROUTER_MODEL)
        response = await self._client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.05,
            max_tokens=512,
            extra_headers={
                "HTTP-Referer": "https://github.com/somat3k/cBocior_as",
                "X-Title": "cBocior_as",
            },
        )

        raw = response.choices[0].message.content or "{}"
        return self._parse_llm_response(raw, payload)
