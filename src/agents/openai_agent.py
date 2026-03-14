"""
src/agents/openai_agent.py — OpenAI GPT-4o agent.

Role: Trade justification + final narrative reasoning.
Traced via LangSmith when LANGCHAIN_TRACING_V2=true.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from constants import OPENAI_API_KEY, OPENAI_MODEL
from src.agents.base_agent import BaseAgent
from src.agents.prompts import build_openai_prompts
from src.utils.logger import get_logger
from src.utils.payload import TradingPayload

logger = get_logger(__name__)


class OpenAIAgent(BaseAgent):
    """GPT-4o powered trade analysis agent."""

    agent_id: str = "openai"

    def __init__(self) -> None:
        super().__init__()
        self._client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def _call(self, payload: TradingPayload) -> TradingPayload:
        market_data = self._format_payload_for_prompt(payload)

        system_prompt, user_prompt = build_openai_prompts(market_data)

        logger.debug("Calling OpenAI", model=OPENAI_MODEL)
        response = await self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or "{}"
        return self._parse_llm_response(raw, payload)
