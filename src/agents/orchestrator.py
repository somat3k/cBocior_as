"""
src/agents/orchestrator.py — Multi-agent orchestrator.

Implements the fan-out strategy described in AGENTS.md:

  1. Groq  → rapid preliminary signal
  2. Gemini → multi-timeframe pattern synthesis
  3. OpenAI → trade justification
  4. OpenRouter → consensus arbitration (receives all three outputs)

Final decision requires majority vote (≥ 3/4 agents agreeing).
Minority or timeout → HOLD.
"""

from __future__ import annotations

import asyncio

from src.agents.base_agent import AgentError, BaseAgent
from src.agents.gemini_agent import GeminiAgent
from src.agents.groq_agent import GroqAgent
from src.agents.openai_agent import OpenAIAgent
from src.agents.openrouter_agent import OpenRouterAgent
from src.utils.logger import get_logger
from src.utils.payload import (
    TradingAction,
    TradingPayload,
    merge_payloads,
)

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Coordinates all four AI agents and produces a consensus TradingPayload.

    Usage::

        orchestrator = AgentOrchestrator()
        final_payload = await orchestrator.run(payload)
    """

    def __init__(self) -> None:
        self._groq = GroqAgent()
        self._gemini = GeminiAgent()
        self._openai = OpenAIAgent()
        self._openrouter = OpenRouterAgent()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self, payload: TradingPayload) -> TradingPayload:
        """
        Execute the full four-agent analysis pipeline.

        Parameters
        ----------
        payload : TradingPayload
            Input payload containing indicator + model signal data.

        Returns
        -------
        TradingPayload
            Final consensus payload from the OpenRouter arbiter.
        """
        symbol = payload.symbol
        logger.info(
            "Orchestrator starting analysis cycle",
            symbol=symbol,
            cycle_id=payload.cycle_id,
        )

        # ── Step 1: Groq (fast preliminary) ─────────────────────────────
        groq_result = await self._safe_call(self._groq, payload)
        logger.debug(
            "Groq signal",
            action=groq_result.action,
            confidence=groq_result.confidence,
        )

        # ── Step 2: Gemini + OpenAI in parallel ──────────────────────────
        gemini_task = asyncio.create_task(
            self._safe_call(self._gemini, payload)
        )
        openai_task = asyncio.create_task(
            self._safe_call(self._openai, payload)
        )
        gemini_result, openai_result = await asyncio.gather(
            gemini_task, openai_task
        )

        # ── Step 3: Merge all three into OpenRouter's context ─────────────
        intermediate = merge_payloads(
            [groq_result, gemini_result, openai_result],
            symbol=symbol,
        )

        # ── Step 4: OpenRouter final arbitration ──────────────────────────
        final_result = await self._safe_call(self._openrouter, intermediate)

        # ── Majority validation ───────────────────────────────────────────
        all_actions = [
            groq_result.action,
            gemini_result.action,
            openai_result.action,
            final_result.action,
        ]
        majority_action = self._majority_vote(all_actions)

        if majority_action != final_result.action:
            logger.warning(
                "OpenRouter decision overridden by majority vote",
                openrouter=final_result.action,
                majority=majority_action,
            )
            final_result = final_result.model_copy(
                update={
                    "action": majority_action,
                    "confidence": max(0.0, final_result.confidence - 0.1),
                    "reasoning": (
                        f"[Majority override: {majority_action}] "
                        + final_result.reasoning
                    ),
                }
            )

        logger.info(
            "Orchestrator cycle complete",
            symbol=symbol,
            cycle_id=payload.cycle_id,
            action=final_result.action,
            confidence=final_result.confidence,
        )
        return final_result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _safe_call(
        self,
        agent: BaseAgent,
        payload: TradingPayload,
    ) -> TradingPayload:
        """Call an agent and return a HOLD payload on failure."""
        try:
            return await agent.analyse(payload)
        except AgentError as exc:
            logger.error(
                "Agent failed, using HOLD fallback",
                agent=agent.agent_id,
                error=str(exc),
            )
            return payload.model_copy(update={
                "source": agent.agent_id,
                "action": TradingAction.HOLD,
                "confidence": 0.0,
                "reasoning": f"Agent error: {exc}",
            })

    @staticmethod
    def _majority_vote(actions: list[TradingAction]) -> TradingAction:
        """Return the action with the most votes; HOLD on tie."""
        from collections import Counter
        counts = Counter(actions)
        best, best_count = counts.most_common(1)[0]
        return best if best_count > len(actions) / 2 else TradingAction.HOLD
