"""
src/agents/orchestrator.py — Groq-only orchestrator.

Uses the Groq agent configured via GROQ_MODEL for signal generation. The
Groq agent acts as the sole decision engine for model outputs; other
providers have been removed from the orchestration flow.
"""

from __future__ import annotations

from src.agents.base_agent import AgentError, BaseAgent
from src.agents.groq_agent import GroqAgent
from src.utils.logger import get_logger
from src.utils.payload import TradingAction, TradingPayload

logger = get_logger(__name__)


class AgentOrchestrator:
    """
    Coordinates the Groq OSS 120B agent and returns its TradingPayload.

    Usage::

        orchestrator = AgentOrchestrator()
        final_payload = await orchestrator.run(payload)
    """

    def __init__(self) -> None:
        self._groq = GroqAgent()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self, payload: TradingPayload) -> TradingPayload:
        """
        Execute the Groq-only analysis pipeline.

        Parameters
        ----------
        payload : TradingPayload
            Input payload containing indicator + model signal data.

        Returns
        -------
        TradingPayload
            Final Groq payload for the cycle.
        """
        symbol = payload.symbol
        logger.info(
            "Orchestrator starting analysis cycle",
            symbol=symbol,
            cycle_id=payload.cycle_id,
        )

        # ── Groq only ─────────────────────────────────────────────────────
        final_result = await self._safe_call(self._groq, payload)
        logger.debug(
            "Groq signal",
            action=final_result.action,
            confidence=final_result.confidence,
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
