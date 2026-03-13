"""
src/trading/decision_engine.py — Trade decision engine.

Combines the signal engine output with the orchestrator's agent consensus
to produce final, risk-gated trade decisions.

Decision rules:
  1. Both signal score AND agent consensus must agree on direction.
  2. Composite confidence must be ≥ 0.65 to trade.
  3. De-duplication: same signal within cooldown period → skip.
  4. All decisions are risk-checked before forwarding to Execution.
"""

from __future__ import annotations

import time
from typing import Any

from constants import BOT_ANALYSIS_COOLDOWN_SECONDS
from src.utils.logger import get_logger
from src.utils.payload import (
    IndicatorSnapshot,
    ModelSignals,
    PayloadBuilder,
    RiskFlags,
    TradingAction,
    TradingPayload,
)

logger = get_logger(__name__)

# Minimum confidence required to execute a trade (not HOLD)
MIN_CONFIDENCE: float = 0.65


class DecisionEngine:
    """
    Combines agent consensus with internal signal analysis to produce
    actionable trading decisions.

    Usage::

        engine = DecisionEngine()
        decision = engine.decide(
            agent_payload=orchestrator_result,
            signal_data=signal_engine_output,
            risk_flags=risk_manager_flags,
        )
    """

    def __init__(self) -> None:
        self._last_decision_time: float = 0.0
        self._last_action: TradingAction = TradingAction.HOLD

    def decide(
        self,
        agent_payload: TradingPayload,
        signal_data: dict[str, Any],
        risk_flags: RiskFlags,
        current_spread_pips: float = 0.0,
    ) -> TradingPayload:
        """
        Produce a final trade decision.

        Parameters
        ----------
        agent_payload : TradingPayload
            Consensus payload from the AgentOrchestrator.
        signal_data : dict
            Output from SignalEngine.compute().
        risk_flags : RiskFlags
            Flags evaluated by RiskManager.
        current_spread_pips : float
            Live spread for the symbol.

        Returns
        -------
        TradingPayload — final actionable payload.
        """
        agent_action = agent_payload.action
        signal_action = signal_data.get("action", TradingAction.HOLD)
        signal_confidence = float(signal_data.get("confidence", 0.0))
        agent_confidence = float(agent_payload.confidence)

        # ── Cooldown check ────────────────────────────────────────────────
        now = time.time()
        elapsed = now - self._last_decision_time
        if elapsed < BOT_ANALYSIS_COOLDOWN_SECONDS:
            logger.debug(
                "In cooldown, returning HOLD",
                elapsed_s=round(elapsed, 1),
                cooldown_s=BOT_ANALYSIS_COOLDOWN_SECONDS,
            )
            return self._hold_payload(
                agent_payload,
                reason=f"Cooldown ({BOT_ANALYSIS_COOLDOWN_SECONDS - elapsed:.0f}s remaining)",
            )

        # ── Agreement check ───────────────────────────────────────────────
        if agent_action != signal_action:
            logger.info(
                "Agent and signal engine disagree → HOLD",
                agent=agent_action,
                signal=signal_action,
            )
            return self._hold_payload(
                agent_payload,
                reason=f"Agent/signal disagreement: {agent_action} vs {signal_action}",
            )

        # ── Combined confidence ────────────────────────────────────────────
        combined_confidence = 0.55 * agent_confidence + 0.45 * signal_confidence

        if combined_confidence < MIN_CONFIDENCE:
            logger.info(
                "Confidence below threshold → HOLD",
                combined=round(combined_confidence, 3),
                threshold=MIN_CONFIDENCE,
            )
            return self._hold_payload(
                agent_payload,
                reason=f"Confidence {combined_confidence:.3f} < {MIN_CONFIDENCE}",
            )

        # ── Risk gate ─────────────────────────────────────────────────────
        if risk_flags.any_flag:
            logger.info(
                "Risk flags set → HOLD",
                flags=risk_flags.model_dump(),
            )
            return self._hold_payload(
                agent_payload,
                reason="Risk flags active",
                risk_flags=risk_flags,
            )

        # ── De-duplication ────────────────────────────────────────────────
        if (
            agent_action == self._last_action
            and elapsed < BOT_ANALYSIS_COOLDOWN_SECONDS * 3
        ):
            logger.debug(
                "Duplicate signal suppressed",
                action=agent_action,
                elapsed_s=round(elapsed, 1),
            )
            return self._hold_payload(
                agent_payload,
                reason="Duplicate signal suppressed",
            )

        # ── Accept trade ──────────────────────────────────────────────────
        self._last_decision_time = now
        self._last_action = agent_action

        final = agent_payload.model_copy(update={
            "source": "decision_engine",
            "action": agent_action,
            "confidence": round(combined_confidence, 4),
            "reasoning": (
                f"[CONFIRMED] Agent: {agent_confidence:.3f}, "
                f"Signal: {signal_confidence:.3f}, "
                f"Combined: {combined_confidence:.3f}. "
                + agent_payload.reasoning
            ),
            "risk_flags": risk_flags,
        })

        logger.info(
            "Trade decision accepted",
            action=agent_action.value,
            confidence=round(combined_confidence, 3),
            symbol=agent_payload.symbol,
        )
        return final

    @staticmethod
    def _hold_payload(
        base: TradingPayload,
        reason: str = "",
        risk_flags: RiskFlags | None = None,
    ) -> TradingPayload:
        return base.model_copy(update={
            "source": "decision_engine",
            "action": TradingAction.HOLD,
            "confidence": 0.0,
            "reasoning": reason,
            "risk_flags": risk_flags or base.risk_flags,
        })
