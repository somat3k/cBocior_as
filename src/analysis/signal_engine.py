"""
src/analysis/signal_engine.py — Trading signal aggregation engine.

Combines:
  - Indicator signals (RSI, MACD, BB, EMA crossovers, Stochastic)
  - Pattern detector signals
  - Model prediction signals
  - Market regime context

Produces a composite signal score (0–100) and a recommended action.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logger import get_logger
from src.utils.payload import (
    IndicatorSnapshot,
    ModelSignals,
    TradingAction,
)

logger = get_logger(__name__)

# Signal strength thresholds
STRONG_BUY_THRESHOLD = 65
WEAK_BUY_THRESHOLD = 55
WEAK_SELL_THRESHOLD = 45
STRONG_SELL_THRESHOLD = 35


class SignalEngine:
    """
    Aggregates all available signals into a composite score and action.

    Score: 0 = strong sell, 50 = neutral, 100 = strong buy
    """

    def compute(
        self,
        indicator_snapshots: list[IndicatorSnapshot],
        model_signals: list[ModelSignals],
        pattern_results: list[dict[str, Any]],
        regime: str = "UNKNOWN",
    ) -> dict[str, Any]:
        """
        Compute composite signal from all inputs.

        Returns
        -------
        dict with: score, action, confidence, components
        """
        components: dict[str, float] = {}

        # ── Indicator signals ────────────────────────────────────────────
        ind_score = self._score_indicators(indicator_snapshots)
        components["indicators"] = ind_score

        # ── Model signals ────────────────────────────────────────────────
        model_score = self._score_models(model_signals)
        components["models"] = model_score

        # ── Pattern signals ───────────────────────────────────────────────
        pattern_score = self._score_patterns(pattern_results)
        components["patterns"] = pattern_score

        # ── Regime adjustment ─────────────────────────────────────────────
        regime_weight = self._regime_weight(regime)

        # ── Weighted composite ────────────────────────────────────────────
        composite = (
            ind_score * 0.35
            + model_score * 0.45
            + pattern_score * 0.20
        )
        # Apply regime scaling (volatile = dampen towards 50)
        composite = 50 + (composite - 50) * regime_weight

        composite = float(np.clip(composite, 0, 100))
        components["composite"] = round(composite, 2)

        # ── Action + confidence ───────────────────────────────────────────
        action, confidence = self._action_from_score(composite)

        logger.debug(
            "Signal computed",
            score=composite,
            action=action,
            confidence=confidence,
            regime=regime,
        )

        return {
            "score": round(composite, 2),
            "action": action,
            "confidence": round(confidence, 4),
            "components": components,
        }

    # ------------------------------------------------------------------
    # Indicator scoring
    # ------------------------------------------------------------------

    def _score_indicators(
        self,
        snapshots: list[IndicatorSnapshot],
    ) -> float:
        """Average indicator score across all timeframes."""
        if not snapshots:
            return 50.0

        scores = []
        for snap in snapshots:
            tf_scores: list[float] = []

            # RSI
            if snap.rsi is not None:
                rsi_score = 50 + (50 - snap.rsi)  # RSI 30 → 70, RSI 70 → 30
                tf_scores.append(float(np.clip(rsi_score, 0, 100)))

            # MACD histogram
            if snap.macd_hist is not None:
                macd_score = 50 + np.tanh(snap.macd_hist * 100) * 50
                tf_scores.append(float(np.clip(macd_score, 0, 100)))

            # BB %B
            if snap.bb_upper and snap.bb_lower and snap.bb_middle:
                close_proxy = snap.bb_middle  # use mid as proxy
                bb_pct = (close_proxy - snap.bb_lower) / (
                    snap.bb_upper - snap.bb_lower + 1e-10
                )
                bb_score = bb_pct * 100
                tf_scores.append(float(np.clip(bb_score, 0, 100)))

            # Stochastic
            if snap.stoch_k is not None:
                stoch_score = snap.stoch_k  # already 0–100
                tf_scores.append(float(np.clip(stoch_score, 0, 100)))

            if tf_scores:
                scores.append(float(np.mean(tf_scores)))

        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------
    # Model scoring
    # ------------------------------------------------------------------

    def _score_models(self, signals: list[ModelSignals]) -> float:
        """Convert model probabilities to 0–100 score."""
        if not signals:
            return 50.0

        probs = []
        for sig in signals:
            p = sig.ensemble_prediction
            if p is None:
                p = sig.nn_prediction
            if p is not None:
                probs.append(float(p))

        return float(np.mean(probs)) * 100 if probs else 50.0

    # ------------------------------------------------------------------
    # Pattern scoring
    # ------------------------------------------------------------------

    def _score_patterns(self, patterns: list[dict[str, Any]]) -> float:
        if not patterns:
            return 50.0
        total_score = sum(
            float(p.get("pattern_score", 0)) for p in patterns
        )
        # pattern_score ranges roughly -3 to +3; normalise to 0-100
        normalised = 50 + (total_score / len(patterns)) * (50 / 3)
        return float(np.clip(normalised, 0, 100))

    # ------------------------------------------------------------------
    # Regime weight
    # ------------------------------------------------------------------

    def _regime_weight(self, regime: str) -> float:
        """
        Scale factor applied to (composite - 50).
        Volatile market → dampen signals (weight < 1).
        Trending → amplify (weight > 1, capped at 1.2).
        """
        return {
            "TRENDING_UP": 1.1,
            "TRENDING_DOWN": 1.1,
            "RANGING": 0.9,
            "VOLATILE": 0.5,
            "UNKNOWN": 0.8,
        }.get(regime, 0.8)

    # ------------------------------------------------------------------
    # Action + confidence
    # ------------------------------------------------------------------

    def _action_from_score(
        self, score: float
    ) -> tuple[TradingAction, float]:
        if score >= STRONG_BUY_THRESHOLD:
            action = TradingAction.BUY
            confidence = (score - STRONG_BUY_THRESHOLD) / (
                100 - STRONG_BUY_THRESHOLD
            ) * 0.4 + 0.6
        elif score >= WEAK_BUY_THRESHOLD:
            action = TradingAction.BUY
            confidence = (score - WEAK_BUY_THRESHOLD) / (
                STRONG_BUY_THRESHOLD - WEAK_BUY_THRESHOLD
            ) * 0.15 + 0.45
        elif score <= STRONG_SELL_THRESHOLD:
            action = TradingAction.SELL
            confidence = (STRONG_SELL_THRESHOLD - score) / STRONG_SELL_THRESHOLD * 0.4 + 0.6
        elif score <= WEAK_SELL_THRESHOLD:
            action = TradingAction.SELL
            confidence = (WEAK_SELL_THRESHOLD - score) / (
                WEAK_SELL_THRESHOLD - STRONG_SELL_THRESHOLD
            ) * 0.15 + 0.45
        else:
            action = TradingAction.HOLD
            confidence = 0.5

        return action, float(np.clip(confidence, 0.0, 1.0))
