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

        # ── Cross-timeframe divergence  (D9) ─────────────────────────────
        divergence_penalty = self._cross_tf_divergence(indicator_snapshots, model_signals)
        components["tf_divergence"] = divergence_penalty
        if divergence_penalty > 0:
            logger.debug(
                "Cross-timeframe divergence detected",
                penalty=divergence_penalty,
            )

        # ── Session filter  (G9) ─────────────────────────────────────────
        session_mult = self._session_multiplier()
        components["session_multiplier"] = round(session_mult, 3)

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
        # Apply session multiplier (off-hours → soften signal)
        composite = 50 + (composite - 50) * session_mult
        # Apply divergence penalty: dampen towards 50 proportionally
        if divergence_penalty > 0:
            dampen = 1.0 - min(divergence_penalty, 0.5)
            composite = 50 + (composite - 50) * dampen

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
            "timeframe_divergence": divergence_penalty > 0,
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
            if snap.bb_upper is not None and snap.bb_lower is not None and snap.bb_middle is not None:
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
    # Cross-timeframe divergence  (D9)
    # ------------------------------------------------------------------

    def _cross_tf_divergence(
        self,
        snapshots: list[IndicatorSnapshot],
        model_signals: list[ModelSignals],
    ) -> float:
        """
        Detect conflicting directional signals across timeframes.

        Returns a penalty in [0, 1]: 0 = no divergence, higher = more conflict.
        Divergence is flagged when:
        * Model predictions span both sides of 0.5 across timeframes, OR
        * RSI is oversold on one TF and overbought on another.
        """
        # Model probability divergence
        model_probs = []
        for sig in model_signals:
            p = sig.ensemble_prediction if sig.ensemble_prediction is not None else sig.nn_prediction
            if p is not None:
                model_probs.append(float(p))

        model_divergence = 0.0
        if len(model_probs) >= 2:
            bullish = sum(1 for p in model_probs if p > 0.55)
            bearish = sum(1 for p in model_probs if p < 0.45)
            if bullish > 0 and bearish > 0:
                model_divergence = min(bullish, bearish) / len(model_probs)

        # RSI divergence: one TF oversold (<35), another overbought (>65)
        rsi_values = [snap.rsi for snap in snapshots if snap.rsi is not None]
        rsi_divergence = 0.0
        if len(rsi_values) >= 2:
            oversold = sum(1 for r in rsi_values if r < 35)
            overbought = sum(1 for r in rsi_values if r > 65)
            if oversold > 0 and overbought > 0:
                rsi_divergence = 0.5

        return float(max(model_divergence, rsi_divergence))

    # ------------------------------------------------------------------
    # Session filter  (G9)
    # ------------------------------------------------------------------

    def _session_multiplier(self) -> float:
        """
        Return a signal strength multiplier based on the current UTC time.

        Sessions and their multipliers:
          - London–NY overlap (13:00–16:00 UTC): 1.20  ← highest liquidity
          - New York (13:00–21:00 UTC):          1.10
          - London (08:00–16:00 UTC):            1.10
          - Asian (00:00–08:00 UTC):             0.80
          - Outside all major sessions:          0.70
        """
        from datetime import datetime
        from datetime import timezone as _tz
        now_utc = datetime.now(tz=_tz.utc)
        hour = now_utc.hour + now_utc.minute / 60.0

        london = 8.0 <= hour < 16.0
        new_york = 13.0 <= hour < 21.0
        overlap = 13.0 <= hour < 16.0
        asian = 0.0 <= hour < 8.0

        if overlap:
            return 1.20
        elif london or new_york:
            return 1.10
        elif asian:
            return 0.80
        return 0.70

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
