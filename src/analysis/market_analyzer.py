"""
src/analysis/market_analyzer.py — Market regime and trend analyser.

Classifies the current market state into one of:
  - TRENDING_UP
  - TRENDING_DOWN
  - RANGING
  - VOLATILE

Also computes momentum, volatility, and session filters.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class TradingSession(str, Enum):
    SYDNEY = "SYDNEY"
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP_LONDON_NY = "OVERLAP_LONDON_NY"
    CLOSED = "CLOSED"


# ---------------------------------------------------------------------------
# MarketAnalyzer
# ---------------------------------------------------------------------------

class MarketAnalyzer:
    """
    Analyses multi-timeframe DataFrames to determine market regime,
    momentum, volatility, and active trading session.
    """

    def analyse(
        self,
        dfs: dict[str, pd.DataFrame],
        indicator_snapshots: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Perform a comprehensive market analysis.

        Parameters
        ----------
        dfs : dict timeframe → OHLCV+indicator DataFrame
        indicator_snapshots : list of snapshot dicts (one per timeframe)

        Returns
        -------
        dict with keys: regime, momentum, volatility, session, divergence
        """
        # Use the 1H DataFrame for regime classification if available
        base_df = dfs.get("H1") or dfs.get("M5") or next(iter(dfs.values()))

        regime = self._classify_regime(base_df)
        momentum = self._compute_momentum(base_df)
        volatility = self._compute_volatility(base_df)
        session = self._current_session(base_df)
        divergence = self._detect_cross_tf_divergence(indicator_snapshots)

        result = {
            "regime": regime.value,
            "momentum": round(momentum, 4),
            "volatility": round(volatility, 4),
            "session": session.value,
            "cross_tf_divergence": divergence,
        }

        logger.debug("Market analysis", **result)
        return result

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def _classify_regime(self, df: pd.DataFrame) -> MarketRegime:
        if df.empty or len(df) < 50:
            return MarketRegime.UNKNOWN

        close = df["close"]
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else ema_50

        # ATR-based volatility
        high, low = df["high"], df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Normalised ATR
        price = float(close.iloc[-1])
        natr = atr / (price + 1e-10)

        last_ema50 = float(ema_50.iloc[-1])
        last_ema200 = float(ema_200.iloc[-1])
        trend_pct = (last_ema50 - last_ema200) / (last_ema200 + 1e-10)

        # Regime logic
        if natr > 0.005:  # normalised ATR > 0.5% → volatile
            return MarketRegime.VOLATILE
        if trend_pct > 0.002:
            return MarketRegime.TRENDING_UP
        if trend_pct < -0.002:
            return MarketRegime.TRENDING_DOWN
        return MarketRegime.RANGING

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------

    def _compute_momentum(self, df: pd.DataFrame, period: int = 14) -> float:
        """Return rate-of-change over `period` bars, normalised to [-1, 1]."""
        if len(df) < period + 1:
            return 0.0
        close = df["close"]
        roc = (close.iloc[-1] - close.iloc[-period - 1]) / (
            close.iloc[-period - 1] + 1e-10
        )
        return float(np.clip(roc * 100, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Volatility
    # ------------------------------------------------------------------

    def _compute_volatility(self, df: pd.DataFrame, period: int = 14) -> float:
        """Return normalised ATR as a volatility measure."""
        if len(df) < period + 1:
            return 0.0
        close = df["close"]
        high = df["high"]
        low = df["low"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.rolling(period).mean().iloc[-1])
        price = float(close.iloc[-1])
        return atr / (price + 1e-10) * 10000  # in pips for FX

    # ------------------------------------------------------------------
    # Trading session detection
    # ------------------------------------------------------------------

    def _current_session(self, df: pd.DataFrame) -> TradingSession:
        """
        Determine the current trading session based on the last bar's
        UTC timestamp.
        """
        if df.empty:
            return TradingSession.CLOSED

        last_ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns else None
        if last_ts is None:
            return TradingSession.CLOSED

        if hasattr(last_ts, "hour"):
            hour = int(last_ts.hour)
        else:
            try:
                hour = pd.Timestamp(last_ts).hour
            except Exception:
                return TradingSession.CLOSED

        # UTC session hours (approximate)
        if 13 <= hour < 17:
            return TradingSession.OVERLAP_LONDON_NY
        if 8 <= hour < 17:
            return TradingSession.LONDON
        if 17 <= hour < 22:
            return TradingSession.NEW_YORK
        if 22 <= hour or hour < 2:
            return TradingSession.SYDNEY
        if 2 <= hour < 8:
            return TradingSession.TOKYO
        return TradingSession.CLOSED

    # ------------------------------------------------------------------
    # Cross-timeframe divergence
    # ------------------------------------------------------------------

    def _detect_cross_tf_divergence(
        self,
        snapshots: list[dict[str, Any]],
    ) -> bool:
        """
        Return True if RSI or MACD signals diverge across timeframes.
        """
        if len(snapshots) < 2:
            return False

        rsi_values = [
            s.get("rsi") for s in snapshots if s.get("rsi") is not None
        ]
        if len(rsi_values) >= 2:
            rsi_range = max(rsi_values) - min(rsi_values)
            if rsi_range > 20:  # significant RSI divergence
                return True

        # MACD histogram sign divergence
        hist_signs = [
            np.sign(s.get("macd_hist", 0))
            for s in snapshots
            if s.get("macd_hist") is not None
        ]
        if len(set(hist_signs)) > 1:
            return True

        return False
