"""
src/analysis/pattern_detector.py — Candlestick and indicator pattern detector.

Detects classic candlestick patterns and price-action formations using
pure pandas/numpy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pattern detection functions (vectorised)
# ---------------------------------------------------------------------------

def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """Return +1 for bullish engulfing, -1 for bearish, 0 otherwise."""
    o, c = df["open"], df["close"]
    po, pc = o.shift(1), c.shift(1)

    bullish = (pc < po) & (c > o) & (o <= pc) & (c >= po)
    bearish = (pc > po) & (c < o) & (o >= pc) & (c <= po)
    return bullish.astype(int) - bearish.astype(int)


def detect_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """Return True where body is ≤ threshold × total range."""
    body = (df["close"] - df["open"]).abs()
    total_range = df["high"] - df["low"]
    return body <= (threshold * (total_range + 1e-10))


def detect_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Hammer: small body, long lower shadow (≥ 2× body), tiny upper shadow.
    Returns True on hammer candles (bullish reversal).
    """
    body = (df["close"] - df["open"]).abs()
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    return (lower_shadow >= 2 * body) & (upper_shadow <= body)


def detect_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Shooting star: small body, long upper shadow (≥ 2× body), tiny lower shadow.
    Returns True on shooting-star candles (bearish reversal).
    """
    body = (df["close"] - df["open"]).abs()
    upper_shadow = df["high"] - df[["open", "close"]].max(axis=1)
    lower_shadow = df[["open", "close"]].min(axis=1) - df["low"]
    return (upper_shadow >= 2 * body) & (lower_shadow <= body)


def detect_double_top(
    df: pd.DataFrame,
    window: int = 20,
    tolerance: float = 0.002,
) -> pd.Series:
    """
    Simplified double-top detector: two local highs within tolerance,
    separated by a trough.
    Returns True where the pattern completion is detected.
    """
    highs = df["high"]
    result = pd.Series(False, index=df.index)

    for i in range(window * 2, len(df)):
        segment = highs.iloc[i - window * 2 : i]
        if len(segment) < window * 2:
            continue
        h1 = segment.iloc[:window].max()
        h2 = segment.iloc[window:].max()
        if abs(h1 - h2) / (h1 + 1e-10) <= tolerance:
            result.iloc[i] = True

    return result


def detect_double_bottom(
    df: pd.DataFrame,
    window: int = 20,
    tolerance: float = 0.002,
) -> pd.Series:
    """Simplified double-bottom detector (two similar lows)."""
    lows = df["low"]
    result = pd.Series(False, index=df.index)

    for i in range(window * 2, len(df)):
        segment = lows.iloc[i - window * 2 : i]
        if len(segment) < window * 2:
            continue
        l1 = segment.iloc[:window].min()
        l2 = segment.iloc[window:].min()
        if abs(l1 - l2) / (l1 + 1e-10) <= tolerance:
            result.iloc[i] = True

    return result


# ---------------------------------------------------------------------------
# PatternDetector class
# ---------------------------------------------------------------------------

class PatternDetector:
    """
    Runs all pattern detectors on a DataFrame and returns a summary dict.
    """

    def detect_all(
        self,
        df: pd.DataFrame,
        timeframe: str = "",
    ) -> dict[str, Any]:
        """
        Detect all patterns and return a summary for the latest bar.

        Returns
        -------
        dict with pattern names as keys and True/False (or int) values.
        """
        if df.empty or len(df) < 5:
            return {}

        result: dict[str, Any] = {}
        last_idx = df.index[-1]

        # Single-bar patterns
        result["engulfing"] = int(detect_engulfing(df).loc[last_idx])
        result["doji"] = bool(detect_doji(df).loc[last_idx])
        result["hammer"] = bool(detect_hammer(df).loc[last_idx])
        result["shooting_star"] = bool(detect_shooting_star(df).loc[last_idx])

        # Multi-bar patterns (require enough data)
        if len(df) >= 40:
            result["double_top"] = bool(detect_double_top(df).loc[last_idx])
            result["double_bottom"] = bool(
                detect_double_bottom(df).loc[last_idx]
            )
        else:
            result["double_top"] = False
            result["double_bottom"] = False

        # Composite bullish/bearish score (-3 to +3)
        score = (
            result["engulfing"]
            + int(result["hammer"])
            - int(result["shooting_star"])
            + int(result["double_bottom"])
            - int(result["double_top"])
        )
        result["pattern_score"] = score
        result["timeframe"] = timeframe

        logger.debug(
            "Patterns detected",
            timeframe=timeframe,
            score=score,
            patterns={k: v for k, v in result.items() if k != "timeframe"},
        )
        return result
