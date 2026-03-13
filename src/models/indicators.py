"""
src/models/indicators.py — Multiplex technical indicator system.

Computes a rich set of technical indicators across multiple timeframes
(1 m, 5 m, 1 H) using the `ta` library backed by pandas.

All indicator values are normalised to [0, 1] where applicable so they
can be used directly as neural-network input features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

from constants import (
    ATR_PERIOD,
    BB_PERIOD,
    BB_STD,
    EMA_PERIODS,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_PERIOD,
    STOCH_D,
    STOCH_K,
    STOCH_SMOOTH,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core indicator computation
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Compute all technical indicators for a given OHLCV DataFrame.

    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        timeframe: e.g. "M1", "M5", "H1" (used for column prefixing)

    Returns:
        Original DataFrame with extra indicator columns prefixed by timeframe,
        e.g. ``M1_rsi``, ``M5_macd``.
    """
    if df.empty or len(df) < max(BB_PERIOD, MACD_SLOW, RSI_PERIOD) + 5:
        logger.warning(
            "Insufficient bars for indicator computation",
            timeframe=timeframe,
            rows=len(df),
        )
        return df

    df = df.copy()
    pfx = f"{timeframe}_"

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df.get("volume", pd.Series(np.ones(len(df))))

    # ── RSI ─────────────────────────────────────────────────────────────
    df[f"{pfx}rsi"] = _rsi(close, RSI_PERIOD)

    # ── MACD ────────────────────────────────────────────────────────────
    macd_line, signal_line, histogram = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    df[f"{pfx}macd"] = macd_line
    df[f"{pfx}macd_signal"] = signal_line
    df[f"{pfx}macd_hist"] = histogram

    # ── Bollinger Bands ──────────────────────────────────────────────────
    bb_upper, bb_middle, bb_lower = _bollinger(close, BB_PERIOD, BB_STD)
    df[f"{pfx}bb_upper"] = bb_upper
    df[f"{pfx}bb_middle"] = bb_middle
    df[f"{pfx}bb_lower"] = bb_lower
    df[f"{pfx}bb_width"] = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    df[f"{pfx}bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # ── EMAs ────────────────────────────────────────────────────────────
    for period in EMA_PERIODS:
        df[f"{pfx}ema_{period}"] = close.ewm(span=period, adjust=False).mean()

    # ── EMA crossovers (binary) ──────────────────────────────────────────
    df[f"{pfx}ema_cross_9_21"] = (
        df[f"{pfx}ema_9"] > df[f"{pfx}ema_21"]
    ).astype(float)
    df[f"{pfx}ema_cross_21_50"] = (
        df[f"{pfx}ema_21"] > df[f"{pfx}ema_50"]
    ).astype(float)

    # ── ATR ─────────────────────────────────────────────────────────────
    df[f"{pfx}atr"] = _atr(high, low, close, ATR_PERIOD)

    # ── Stochastic ───────────────────────────────────────────────────────
    stoch_k_vals, stoch_d_vals = _stochastic(
        high, low, close, STOCH_K, STOCH_D, STOCH_SMOOTH
    )
    df[f"{pfx}stoch_k"] = stoch_k_vals
    df[f"{pfx}stoch_d"] = stoch_d_vals

    # ── OBV (On-Balance Volume) ──────────────────────────────────────────
    df[f"{pfx}obv"] = _obv(close, volume)

    # ── Price position features ──────────────────────────────────────────
    df[f"{pfx}price_vs_ema50"] = (close - df[f"{pfx}ema_50"]) / (
        df[f"{pfx}atr"] + 1e-10
    )
    df[f"{pfx}price_vs_ema200"] = (close - df[f"{pfx}ema_200"]) / (
        df[f"{pfx}atr"] + 1e-10
    )

    # ── Normalised RSI & Stochastic to [0,1] ─────────────────────────────
    df[f"{pfx}rsi_norm"] = df[f"{pfx}rsi"] / 100.0
    df[f"{pfx}stoch_k_norm"] = df[f"{pfx}stoch_k"] / 100.0
    df[f"{pfx}stoch_d_norm"] = df[f"{pfx}stoch_d"] / 100.0

    logger.debug(
        "Indicators computed",
        timeframe=timeframe,
        rows=len(df),
        new_cols=len([c for c in df.columns if c.startswith(pfx)]),
    )
    return df


def get_feature_columns(timeframe: str) -> list[str]:
    """Return list of all indicator column names for a timeframe."""
    pfx = f"{timeframe}_"
    base = [
        "rsi_norm",
        "macd", "macd_signal", "macd_hist",
        "bb_width", "bb_pct",
        "ema_cross_9_21", "ema_cross_21_50",
        "atr",
        "stoch_k_norm", "stoch_d_norm",
        "price_vs_ema50", "price_vs_ema200",
    ]
    return [f"{pfx}{b}" for b in base]


def build_feature_matrix(
    dfs: dict[str, pd.DataFrame],
    timeframes: tuple[str, ...],
) -> pd.DataFrame:
    """
    Merge indicator DataFrames from multiple timeframes into a single
    feature matrix aligned on the 1-minute index.

    Args:
        dfs: dict mapping timeframe → DataFrame with indicator columns
        timeframes: ordered tuple of timeframes to include

    Returns:
        Wide feature DataFrame with all timeframe columns merged.
    """
    if not dfs:
        return pd.DataFrame()

    # Use the 1M timeframe as the base index (finest granularity)
    base_tf = timeframes[0] if timeframes else next(iter(dfs))
    base = dfs[base_tf].copy()

    for tf in timeframes[1:]:
        if tf not in dfs:
            continue
        feat_cols = get_feature_columns(tf)
        available = [c for c in feat_cols if c in dfs[tf].columns]
        if not available:
            continue

        # Forward-fill the lower timeframe features onto the base index
        other = dfs[tf][["timestamp"] + available].copy()
        other = other.set_index("timestamp").sort_index()
        base_indexed = base.set_index("timestamp").sort_index()
        merged = base_indexed.join(
            other, how="left"
        ).ffill()
        base = merged.reset_index()

    return base


# ---------------------------------------------------------------------------
# Primitive indicator implementations (numpy-based, no TA dependency needed)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(
    close: pd.Series,
    period: int,
    num_std: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int,
) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int,
    d_period: int,
    smooth: int,
) -> tuple[pd.Series, pd.Series]:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k_smooth = k.rolling(smooth).mean()
    d = k_smooth.rolling(d_period).mean()
    return k_smooth, d


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Multiplex indicator snapshot (used by agents)
# ---------------------------------------------------------------------------

def snapshot_for_payload(
    df: pd.DataFrame,
    timeframe: str,
) -> dict[str, Any]:
    """
    Extract the latest indicator values for a timeframe as a plain dict
    suitable for injecting into a TradingPayload IndicatorSnapshot.
    """
    if df.empty:
        return {"timeframe": timeframe}

    last = df.iloc[-1]
    pfx = f"{timeframe}_"

    def _get(col: str) -> float | None:
        full = f"{pfx}{col}"
        val = last.get(full)
        return None if val is None or (isinstance(val, float) and np.isnan(val)) else float(val)

    return {
        "timeframe": timeframe,
        "rsi": _get("rsi"),
        "macd": _get("macd"),
        "macd_signal": _get("macd_signal"),
        "macd_hist": _get("macd_hist"),
        "bb_upper": _get("bb_upper"),
        "bb_middle": _get("bb_middle"),
        "bb_lower": _get("bb_lower"),
        "ema_9": _get("ema_9"),
        "ema_21": _get("ema_21"),
        "ema_50": _get("ema_50"),
        "ema_200": _get("ema_200"),
        "atr": _get("atr"),
        "stoch_k": _get("stoch_k"),
        "stoch_d": _get("stoch_d"),
        "volume": float(last.get("volume", 0) or 0),
    }
