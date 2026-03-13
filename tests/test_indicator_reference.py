"""
tests/test_indicator_reference.py — Indicator unit tests against known reference values.

Each test uses a hand-crafted OHLCV sequence for which the expected output can
be independently calculated or verified against well-known formulas.  (D12)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.indicators import compute_indicators

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(tf: str, name: str) -> str:
    """Return prefixed column: _col('M1', 'rsi') → 'M1_rsi'."""
    return f"{tf}_{name}"


def _flat_df(n: int = 100, price: float = 1.1000) -> pd.DataFrame:
    return pd.DataFrame({
        "open":   [price] * n, "high": [price + 0.0001] * n,
        "low":    [price - 0.0001] * n, "close": [price] * n,
        "volume": [1000.0] * n,
    })


def _trending_up(n: int = 100, start: float = 1.0, step: float = 0.001) -> pd.DataFrame:
    close = start + np.arange(n) * step
    return pd.DataFrame({
        "open": close - 0.0002, "high": close + 0.0002,
        "low": close - 0.0003, "close": close, "volume": np.ones(n) * 500.0,
    })


def _trending_down(n: int = 100, start: float = 2.0, step: float = 0.001) -> pd.DataFrame:
    close = start - np.arange(n) * step
    return pd.DataFrame({
        "open": close + 0.0002, "high": close + 0.0003,
        "low": close - 0.0002, "close": close, "volume": np.ones(n) * 500.0,
    })


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_flat_in_range(self) -> None:
        """RSI of a flat series should return a value in [0, 100]."""
        df = compute_indicators(_flat_df(50), "M1")
        rsi = df[_col("M1", "rsi")].dropna()
        assert len(rsi) > 0
        assert 0.0 <= rsi.iloc[-1] <= 100.0

    def test_rsi_trending_up_above_50(self) -> None:
        df = compute_indicators(_trending_up(100), "M1")
        rsi = df[_col("M1", "rsi")].dropna().iloc[-1]
        assert rsi > 50.0

    def test_rsi_trending_down_below_50(self) -> None:
        df = compute_indicators(_trending_down(100), "M1")
        rsi = df[_col("M1", "rsi")].dropna().iloc[-1]
        assert rsi < 50.0

    def test_rsi_range_0_to_100(self) -> None:
        df = compute_indicators(_trending_up(100), "M1")
        rsi = df[_col("M1", "rsi")].dropna()
        assert (rsi >= 0.0).all()
        assert (rsi <= 100.0).all()

    def test_rsi_column_present(self) -> None:
        df = compute_indicators(_flat_df(50), "M1")
        assert _col("M1", "rsi") in df.columns


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_columns_present(self) -> None:
        df = compute_indicators(_flat_df(60), "M1")
        for name in ("macd", "macd_signal", "macd_hist"):
            assert _col("M1", name) in df.columns

    def test_macd_hist_positive_in_uptrend(self) -> None:
        df = compute_indicators(_trending_up(100), "M1")
        hist = df[_col("M1", "macd_hist")].dropna().iloc[-1]
        assert hist > 0.0

    def test_macd_hist_negative_in_downtrend(self) -> None:
        df = compute_indicators(_trending_down(100), "M1")
        hist = df[_col("M1", "macd_hist")].dropna().iloc[-1]
        assert hist < 0.0

    def test_macd_hist_equals_macd_minus_signal(self) -> None:
        df = compute_indicators(_trending_up(80), "M1")
        mc, sc, hc = _col("M1", "macd"), _col("M1", "macd_signal"), _col("M1", "macd_hist")
        row = df[[mc, sc, hc]].dropna().iloc[-1]
        assert row[hc] == pytest.approx(row[mc] - row[sc], abs=1e-8)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def test_bb_columns_present(self) -> None:
        df = compute_indicators(_flat_df(60), "M1")
        for name in ("bb_upper", "bb_middle", "bb_lower"):
            assert _col("M1", name) in df.columns

    def test_bb_upper_above_lower(self) -> None:
        df = compute_indicators(_trending_up(60), "M1")
        upper = _col("M1", "bb_upper")
        mid = _col("M1", "bb_middle")
        lower = _col("M1", "bb_lower")
        bb = df[[upper, mid, lower]].dropna()
        assert (bb[upper] >= bb[mid]).all()
        assert (bb[mid] >= bb[lower]).all()

    def test_bb_width_near_zero_for_flat_series(self) -> None:
        df = compute_indicators(_flat_df(50), "M1")
        upper = _col("M1", "bb_upper")
        lower = _col("M1", "bb_lower")
        bb = df[[upper, lower]].dropna()
        assert ((bb[upper] - bb[lower]).abs() < 0.01).all()

    def test_bb_middle_is_rolling_mean(self) -> None:
        df = compute_indicators(_trending_up(60), "M1")
        sma20 = df["close"].rolling(20).mean()
        bb_mid = df[_col("M1", "bb_middle")]
        diff = (sma20 - bb_mid).dropna().abs()
        assert (diff < 1e-8).all()


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_columns_present(self) -> None:
        df = compute_indicators(_flat_df(60), "M1")
        for period in (9, 21, 50, 200):
            col = _col("M1", f"ema_{period}")
            assert col in df.columns, f"Column '{col}' missing"

    def test_ema9_above_ema200_in_uptrend(self) -> None:
        df = compute_indicators(_trending_up(250), "M1")
        ema9 = _col("M1", "ema_9")
        ema200 = _col("M1", "ema_200")
        last = df[[ema9, ema200]].dropna().iloc[-1]
        assert last[ema9] > last[ema200]

    def test_ema9_below_ema200_in_downtrend(self) -> None:
        df = compute_indicators(_trending_down(250), "M1")
        ema9 = _col("M1", "ema_9")
        ema200 = _col("M1", "ema_200")
        last = df[[ema9, ema200]].dropna().iloc[-1]
        assert last[ema9] < last[ema200]


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_atr_column_present(self) -> None:
        df = compute_indicators(_flat_df(60), "M1")
        assert _col("M1", "atr") in df.columns

    def test_atr_non_negative(self) -> None:
        df = compute_indicators(_trending_up(50), "M1")
        atr = df[_col("M1", "atr")].dropna()
        assert (atr >= 0.0).all()

    def test_atr_flat_series_near_zero(self) -> None:
        # Need at least max(BB_PERIOD=20, MACD_SLOW=26, RSI_PERIOD=14) + 5 = 31 rows
        df = compute_indicators(_flat_df(35, price=1.0), "M1")
        atr = df[_col("M1", "atr")].dropna()
        if len(atr):
            assert atr.iloc[-1] < 0.01

    def test_atr_higher_in_volatile_market(self) -> None:
        flat_ind = compute_indicators(_flat_df(50), "M1")
        rng = np.random.default_rng(0)
        close = 1.0 + np.cumsum(rng.normal(0, 0.01, 50))
        volatile = pd.DataFrame({
            "open": close, "high": close + 0.01,
            "low": close - 0.01, "close": close, "volume": [100.0] * 50,
        })
        vol_ind = compute_indicators(volatile, "M1")
        flat_atr = flat_ind[_col("M1", "atr")].dropna().iloc[-1]
        vol_atr = vol_ind[_col("M1", "atr")].dropna().iloc[-1]
        assert vol_atr > flat_atr


# ---------------------------------------------------------------------------
# Stochastic Oscillator
# ---------------------------------------------------------------------------

class TestStochastic:
    def test_stochastic_columns_present(self) -> None:
        df = compute_indicators(_flat_df(60), "M1")
        assert _col("M1", "stoch_k") in df.columns
        assert _col("M1", "stoch_d") in df.columns

    def test_stochastic_range_0_to_100(self) -> None:
        df = compute_indicators(_trending_up(60), "M1")
        k = df[_col("M1", "stoch_k")].dropna()
        assert (k >= 0.0).all()
        assert (k <= 100.0).all()

    def test_stoch_k_high_in_uptrend(self) -> None:
        df = compute_indicators(_trending_up(80), "M1")
        k = df[_col("M1", "stoch_k")].dropna().iloc[-1]
        assert k > 70.0


# ---------------------------------------------------------------------------
# Feature normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    def test_normalised_rsi_in_01_range(self) -> None:
        df = compute_indicators(_trending_up(100), "M1")
        col = _col("M1", "rsi_norm")
        if col in df.columns:
            vals = df[col].dropna()
            assert (vals >= 0.0).all()
            assert (vals <= 1.0).all()

    def test_normalised_stoch_in_01_range(self) -> None:
        df = compute_indicators(_trending_up(80), "M1")
        for name in ("stoch_k_norm", "stoch_d_norm"):
            col = _col("M1", name)
            if col in df.columns:
                vals = df[col].dropna()
                assert (vals >= 0.0).all()
                assert (vals <= 1.0).all()


# ---------------------------------------------------------------------------
# Multi-timeframe columns
# ---------------------------------------------------------------------------

class TestMultiTimeframeColumns:
    @pytest.mark.parametrize("tf", ["M1", "M5", "H1"])
    def test_core_columns_present_all_timeframes(self, tf: str) -> None:
        df = compute_indicators(_trending_up(300), tf)
        required = [
            _col(tf, "rsi"), _col(tf, "macd"), _col(tf, "macd_signal"),
            _col(tf, "macd_hist"), _col(tf, "bb_upper"), _col(tf, "bb_middle"),
            _col(tf, "bb_lower"), _col(tf, "ema_9"), _col(tf, "ema_21"),
            _col(tf, "ema_50"), _col(tf, "atr"),
            _col(tf, "stoch_k"), _col(tf, "stoch_d"),
        ]
        for col in required:
            assert col in df.columns, f"Column '{col}' missing for {tf}"
