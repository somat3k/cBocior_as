"""
tests/test_indicators.py — Unit tests for multiplex indicator system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.indicators import (
    _atr,
    _bollinger,
    _macd,
    _rsi,
    _stochastic,
    compute_indicators,
    get_feature_columns,
    snapshot_for_payload,
)


def make_ohlcv(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + rng.uniform(0.0001, 0.0010, n)
    low = close - rng.uniform(0.0001, 0.0010, n)
    open_ = close - rng.normal(0, 0.0003, n)
    volume = rng.integers(100, 1000, n).astype(float)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestRSI:
    def test_range(self) -> None:
        df = make_ohlcv(100)
        rsi = _rsi(df["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_length(self) -> None:
        df = make_ohlcv(50)
        rsi = _rsi(df["close"], 14)
        assert len(rsi) == 50


class TestMACD:
    def test_output_length(self) -> None:
        df = make_ohlcv(100)
        macd, signal, hist = _macd(df["close"], 12, 26, 9)
        assert len(macd) == len(df)
        assert len(signal) == len(df)
        assert len(hist) == len(df)

    def test_histogram_equals_diff(self) -> None:
        df = make_ohlcv(100)
        macd, signal, hist = _macd(df["close"], 12, 26, 9)
        expected = macd - signal
        pd.testing.assert_series_equal(hist, expected)


class TestBollinger:
    def test_upper_above_lower(self) -> None:
        df = make_ohlcv(100)
        upper, middle, lower = _bollinger(df["close"], 20, 2.0)
        valid = ~upper.isna() & ~lower.isna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()


class TestATR:
    def test_non_negative(self) -> None:
        df = make_ohlcv(100)
        atr = _atr(df["high"], df["low"], df["close"], 14)
        assert (atr.dropna() >= 0).all()


class TestComputeIndicators:
    def test_columns_added(self) -> None:
        df = make_ohlcv(200)
        result = compute_indicators(df, "M1")
        expected_cols = get_feature_columns("M1")
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_all_nan_columns(self) -> None:
        df = make_ohlcv(200)
        result = compute_indicators(df, "M1")
        feat_cols = get_feature_columns("M1")
        available = [c for c in feat_cols if c in result.columns]
        for col in available:
            non_nan = result[col].dropna()
            assert len(non_nan) > 0, f"Column {col} is all NaN"


class TestSnapshotForPayload:
    def test_required_keys(self) -> None:
        df = make_ohlcv(200)
        df = compute_indicators(df, "H1")
        snap = snapshot_for_payload(df, "H1")
        assert snap["timeframe"] == "H1"
        for key in ("rsi", "macd", "atr"):
            assert key in snap

    def test_empty_df(self) -> None:
        snap = snapshot_for_payload(pd.DataFrame(), "M5")
        assert snap == {"timeframe": "M5"}
