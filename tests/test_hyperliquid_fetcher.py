"""
tests/test_hyperliquid_fetcher.py — Unit tests for HyperLiquidFetcher.

All network calls are mocked; no real HTTP requests are made.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.ctrader_client import OHLCVBar
from src.data.hyperliquid_fetcher import HyperLiquidFetcher
from src.utils.cache import reset_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candle(ts_ms: int, price: float, volume: float = 100.0) -> dict:
    return {
        "t": str(ts_ms),
        "o": str(price),
        "h": str(price + 1),
        "l": str(price - 1),
        "c": str(price + 0.5),
        "v": str(volume),
    }


def _make_mock_session(candles: list[dict], status_code: int = 200) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = candles
    mock_resp.raise_for_status = MagicMock()
    mock_session = MagicMock()
    mock_session.post.return_value = mock_resp
    return mock_session


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHyperLiquidFetcherInit:
    def test_hl_coin_mapping(self) -> None:
        fetcher = HyperLiquidFetcher("BTCUSD")
        assert fetcher.hl_coin == "BTC"

    def test_unknown_symbol_uses_symbol_as_coin(self) -> None:
        fetcher = HyperLiquidFetcher("UNKNOWNSYMBOL")
        assert fetcher.hl_coin == "UNKNOWNSYMBOL"


class TestFetchHistoricalBars:
    def test_returns_bars_on_success(self, tmp_data_dir: Path) -> None:
        now_ms = int(time.time() * 1000)
        candles = [_make_candle(now_ms - i * 60_000, 50000.0 + i) for i in range(5)]
        session = _make_mock_session(candles)

        fetcher = HyperLiquidFetcher("BTCUSD", data_dir=tmp_data_dir, session=session)
        bars = fetcher.fetch_historical_bars("M1", count=5)

        assert len(bars) == 5
        assert all(isinstance(b, OHLCVBar) for b in bars)
        assert bars[0].symbol == "BTCUSD"

    def test_bars_sorted_oldest_first(self, tmp_data_dir: Path) -> None:
        now_ms = int(time.time() * 1000)
        # Provide candles in reverse order
        candles = [_make_candle(now_ms - i * 60_000, 100.0) for i in range(5)]
        session = _make_mock_session(candles)

        fetcher = HyperLiquidFetcher("ETHUSD", data_dir=tmp_data_dir, session=session)
        bars = fetcher.fetch_historical_bars("M1", count=5)

        timestamps = [b.timestamp for b in bars]
        assert timestamps == sorted(timestamps)

    def test_returns_empty_on_api_error(self, tmp_data_dir: Path) -> None:
        session = MagicMock()
        session.post.side_effect = Exception("network error")

        fetcher = HyperLiquidFetcher("BTCUSD", data_dir=tmp_data_dir, session=session)
        bars = fetcher.fetch_historical_bars("M1", count=5)

        assert bars == []

    def test_returns_empty_on_http_error(self, tmp_data_dir: Path) -> None:
        import requests as req_lib

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.RequestException("404")
        session = MagicMock()
        session.post.return_value = mock_resp

        fetcher = HyperLiquidFetcher("BTCUSD", data_dir=tmp_data_dir, session=session)
        bars = fetcher.fetch_historical_bars("M1", count=5)

        assert bars == []

    def test_unsupported_timeframe_returns_empty(self, tmp_data_dir: Path) -> None:
        session = _make_mock_session([])
        fetcher = HyperLiquidFetcher("BTCUSD", data_dir=tmp_data_dir, session=session)
        bars = fetcher.fetch_historical_bars("UNSUPPORTED_TF", count=5)

        assert bars == []
        session.post.assert_not_called()

    def test_cache_hit_skips_api_call(self, tmp_data_dir: Path) -> None:
        now_ms = int(time.time() * 1000)
        candles = [_make_candle(now_ms - i * 60_000, 200.0) for i in range(3)]
        session = _make_mock_session(candles)

        fetcher = HyperLiquidFetcher("SOLUSD", data_dir=tmp_data_dir, session=session)
        # First call populates cache
        bars1 = fetcher.fetch_historical_bars("M1", count=3)
        # Second call should hit cache
        bars2 = fetcher.fetch_historical_bars("M1", count=3)

        assert len(bars1) == len(bars2) == 3
        assert session.post.call_count == 1  # only one real API call


class TestFetchAllTimeframes:
    def test_saves_csv_for_each_timeframe(self, tmp_data_dir: Path) -> None:
        now_ms = int(time.time() * 1000)
        candles_m1 = [_make_candle(now_ms - i * 60_000, 100.0) for i in range(5)]
        candles_m5 = [_make_candle(now_ms - i * 300_000, 100.0) for i in range(5)]
        candles_h1 = [_make_candle(now_ms - i * 3_600_000, 100.0) for i in range(5)]

        call_count = [0]
        payloads = [candles_m1, candles_m5, candles_h1]

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        def fake_post(*args, **kwargs):
            mock_resp.json.return_value = payloads[call_count[0] % len(payloads)]
            call_count[0] += 1
            return mock_resp

        session = MagicMock()
        session.post.side_effect = fake_post

        fetcher = HyperLiquidFetcher("EURUSD", data_dir=tmp_data_dir, session=session)
        result = fetcher.fetch_all_timeframes(("M1", "M5", "H1"))

        assert set(result.keys()) == {"M1", "M5", "H1"}
        csv_files = list(tmp_data_dir.glob("EURUSD_*.csv"))
        assert len(csv_files) == 3


class TestCandleToBar:
    def test_field_mapping(self) -> None:
        fetcher = HyperLiquidFetcher("XRPUSD")
        candle = {
            "t": "1700000000000",
            "o": "0.55",
            "h": "0.57",
            "l": "0.53",
            "c": "0.56",
            "v": "12345.6",
        }
        bar = fetcher._candle_to_bar(candle, "M1")
        assert bar.symbol == "XRPUSD"
        assert bar.timeframe == "M1"
        assert bar.open == pytest.approx(0.55)
        assert bar.high == pytest.approx(0.57)
        assert bar.low == pytest.approx(0.53)
        assert bar.close == pytest.approx(0.56)
        assert bar.volume == pytest.approx(12345.6)
        assert bar.timestamp.tzinfo is not None
