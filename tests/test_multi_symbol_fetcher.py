"""
tests/test_multi_symbol_fetcher.py — Unit tests for MultiSymbolFetcher.

All cTrader and HyperLiquid network calls are mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.ctrader_client import OHLCVBar
from src.data.multi_symbol_fetcher import MultiSymbolFetcher
from src.utils.cache import reset_cache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bar(symbol: str, timeframe: str, price: float = 100.0) -> OHLCVBar:
    return OHLCVBar(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        open=price,
        high=price + 1,
        low=price - 1,
        close=price + 0.5,
        volume=1000.0,
    )


def _make_df(symbol: str, timeframe: str, n: int = 5) -> pd.DataFrame:
    bars = [_make_bar(symbol, timeframe, 100.0 + i) for i in range(n)]
    data = [b.to_dict() for b in bars]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Tests — cTrader primary path
# ---------------------------------------------------------------------------

class TestCTraderPrimaryPath:
    def test_uses_ctrader_when_available(self, tmp_data_dir: Path) -> None:
        mock_client = MagicMock()
        mock_client.fetch_historical_bars.return_value = [
            _make_bar("EURUSD", "M1")
        ]

        fetcher = MultiSymbolFetcher(
            ctrader_client=mock_client,
            data_dir=tmp_data_dir,
            symbols=("EURUSD",),
            timeframes=("M1",),
        )
        result = fetcher.fetch_all()

        assert "EURUSD" in result
        assert "M1" in result["EURUSD"]
        mock_client.fetch_historical_bars.assert_called_once()

    def test_falls_back_to_hyperliquid_when_ctrader_returns_empty(
        self, tmp_data_dir: Path
    ) -> None:
        mock_client = MagicMock()
        mock_client.fetch_historical_bars.return_value = []

        with patch(
            "src.data.multi_symbol_fetcher.HyperLiquidFetcher"
        ) as MockHL:
            mock_hl_instance = MagicMock()
            mock_hl_instance.fetch_historical_bars.return_value = [
                _make_bar("EURUSD", "M1")
            ]
            MockHL.return_value = mock_hl_instance

            fetcher = MultiSymbolFetcher(
                ctrader_client=mock_client,
                data_dir=tmp_data_dir,
                symbols=("EURUSD",),
                timeframes=("M1",),
            )
            result = fetcher.fetch_all()

        assert "EURUSD" in result
        mock_hl_instance.fetch_historical_bars.assert_called_once()

    def test_falls_back_to_hyperliquid_when_ctrader_raises(
        self, tmp_data_dir: Path
    ) -> None:
        mock_client = MagicMock()
        mock_client.fetch_historical_bars.side_effect = Exception("TCP error")

        with patch(
            "src.data.multi_symbol_fetcher.HyperLiquidFetcher"
        ) as MockHL:
            mock_hl_instance = MagicMock()
            mock_hl_instance.fetch_historical_bars.return_value = [
                _make_bar("BTCUSD", "M1")
            ]
            MockHL.return_value = mock_hl_instance

            fetcher = MultiSymbolFetcher(
                ctrader_client=mock_client,
                data_dir=tmp_data_dir,
                symbols=("BTCUSD",),
                timeframes=("M1",),
            )
            result = fetcher.fetch_all()

        assert "BTCUSD" in result


# ---------------------------------------------------------------------------
# Tests — no cTrader client (HyperLiquid direct)
# ---------------------------------------------------------------------------

class TestHyperLiquidDirectPath:
    def test_uses_hyperliquid_when_no_ctrader_client(
        self, tmp_data_dir: Path
    ) -> None:
        with patch(
            "src.data.multi_symbol_fetcher.HyperLiquidFetcher"
        ) as MockHL:
            mock_hl_instance = MagicMock()
            mock_hl_instance.fetch_historical_bars.return_value = [
                _make_bar("SOLUSD", "M1")
            ]
            MockHL.return_value = mock_hl_instance

            fetcher = MultiSymbolFetcher(
                ctrader_client=None,
                data_dir=tmp_data_dir,
                symbols=("SOLUSD",),
                timeframes=("M1",),
            )
            result = fetcher.fetch_all()

        assert "SOLUSD" in result
        assert "M1" in result["SOLUSD"]


# ---------------------------------------------------------------------------
# Tests — CSV fallback
# ---------------------------------------------------------------------------

class TestCsvFallback:
    def test_loads_csv_when_api_sources_fail(self, tmp_data_dir: Path) -> None:
        # Pre-write a CSV file
        df = _make_df("XRPUSD", "M1", n=5)
        from datetime import date

        from constants import CSV_TEMPLATE

        csv_path = tmp_data_dir / CSV_TEMPLATE.format(
            symbol="XRPUSD",
            timeframe="M1",
            date=date.today().isoformat(),
        )
        df.to_csv(csv_path, index=False)

        with patch(
            "src.data.multi_symbol_fetcher.HyperLiquidFetcher"
        ) as MockHL:
            mock_hl_instance = MagicMock()
            mock_hl_instance.fetch_historical_bars.return_value = []
            MockHL.return_value = mock_hl_instance

            fetcher = MultiSymbolFetcher(
                ctrader_client=None,
                data_dir=tmp_data_dir,
                symbols=("XRPUSD",),
                timeframes=("M1",),
            )
            result = fetcher.fetch_all()

        assert "XRPUSD" in result
        assert len(result["XRPUSD"]["M1"]) == 5

    def test_returns_empty_when_all_sources_fail(
        self, tmp_data_dir: Path
    ) -> None:
        with patch(
            "src.data.multi_symbol_fetcher.HyperLiquidFetcher"
        ) as MockHL:
            mock_hl_instance = MagicMock()
            mock_hl_instance.fetch_historical_bars.return_value = []
            MockHL.return_value = mock_hl_instance

            fetcher = MultiSymbolFetcher(
                ctrader_client=None,
                data_dir=tmp_data_dir,
                symbols=("ADAUSD",),
                timeframes=("M1",),
            )
            result = fetcher.fetch_all()

        # Symbol present with no timeframes, or absent
        assert result.get("ADAUSD", {}) == {}


# ---------------------------------------------------------------------------
# Tests — caching
# ---------------------------------------------------------------------------

class TestCaching:
    def test_second_fetch_uses_cache(self, tmp_data_dir: Path) -> None:
        mock_client = MagicMock()
        mock_client.fetch_historical_bars.return_value = [
            _make_bar("EURUSD", "M1")
        ]

        fetcher = MultiSymbolFetcher(
            ctrader_client=mock_client,
            data_dir=tmp_data_dir,
            symbols=("EURUSD",),
            timeframes=("M1",),
        )
        fetcher.fetch_all()
        fetcher.fetch_all()

        # cTrader should only be called on the first fetch; second uses cache
        assert mock_client.fetch_historical_bars.call_count == 1


# ---------------------------------------------------------------------------
# Tests — multi-symbol output structure
# ---------------------------------------------------------------------------

class TestOutputStructure:
    def test_result_keys_are_symbols(self, tmp_data_dir: Path) -> None:
        symbols = ("EURUSD", "GBPUSD")

        mock_client = MagicMock()
        mock_client.fetch_historical_bars.return_value = [
            _make_bar("EURUSD", "M1"),
            _make_bar("GBPUSD", "M1"),
        ]

        # Side-effect: return the bar matching the called symbol
        def _side_effect(symbol, timeframe, count):
            return [_make_bar(symbol, timeframe)]

        mock_client.fetch_historical_bars.side_effect = _side_effect

        fetcher = MultiSymbolFetcher(
            ctrader_client=mock_client,
            data_dir=tmp_data_dir,
            symbols=symbols,
            timeframes=("M1",),
        )
        result = fetcher.fetch_all()

        assert set(result.keys()) == set(symbols)
        for sym in symbols:
            assert "M1" in result[sym]
            assert isinstance(result[sym]["M1"], pd.DataFrame)
