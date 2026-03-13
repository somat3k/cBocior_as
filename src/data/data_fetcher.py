"""
src/data/data_fetcher.py — OHLCV data fetcher and CSV exporter.

Fetches historical and live data from cTrader, persists to CSV files
in DATA_DIR with the naming convention defined in constants.py.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from constants import (
    CSV_TEMPLATE,
    DATA_DIR,
    SUPPORTED_TIMEFRAMES,
    TF_1H,
    TF_1M,
    TF_5M,
    TRAIN_1H_TRADES,
    TRAIN_1M_TRADES,
    TRAIN_5M_TRADES,
)
from src.data.ctrader_client import CTraderClient, OHLCVBar
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Trades per timeframe
_TF_TRADE_COUNT: dict[str, int] = {
    TF_1M: TRAIN_1M_TRADES,
    TF_5M: TRAIN_5M_TRADES,
    TF_1H: TRAIN_1H_TRADES,
}


class DataFetcher:
    """
    Fetches OHLCV data from cTrader and exports it to CSV.

    Can be used in two modes:
    1. **Historical fetch** — download N bars for training data.
    2. **Live subscription** — continuously append new bars to in-memory
       buffers and periodically flush to CSV.
    """

    def __init__(
        self,
        client: CTraderClient,
        symbol: str,
        data_dir: Path = DATA_DIR,
    ) -> None:
        self.client = client
        self.symbol = symbol
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffer: timeframe → list of bar dicts
        self._buffers: dict[str, list[dict[str, Any]]] = {
            tf: [] for tf in SUPPORTED_TIMEFRAMES
        }

    # ------------------------------------------------------------------
    # Historical fetch
    # ------------------------------------------------------------------

    def fetch_all_timeframes(
        self,
        timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch historical bars for each timeframe and return DataFrames.
        Also saves each timeframe to CSV.
        """
        result: dict[str, pd.DataFrame] = {}

        for tf in timeframes:
            count = _TF_TRADE_COUNT.get(tf, 500)
            logger.info(
                "Fetching historical bars",
                symbol=self.symbol,
                timeframe=tf,
                count=count,
            )
            bars = self.client.fetch_historical_bars(
                symbol=self.symbol,
                timeframe=tf,
                count=count,
            )
            if not bars:
                logger.warning(
                    "No bars received",
                    symbol=self.symbol,
                    timeframe=tf,
                )
                continue

            df = self._bars_to_dataframe(bars)
            csv_path = self._csv_path(tf)
            self._save_csv(df, csv_path)
            result[tf] = df
            logger.info(
                "Saved historical data",
                symbol=self.symbol,
                timeframe=tf,
                rows=len(df),
                path=str(csv_path),
            )

        return result

    # ------------------------------------------------------------------
    # Live subscription
    # ------------------------------------------------------------------

    def subscribe_live(
        self,
        timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    ) -> None:
        """Subscribe to live bars for all requested timeframes."""
        self.client.on_bar_callback = self._on_live_bar
        for tf in timeframes:
            self.client.subscribe_live_bars(self.symbol, tf)

    def _on_live_bar(self, bar: OHLCVBar) -> None:
        """Append a live bar to the in-memory buffer and flush to CSV."""
        if bar.symbol != self.symbol:
            return
        tf = bar.timeframe
        self._buffers.setdefault(tf, []).append(bar.to_dict())
        logger.debug(
            "Live bar received",
            symbol=self.symbol,
            timeframe=tf,
            close=bar.close,
        )

    def flush_to_csv(self, timeframe: str) -> Path | None:
        """Flush the in-memory buffer for a timeframe to CSV."""
        buf = self._buffers.get(timeframe, [])
        if not buf:
            return None
        df = pd.DataFrame(buf)
        df = self._standardise_df(df)
        csv_path = self._csv_path(timeframe)
        if csv_path.exists():
            existing = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df = pd.concat([existing, df]).drop_duplicates(
                subset="timestamp"
            ).sort_values("timestamp")
        self._save_csv(df, csv_path)
        self._buffers[timeframe] = []
        return csv_path

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    def load_csv(self, timeframe: str) -> pd.DataFrame:
        """Load a previously saved CSV for a given timeframe."""
        csv_path = self._csv_path(timeframe)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}. "
                "Run fetch_all_timeframes() first."
            )
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return self._standardise_df(df)

    def _csv_path(self, timeframe: str) -> Path:
        fname = CSV_TEMPLATE.format(
            symbol=self.symbol,
            timeframe=timeframe,
            date=date.today().isoformat(),
        )
        return self.data_dir / fname

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_dataframe(bars: list[OHLCVBar]) -> pd.DataFrame:
        data = [b.to_dict() for b in bars]
        df = pd.DataFrame(data)
        return DataFetcher._standardise_df(df)

    @staticmethod
    def _standardise_df(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure consistent dtypes and column order."""
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
