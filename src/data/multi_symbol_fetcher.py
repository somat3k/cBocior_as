"""
src/data/multi_symbol_fetcher.py — Multi-symbol OHLCV data orchestrator.

Fetches historical OHLCV bars for every symbol in :data:`~constants.TRAINING_SYMBOLS`
from the primary cTrader / cAlgo feed.  If that feed fails for a symbol the
module transparently falls back to HyperLiquid.  All results are cached in
Redis (or the in-memory fallback) and written to CSV files.

Typical usage::

    from src.data.multi_symbol_fetcher import MultiSymbolFetcher

    fetcher = MultiSymbolFetcher(ctrader_client=client)
    data = fetcher.fetch_all()
    # data["EURUSD"]["M1"] → pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from constants import (
    CSV_TEMPLATE,
    DATA_DIR,
    REDIS_CACHE_TTL_SECONDS,
    SUPPORTED_TIMEFRAMES,
    TRAIN_1H_TRADES,
    TRAIN_1M_TRADES,
    TRAIN_5M_TRADES,
    TRAINING_SYMBOLS,
    TF_1H,
    TF_1M,
    TF_5M,
)
from src.data.ctrader_client import CTraderClient, OHLCVBar
from src.data.hyperliquid_fetcher import HyperLiquidFetcher
from src.utils.cache import cache_get_json, cache_set_json, get_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

_TF_TRADE_COUNT: dict[str, int] = {
    TF_1M: TRAIN_1M_TRADES,
    TF_5M: TRAIN_5M_TRADES,
    TF_1H: TRAIN_1H_TRADES,
}


class MultiSymbolFetcher:
    """
    Orchestrates OHLCV data collection for all training symbols.

    For each symbol and timeframe the fetch order is:
    1. Redis cache (fast path – avoids redundant API calls within TTL).
    2. cTrader / cAlgo primary feed via :class:`~src.data.ctrader_client.CTraderClient`.
    3. HyperLiquid REST API fallback.
    4. Previously saved CSV file (last resort, no network required).

    Parameters
    ----------
    ctrader_client : CTraderClient, optional
        An already-connected cTrader client.  When ``None`` the cTrader path
        is skipped and HyperLiquid is used directly.
    data_dir : Path
        Root directory for CSV persistence.
    symbols : tuple[str, ...], optional
        Override the default :data:`~constants.TRAINING_SYMBOLS` list.
    timeframes : tuple[str, ...], optional
        Override the default :data:`~constants.SUPPORTED_TIMEFRAMES` list.
    """

    def __init__(
        self,
        ctrader_client: CTraderClient | None = None,
        data_dir: Path = DATA_DIR,
        symbols: tuple[str, ...] = TRAINING_SYMBOLS,
        timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    ) -> None:
        self.ctrader_client = ctrader_client
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = symbols
        self.timeframes = timeframes
        self._cache = get_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_all(self) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Fetch OHLCV data for every symbol and timeframe.

        Returns
        -------
        dict[symbol, dict[timeframe, DataFrame]]
        """
        result: dict[str, dict[str, pd.DataFrame]] = {}
        for symbol in self.symbols:
            logger.info("Fetching symbol", symbol=symbol)
            symbol_data = self.fetch_symbol(symbol)
            if symbol_data:
                result[symbol] = symbol_data
            else:
                logger.warning("No data collected for symbol", symbol=symbol)
        return result

    def fetch_symbol(
        self, symbol: str
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all timeframes for a single *symbol*.

        Returns
        -------
        dict[timeframe, DataFrame]  — may be empty if all sources fail.
        """
        result: dict[str, pd.DataFrame] = {}
        for tf in self.timeframes:
            df = self._fetch_timeframe(symbol, tf)
            if df is not None and not df.empty:
                result[tf] = df
        return result

    # ------------------------------------------------------------------
    # Per-timeframe fetch with fallback chain
    # ------------------------------------------------------------------

    def _fetch_timeframe(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame | None:
        count = _TF_TRADE_COUNT.get(timeframe, 500)
        cache_key = f"ohlcv:{symbol}:{timeframe}:{count}"

        # ── 1. Redis / in-memory cache ───────────────────────────────
        cached = cache_get_json(self._cache, cache_key)
        if cached is not None:
            logger.debug(
                "OHLCV served from cache",
                symbol=symbol,
                timeframe=timeframe,
            )
            return MultiSymbolFetcher._standardise_df(pd.DataFrame(cached))

        # ── 2. cTrader primary feed ───────────────────────────────────
        df = self._try_ctrader(symbol, timeframe, count)

        # ── 3. HyperLiquid fallback ───────────────────────────────────
        if df is None or df.empty:
            logger.info(
                "cTrader unavailable, falling back to HyperLiquid",
                symbol=symbol,
                timeframe=timeframe,
            )
            df = self._try_hyperliquid(symbol, timeframe, count)

        # ── 4. Load from existing CSV (last resort) ───────────────────
        if df is None or df.empty:
            df = self._try_load_csv(symbol, timeframe)
            if df is not None and not df.empty:
                logger.info(
                    "Loaded data from existing CSV",
                    symbol=symbol,
                    timeframe=timeframe,
                )

        if df is None or df.empty:
            logger.error(
                "All data sources exhausted; no data for symbol/timeframe",
                symbol=symbol,
                timeframe=timeframe,
            )
            return None

        # ── Persist to cache and CSV ───────────────────────────────────
        # Convert timestamps to ISO strings during serialisation so the
        # records dict is JSON-serialisable without copying the DataFrame.
        records = [
            {
                k: (v.isoformat() if hasattr(v, "isoformat") else v)
                for k, v in row.items()
            }
            for row in df.to_dict(orient="records")
        ]
        cache_set_json(
            self._cache, cache_key, records, ttl=REDIS_CACHE_TTL_SECONDS
        )
        self._save_csv(df, self._csv_path(symbol, timeframe))

        logger.info(
            "OHLCV data ready",
            symbol=symbol,
            timeframe=timeframe,
            rows=len(df),
        )
        return df

    # ------------------------------------------------------------------
    # Source adapters
    # ------------------------------------------------------------------

    def _try_ctrader(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame | None:
        if self.ctrader_client is None:
            return None
        try:
            bars: list[OHLCVBar] = self.ctrader_client.fetch_historical_bars(
                symbol=symbol,
                timeframe=timeframe,
                count=count,
            )
            if not bars:
                return None
            return self._bars_to_dataframe(bars)
        except Exception as exc:
            logger.warning(
                "cTrader fetch failed",
                symbol=symbol,
                timeframe=timeframe,
                error=str(exc),
            )
            return None

    def _try_hyperliquid(
        self, symbol: str, timeframe: str, count: int
    ) -> pd.DataFrame | None:
        try:
            hl = HyperLiquidFetcher(symbol=symbol, data_dir=self.data_dir)
            bars = hl.fetch_historical_bars(timeframe=timeframe, count=count)
            if not bars:
                return None
            return self._bars_to_dataframe(bars)
        except Exception as exc:
            logger.warning(
                "HyperLiquid fetch failed",
                symbol=symbol,
                timeframe=timeframe,
                error=str(exc),
            )
            return None

    def _try_load_csv(
        self, symbol: str, timeframe: str
    ) -> pd.DataFrame | None:
        csv_path = self._latest_csv_path(symbol, timeframe)
        if csv_path is None or not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            return self._standardise_df(df)
        except Exception as exc:
            logger.warning(
                "CSV load failed",
                path=str(csv_path),
                error=str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _latest_csv_path(self, symbol: str, timeframe: str) -> Path | None:
        """
        Return the path of the most recently saved CSV for *symbol* / *timeframe*.

        Falls back to today's expected path so the write-side still works.
        """
        stem = f"{symbol}_{timeframe}_"
        matches = sorted(
            self.data_dir.glob(f"{stem}*.csv"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        if matches:
            return matches[0]
        # No prior CSV found — return today's path so saves still land correctly
        return self._csv_path(symbol, timeframe)

    def _csv_path(self, symbol: str, timeframe: str) -> Path:
        from datetime import date

        fname = CSV_TEMPLATE.format(
            symbol=symbol,
            timeframe=timeframe,
            date=date.today().isoformat(),
        )
        return self.data_dir / fname

    @staticmethod
    def _save_csv(df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def _bars_to_dataframe(bars: list[OHLCVBar]) -> pd.DataFrame:
        data = [b.to_dict() for b in bars]
        df = pd.DataFrame(data)
        return MultiSymbolFetcher._standardise_df(df)

    @staticmethod
    def _standardise_df(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
