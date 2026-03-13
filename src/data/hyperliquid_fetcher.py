"""
src/data/hyperliquid_fetcher.py — HyperLiquid OHLCV fallback data fetcher.

Used when the primary cTrader / cAlgo feed is unavailable.  Fetches candle
(OHLCV) data from the public HyperLiquid REST API and returns a list of
:class:`~src.data.ctrader_client.OHLCVBar` objects so it is a drop-in
replacement for :meth:`CTraderClient.fetch_historical_bars`.

HyperLiquid candle API reference:
  POST https://api.hyperliquid.xyz/info
  body: {"type": "candleSnapshot", "req": {"coin": "<COIN>", "interval": "<INTERVAL>", "startTime": <ms>, "endTime": <ms>}}

Supported intervals: 1m, 5m, 15m, 1h, 4h, 1d
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from constants import (
    CSV_TEMPLATE,
    DATA_DIR,
    HYPERLIQUID_SYMBOL_MAP,
    REDIS_CACHE_TTL_SECONDS,
    SUPPORTED_TIMEFRAMES,
    TF_1H,
    TF_1M,
    TF_5M,
    TRAIN_1H_TRADES,
    TRAIN_1M_TRADES,
    TRAIN_5M_TRADES,
)
from src.data.ctrader_client import OHLCVBar
from src.utils.cache import cache_get_json, cache_set_json, get_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)

HYPERLIQUID_BASE_URL: str = "https://api.hyperliquid.xyz/info"

# Map internal timeframe codes → HyperLiquid interval strings
_TF_TO_HL_INTERVAL: dict[str, str] = {
    TF_1M: "1m",
    TF_5M: "5m",
    TF_1H: "1h",
    # Extended mappings for future use
    "M15": "15m",
    "H4": "4h",
    "D1": "1d",
}

# Trades per timeframe (mirrors DataFetcher defaults)
_TF_TRADE_COUNT: dict[str, int] = {
    TF_1M: TRAIN_1M_TRADES,
    TF_5M: TRAIN_5M_TRADES,
    TF_1H: TRAIN_1H_TRADES,
}

# Minutes per bar for each timeframe (used to compute startTime)
_TF_MINUTES: dict[str, int] = {
    TF_1M: 1,
    TF_5M: 5,
    TF_1H: 60,
    "M15": 15,
    "H4": 240,
    "D1": 1440,
}


class HyperLiquidFetcher:
    """
    Fetches OHLCV candle data from the public HyperLiquid REST API.

    Parameters
    ----------
    symbol : str
        Internal trading symbol (e.g. ``"BTCUSD"``).  The class uses
        :data:`~constants.HYPERLIQUID_SYMBOL_MAP` to translate to the
        HyperLiquid coin name.
    data_dir : Path
        Directory where CSV fallback files are written.
    session : requests.Session, optional
        Allows injection of a mock session in tests.
    """

    def __init__(
        self,
        symbol: str,
        data_dir: Path = DATA_DIR,
        session: requests.Session | None = None,
    ) -> None:
        self.symbol = symbol
        self.hl_coin = HYPERLIQUID_SYMBOL_MAP.get(symbol, symbol)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._session = session or requests.Session()
        self._cache = get_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_historical_bars(
        self,
        timeframe: str,
        count: int | None = None,
    ) -> list[OHLCVBar]:
        """
        Fetch historical OHLCV bars from HyperLiquid.

        Parameters
        ----------
        timeframe : str
            One of the :data:`~constants.SUPPORTED_TIMEFRAMES` codes or an
            extended code (``"M15"``, ``"H4"``, ``"D1"``).
        count : int, optional
            Number of bars to fetch.  Defaults to the training bar count for
            the given timeframe defined in :mod:`constants`.

        Returns
        -------
        list[OHLCVBar]
            Candle bars sorted oldest-first.  Returns an empty list on error.
        """
        if count is None:
            count = _TF_TRADE_COUNT.get(timeframe, 500)

        cache_key = f"hl:{self.symbol}:{timeframe}:{count}"
        cached = cache_get_json(self._cache, cache_key)
        if cached is not None:
            logger.debug(
                "HyperLiquid bars served from cache",
                symbol=self.symbol,
                timeframe=timeframe,
                count=len(cached),
            )
            return [self._dict_to_bar(d) for d in cached]

        interval = _TF_TO_HL_INTERVAL.get(timeframe)
        if interval is None:
            logger.error(
                "Unsupported timeframe for HyperLiquid", timeframe=timeframe
            )
            return []

        bars = self._fetch_from_api(interval, count, timeframe)
        if bars:
            bars_serialisable = []
            for b in bars:
                d = b.to_dict()
                bars_serialisable.append(d)
            cache_set_json(
                self._cache,
                cache_key,
                bars_serialisable,
                ttl=REDIS_CACHE_TTL_SECONDS,
            )
        return bars

    def fetch_all_timeframes(
        self,
        timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch bars for multiple timeframes and save each to CSV.

        Returns
        -------
        dict mapping timeframe → OHLCV DataFrame
        """
        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            count = _TF_TRADE_COUNT.get(tf, 500)
            logger.info(
                "HyperLiquid fallback: fetching bars",
                symbol=self.symbol,
                timeframe=tf,
                count=count,
            )
            bars = self.fetch_historical_bars(tf, count)
            if not bars:
                logger.warning(
                    "HyperLiquid returned no bars",
                    symbol=self.symbol,
                    timeframe=tf,
                )
                continue
            df = self._bars_to_dataframe(bars)
            csv_path = self._csv_path(tf)
            self._save_csv(df, csv_path)
            result[tf] = df
            logger.info(
                "HyperLiquid data saved",
                symbol=self.symbol,
                timeframe=tf,
                rows=len(df),
                path=str(csv_path),
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_from_api(
        self, interval: str, count: int, timeframe: str
    ) -> list[OHLCVBar]:
        """Call the HyperLiquid candle API and return OHLCVBar list."""
        now_ms = int(time.time() * 1000)
        bar_minutes = _TF_MINUTES.get(timeframe, 1)
        start_ms = now_ms - count * bar_minutes * 60 * 1000

        payload: dict[str, Any] = {
            "type": "candleSnapshot",
            "req": {
                "coin": self.hl_coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": now_ms,
            },
        }

        try:
            resp = self._session.post(
                HYPERLIQUID_BASE_URL,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            raw: list[dict[str, Any]] = resp.json()
        except requests.RequestException as exc:
            logger.error(
                "HyperLiquid API request failed",
                symbol=self.symbol,
                timeframe=timeframe,
                error=str(exc),
            )
            return []
        except Exception as exc:
            logger.error(
                "HyperLiquid unexpected error",
                symbol=self.symbol,
                timeframe=timeframe,
                error=str(exc),
            )
            return []

        bars: list[OHLCVBar] = []
        for candle in raw:
            try:
                bar = self._candle_to_bar(candle, timeframe)
                bars.append(bar)
            except Exception as exc:
                logger.warning(
                    "HyperLiquid candle parse error", error=str(exc)
                )
                continue

        bars.sort(key=lambda b: b.timestamp)
        logger.debug(
            "HyperLiquid bars fetched",
            symbol=self.symbol,
            timeframe=timeframe,
            count=len(bars),
        )
        return bars

    def _candle_to_bar(self, candle: dict[str, Any], timeframe: str) -> OHLCVBar:
        """
        Parse a HyperLiquid candle dict into an OHLCVBar.

        HyperLiquid candle fields (as of 2024 API):
          t  → open time (ms epoch)
          o  → open price (string)
          h  → high price (string)
          l  → low price (string)
          c  → close price (string)
          v  → volume (string)
        """
        ts = datetime.fromtimestamp(int(candle["t"]) / 1000, tz=timezone.utc)
        return OHLCVBar(
            symbol=self.symbol,
            timeframe=timeframe,
            timestamp=ts,
            open=float(candle["o"]),
            high=float(candle["h"]),
            low=float(candle["l"]),
            close=float(candle["c"]),
            volume=float(candle["v"]),
        )

    @staticmethod
    def _dict_to_bar(d: dict[str, Any]) -> OHLCVBar:
        """Reconstruct an OHLCVBar from its to_dict() representation."""
        return OHLCVBar(
            symbol=d["symbol"],
            timeframe=d["timeframe"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            open=float(d["open"]),
            high=float(d["high"]),
            low=float(d["low"]),
            close=float(d["close"]),
            volume=float(d["volume"]),
        )

    def _csv_path(self, timeframe: str) -> Path:
        from datetime import date

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

    @staticmethod
    def _bars_to_dataframe(bars: list[OHLCVBar]) -> pd.DataFrame:
        data = [b.to_dict() for b in bars]
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
