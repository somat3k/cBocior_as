"""
src/data/ctrader_client.py — cTrader Open API client wrapper.

Uses the official `ctrader-open-api` SDK (Twisted-based TCP/Protobuf).
Provides async data subscription and historical trendbar fetching.

Protobuf message references:
  https://help.ctrader.com/open-api/python-SDK/python-sdk-index/
  https://github.com/spotware/OpenApiPy
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# cTrader period IDs (ProtoOATrendbarPeriod enum values)
# ---------------------------------------------------------------------------
PERIOD_M1: int = 1
PERIOD_M5: int = 5
PERIOD_M15: int = 15
PERIOD_H1: int = 60
PERIOD_H4: int = 240
PERIOD_D1: int = 1440

TF_TO_PERIOD: dict[str, int] = {
    "M1": PERIOD_M1,
    "M5": PERIOD_M5,
    "M15": PERIOD_M15,
    "H1": PERIOD_H1,
    "H4": PERIOD_H4,
    "D1": PERIOD_D1,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class OHLCVBar:
    """A single OHLCV candle bar."""
    symbol: str
    timeframe: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass
class ConnectionState:
    connected: bool = False
    authenticated: bool = False
    account_authorized: bool = False
    last_heartbeat: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# CTraderClient
# ---------------------------------------------------------------------------

class CTraderClient:
    """
    Wrapper around the cTrader Open API Python SDK.

    Usage (standalone — the SDK uses Twisted reactor):

        client = CTraderClient(
            client_id=..., client_secret=...,
            access_token=..., account_id=...,
            host=..., port=...,
        )
        client.connect()  # blocks until disconnected

    Usage (callback-driven):

        client.on_bar_callback = my_handler
        client.connect()
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        access_token: str,
        account_id: int,
        host: str,
        port: int,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = access_token
        self.account_id = account_id
        self.host = host
        self.port = port

        self.state = ConnectionState()
        self._client: Any = None  # ctrader_open_api.Client
        self._symbol_map: dict[str, int] = {}  # name → symbolId
        self._pending_bars: dict[str, list[OHLCVBar]] = {}

        # Callbacks — set by consumers
        self.on_bar_callback: Callable[[OHLCVBar], None] | None = None
        self.on_connected_callback: Callable[[], None] | None = None
        self.on_error_callback: Callable[[Exception], None] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self, install_signal_handlers: bool = True) -> None:
        """Start the Twisted reactor and connect to cTrader servers.

        Parameters
        ----------
        install_signal_handlers : bool
            Pass ``False`` when the reactor is started from a non-main thread
            (e.g., a daemon background thread).  Twisted's default signal
            handler installation raises ``ValueError`` in that case.
        """
        try:
            from ctrader_open_api import Client, TcpProtocol
        except ImportError as exc:
            raise ImportError(
                "ctrader-open-api is not installed. "
                "Run: pip install ctrader-open-api"
            ) from exc

        from twisted.internet import reactor

        self._client = Client(self.host, self.port, TcpProtocol)
        self._client.setConnectedCallback(self._on_connected)
        self._client.setDisconnectedCallback(self._on_disconnected)
        self._client.setMessageReceivedCallback(self._on_message)
        self._client.startService()

        logger.info(
            "cTrader client connecting",
            host=self.host,
            port=self.port,
        )
        reactor.run(installSignalHandlers=install_signal_handlers)

    def subscribe_live_bars(self, symbol: str, timeframe: str) -> None:
        """Subscribe to live trendbar updates for a symbol/timeframe."""
        symbol_id = self._symbol_map.get(symbol)
        if symbol_id is None:
            logger.warning(
                "Symbol not found in map; requesting symbol list first",
                symbol=symbol,
            )
            return
        self._send_subscribe_live_trendbar(symbol_id, timeframe)

    def fetch_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> list[OHLCVBar]:
        """
        Request historical trendbar data and return bars.
        This is a blocking call; it waits up to 30 s for the response.
        """
        symbol_id = self._symbol_map.get(symbol)
        if symbol_id is None:
            logger.error("Symbol not in map", symbol=symbol)
            return []

        self._pending_bars[f"{symbol}_{timeframe}"] = []
        self._send_get_trendbars(symbol_id, symbol, timeframe, count)

        deadline = time.time() + 30
        key = f"{symbol}_{timeframe}"
        while time.time() < deadline:
            bars = self._pending_bars.get(key, [])
            if len(bars) >= count:
                break
            time.sleep(0.1)

        return self._pending_bars.pop(key, [])

    # ------------------------------------------------------------------
    # Internal Twisted callbacks
    # ------------------------------------------------------------------

    def _on_connected(self, client: Any) -> None:
        """Called when TCP connection is established."""
        logger.info("TCP connection established, sending app auth")
        self.state.connected = True
        self._send_app_auth()

    def _on_disconnected(self, client: Any, reason: Any) -> None:
        logger.warning("Disconnected from cTrader", reason=str(reason))
        self.state.connected = False
        self.state.authenticated = False
        self.state.account_authorized = False

    def _on_message(self, client: Any, message: Any) -> None:
        """Route incoming Protobuf messages to handlers."""
        try:
            from ctrader_open_api import Protobuf
            from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
                ProtoOAApplicationAuthRes,
            )
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAAccountAuthRes,
                ProtoOAGetSymbolsListRes,
                ProtoOAGetTrendbarsRes,
                ProtoOASubscribeLiveTrendbarRes,
            )

            pb = Protobuf.extract(message)

            if isinstance(pb, ProtoOAApplicationAuthRes):
                self._handle_app_auth_res(pb)
            elif isinstance(pb, ProtoOAAccountAuthRes):
                self._handle_account_auth_res(pb)
            elif isinstance(pb, ProtoOAGetSymbolsListRes):
                self._handle_symbols_list(pb)
            elif isinstance(pb, ProtoOAGetTrendbarsRes):
                self._handle_trendbars_res(pb)
            elif isinstance(pb, ProtoOASubscribeLiveTrendbarRes):
                logger.debug("Live trendbar subscription confirmed")
            else:
                self._handle_live_trendbar(pb)
        except Exception as exc:
            logger.error("Error processing message", error=str(exc))
            if self.on_error_callback:
                self.on_error_callback(exc)

    # ------------------------------------------------------------------
    # Message senders
    # ------------------------------------------------------------------

    def _send_app_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import (
            ProtoOAApplicationAuthReq,
        )
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.client_id
        req.clientSecret = self.client_secret
        deferred = self._client.send(req)
        deferred.addErrback(lambda f: logger.error("App auth error", error=str(f)))

    def _send_account_auth(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAAccountAuthReq,
        )
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        req.accessToken = self.access_token
        self._client.send(req)

    def _send_get_symbols(self) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAGetSymbolsListReq,
        )
        req = ProtoOAGetSymbolsListReq()
        req.ctidTraderAccountId = self.account_id
        self._client.send(req)

    def _send_subscribe_live_trendbar(
        self, symbol_id: int, timeframe: str
    ) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOASubscribeLiveTrendbarReq,
        )
        period = TF_TO_PERIOD.get(timeframe, PERIOD_M1)
        req = ProtoOASubscribeLiveTrendbarReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.period = period
        self._client.send(req)
        logger.info(
            "Subscribed to live trendbars",
            symbol_id=symbol_id,
            timeframe=timeframe,
        )

    def _send_get_trendbars(
        self,
        symbol_id: int,
        symbol: str,
        timeframe: str,
        count: int,
    ) -> None:
        from ctrader_open_api.messages.OpenApiMessages_pb2 import (
            ProtoOAGetTrendbarsReq,
        )
        period = TF_TO_PERIOD.get(timeframe, PERIOD_M1)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        # Rough estimate: each bar = period minutes
        period_minutes = period
        start_ms = now_ms - count * period_minutes * 60 * 1000

        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.symbolId = symbol_id
        req.period = period
        req.fromTimestamp = start_ms
        req.toTimestamp = now_ms
        req.count = count
        self._client.send(req)

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    def _handle_app_auth_res(self, pb: Any) -> None:
        logger.info("Application authenticated with cTrader")
        self.state.authenticated = True
        self._send_account_auth()

    def _handle_account_auth_res(self, pb: Any) -> None:
        logger.info("Account authorised", account_id=self.account_id)
        self.state.account_authorized = True
        self._send_get_symbols()
        # on_connected_callback is fired from _handle_symbols_list, once the
        # symbol map is populated and subscriptions / fetches can succeed.

    def _handle_symbols_list(self, pb: Any) -> None:
        for sym in pb.symbol:
            self._symbol_map[sym.symbolName] = sym.symbolId
        logger.info(
            "Symbol map loaded", total_symbols=len(self._symbol_map)
        )
        # Fire the connected callback now that the symbol map is ready,
        # so callers can safely call subscribe_live_bars / fetch_historical_bars.
        if self.on_connected_callback:
            self.on_connected_callback()

    def _handle_trendbars_res(self, pb: Any) -> None:
        symbol_id = pb.symbolId
        timeframe_period = pb.period

        # Reverse map: period → timeframe string
        period_to_tf = {v: k for k, v in TF_TO_PERIOD.items()}
        timeframe = period_to_tf.get(timeframe_period, str(timeframe_period))

        # Reverse map: symbol_id → symbol name
        symbol_name = next(
            (k for k, v in self._symbol_map.items() if v == symbol_id),
            str(symbol_id),
        )

        key = f"{symbol_name}_{timeframe}"
        bars: list[OHLCVBar] = self._pending_bars.get(key, [])

        for tb in pb.trendbar:
            ts = datetime.fromtimestamp(
                tb.utcTimestampInMinutes * 60, tz=timezone.utc
            )
            bar = OHLCVBar(
                symbol=symbol_name,
                timeframe=timeframe,
                timestamp=ts,
                open=tb.low + tb.deltaOpen,
                high=tb.low + tb.deltaHigh,
                low=tb.low,
                close=tb.low + tb.deltaClose,
                volume=tb.volume,
            )
            bars.append(bar)

        self._pending_bars[key] = bars
        logger.debug(
            "Received historical bars",
            symbol=symbol_name,
            timeframe=timeframe,
            count=len(bars),
        )

    def _handle_live_trendbar(self, pb: Any) -> None:
        """Handle live trendbar updates (ProtoOASpotEvent or similar)."""
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOALiveTrendbarEvent,
            )
            if not isinstance(pb, ProtoOALiveTrendbarEvent):
                return
        except ImportError:
            return

        symbol_id = pb.symbolId
        symbol_name = next(
            (k for k, v in self._symbol_map.items() if v == symbol_id),
            str(symbol_id),
        )
        period_to_tf = {v: k for k, v in TF_TO_PERIOD.items()}
        timeframe = period_to_tf.get(pb.period, str(pb.period))

        tb = pb.trendbar
        ts = datetime.fromtimestamp(
            tb.utcTimestampInMinutes * 60, tz=timezone.utc
        )
        bar = OHLCVBar(
            symbol=symbol_name,
            timeframe=timeframe,
            timestamp=ts,
            open=tb.low + tb.deltaOpen,
            high=tb.low + tb.deltaHigh,
            low=tb.low,
            close=tb.low + tb.deltaClose,
            volume=tb.volume,
        )

        if self.on_bar_callback:
            self.on_bar_callback(bar)
