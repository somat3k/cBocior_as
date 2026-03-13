"""
src/trading/execution.py — Trade execution via cTrader Open API.

Translates DecisionEngine decisions into cTrader order messages and
manages open position tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from constants import (
    TRADING_STOP_LOSS_PIPS,
    TRADING_TAKE_PROFIT_PIPS,
    TRADING_VOLUME,
)
from src.utils.logger import get_logger
from src.utils.payload import TradingAction, TradingPayload

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents an open position."""
    order_id: str
    symbol: str
    action: TradingAction
    volume: int
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None
    close_price: float | None = None
    pnl: float | None = None


class Execution:
    """
    Manages order submission and position tracking.

    The ``client`` parameter should be a connected ``CTraderClient`` instance.
    When ``dry_run=True`` no actual orders are placed (useful for testing).
    """

    def __init__(self, client: Any = None, dry_run: bool = False) -> None:
        self._client = client
        self._dry_run = dry_run
        self._open_positions: dict[str, Position] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        payload: TradingPayload,
        volume: int = TRADING_VOLUME,
        stop_loss_pips: float = TRADING_STOP_LOSS_PIPS,
        take_profit_pips: float = TRADING_TAKE_PROFIT_PIPS,
        current_price: float = 0.0,
    ) -> Position | None:
        """
        Execute the trade decision in the payload.

        Parameters
        ----------
        payload : TradingPayload
            Decision from DecisionEngine.
        volume : int
            Volume in cTrader centilots (lots × 100).  Default from
            ``TRADING_VOLUME`` constant.  E.g. ``1`` = 0.01 lot (micro-lot).
        stop_loss_pips : float
            Stop-loss distance in pips.
        take_profit_pips : float
            Take-profit distance in pips.
        current_price : float
            Live bid/ask mid-price used to compute absolute SL/TP levels.
            When ``0.0`` the order is sent without pre-computed SL/TP and
            protection levels must be attached after the fill price is known.

        Returns the new Position or None (on HOLD).
        """
        if payload.action == TradingAction.HOLD:
            logger.debug("HOLD — no order placed")
            return None

        symbol = payload.symbol
        action = payload.action

        # Compute SL/TP absolute prices when a live price is available.
        # If current_price is unknown (0.0) the protection levels are omitted
        # and must be set via a separate modify-position request after fill.
        pip = 0.0001  # for 4/5-decimal FX pairs
        if current_price > 0.0:
            if action == TradingAction.BUY:
                sl = current_price - stop_loss_pips * pip
                tp = current_price + take_profit_pips * pip
            else:
                sl = current_price + stop_loss_pips * pip
                tp = current_price - take_profit_pips * pip
        else:
            sl = 0.0
            tp = 0.0
            logger.warning(
                "current_price is 0.0 — SL/TP omitted from order; "
                "attach protection levels after fill",
                symbol=symbol,
                action=action.value,
            )

        order_id = f"{symbol}_{action}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        position = Position(
            order_id=order_id,
            symbol=symbol,
            action=action,
            volume=volume,
            entry_price=current_price,
            stop_loss=sl,
            take_profit=tp,
        )

        if self._dry_run:
            log_kwargs: dict = {
                "order_id": order_id,
                "action": action.value,
                "symbol": symbol,
                "volume": volume,
            }
            if sl:
                log_kwargs["sl"] = sl
            if tp:
                log_kwargs["tp"] = tp
            if not sl and not tp:
                log_kwargs["protection"] = "pending_post_fill"
            logger.info("DRY RUN — order not placed", **log_kwargs)
        else:
            self._send_market_order(symbol, action, volume, sl, tp, order_id)

        self._open_positions[order_id] = position
        return position

    def close_position(self, order_id: str, close_price: float) -> None:
        """Mark a position as closed and compute PnL."""
        pos = self._open_positions.get(order_id)
        if not pos:
            logger.warning("Position not found", order_id=order_id)
            return

        pos.closed_at = datetime.now(timezone.utc)
        pos.close_price = close_price

        # Simple PnL estimate (ignoring spread)
        pip = 0.0001
        if pos.action == TradingAction.BUY:
            pip_gain = (close_price - pos.entry_price) / pip
        else:
            pip_gain = (pos.entry_price - close_price) / pip

        pos.pnl = pip_gain * pos.volume * pip  # approximate USD
        del self._open_positions[order_id]

        logger.info(
            "Position closed",
            order_id=order_id,
            pnl=round(pos.pnl, 2),
        )

    @property
    def open_positions(self) -> dict[str, Position]:
        return dict(self._open_positions)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _send_market_order(
        self,
        symbol: str,
        action: TradingAction,
        volume: int,
        stop_loss: float,
        take_profit: float,
        order_id: str,
    ) -> None:
        """Send a market order via the cTrader Open API client."""
        if self._client is None:
            logger.error("No cTrader client attached to Execution")
            return

        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOANewOrderReq,
                ProtoOAOrderType,
                ProtoOATradeSide,
            )

            req = ProtoOANewOrderReq()
            req.ctidTraderAccountId = self._client.account_id
            req.symbolId = self._client._symbol_map.get(symbol, 0)
            req.orderType = ProtoOAOrderType.MARKET
            req.tradeSide = (
                ProtoOATradeSide.BUY
                if action == TradingAction.BUY
                else ProtoOATradeSide.SELL
            )
            req.volume = volume  # already in centilots (lots × 100) as per TRADING_VOLUME constant
            if stop_loss:
                req.stopLoss = stop_loss
            if take_profit:
                req.takeProfit = take_profit
            req.comment = order_id[:31]  # max 31 chars

            self._client._client.send(req)
            logger.info(
                "Market order sent",
                symbol=symbol,
                action=action.value,
                volume=volume,
                order_id=order_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to send market order",
                error=str(exc),
                order_id=order_id,
            )
            raise
