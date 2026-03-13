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
    ) -> Position | None:
        """
        Execute the trade decision in the payload.

        Returns the new Position or None (on HOLD or dry run).
        """
        if payload.action == TradingAction.HOLD:
            logger.debug("HOLD — no order placed")
            return None

        symbol = payload.symbol
        action = payload.action

        # Estimate entry price from last known mid (or 0.0 as placeholder)
        entry_price = 0.0  # Will be filled by cTrader on execution

        # Compute SL/TP prices (directional)
        pip = 0.0001  # for 5-decimal FX pairs
        if action == TradingAction.BUY:
            sl = entry_price - stop_loss_pips * pip if entry_price else 0.0
            tp = entry_price + take_profit_pips * pip if entry_price else 0.0
        else:
            sl = entry_price + stop_loss_pips * pip if entry_price else 0.0
            tp = entry_price - take_profit_pips * pip if entry_price else 0.0

        order_id = f"{symbol}_{action}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        position = Position(
            order_id=order_id,
            symbol=symbol,
            action=action,
            volume=volume,
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
        )

        if self._dry_run:
            logger.info(
                "DRY RUN — order not placed",
                order_id=order_id,
                action=action.value,
                symbol=symbol,
                volume=volume,
            )
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
            req.volume = volume * 100  # cTrader uses cents of lots
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
