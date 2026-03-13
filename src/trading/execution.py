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

    Parameters
    ----------
    client : Any, optional
        Connected ``CTraderClient``.
    dry_run : bool
        When ``True`` orders are logged but not sent to the broker.
    account_id : int, optional
        cTrader account ID to trade on.  Defaults to ``client.account_id``
        when not supplied.  Pass explicitly to target a secondary account
        while reusing the same cTrader connection.
    """

    def __init__(
        self,
        client: Any = None,
        dry_run: bool = False,
        account_id: int | None = None,
    ) -> None:
        self._client = client
        self._dry_run = dry_run
        # Resolve the account ID: explicit > client's own > 0 fallback
        if account_id is not None:
            self._account_id = account_id
        elif client is not None:
            self._account_id = getattr(client, "account_id", 0)
        else:
            self._account_id = 0
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
        atr: float = 0.0,
        atr_sl_mult: float = 1.5,
        atr_tp_mult: float = 2.5,
    ) -> "Position | None":
        """
        Execute the trade decision in the payload.

        Parameters
        ----------
        payload : TradingPayload
            Decision from DecisionEngine.
        volume : int
            Volume in cTrader centilots (lots × 100).
        stop_loss_pips : float
            Fixed stop-loss distance in pips.  Overridden by ATR when *atr* > 0.
        take_profit_pips : float
            Fixed take-profit distance in pips.  Overridden by ATR when *atr* > 0.
        current_price : float
            Live bid/ask mid-price for SL/TP calculation.
        atr : float
            Current Average True Range value in price units.  When > 0 the
            SL/TP distances are computed as ATR multiples (H8).
        atr_sl_mult : float
            SL = entry ± atr * atr_sl_mult (default 1.5).
        atr_tp_mult : float
            TP = entry ± atr * atr_tp_mult (default 2.5).

        Returns the new Position or None (on HOLD).
        """
        if payload.action == TradingAction.HOLD:
            logger.debug("HOLD — no order placed")
            return None

        symbol = payload.symbol
        action = payload.action

        pip = 0.0001  # for 4/5-decimal FX pairs
        if current_price > 0.0:
            # H8: ATR-based SL/TP overrides fixed pips when ATR is available
            if atr > 0.0:
                sl_dist = atr * atr_sl_mult
                tp_dist = atr * atr_tp_mult
                logger.debug(
                    "Using ATR-based SL/TP",
                    atr=round(atr, 5),
                    sl_dist=round(sl_dist, 5),
                    tp_dist=round(tp_dist, 5),
                )
            else:
                sl_dist = stop_loss_pips * pip
                tp_dist = take_profit_pips * pip

            if action == TradingAction.BUY:
                sl = current_price - sl_dist
                tp = current_price + tp_dist
            else:
                sl = current_price + sl_dist
                tp = current_price - tp_dist
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
            if atr > 0.0:
                log_kwargs["atr"] = round(atr, 5)
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

    def update_trailing_stops(
        self,
        current_price: float,
        trail_pips: float = TRADING_STOP_LOSS_PIPS,
    ) -> list[str]:
        """
        Adjust stop-loss on all open positions using a trailing stop.  (H9)

        The trailing stop follows the price at a fixed *trail_pips* distance:
        - For BUY positions the SL is raised when *current_price* moves above
          the highest seen close so far by more than *trail_pips*.
        - For SELL positions the SL is lowered when *current_price* moves below
          the lowest seen close so far by more than *trail_pips*.

        Parameters
        ----------
        current_price : float
            Latest mid-price.
        trail_pips : float
            Trailing distance in pips.

        Returns
        -------
        list[str]
            Order IDs of positions whose SL was updated.
        """
        pip = 0.0001
        trail_dist = trail_pips * pip
        updated: list[str] = []

        for order_id, pos in self._open_positions.items():
            if pos.action == TradingAction.BUY:
                new_sl = current_price - trail_dist
                if new_sl > pos.stop_loss:
                    logger.debug(
                        "Trailing stop raised (BUY)",
                        order_id=order_id,
                        old_sl=round(pos.stop_loss, 5),
                        new_sl=round(new_sl, 5),
                        price=round(current_price, 5),
                    )
                    pos.stop_loss = new_sl
                    updated.append(order_id)
                    self._send_modify_sl(order_id, new_sl)
            else:
                new_sl = current_price + trail_dist
                if new_sl < pos.stop_loss or pos.stop_loss == 0.0:
                    logger.debug(
                        "Trailing stop lowered (SELL)",
                        order_id=order_id,
                        old_sl=round(pos.stop_loss, 5),
                        new_sl=round(new_sl, 5),
                        price=round(current_price, 5),
                    )
                    pos.stop_loss = new_sl
                    updated.append(order_id)
                    self._send_modify_sl(order_id, new_sl)

        return updated

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
            req.ctidTraderAccountId = self._account_id
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
                account_id=self._account_id,
            )
        except Exception as exc:
            logger.error(
                "Failed to send market order",
                error=str(exc),
                order_id=order_id,
            )
            raise

    def _send_modify_sl(self, order_id: str, new_sl: float) -> None:
        """
        Send a position-amend request to update the stop-loss price.  (H9)

        In dry-run mode or when no client is attached the call is skipped
        (the in-memory ``Position.stop_loss`` has already been updated by the
        caller so the next fill-check will use the correct level).
        """
        if self._dry_run or self._client is None:
            return
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAAmendPositionSLTPReq,
            )
            req = ProtoOAAmendPositionSLTPReq()
            req.ctidTraderAccountId = self._account_id
            # The order_id is an internal string; the real position ID would
            # be tracked once the broker confirms the fill.  This is a stub
            # that should be wired to the broker-returned positionId.
            req.stopLoss = new_sl
            req.comment = order_id[:31]
            self._client._client.send(req)
            logger.debug(
                "Trailing stop amended",
                order_id=order_id,
                new_sl=round(new_sl, 5),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to send trailing-stop amendment",
                order_id=order_id,
                error=str(exc),
            )
