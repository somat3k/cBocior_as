"""
src/trading/risk_manager.py — Risk management enforcer.

Checks every candidate trade against configured risk limits before
allowing execution.  All checks return a (allowed: bool, reason: str)
tuple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from constants import (
    RISK_DAILY_LOSS_LIMIT_USD,
    RISK_MAX_DRAWDOWN_PCT,
    RISK_MAX_POSITION_SIZE,
    TRADING_MAX_SPREAD_PIPS,
    TRADING_STOP_LOSS_PIPS,
    TRADING_TAKE_PROFIT_PIPS,
    TRADING_VOLUME,
)
from src.utils.logger import get_logger
from src.utils.payload import RiskFlags, TradingAction, TradingPayload

logger = get_logger(__name__)


@dataclass
class AccountState:
    balance: float = 0.0
    equity: float = 0.0
    margin: float = 0.0
    unrealised_pnl: float = 0.0
    daily_pnl: float = 0.0
    peak_balance: float = 0.0
    open_positions: list[dict[str, Any]] = field(default_factory=list)


class RiskManager:
    """
    Evaluates risk before trade execution.

    Methods return (allowed: bool, reason: str).
    """

    def __init__(self) -> None:
        self.account = AccountState()

    def update_account(self, state: dict[str, Any]) -> None:
        """Update internal account state from cTrader account data."""
        self.account.balance = float(state.get("balance", self.account.balance))
        self.account.equity = float(state.get("equity", self.account.equity))
        self.account.margin = float(state.get("margin", self.account.margin))
        self.account.unrealised_pnl = float(
            state.get("unrealisedPnl", self.account.unrealised_pnl)
        )
        self.account.daily_pnl = float(
            state.get("dailyPnl", self.account.daily_pnl)
        )
        if self.account.balance > self.account.peak_balance:
            self.account.peak_balance = self.account.balance

    def evaluate(
        self,
        payload: TradingPayload,
        current_spread_pips: float = 0.0,
    ) -> tuple[bool, str, RiskFlags]:
        """
        Evaluate all risk rules for the proposed trade.

        Returns
        -------
        (allowed, reason, risk_flags)
        """
        flags = RiskFlags()
        reasons: list[str] = []

        # 1. Spread check
        if current_spread_pips > TRADING_MAX_SPREAD_PIPS:
            flags.spread_exceeded = True
            reasons.append(
                f"Spread {current_spread_pips:.1f} > limit {TRADING_MAX_SPREAD_PIPS}"
            )

        # 2. Daily loss limit
        if self.account.daily_pnl < -RISK_DAILY_LOSS_LIMIT_USD:
            flags.daily_loss_limit_approaching = True
            reasons.append(
                f"Daily loss {self.account.daily_pnl:.2f} USD exceeds limit "
                f"−{RISK_DAILY_LOSS_LIMIT_USD}"
            )

        # 3. Max drawdown
        if self.account.peak_balance > 0:
            drawdown_pct = (
                (self.account.peak_balance - self.account.equity)
                / self.account.peak_balance
                * 100
            )
            if drawdown_pct >= RISK_MAX_DRAWDOWN_PCT * 0.9:
                flags.drawdown_warning = True
                reasons.append(
                    f"Drawdown {drawdown_pct:.1f}% approaching limit "
                    f"{RISK_MAX_DRAWDOWN_PCT}%"
                )
            if drawdown_pct >= RISK_MAX_DRAWDOWN_PCT:
                reasons.append(
                    f"Max drawdown {drawdown_pct:.1f}% exceeded — HALT"
                )

        # 4. Low confidence
        if payload.confidence < 0.55:
            flags.low_confidence = True
            reasons.append(
                f"Confidence {payload.confidence:.2f} < 0.55 threshold"
            )

        # 5. HOLD signal — always allowed (it's doing nothing)
        if payload.action == TradingAction.HOLD:
            return True, "HOLD — no trade executed", flags

        # 6. Any blocking flag → deny
        blocking = (
            flags.spread_exceeded
            or flags.daily_loss_limit_approaching
            or flags.drawdown_warning
            or flags.low_confidence
        )
        if blocking:
            reason_str = "; ".join(reasons) if reasons else "Risk limit triggered"
            logger.warning(
                "Trade blocked by risk manager",
                action=payload.action,
                reasons=reasons,
            )
            return False, reason_str, flags

        return True, "Risk checks passed", flags

    def compute_position_size(
        self,
        stop_loss_pips: float = TRADING_STOP_LOSS_PIPS,
        risk_pct: float = 1.0,
    ) -> int:
        """
        Kelly-fraction position sizing.

        Parameters
        ----------
        stop_loss_pips : float
        risk_pct : float  — % of balance to risk per trade

        Returns
        -------
        int  — units to trade
        """
        if self.account.balance <= 0 or stop_loss_pips <= 0:
            return TRADING_VOLUME

        risk_amount = self.account.balance * risk_pct / 100
        # Simplified: assume 1 pip = $0.0001 per unit for major pairs
        pip_value = 0.0001
        size = int(risk_amount / (stop_loss_pips * pip_value))
        size = max(1000, min(size, RISK_MAX_POSITION_SIZE))
        return size
