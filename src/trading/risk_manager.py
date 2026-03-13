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

    Parameters
    ----------
    initial_capital : float, optional
        Account's starting balance.  Used for position sizing when the live
        balance has not yet been received from the broker (``account.balance
        == 0``).  Defaults to ``RISK_MAX_POSITION_SIZE`` guard.
    """

    def __init__(self, initial_capital: float = 0.0) -> None:
        self.account = AccountState()
        self._initial_capital = initial_capital

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
        required_margin: float = 0.0,
    ) -> tuple[bool, str, RiskFlags]:
        """
        Evaluate all risk rules for the proposed trade.

        Parameters
        ----------
        payload : TradingPayload
        current_spread_pips : float
        required_margin : float
            Estimated margin required for the trade in account currency.
            When > 0, a margin sufficiency check is performed  (I6).

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

        # 3. Max drawdown + emergency stop
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
            # I10: Emergency stop — drawdown > 2× configured limit
            if drawdown_pct >= RISK_MAX_DRAWDOWN_PCT * 2.0:
                flags.emergency_halt = True
                reasons.append(
                    f"EMERGENCY HALT: drawdown {drawdown_pct:.1f}% > "
                    f"2× limit ({RISK_MAX_DRAWDOWN_PCT * 2.0:.1f}%) — "
                    "all trading suspended"
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

        # 6. Margin check  (I6) — only when equity and required_margin are known
        if required_margin > 0.0 and self.account.equity > 0.0:
            free_margin = self.account.equity - self.account.margin
            if free_margin < required_margin:
                reasons.append(
                    f"Insufficient margin: free={free_margin:.2f}, "
                    f"required={required_margin:.2f}"
                )
                logger.warning(
                    "Trade blocked: insufficient margin",
                    free_margin=round(free_margin, 2),
                    required_margin=round(required_margin, 2),
                )
                return False, "; ".join(reasons), flags

        # 7. Any blocking flag → deny
        blocking = (
            flags.spread_exceeded
            or flags.daily_loss_limit_approaching
            or flags.drawdown_warning
            or flags.low_confidence
            or flags.emergency_halt
        )
        if blocking:
            reason_str = "; ".join(reasons) if reasons else "Risk limit triggered"
            logger.warning(
                "Trade blocked by risk manager",
                action=payload.action,
                reasons=reasons,
                emergency_halt=flags.emergency_halt,
            )
            return False, reason_str, flags

        return True, "Risk checks passed", flags

    def build_risk_report(self) -> dict:
        """
        Build a structured risk report snapshot for inclusion in trade payloads.  (I9)

        Returns a plain dict that can be stored in ``TradingPayload.metadata``
        under the key ``"risk_report"``.
        """
        balance = self.account.balance or self._initial_capital
        peak = self.account.peak_balance or balance
        drawdown_pct = 0.0
        if peak > 0:
            drawdown_pct = (peak - self.account.equity) / peak * 100

        return {
            "balance": round(balance, 2),
            "equity": round(self.account.equity, 2),
            "margin": round(self.account.margin, 2),
            "free_margin": round(self.account.equity - self.account.margin, 2),
            "unrealised_pnl": round(self.account.unrealised_pnl, 2),
            "daily_pnl": round(self.account.daily_pnl, 2),
            "peak_balance": round(peak, 2),
            "drawdown_pct": round(drawdown_pct, 2),
            "open_positions": len(self.account.open_positions),
            "daily_loss_limit_usd": RISK_DAILY_LOSS_LIMIT_USD,
            "max_drawdown_pct": RISK_MAX_DRAWDOWN_PCT,
            "emergency_halt": drawdown_pct >= RISK_MAX_DRAWDOWN_PCT * 2.0,
        }

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
        # Use live balance when available, fall back to initial_capital,
        # and finally to the default TRADING_VOLUME.
        balance = self.account.balance or self._initial_capital
        if balance <= 0 or stop_loss_pips <= 0:
            return TRADING_VOLUME

        risk_amount = balance * risk_pct / 100
        # Simplified: assume 1 pip = $0.0001 per unit for major pairs
        pip_value = 0.0001
        size = int(risk_amount / (stop_loss_pips * pip_value))
        size = max(1000, min(size, RISK_MAX_POSITION_SIZE))
        return size
