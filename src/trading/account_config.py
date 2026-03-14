"""
src/trading/account_config.py — Account configuration for dual-account trading.

Defines :class:`AccountConfig` which bundles a cTrader account ID with its
corresponding initial capital.  Two accounts are pre-configured:

* **Account 1** — ``CTRADER_ACCOUNT_ID`` / ``INITIAL_CAPITAL_ACC1``  (default 10 000 USD)
* **Account 2** — ``CTRADER_ACCOUNT_ID_ACC2`` / ``INITIAL_CAPITAL_ACC2`` (default 50 USD)

Both accounts receive the same trading signals; position sizes are scaled
independently to each account's capital.

Usage::

    configs = get_account_configs()
    for cfg in configs:
        size = cfg.position_size_from_risk(stop_loss_pips=30, risk_pct=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass

from constants import (
    CTRADER_ACCOUNT_ID,
    CTRADER_ACCOUNT_ID_ACC2,
    INITIAL_CAPITAL_ACC1,
    INITIAL_CAPITAL_ACC2,
    RISK_MAX_POSITION_SIZE,
    TRADING_STOP_LOSS_PIPS,
    TRADING_VOLUME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AccountConfig:
    """
    Immutable configuration bundle for a single trading account.

    Attributes
    ----------
    account_id : int
        cTrader account ID.
    initial_capital : float
        Starting balance used for position sizing when the live balance
        has not yet been received from the broker.
    label : str
        Human-readable name (e.g. ``"acc1"``), used in log messages and
        artefact naming.
    """

    account_id: int
    initial_capital: float
    label: str

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size_from_risk(
        self,
        current_balance: float | None = None,
        stop_loss_pips: float = TRADING_STOP_LOSS_PIPS,
        risk_pct: float = 1.0,
    ) -> int:
        """
        Compute position size (centilots) using fractional-Kelly sizing.

        Parameters
        ----------
        current_balance : float, optional
            Live account balance.  Falls back to ``initial_capital`` when
            ``None`` or ``0``.
        stop_loss_pips : float
            Stop-loss distance in pips used to cap per-trade risk.
        risk_pct : float
            Percentage of balance to risk per trade (default 1 %).

        Returns
        -------
        int  — centilots (lots × 100); clamped to
               ``[1, RISK_MAX_POSITION_SIZE]``.
        """
        balance = current_balance if current_balance else self.initial_capital
        if balance <= 0 or stop_loss_pips <= 0:
            logger.warning(
                "Invalid balance or stop-loss; returning minimum volume",
                account=self.label,
            )
            return TRADING_VOLUME

        risk_amount = balance * risk_pct / 100
        # Simplified pip-value: 1 pip ≈ $0.0001 per unit for 4-decimal FX.
        pip_value = 0.0001
        size = int(risk_amount / (stop_loss_pips * pip_value))
        size = max(1, min(size, RISK_MAX_POSITION_SIZE))
        logger.debug(
            "Position size calculated",
            account=self.label,
            balance=balance,
            risk_pct=risk_pct,
            stop_loss_pips=stop_loss_pips,
            centilots=size,
        )
        return size

    def __str__(self) -> str:
        return (
            f"AccountConfig(label={self.label!r}, "
            f"account_id={self.account_id}, "
            f"initial_capital={self.initial_capital})"
        )


# ---------------------------------------------------------------------------
# Pre-built account objects
# ---------------------------------------------------------------------------

#: Account 1 — larger balance, default 10 000 USD
ACCOUNT_1 = AccountConfig(
    account_id=CTRADER_ACCOUNT_ID,
    initial_capital=INITIAL_CAPITAL_ACC1,
    label="acc1",
)

#: Account 2 — smaller balance, default 50 USD
ACCOUNT_2 = AccountConfig(
    account_id=CTRADER_ACCOUNT_ID_ACC2,
    initial_capital=INITIAL_CAPITAL_ACC2,
    label="acc2",
)


def get_account_configs() -> list[AccountConfig]:
    """
    Return the list of configured trading accounts.

    Returns both accounts.  If both share the same account ID (i.e.
    ``CTRADER_ACCOUNT_ID_ACC2`` was not set), de-duplicates so the same
    account is not traded twice.
    """
    configs: list[AccountConfig] = [ACCOUNT_1]
    if ACCOUNT_2.account_id != ACCOUNT_1.account_id:
        configs.append(ACCOUNT_2)
    else:
        logger.info(
            "CTRADER_ACCOUNT_ID_ACC2 not configured; trading single account only",
            account_id=ACCOUNT_1.account_id,
        )
    return configs
