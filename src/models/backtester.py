"""
src/models/backtester.py — Trade simulation and evaluation engine.

Runs a bar-by-bar backtest on the model validation set to evaluate:

  - **Win rate**    — percentage of closed trades that were profitable.
  - **%MDD**        — Maximum Drawdown as a percentage of peak equity.
  - **Balance growth** — total return on initial capital.
  - **Trade cancellation** — early exits triggered when the model
    re-calculates its position estimate and the signal reverses.
    This implements the strategy of "collecting profits more often and
    canceling trades due to re-calculated position and possibility to
    reverse in favour".
  - **Profit factor** — gross profit divided by gross loss.

Trade exit priority
-------------------
1. Take-profit price reached → **WIN**
2. Stop-loss price reached   → **LOSS**
3. Signal reversal (ensemble probability crosses the reversal boundary)
   → **CANCELLED** (early close; PnL may be positive or negative).
   If the reversed signal is strong enough, a new position in the
   opposite direction is opened immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from constants import (
    TRADING_STOP_LOSS_PIPS,
    TRADING_TAKE_PROFIT_PIPS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard 4/5-decimal FX pip value; crypto positions use the same value
# (pip-based SL/TP becomes effectively inactive for high-price assets,
# making signal-reversal the dominant exit mechanism — which is fine).
_PIP: float = 0.0001


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Summary statistics from one backtest run."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    cancelled_trades: int = 0     # early exits via signal reversal

    win_rate: float = 0.0         # winning / (winning + losing) × 100
    max_drawdown_pct: float = 0.0 # peak-to-trough as % of peak equity
    total_return_pct: float = 0.0 # (final_balance - initial) / initial × 100
    profit_factor: float = 0.0    # gross profit / gross loss
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    cancelled_rate: float = 0.0   # cancelled / total_trades × 100

    equity_curve: list[float] = field(default_factory=list)
    initial_capital: float = 10_000.0
    symbol: str = ""
    timeframe: str = ""

    def __str__(self) -> str:
        return (
            f"Trades={self.total_trades:3d} | "
            f"WR={self.win_rate:5.1f}% | "
            f"MDD={self.max_drawdown_pct:5.1f}% | "
            f"Ret={self.total_return_pct:+6.1f}% | "
            f"PF={self.profit_factor:.2f} | "
            f"Cancelled={self.cancelled_rate:.1f}%"
        )


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class Backtester:
    """
    Bar-by-bar trade simulator using pre-computed model probabilities.

    Parameters
    ----------
    initial_capital : float
        Starting account balance in USD.
    stop_loss_pips : float
        Stop-loss distance in pips.
    take_profit_pips : float
        Take-profit distance in pips.
    buy_threshold : float
        Ensemble probability ≥ this triggers a BUY entry (default 0.62).
    sell_threshold : float
        Ensemble probability ≤ this triggers a SELL entry (default 0.38).
    reversal_exit_threshold : float
        When holding a BUY and probability drops below this value (or
        holding a SELL and probability rises above ``1 - threshold``), the
        trade is closed early (cancelled) and, if the reversed signal also
        crosses the entry threshold, a new position is opened immediately
        in the opposite direction.  Default 0.50.
    risk_pct : float
        Fraction of balance risked per trade, used for position sizing
        (default 1 %).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        stop_loss_pips: float = TRADING_STOP_LOSS_PIPS,
        take_profit_pips: float = TRADING_TAKE_PROFIT_PIPS,
        buy_threshold: float = 0.62,
        sell_threshold: float = 0.38,
        reversal_exit_threshold: float = 0.50,
        risk_pct: float = 1.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.stop_loss_pips = stop_loss_pips
        self.take_profit_pips = take_profit_pips
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.reversal_exit_threshold = reversal_exit_threshold
        self.risk_pct = risk_pct

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        X_scaled: np.ndarray,
        close_prices: np.ndarray,
        nn: Any,
        gbm: Any,
        symbol: str = "",
        timeframe: str = "",
    ) -> BacktestResult:
        """
        Simulate trading bar-by-bar on the validation set.

        Parameters
        ----------
        X_scaled : np.ndarray, shape (n_bars, n_features)
            Pre-scaled feature matrix (use the same scaler as training).
        close_prices : np.ndarray, shape (n_bars,)
            Close prices aligned with each row of ``X_scaled``.
        nn : NeuralNetwork
            Trained neural network with ``predict_proba`` method.
        gbm : fitted classifier
            Trained GBM model with ``predict_proba`` method.
        symbol : str
        timeframe : str

        Returns
        -------
        BacktestResult
        """
        n = len(X_scaled)
        if n < 10:
            logger.warning(
                "Too few samples for backtest; returning empty result",
                n=n,
                symbol=symbol,
                timeframe=timeframe,
            )
            return BacktestResult(
                symbol=symbol,
                timeframe=timeframe,
                initial_capital=self.initial_capital,
            )

        # ── Batch inference ──────────────────────────────────────────
        nn_probs: np.ndarray = np.asarray(
            nn.predict_proba(X_scaled), dtype=float
        ).ravel()
        gbm_probs: np.ndarray = gbm.predict_proba(X_scaled)[:, 1]
        ensemble: np.ndarray = 0.6 * nn_probs + 0.4 * gbm_probs

        # ── Simulation state ─────────────────────────────────────────
        balance = float(self.initial_capital)
        equity_curve: list[float] = [balance]
        peak_equity = balance

        wins: list[float] = []    # pip gains for closed winning trades
        losses: list[float] = []  # pip losses for closed losing trades (positive)
        cancelled_pnls: list[float] = []

        total_trades = 0
        winning_trades = 0
        losing_trades = 0
        cancelled_trades = 0

        in_trade = False
        trade_direction: int = 0  # +1 = BUY, -1 = SELL
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0

        for i in range(n):
            prob = float(ensemble[i])
            price = float(close_prices[i])

            if in_trade:
                if trade_direction == 1:  # BUY open
                    if price >= tp_price:
                        balance += self._pnl(balance, self.take_profit_pips)
                        wins.append(self.take_profit_pips)
                        winning_trades += 1
                        in_trade = False

                    elif price <= sl_price:
                        balance -= self._pnl(balance, self.stop_loss_pips)
                        losses.append(self.stop_loss_pips)
                        losing_trades += 1
                        in_trade = False

                    elif prob < self.reversal_exit_threshold:
                        # Signal reversed → cancel trade early
                        raw_pips = (price - entry_price) / _PIP
                        sign = float(np.sign(raw_pips)) if raw_pips != 0 else 0.0
                        pnl_val = self._pnl(balance, abs(raw_pips)) * sign
                        balance += pnl_val
                        cancelled_pnls.append(raw_pips)
                        cancelled_trades += 1
                        in_trade = False
                        # If strongly bearish: reverse immediately
                        if prob <= self.sell_threshold:
                            in_trade = True
                            trade_direction = -1
                            entry_price = price
                            sl_price = price + self.stop_loss_pips * _PIP
                            tp_price = price - self.take_profit_pips * _PIP
                            total_trades += 1

                else:  # SELL open
                    if price <= tp_price:
                        balance += self._pnl(balance, self.take_profit_pips)
                        wins.append(self.take_profit_pips)
                        winning_trades += 1
                        in_trade = False

                    elif price >= sl_price:
                        balance -= self._pnl(balance, self.stop_loss_pips)
                        losses.append(self.stop_loss_pips)
                        losing_trades += 1
                        in_trade = False

                    elif prob > (1.0 - self.reversal_exit_threshold):
                        # Signal reversed → cancel trade early
                        raw_pips = (entry_price - price) / _PIP
                        sign = float(np.sign(raw_pips)) if raw_pips != 0 else 0.0
                        pnl_val = self._pnl(balance, abs(raw_pips)) * sign
                        balance += pnl_val
                        cancelled_pnls.append(raw_pips)
                        cancelled_trades += 1
                        in_trade = False
                        # If strongly bullish: reverse immediately
                        if prob >= self.buy_threshold:
                            in_trade = True
                            trade_direction = 1
                            entry_price = price
                            sl_price = price - self.stop_loss_pips * _PIP
                            tp_price = price + self.take_profit_pips * _PIP
                            total_trades += 1

            if not in_trade:
                if prob >= self.buy_threshold:
                    in_trade = True
                    trade_direction = 1
                    entry_price = price
                    sl_price = price - self.stop_loss_pips * _PIP
                    tp_price = price + self.take_profit_pips * _PIP
                    total_trades += 1
                elif prob <= self.sell_threshold:
                    in_trade = True
                    trade_direction = -1
                    entry_price = price
                    sl_price = price + self.stop_loss_pips * _PIP
                    tp_price = price - self.take_profit_pips * _PIP
                    total_trades += 1

            balance = max(balance, 1.0)
            equity_curve.append(balance)
            if balance > peak_equity:
                peak_equity = balance

        # ── Metrics ──────────────────────────────────────────────────
        closed = winning_trades + losing_trades
        win_rate = (winning_trades / closed * 100.0) if closed > 0 else 0.0
        cancelled_rate = (
            cancelled_trades / total_trades * 100.0
        ) if total_trades > 0 else 0.0
        avg_win_pips = float(np.mean(wins)) if wins else 0.0
        avg_loss_pips = float(np.mean(losses)) if losses else 0.0
        gross_profit = float(sum(wins))
        gross_loss = float(sum(losses)) if losses else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float(gross_profit > 0)
        )
        max_dd = _max_drawdown(equity_curve)
        total_return = (balance - self.initial_capital) / self.initial_capital * 100.0

        result = BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            cancelled_trades=cancelled_trades,
            win_rate=round(win_rate, 2),
            max_drawdown_pct=round(max_dd, 2),
            total_return_pct=round(total_return, 2),
            profit_factor=round(profit_factor, 3),
            avg_win_pips=round(avg_win_pips, 1),
            avg_loss_pips=round(avg_loss_pips, 1),
            cancelled_rate=round(cancelled_rate, 2),
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            symbol=symbol,
            timeframe=timeframe,
        )

        logger.info(
            "Backtest complete",
            symbol=symbol or "(unknown)",
            timeframe=timeframe or "(unknown)",
            result=str(result),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pnl(self, balance: float, pips: float) -> float:
        """
        Approximate USD PnL for a ``pips``-size move.

        Sizes the trade to risk ``risk_pct`` % of the current balance
        on ``stop_loss_pips``.
        """
        risk_amount = balance * self.risk_pct / 100.0
        # volume (units) to risk exactly risk_amount on a stop_loss_pips move
        volume = risk_amount / (self.stop_loss_pips * _PIP + 1e-10)
        return pips * _PIP * volume


# ---------------------------------------------------------------------------
# Module-level helper (usable outside Backtester too)
# ---------------------------------------------------------------------------

def _max_drawdown(equity_curve: list[float]) -> float:
    """Return the maximum peak-to-trough drawdown as a percentage."""
    if len(equity_curve) < 2:
        return 0.0
    eq = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    drawdown = (peak - eq) / (peak + 1e-10) * 100.0
    return float(drawdown.max())
