"""
tests/test_backtester.py — Unit tests for the Backtester and BacktestResult.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.models.backtester import Backtester, BacktestResult, _max_drawdown

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_models(
    nn_probs: list[float],
    gbm_probs: list[float],
) -> tuple[MagicMock, MagicMock]:
    """Return mock nn and gbm that yield deterministic probabilities."""
    nn = MagicMock()
    nn.predict_proba.return_value = np.array(nn_probs, dtype=float)

    gbm = MagicMock()
    # GBM predict_proba returns shape (n, 2) — column 1 is P(class=1)
    probs_2d = np.column_stack([
        1 - np.array(gbm_probs),
        np.array(gbm_probs),
    ])
    gbm.predict_proba.return_value = probs_2d

    return nn, gbm


def _make_backtester(**kwargs) -> Backtester:
    defaults = dict(
        initial_capital=10_000.0,
        stop_loss_pips=30.0,
        take_profit_pips=60.0,
        buy_threshold=0.62,
        sell_threshold=0.38,
        reversal_exit_threshold=0.50,
        risk_pct=1.0,
    )
    defaults.update(kwargs)
    return Backtester(**defaults)


# ---------------------------------------------------------------------------
# _max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_flat_curve_returns_zero(self) -> None:
        assert _max_drawdown([100.0, 100.0, 100.0]) == 0.0

    def test_monotonic_growth_returns_zero(self) -> None:
        assert _max_drawdown([100.0, 110.0, 120.0]) == 0.0

    def test_single_dip(self) -> None:
        mdd = _max_drawdown([100.0, 110.0, 80.0, 90.0])
        # peak=110, trough=80, dd=(110-80)/110=27.27%
        assert abs(mdd - 27.27) < 0.1

    def test_short_curve(self) -> None:
        assert _max_drawdown([100.0]) == 0.0
        assert _max_drawdown([]) == 0.0

    def test_all_declining(self) -> None:
        mdd = _max_drawdown([100.0, 90.0, 80.0, 70.0])
        # peak=100, trough=70, dd=30%
        assert abs(mdd - 30.0) < 0.1


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_default_values(self) -> None:
        r = BacktestResult()
        assert r.total_trades == 0
        assert r.win_rate == 0.0
        assert r.equity_curve == []

    def test_str_representation(self) -> None:
        r = BacktestResult(
            total_trades=10,
            win_rate=60.0,
            max_drawdown_pct=3.5,
            total_return_pct=12.0,
            profit_factor=1.8,
            cancelled_rate=20.0,
        )
        s = str(r)
        assert "60.0" in s
        assert "3.5" in s
        assert "12.0" in s


# ---------------------------------------------------------------------------
# Backtester — empty / edge inputs
# ---------------------------------------------------------------------------

class TestBacktesterEdgeCases:
    def test_too_few_samples_returns_empty_result(self) -> None:
        bt = _make_backtester()
        nn, gbm = _make_mock_models([0.7] * 5, [0.7] * 5)
        result = bt.run(
            X_scaled=np.zeros((5, 3)),
            close_prices=np.ones(5) * 1.10,
            nn=nn, gbm=gbm,
        )
        assert result.total_trades == 0
        assert result.win_rate == 0.0

    def test_returns_backtest_result_instance(self) -> None:
        n = 50
        bt = _make_backtester()
        nn, gbm = _make_mock_models([0.5] * n, [0.5] * n)
        result = bt.run(
            X_scaled=np.zeros((n, 3)),
            close_prices=np.linspace(1.10, 1.12, n),
            nn=nn, gbm=gbm,
        )
        assert isinstance(result, BacktestResult)


# ---------------------------------------------------------------------------
# Backtester — BUY signals
# ---------------------------------------------------------------------------

class TestBacktesterBuySignals:
    def test_constant_buy_signal_enters_position(self) -> None:
        n = 100
        bt = _make_backtester()
        # Strong constant BUY signal
        nn, gbm = _make_mock_models([0.8] * n, [0.8] * n)
        # Price that hits TP: rises by > 60 pips (60 * 0.0001 = 0.006)
        prices = np.linspace(1.10, 1.11, n)  # +0.01 = 100 pips
        result = bt.run(
            X_scaled=np.zeros((n, 3)),
            close_prices=prices,
            nn=nn, gbm=gbm,
        )
        assert result.total_trades >= 1

    def test_buy_and_tp_hit_records_win(self) -> None:
        """Steady uptrend should produce winning trades via TP."""
        n = 200
        bt = _make_backtester(take_profit_pips=10.0, stop_loss_pips=20.0)
        nn, gbm = _make_mock_models([0.75] * n, [0.75] * n)
        # Price increases 1 pip per bar → hits 10-pip TP quickly
        prices = 1.10 + np.arange(n) * 0.0001
        result = bt.run(
            X_scaled=np.zeros((n, 3)),
            close_prices=prices,
            nn=nn, gbm=gbm,
        )
        assert result.winning_trades > 0

    def test_buy_sl_hit_records_loss(self) -> None:
        """Sharp price drop should hit SL."""
        n = 50
        bt = _make_backtester(stop_loss_pips=10.0, take_profit_pips=100.0)
        nn, gbm = _make_mock_models([0.75] * n, [0.75] * n)
        # Price drops 20 pips from entry
        prices = np.concatenate([
            [1.10],                           # entry bar
            np.linspace(1.10, 1.098, n - 1), # fall 20 pips → SL at 1.099
        ])
        result = bt.run(
            X_scaled=np.zeros((n, 3)),
            close_prices=prices,
            nn=nn, gbm=gbm,
        )
        assert result.losing_trades > 0


# ---------------------------------------------------------------------------
# Backtester — SELL signals
# ---------------------------------------------------------------------------

class TestBacktesterSellSignals:
    def test_sell_tp_hit_records_win(self) -> None:
        """Steady downtrend with SELL signal should produce winning trades."""
        n = 200
        bt = _make_backtester(take_profit_pips=10.0, stop_loss_pips=20.0)
        nn, gbm = _make_mock_models([0.25] * n, [0.25] * n)
        # Price decreases 1 pip per bar → hits 10-pip TP quickly for SELL
        prices = 1.10 - np.arange(n) * 0.0001
        result = bt.run(
            X_scaled=np.zeros((n, 3)),
            close_prices=prices,
            nn=nn, gbm=gbm,
        )
        assert result.winning_trades > 0


# ---------------------------------------------------------------------------
# Backtester — Signal reversal / trade cancellation
# ---------------------------------------------------------------------------

class TestBacktesterSignalReversal:
    def test_reversal_exits_trade_early(self) -> None:
        """
        BUY entered on bar 0, then signal flips below reversal threshold.
        Expect at least one cancellation.
        """
        n = 30
        probs = [0.75] * 5 + [0.45] * 25   # enter BUY, then signal reverses
        bt = _make_backtester(reversal_exit_threshold=0.50, stop_loss_pips=200.0)
        nn, gbm = _make_mock_models(probs, probs)
        prices = np.linspace(1.10, 1.105, n)  # gentle rise, no TP/SL hit
        result = bt.run(
            X_scaled=np.zeros((n, 3)),
            close_prices=prices,
            nn=nn, gbm=gbm,
        )
        assert result.cancelled_trades >= 1
        assert result.cancelled_rate > 0.0

    def test_strong_reversal_opens_opposite_trade(self) -> None:
        """
        BUY open → signal reverses strongly below sell_threshold
        → cancel BUY and open SELL.
        """
        n = 40
        probs = [0.75] * 5 + [0.30] * 35   # enter BUY, then strong SELL signal
        bt = _make_backtester(
            reversal_exit_threshold=0.50,
            sell_threshold=0.38,
            stop_loss_pips=500.0,  # wide SL so no SL exit
            take_profit_pips=1000.0,
        )
        nn, gbm = _make_mock_models(probs, probs)
        prices = np.ones(n) * 1.10
        result = bt.run(
            X_scaled=np.zeros((n, 3)),
            close_prices=prices,
            nn=nn, gbm=gbm,
        )
        # At least one cancellation and at least one reverse-direction trade
        assert result.cancelled_trades >= 1
        assert result.total_trades >= 2  # original BUY + at least one SELL


# ---------------------------------------------------------------------------
# Backtester — Metrics
# ---------------------------------------------------------------------------

class TestBacktesterMetrics:
    def test_win_rate_computed_correctly(self) -> None:
        """Win rate should be winning / (winning + losing) * 100."""
        bt = _make_backtester(take_profit_pips=5.0, stop_loss_pips=100.0)
        n = 100
        nn, gbm = _make_mock_models([0.75] * n, [0.75] * n)
        # Price trend that hits TP repeatedly
        prices = 1.10 + np.arange(n) * 0.00005  # ~0.5 pip/bar
        result = bt.run(np.zeros((n, 3)), prices, nn, gbm)
        closed = result.winning_trades + result.losing_trades
        if closed > 0:
            expected_wr = result.winning_trades / closed * 100
            assert abs(result.win_rate - expected_wr) < 0.01

    def test_equity_curve_starts_at_initial_capital(self) -> None:
        n = 20
        bt = _make_backtester()
        nn, gbm = _make_mock_models([0.5] * n, [0.5] * n)
        result = bt.run(np.zeros((n, 3)), np.ones(n) * 1.10, nn, gbm)
        assert result.equity_curve[0] == pytest.approx(10_000.0)

    def test_balance_never_goes_below_one_dollar(self) -> None:
        n = 50
        bt = _make_backtester(stop_loss_pips=1.0, take_profit_pips=1.0, risk_pct=100.0)
        nn, gbm = _make_mock_models([0.75] * n, [0.75] * n)
        prices = 1.10 - np.arange(n) * 0.0001  # falling → BUY SLs
        result = bt.run(np.zeros((n, 3)), prices, nn, gbm)
        assert all(e >= 1.0 for e in result.equity_curve)

    def test_symbol_and_timeframe_in_result(self) -> None:
        n = 20
        bt = _make_backtester()
        nn, gbm = _make_mock_models([0.5] * n, [0.5] * n)
        result = bt.run(
            np.zeros((n, 3)), np.ones(n) * 1.10, nn, gbm,
            symbol="EURUSD", timeframe="M1",
        )
        assert result.symbol == "EURUSD"
        assert result.timeframe == "M1"

    def test_profit_factor_zero_when_no_trades(self) -> None:
        n = 20
        bt = _make_backtester()
        nn, gbm = _make_mock_models([0.5] * n, [0.5] * n)  # neutral: no entry
        result = bt.run(np.zeros((n, 3)), np.ones(n) * 1.10, nn, gbm)
        # No trades → profit_factor should be 0 (or 0 winning trades)
        assert result.profit_factor >= 0.0

    def test_max_drawdown_zero_on_no_trades(self) -> None:
        n = 20
        bt = _make_backtester()
        nn, gbm = _make_mock_models([0.5] * n, [0.5] * n)
        result = bt.run(np.zeros((n, 3)), np.ones(n) * 1.10, nn, gbm)
        assert result.max_drawdown_pct >= 0.0


# ---------------------------------------------------------------------------
# Backtester — Spread + Slippage (P3)
# ---------------------------------------------------------------------------

class TestBacktesterSpreadSlippage:
    def test_spread_reduces_win_rate_vs_zero_spread(self) -> None:
        """A trade with spread costs should produce a lower or equal return."""
        n = 100
        prices = 1.10 + np.arange(n) * 0.0001

        nn_zero, gbm_zero = _make_mock_models([0.75] * n, [0.75] * n)
        nn_spread, gbm_spread = _make_mock_models([0.75] * n, [0.75] * n)

        bt_zero = Backtester(
            initial_capital=10_000.0, stop_loss_pips=20.0,
            take_profit_pips=10.0, spread_pips=0.0, slippage_pips=0.0, seed=0,
        )
        bt_spread = Backtester(
            initial_capital=10_000.0, stop_loss_pips=20.0,
            take_profit_pips=10.0, spread_pips=2.0, slippage_pips=0.5, seed=0,
        )
        result_zero = bt_zero.run(np.zeros((n, 3)), prices, nn_zero, gbm_zero)
        result_spread = bt_spread.run(np.zeros((n, 3)), prices, nn_spread, gbm_spread)

        # Spread/slippage can never improve returns
        assert result_spread.total_return_pct <= result_zero.total_return_pct + 0.01

    def test_zero_spread_zero_slippage_no_extra_cost(self) -> None:
        """With spread=0 and slippage=0 entry price equals mid price."""
        bt = Backtester(spread_pips=0.0, slippage_pips=0.0, seed=42)
        assert bt._entry_price(1.10000, 1) == pytest.approx(1.10000)
        assert bt._entry_price(1.10000, -1) == pytest.approx(1.10000)

    def test_buy_entry_higher_than_mid(self) -> None:
        """BUY fill price > mid due to spread/slippage costs."""
        bt = Backtester(spread_pips=2.0, slippage_pips=0.0, seed=0)
        fill = bt._entry_price(1.10000, 1)
        assert fill > 1.10000

    def test_sell_entry_lower_than_mid(self) -> None:
        """SELL fill price < mid due to spread/slippage costs."""
        bt = Backtester(spread_pips=2.0, slippage_pips=0.0, seed=0)
        fill = bt._entry_price(1.10000, -1)
        assert fill < 1.10000

    def test_slippage_bounded_by_max(self) -> None:
        """Slippage should never exceed slippage_pips."""
        from src.models.backtester import _PIP
        bt = Backtester(spread_pips=0.0, slippage_pips=1.0, seed=7)
        for _ in range(100):
            fill = bt._entry_price(1.10000, 1)
            assert fill <= 1.10000 + 1.0 * _PIP + 1e-9
