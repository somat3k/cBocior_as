"""
tests/test_dual_account_trading.py — Tests for dual-account execution pipeline.

Validates that:
* RiskManager uses initial_capital when balance is 0.
* Execution uses the correct account_id when sending orders.
* Both accounts receive independent position sizes proportional to capital.
* A BUY/SELL decision reaches both accounts; HOLD reaches neither.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.trading.account_config import AccountConfig, get_account_configs
from src.trading.execution import Execution, Position
from src.trading.risk_manager import RiskManager
from src.utils.payload import TradingAction, TradingPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(action: TradingAction, confidence: float = 0.9) -> TradingPayload:
    from src.utils.payload import PayloadBuilder

    return (
        PayloadBuilder()
        .source("test")
        .symbol("EURUSD")
        .action(action)
        .confidence(confidence)
        .reasoning("unit test")
        .build()
    )


# ---------------------------------------------------------------------------
# RiskManager — initial_capital integration
# ---------------------------------------------------------------------------

class TestRiskManagerInitialCapital:
    def test_uses_initial_capital_when_balance_zero(self) -> None:
        rm = RiskManager(initial_capital=10_000.0)
        size = rm.compute_position_size(stop_loss_pips=30.0, risk_pct=1.0)
        # risk=100, size=100/(30*0.0001)=33333 → clamped to 10000
        assert size == 10_000

    def test_small_initial_capital_50(self) -> None:
        rm = RiskManager(initial_capital=50.0)
        size = rm.compute_position_size(stop_loss_pips=30.0, risk_pct=1.0)
        # risk=0.5, size=0.5/(30*0.0001)=166 → within [1000, 10000] lower bound = 1000
        assert size == 1000  # lower bound from RiskManager is 1000

    def test_live_balance_overrides_initial_capital(self) -> None:
        rm = RiskManager(initial_capital=50.0)
        rm.update_account({"balance": 10_000.0})
        size_with_live = rm.compute_position_size(stop_loss_pips=30.0)
        size_initial = RiskManager(initial_capital=50.0).compute_position_size(stop_loss_pips=30.0)
        assert size_with_live > size_initial

    def test_no_initial_capital_no_live_balance_returns_default(self) -> None:
        from constants import TRADING_VOLUME

        rm = RiskManager(initial_capital=0.0)
        size = rm.compute_position_size(stop_loss_pips=30.0)
        assert size == TRADING_VOLUME

    def test_acc1_larger_than_acc2(self) -> None:
        rm1 = RiskManager(initial_capital=10_000.0)
        rm2 = RiskManager(initial_capital=50.0)
        size1 = rm1.compute_position_size(stop_loss_pips=30.0, risk_pct=1.0)
        size2 = rm2.compute_position_size(stop_loss_pips=30.0, risk_pct=1.0)
        assert size1 >= size2


# ---------------------------------------------------------------------------
# Execution — account_id targeting
# ---------------------------------------------------------------------------

class TestExecutionAccountId:
    def test_uses_explicit_account_id(self) -> None:
        ex = Execution(client=None, dry_run=True, account_id=99999)
        assert ex._account_id == 99999

    def test_falls_back_to_client_account_id(self) -> None:
        mock_client = MagicMock()
        mock_client.account_id = 12345
        ex = Execution(client=mock_client, dry_run=True)
        assert ex._account_id == 12345

    def test_zero_when_no_client_no_explicit_id(self) -> None:
        ex = Execution(client=None, dry_run=True)
        assert ex._account_id == 0

    def test_dry_run_returns_position(self) -> None:
        ex = Execution(client=None, dry_run=True, account_id=111)
        payload = _make_payload(TradingAction.BUY)
        pos = ex.execute(payload=payload, volume=100, current_price=1.10)
        assert isinstance(pos, Position)
        assert pos.volume == 100

    def test_hold_returns_none(self) -> None:
        ex = Execution(client=None, dry_run=True, account_id=111)
        payload = _make_payload(TradingAction.HOLD)
        assert ex.execute(payload=payload) is None


# ---------------------------------------------------------------------------
# Dual-account execution — same signal, different sizes
# ---------------------------------------------------------------------------

class TestDualAccountExecution:
    def test_both_accounts_receive_trade(self) -> None:
        payload = _make_payload(TradingAction.BUY)

        ex1 = Execution(client=None, dry_run=True, account_id=12345678)
        ex2 = Execution(client=None, dry_run=True, account_id=87654321)

        pos1 = ex1.execute(payload=payload, volume=500, current_price=1.10)
        pos2 = ex2.execute(payload=payload, volume=10, current_price=1.10)

        assert pos1 is not None
        assert pos2 is not None
        assert pos1.volume == 500
        assert pos2.volume == 10

    def test_hold_blocks_both_accounts(self) -> None:
        payload = _make_payload(TradingAction.HOLD)
        ex1 = Execution(client=None, dry_run=True, account_id=12345678)
        ex2 = Execution(client=None, dry_run=True, account_id=87654321)

        assert ex1.execute(payload=payload) is None
        assert ex2.execute(payload=payload) is None

    def test_position_sizes_proportional_to_capital(self) -> None:
        """Account 1 (10k) must produce a larger size than Account 2 (50)."""
        rm1 = RiskManager(initial_capital=10_000.0)
        rm2 = RiskManager(initial_capital=50.0)

        size1 = rm1.compute_position_size(stop_loss_pips=30.0, risk_pct=1.0)
        size2 = rm2.compute_position_size(stop_loss_pips=30.0, risk_pct=1.0)

        payload = _make_payload(TradingAction.BUY)
        ex1 = Execution(client=None, dry_run=True, account_id=12345678)
        ex2 = Execution(client=None, dry_run=True, account_id=87654321)

        pos1 = ex1.execute(payload=payload, volume=size1, current_price=1.10)
        pos2 = ex2.execute(payload=payload, volume=size2, current_price=1.10)

        assert pos1.volume >= pos2.volume

    def test_sell_position_sl_tp_directions(self) -> None:
        """For a SELL, SL > entry_price and TP < entry_price."""
        payload = _make_payload(TradingAction.SELL)
        ex = Execution(client=None, dry_run=True, account_id=12345678)
        pos = ex.execute(payload=payload, volume=100, current_price=1.10)
        assert pos is not None
        assert pos.stop_loss > 1.10
        assert pos.take_profit < 1.10

    def test_buy_position_sl_tp_directions(self) -> None:
        """For a BUY, SL < entry_price and TP > entry_price."""
        payload = _make_payload(TradingAction.BUY)
        ex = Execution(client=None, dry_run=True, account_id=12345678)
        pos = ex.execute(payload=payload, volume=100, current_price=1.10)
        assert pos is not None
        assert pos.stop_loss < 1.10
        assert pos.take_profit > 1.10


# ---------------------------------------------------------------------------
# get_account_configs integration
# ---------------------------------------------------------------------------

class TestGetAccountConfigsIntegration:
    def test_configs_have_correct_capitals(self) -> None:
        configs = get_account_configs()
        capitals = {c.label: c.initial_capital for c in configs}
        assert capitals["acc1"] == 10_000.0
        assert capitals["acc2"] == 50.0

    def test_configs_have_distinct_account_ids(self) -> None:
        configs = get_account_configs()
        ids = [c.account_id for c in configs]
        assert len(set(ids)) == len(ids), "Account IDs must be unique"
