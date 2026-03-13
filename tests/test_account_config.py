"""
tests/test_account_config.py — Unit tests for AccountConfig and related helpers.
"""

from __future__ import annotations

import pytest

from src.trading.account_config import (
    ACCOUNT_1,
    ACCOUNT_2,
    AccountConfig,
    get_account_configs,
)


# ---------------------------------------------------------------------------
# AccountConfig construction
# ---------------------------------------------------------------------------

class TestAccountConfigConstruction:
    def test_acc1_defaults(self) -> None:
        assert ACCOUNT_1.label == "acc1"
        assert ACCOUNT_1.initial_capital == 10_000.0
        assert ACCOUNT_1.account_id == 12345678  # from test conftest env

    def test_acc2_defaults(self) -> None:
        assert ACCOUNT_2.label == "acc2"
        assert ACCOUNT_2.initial_capital == 50.0
        assert ACCOUNT_2.account_id == 87654321  # from test conftest env

    def test_frozen(self) -> None:
        with pytest.raises((AttributeError, TypeError)):
            ACCOUNT_1.initial_capital = 999.0  # type: ignore[misc]

    def test_str_repr(self) -> None:
        s = str(ACCOUNT_1)
        assert "acc1" in s
        assert "10000" in s


# ---------------------------------------------------------------------------
# get_account_configs
# ---------------------------------------------------------------------------

class TestGetAccountConfigs:
    def test_returns_two_distinct_accounts(self) -> None:
        configs = get_account_configs()
        assert len(configs) == 2
        labels = [c.label for c in configs]
        assert "acc1" in labels
        assert "acc2" in labels

    def test_returns_one_account_when_ids_same(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.trading.account_config as ac_mod

        # Simulate same ID for both accounts
        same_id = AccountConfig(account_id=12345678, initial_capital=10_000, label="acc1")
        dup_id = AccountConfig(account_id=12345678, initial_capital=50, label="acc2")

        monkeypatch.setattr(ac_mod, "ACCOUNT_1", same_id)
        monkeypatch.setattr(ac_mod, "ACCOUNT_2", dup_id)

        configs = ac_mod.get_account_configs()
        assert len(configs) == 1
        assert configs[0].label == "acc1"


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizeFromRisk:
    def test_uses_initial_capital_when_no_live_balance(self) -> None:
        cfg = AccountConfig(account_id=1, initial_capital=10_000.0, label="test")
        size = cfg.position_size_from_risk(
            current_balance=None,
            stop_loss_pips=30.0,
            risk_pct=1.0,
        )
        # risk_amount = 10000 * 0.01 = 100, size = 100 / (30 * 0.0001) = 33_333
        # clamped to RISK_MAX_POSITION_SIZE=10000
        assert size == 10_000

    def test_uses_live_balance_when_provided(self) -> None:
        cfg = AccountConfig(account_id=1, initial_capital=10_000.0, label="test")
        # Use 5000 vs 500 — both below the cap so difference is visible
        size_500 = cfg.position_size_from_risk(current_balance=500.0, stop_loss_pips=30.0)
        size_5k = cfg.position_size_from_risk(current_balance=5_000.0, stop_loss_pips=30.0)
        assert size_500 < size_5k

    def test_small_account_50_produces_small_size(self) -> None:
        cfg = AccountConfig(account_id=2, initial_capital=50.0, label="acc2")
        size = cfg.position_size_from_risk(
            current_balance=None,
            stop_loss_pips=30.0,
            risk_pct=1.0,
        )
        # risk_amount = 50 * 0.01 = 0.5, size = 0.5 / (30 * 0.0001) = 166
        # clamped to minimum 1
        assert 1 <= size <= 1000

    def test_small_account_size_less_than_large_account(self) -> None:
        cfg_large = AccountConfig(account_id=1, initial_capital=10_000.0, label="large")
        cfg_small = AccountConfig(account_id=2, initial_capital=50.0, label="small")
        size_large = cfg_large.position_size_from_risk(stop_loss_pips=30.0)
        size_small = cfg_small.position_size_from_risk(stop_loss_pips=30.0)
        assert size_small < size_large

    def test_zero_balance_returns_default_volume(self) -> None:
        from constants import TRADING_VOLUME

        cfg = AccountConfig(account_id=1, initial_capital=0.0, label="zero")
        size = cfg.position_size_from_risk(current_balance=0.0, stop_loss_pips=30.0)
        assert size == TRADING_VOLUME

    def test_zero_stop_loss_returns_default_volume(self) -> None:
        from constants import TRADING_VOLUME

        cfg = AccountConfig(account_id=1, initial_capital=10_000.0, label="test")
        size = cfg.position_size_from_risk(stop_loss_pips=0.0)
        assert size == TRADING_VOLUME

    def test_result_within_bounds(self) -> None:
        from constants import RISK_MAX_POSITION_SIZE

        cfg = AccountConfig(account_id=1, initial_capital=10_000.0, label="test")
        size = cfg.position_size_from_risk(stop_loss_pips=30.0, risk_pct=1.0)
        assert 1 <= size <= RISK_MAX_POSITION_SIZE
