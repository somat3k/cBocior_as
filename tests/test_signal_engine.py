"""
tests/test_signal_engine.py — Unit tests for SignalEngine.
"""

from __future__ import annotations

import pytest

from src.analysis.signal_engine import SignalEngine
from src.utils.payload import IndicatorSnapshot, ModelSignals, TradingAction


class TestSignalEngine:
    @pytest.fixture
    def engine(self) -> SignalEngine:
        return SignalEngine()

    def test_returns_required_keys(self, engine: SignalEngine) -> None:
        result = engine.compute([], [], [], "UNKNOWN")
        assert "score" in result
        assert "action" in result
        assert "confidence" in result
        assert "components" in result

    def test_empty_inputs_neutral(self, engine: SignalEngine) -> None:
        result = engine.compute([], [], [])
        assert result["action"] == TradingAction.HOLD
        assert abs(result["score"] - 50.0) < 10.0

    def test_strong_buy_signal(self, engine: SignalEngine) -> None:
        # RSI = 25 (oversold → bullish), stoch_k = 15, MACD hist positive
        snap = IndicatorSnapshot(
            timeframe="M1",
            rsi=25.0,        # oversold
            macd_hist=0.001, # positive
            stoch_k=15.0,    # oversold
        )
        model_sig = ModelSignals(
            timeframe="M1",
            ensemble_prediction=0.85,
            ensemble_confidence=0.85,
        )
        result = engine.compute([snap], [model_sig], [])
        assert result["score"] >= 55.0
        assert result["action"] == TradingAction.BUY

    def test_strong_sell_signal(self, engine: SignalEngine) -> None:
        snap = IndicatorSnapshot(
            timeframe="M1",
            rsi=78.0,        # overbought
            macd_hist=-0.002,
            stoch_k=85.0,    # overbought
        )
        model_sig = ModelSignals(
            timeframe="M1",
            ensemble_prediction=0.15,
            ensemble_confidence=0.85,
        )
        result = engine.compute([snap], [model_sig], [])
        assert result["score"] <= 45.0
        assert result["action"] == TradingAction.SELL

    def test_volatile_regime_dampens(self, engine: SignalEngine) -> None:
        snap = IndicatorSnapshot(timeframe="M1", rsi=25.0)
        model_sig = ModelSignals(timeframe="M1", ensemble_prediction=0.9)

        normal_result = engine.compute([snap], [model_sig], [], "TRENDING_UP")
        volatile_result = engine.compute([snap], [model_sig], [], "VOLATILE")

        # Volatile should push score closer to 50
        normal_dist = abs(normal_result["score"] - 50.0)
        volatile_dist = abs(volatile_result["score"] - 50.0)
        assert volatile_dist < normal_dist


class TestDecisionEngine:
    def test_below_confidence_returns_hold(self) -> None:
        from src.trading.decision_engine import DecisionEngine
        from src.utils.payload import TradingPayload

        engine = DecisionEngine()
        payload = TradingPayload(
            action=TradingAction.BUY,
            confidence=0.4,  # below threshold
        )
        signal_data = {
            "action": TradingAction.BUY,
            "confidence": 0.4,
        }
        result = engine.decide(payload, signal_data, __import__('src.utils.payload', fromlist=['RiskFlags']).RiskFlags())
        assert result.action == TradingAction.HOLD
