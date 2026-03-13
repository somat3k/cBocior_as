"""
tests/test_payload.py — Unit tests for the TradingPayload system.
"""

from __future__ import annotations

import pytest

from src.utils.payload import (
    AgentID,
    IndicatorSnapshot,
    ModelSignals,
    PayloadBuilder,
    RiskFlags,
    TradingAction,
    TradingPayload,
    merge_payloads,
)


class TestTradingPayload:
    def test_defaults(self) -> None:
        p = TradingPayload()
        assert p.action == TradingAction.HOLD
        assert p.confidence == 0.0
        assert p.version == "1.0"

    def test_confidence_clamp(self) -> None:
        p = TradingPayload(confidence=1.5)
        assert p.confidence == 1.0
        p2 = TradingPayload(confidence=-0.5)
        assert p2.confidence == 0.0

    def test_json_roundtrip(self) -> None:
        original = TradingPayload(
            source="openai",
            symbol="EURUSD",
            action=TradingAction.BUY,
            confidence=0.75,
            reasoning="Strong bullish signal",
        )
        json_str = original.to_json()
        restored = TradingPayload.from_json(json_str)
        assert restored.action == TradingAction.BUY
        assert restored.confidence == 0.75
        assert restored.symbol == "EURUSD"


class TestPayloadBuilder:
    def test_fluent_build(self) -> None:
        payload = (
            PayloadBuilder()
            .source("groq")
            .symbol("GBPUSD")
            .action(TradingAction.SELL)
            .confidence(0.68)
            .reasoning("Bearish MACD")
            .build()
        )
        assert payload.source == "groq"
        assert payload.action == TradingAction.SELL
        assert payload.confidence == 0.68

    def test_add_indicator(self) -> None:
        snap = IndicatorSnapshot(timeframe="M1", rsi=45.0)
        payload = (
            PayloadBuilder()
            .add_indicator(snap)
            .build()
        )
        assert len(payload.indicators) == 1
        assert payload.indicators[0].rsi == 45.0

    def test_add_model_signal(self) -> None:
        sig = ModelSignals(timeframe="M5", ensemble_prediction=0.62)
        payload = PayloadBuilder().add_model_signal(sig).build()
        assert len(payload.model_signals) == 1
        assert payload.model_signals[0].ensemble_prediction == 0.62


class TestRiskFlags:
    def test_any_flag_false(self) -> None:
        flags = RiskFlags()
        assert not flags.any_flag

    def test_any_flag_spread(self) -> None:
        flags = RiskFlags(spread_exceeded=True)
        assert flags.any_flag

    def test_custom_flag(self) -> None:
        flags = RiskFlags(custom=["news_event"])
        assert flags.any_flag


class TestMergePayloads:
    def _make(self, action: TradingAction, confidence: float) -> TradingPayload:
        return TradingPayload(
            action=action,
            confidence=confidence,
            source="test",
        )

    def test_majority_buy(self) -> None:
        payloads = [
            self._make(TradingAction.BUY, 0.7),
            self._make(TradingAction.BUY, 0.8),
            self._make(TradingAction.SELL, 0.6),
        ]
        merged = merge_payloads(payloads, symbol="EURUSD")
        assert merged.action == TradingAction.BUY

    def test_no_majority_defaults_to_hold(self) -> None:
        payloads = [
            self._make(TradingAction.BUY, 0.7),
            self._make(TradingAction.SELL, 0.7),
        ]
        merged = merge_payloads(payloads, symbol="EURUSD")
        assert merged.action == TradingAction.HOLD

    def test_empty_returns_hold(self) -> None:
        merged = merge_payloads([], symbol="EURUSD")
        assert merged.action == TradingAction.HOLD

    def test_risk_flags_union(self) -> None:
        p1 = TradingPayload(
            action=TradingAction.BUY,
            confidence=0.7,
            risk_flags=RiskFlags(spread_exceeded=True),
        )
        p2 = TradingPayload(
            action=TradingAction.BUY,
            confidence=0.7,
            risk_flags=RiskFlags(low_confidence=True),
        )
        merged = merge_payloads([p1, p2])
        assert merged.risk_flags.spread_exceeded
        assert merged.risk_flags.low_confidence


class TestPayloadSigning:
    def test_sign_and_verify(self) -> None:
        p = TradingPayload(action=TradingAction.BUY, confidence=0.8)
        signed = p.sign("my_secret_key")
        assert signed.signature is not None
        assert signed.verify_signature("my_secret_key")

    def test_wrong_key_fails(self) -> None:
        p = TradingPayload(action=TradingAction.BUY, confidence=0.8)
        signed = p.sign("correct_key")
        assert not signed.verify_signature("wrong_key")

    def test_unsigned_fails_verify(self) -> None:
        p = TradingPayload()
        assert not p.verify_signature("any_key")
