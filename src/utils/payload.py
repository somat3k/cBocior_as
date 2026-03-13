"""
src/utils/payload.py — JSON payload builder and validator.

All inter-agent and inter-module communication uses this canonical payload
schema.  Payloads are serialised with ujson for performance, validated with
Pydantic, and optionally signed with HMAC-SHA256 for integrity.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

try:
    import ujson as json
except ImportError:
    import json  # type: ignore[no-redef]

from pydantic import BaseModel, Field, field_validator

from constants import PAYLOAD_ENCODING, PAYLOAD_VERSION


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TradingAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AgentID(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    ORCHESTRATOR = "orchestrator"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class IndicatorSnapshot(BaseModel):
    timeframe: str
    rsi: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    ema_9: float | None = None
    ema_21: float | None = None
    ema_50: float | None = None
    ema_200: float | None = None
    atr: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None
    volume: float | None = None


class ModelSignals(BaseModel):
    timeframe: str
    nn_prediction: float | None = None          # probability of UP move
    nn_confidence: float | None = None
    xgb_prediction: float | None = None
    xgb_confidence: float | None = None
    ensemble_prediction: float | None = None    # weighted average
    ensemble_confidence: float | None = None


class RiskFlags(BaseModel):
    spread_exceeded: bool = False
    drawdown_warning: bool = False
    daily_loss_limit_approaching: bool = False
    timeframe_divergence: bool = False
    low_confidence: bool = False
    custom: list[str] = Field(default_factory=list)

    @property
    def any_flag(self) -> bool:
        return any([
            self.spread_exceeded,
            self.drawdown_warning,
            self.daily_loss_limit_approaching,
            self.timeframe_divergence,
            self.low_confidence,
            bool(self.custom),
        ])


class TradingPayload(BaseModel):
    """Canonical trading signal payload exchanged between all components."""

    version: str = PAYLOAD_VERSION
    cycle_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source: str = AgentID.SYSTEM
    symbol: str = ""
    action: TradingAction = TradingAction.HOLD
    confidence: float = Field(default=0.0)
    reasoning: str = ""
    indicators: list[IndicatorSnapshot] = Field(default_factory=list)
    model_signals: list[ModelSignals] = Field(default_factory=list)
    risk_flags: RiskFlags = Field(default_factory=RiskFlags)
    metadata: dict[str, Any] = Field(default_factory=dict)
    signature: str | None = None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    def to_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "TradingPayload":
        data = json.loads(raw)
        return cls(**data)

    def sign(self, secret: str) -> "TradingPayload":
        """Attach HMAC-SHA256 signature to the payload."""
        body = self.model_dump()
        body.pop("signature", None)
        message = json.dumps(body, sort_keys=True).encode(PAYLOAD_ENCODING)
        sig = hmac.new(
            secret.encode(PAYLOAD_ENCODING), message, hashlib.sha256
        ).hexdigest()
        return self.model_copy(update={"signature": sig})

    def verify_signature(self, secret: str) -> bool:
        """Verify the HMAC-SHA256 signature."""
        if not self.signature:
            return False
        expected = self.sign(secret).signature
        return hmac.compare_digest(self.signature or "", expected or "")


# ---------------------------------------------------------------------------
# Builder (fluent API)
# ---------------------------------------------------------------------------

class PayloadBuilder:
    """Fluent builder for constructing TradingPayload instances."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def source(self, agent_id: str) -> "PayloadBuilder":
        self._data["source"] = agent_id
        return self

    def symbol(self, symbol: str) -> "PayloadBuilder":
        self._data["symbol"] = symbol
        return self

    def action(self, action: TradingAction) -> "PayloadBuilder":
        self._data["action"] = action
        return self

    def confidence(self, value: float) -> "PayloadBuilder":
        self._data["confidence"] = value
        return self

    def reasoning(self, text: str) -> "PayloadBuilder":
        self._data["reasoning"] = text
        return self

    def add_indicator(self, snapshot: IndicatorSnapshot) -> "PayloadBuilder":
        self._data.setdefault("indicators", []).append(snapshot)
        return self

    def add_model_signal(self, signal: ModelSignals) -> "PayloadBuilder":
        self._data.setdefault("model_signals", []).append(signal)
        return self

    def risk_flags(self, flags: RiskFlags) -> "PayloadBuilder":
        self._data["risk_flags"] = flags
        return self

    def metadata(self, **kwargs: Any) -> "PayloadBuilder":
        self._data.setdefault("metadata", {}).update(kwargs)
        return self

    def build(self) -> TradingPayload:
        return TradingPayload(**self._data)


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def merge_payloads(
    payloads: list[TradingPayload],
    symbol: str = "",
) -> TradingPayload:
    """
    Merge multiple agent payloads into a single consensus payload.

    Strategy:
    - Action: majority vote (tie → HOLD)
    - Confidence: weighted average by individual confidence
    - Risk flags: union (any flag set → set in merged)
    - Reasoning: concatenated summaries
    """
    if not payloads:
        return PayloadBuilder().symbol(symbol).build()

    from collections import Counter

    action_votes = Counter(p.action for p in payloads)
    best_action, best_count = action_votes.most_common(1)[0]

    # Majority rule: need > half the votes
    majority_action = (
        best_action if best_count > len(payloads) / 2 else TradingAction.HOLD
    )

    total_conf_weight = sum(p.confidence for p in payloads)
    if total_conf_weight > 0:
        avg_confidence = sum(
            p.action == majority_action and p.confidence or 0.0
            for p in payloads
        ) / total_conf_weight
    else:
        avg_confidence = 0.0

    merged_risk = RiskFlags(
        spread_exceeded=any(p.risk_flags.spread_exceeded for p in payloads),
        drawdown_warning=any(p.risk_flags.drawdown_warning for p in payloads),
        daily_loss_limit_approaching=any(
            p.risk_flags.daily_loss_limit_approaching for p in payloads
        ),
        timeframe_divergence=any(
            p.risk_flags.timeframe_divergence for p in payloads
        ),
        low_confidence=any(p.risk_flags.low_confidence for p in payloads),
        custom=list(
            {flag for p in payloads for flag in p.risk_flags.custom}
        ),
    )

    reasoning_parts = [
        f"[{p.source}] {p.reasoning}" for p in payloads if p.reasoning
    ]

    # Collect all indicators and model signals
    all_indicators: list[IndicatorSnapshot] = []
    all_signals: list[ModelSignals] = []
    for p in payloads:
        all_indicators.extend(p.indicators)
        all_signals.extend(p.model_signals)

    return (
        PayloadBuilder()
        .source(AgentID.ORCHESTRATOR)
        .symbol(symbol)
        .action(majority_action)
        .confidence(round(avg_confidence, 4))
        .reasoning(" | ".join(reasoning_parts))
        .risk_flags(merged_risk)
        .metadata(
            agent_count=len(payloads),
            vote_distribution={k.value: v for k, v in action_votes.items()},
        )
        .build()
        ._model_copy_with_lists(all_indicators, all_signals)
    )


# Monkey-patch helper (avoid modifying Pydantic internals)
def _model_copy_with_lists(
    payload: TradingPayload,
    indicators: list[IndicatorSnapshot],
    signals: list[ModelSignals],
) -> TradingPayload:
    return payload.model_copy(
        update={"indicators": indicators, "model_signals": signals}
    )


TradingPayload._model_copy_with_lists = _model_copy_with_lists  # type: ignore[attr-defined]
