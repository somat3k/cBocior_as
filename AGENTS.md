# AGENTS — Delegation Rules & Instructions

This file defines the roles, permissions, communication protocols, and
behavioral constraints for every AI agent operating inside `cBocior_as`.
It is the authoritative reference for agent orchestration while the bot is
in production.

---

## 1. Agent Roster

| Agent ID | Provider | Primary Role | Secondary Role |
|---|---|---|---|
| `groq` | Groq (OSS 120B via `GROQ_MODEL`) | Single-agent signal generation | Low-latency signal scoring |

---

## 2. Orchestration Rules

### 2.1 Normal Operation (Groq-only)
1. **Groq OSS 120B** (via `GROQ_MODEL`) produces the trading signal (`BUY` / `SELL` / `HOLD`
   + confidence 0–1).
2. The orchestrator returns the Groq payload directly.

### 2.2 Degraded Operation (Groq unavailable)
* If Groq fails or times out → return a HOLD payload and log a critical alert.

### 2.3 Agent Call Budget
* Maximum `BOT_MAX_CONCURRENT_AGENTS` (default 1) simultaneous API calls.
* Groq has a per-call timeout of **10 seconds**.
* Rate-limit back-off: exponential, base 2 s, max 60 s, max 5 retries.

---

## 3. Communication Protocol

All inter-agent messages are **JSON payloads** conforming to the schema
defined in `src/utils/payload.py`.  Key fields:

```json
{
  "version": "1.0",
  "timestamp": "ISO-8601",
  "source": "<agent_id>",
  "action": "BUY | SELL | HOLD",
  "confidence": 0.0,
  "reasoning": "...",
  "indicators": { ... },
  "model_signals": { ... },
  "risk_flags": []
}
```

* Agents **must** produce a valid payload; malformed responses are rejected
  and the agent is marked degraded for the current cycle.
* Payloads are logged via LangSmith for audit and prompt refinement.

---

## 4. Agent Behavioral Constraints

### All Agents
* **Never** place a trade directly.  All agents output *recommendations*;
  only the `DecisionEngine` (`src/trading/decision_engine.py`) executes.
* **Never** bypass the `RiskManager` (`src/trading/risk_manager.py`).
* Agents must include `risk_flags` for any of the following conditions:
  - Spread > `TRADING_MAX_SPREAD_PIPS`
  - Drawdown approaching `RISK_MAX_DRAWDOWN_PCT`
  - Conflicting signals across timeframes
  - Model confidence < 0.55

### Groq Agent (`groq_agent.py`)
* Single-agent signal generator using the OSS 120B model (via `GROQ_MODEL`).
* Must respond within **10 seconds** or the cycle returns HOLD.
* Output language: English, concise (< 200 words in `reasoning`).

---

## 5. Model Integration Rules

* Agents **may** request model re-inference via the orchestrator, but
  **may not** trigger model retraining during live trading.
* Retraining is only allowed during the scheduled maintenance window
  (defined in `TODO.md`, Epic C).
* Model predictions are injected into agent prompts as structured JSON
  under the `model_signals` key.

---

## 6. LangSmith Tracing Policy

* Every agent call **must** be traced when `LANGCHAIN_TRACING_V2=true`.
* Traces include: prompt, response, latency, token usage, and final payload.
* Traces are tagged with `symbol`, `timeframe`, and `cycle_id` for
  filtering in the LangSmith dashboard.

---

## 7. Escalation Path

```
Agent failure
  └─► Orchestrator marks agent DEGRADED
        └─► Orchestrator retries (back-off)
              └─► If max retries exceeded → HOLD trade
                    └─► Alert logged (structlog + LangSmith)
                          └─► Human review (monitor LangSmith dashboard)
```

---

## 8. Versioning

This file is versioned alongside the codebase.  Any change to agent roles,
protocols, or constraints **must** be reviewed via pull request before
merging to `main`.
