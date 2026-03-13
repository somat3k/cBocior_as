# AGENTS — Delegation Rules & Instructions

This file defines the roles, permissions, communication protocols, and
behavioral constraints for every AI agent operating inside `cBocior_as`.
It is the authoritative reference for agent orchestration while the bot is
in production.

---

## 1. Agent Roster

| Agent ID | Provider | Primary Role | Secondary Role |
|---|---|---|---|
| `openai` | OpenAI (GPT-4o) | Decision reasoning, trade justification | Prompt refinement |
| `gemini` | Google Gemini 1.5 Pro | Pattern synthesis, multi-modal analysis | Indicator interpretation |
| `groq` | Groq (LLaMA 3 70B) | High-speed real-time inference | Low-latency signal scoring |
| `openrouter` | OpenRouter (Claude 3.5) | Cross-model consensus, fallback reasoning | Risk assessment |

---

## 2. Orchestration Rules

### 2.1 Normal Operation (All agents available)
1. **Groq** is called first (lowest latency) to produce a rapid preliminary
   signal score (`BUY` / `SELL` / `HOLD` + confidence 0–1).
2. **Gemini** analyses the latest multi-timeframe indicator payload and
   synthesises a pattern narrative.
3. **OpenAI** evaluates both outputs, adds trade justification, and proposes
   the final action with reasoning.
4. **OpenRouter** acts as consensus arbiter: it receives all three previous
   outputs and emits the final decision payload.
5. The orchestrator accepts the final payload only when ≥ 3 of 4 agents
   agree (majority rule).  Disagreements are logged and trigger a HOLD.

### 2.2 Degraded Operation (1–2 agents unavailable)
* If only **Groq** is available → accept its signal at reduced confidence
  (confidence capped at 0.6).
* If only **OpenAI** is available → accept its signal (confidence capped at
  0.7).
* If no agents respond within `BOT_ANALYSIS_COOLDOWN_SECONDS` → HOLD and
  log a critical alert.

### 2.3 Agent Call Budget
* Maximum `BOT_MAX_CONCURRENT_AGENTS` (default 4) simultaneous API calls.
* Each agent has a per-call timeout of **30 seconds**.
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

### OpenAI Agent (`openai_agent.py`)
* Responsible for **final narrative justification** of every trade.
* Must cite at least one technical indicator and one model signal.
* Output language: English, concise (< 200 words in `reasoning`).

### Gemini Agent (`gemini_agent.py`)
* Specialises in **multi-timeframe pattern synthesis**.
* Receives full OHLCV + indicator payloads for 1 m, 5 m, and 1 H.
* Must flag "timeframe divergence" if signals differ across timeframes.

### Groq Agent (`groq_agent.py`)
* Optimised for **speed**; uses the fastest available model.
* Provides a preliminary score used as a tiebreaker.
* Must respond within **10 seconds** or be skipped for this cycle.

### OpenRouter Agent (`openrouter_agent.py`)
* Acts as **consensus arbiter**.
* Receives outputs from all other agents in its prompt context.
* Emits the final signed-off payload with `source: "openrouter"`.

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
