# CHANGELOG

All notable changes to **cBocior_as** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Dual-account trading** — `AccountConfig` + `_AccountContext` in `CBotRunner`
  trade 10 000 USD and 50 USD accounts simultaneously over a single cTrader
  connection; position sizes scale independently via fractional Kelly.
- **Maximal training data** — bar counts raised to 10 000 (M1), 5 000 (M5),
  2 000 (H1) for maximum HyperLiquid history coverage.
- **HyperLiquid pagination** — `_fetch_from_api` automatically splits requests
  larger than the 5 000-bar API cap across multiple time-windowed pages.
- **Backtester** (`src/models/backtester.py`) — bar-by-bar simulation on the
  validation set reporting win rate, %MDD, balance growth, profit factor, and
  trade cancellation rate.  Spread + slippage simulation added for realistic
  fill modelling (P3).
- **Model registry** (`src/models/registry.py`) — `ModelRegistry` records
  per-model metadata JSON (K3), a central `exports/registry.json` (K4),
  and SHA-256 integrity verification on model load (K8).  Older versions are
  auto-archived with the latest three retained (K2).
- **Training script** (`scripts/train_all.py`) — standalone training and export
  with formatted summary table.
- **Cross-timeframe divergence detector** (D9) in `SignalEngine` — detects
  conflicting model probabilities or RSI signals across timeframes and dampens
  the composite signal accordingly.  Divergence flag propagated to
  `RiskFlags.timeframe_divergence` in `CBotRunner`.
- **Session filter** (G9) in `SignalEngine` — London/NY overlap multiplier
  (1.20), single-session (1.10), Asian (0.80), off-hours (0.70).
- **ATR-based SL/TP** (H8) in `Execution.execute()` — `atr`, `atr_sl_mult`,
  `atr_tp_mult` parameters override fixed-pip distances when ATR is provided.
- **Trailing stop** (H9) — `Execution.update_trailing_stops()` adjusts open
  position SLs as price moves favourably.
- **Margin check** (I6) in `RiskManager.evaluate()` — blocks trades when free
  margin is insufficient for the estimated required margin.
- **Risk report payload** (I9) — `RiskManager.build_risk_report()` returns a
  structured snapshot attached to `TradingPayload.metadata["risk_report"]` at
  each execution.
- **Emergency stop** (I10) — `RiskFlags.emergency_halt` set when drawdown
  exceeds 2× `RISK_MAX_DRAWDOWN_PCT`; blocks all trading.
- **Health-check endpoint** (J9) — `CBotRunner` starts a minimal HTTP server
  on port 8765 serving `GET /health` with JSON status (uptime, cycle count,
  symbol, accounts, running flag).
- **Ruff lint job** (M1) in `.github/workflows/ci.yml`.
- **CI badge** (M9) in README.md.
- **Dependabot** (M10) — `.github/dependabot.yml` for weekly pip and
  GitHub Actions dependency updates.
- **Indicator reference tests** (D12) — `tests/test_indicator_reference.py`
  validates RSI, MACD, Bollinger Bands, EMA, ATR, and Stochastic against
  known reference behaviours.
- **Agent unit tests** (E8) and **orchestrator integration tests** (E9) —
  `tests/test_agents.py` mocks the Groq provider and asserts fallback
  behaviour, retry/degraded logic, and payload schema.
- **Trainer integration test** (C14) — `tests/test_trainer_integration.py`
  trains on synthetic data and asserts artefact creation, registry, and predict.
- **Rich progress bar** (C15) — `ModelTrainer.train_all()` shows a `rich`
  progress bar when the library is available.

### Changed
- Groq OSS 120B is now the sole agent provider; other agent integrations were
  removed along with their dependencies and configuration requirements.
- `RiskFlags` in `payload.py` gains `emergency_halt: bool` field; `merge_payloads`
  and `any_flag` updated accordingly.

---

## [0.1.0] — 2026-03-01

### Added
- Initial project scaffold: `src/`, `calgo/`, `data/`, `exports/` layout.
- `constants.py` with all required environment variables validated at import.
- `.env.example` with every configurable variable documented.
- `SECRETS.md` listing required GitHub Actions secrets.
- `AGENTS.md` defining delegation rules and agent behavioral constraints.
- `requirements.txt` — numpy, pandas, scikit-learn, lightgbm, structlog,
  pydantic, and other runtime dependencies (no TF/Torch/ONNX).
- `.gitignore` covering secrets, CSVs, virtual environments, and build artefacts.
- GitHub Actions CI workflow — pytest on Python 3.11 and 3.12, nightly training.
- `README.md` with full architecture overview.
- `structlog` + `rich` logging pipeline (`src/utils/logger.py`).
- `pyproject.toml` with ruff linting configuration.
- `CTraderClient` wrapper for cTrader Open API TCP/Protobuf (`src/data/ctrader_client.py`).
- `DataFetcher` orchestrating multi-timeframe OHLCV collection.
- Multiplex indicator system (RSI, MACD, BB, EMA 9/21/50/200, ATR, Stochastic,
  OBV) across M1, M5, H1 (`src/models/indicators.py`).
- numpy-based `NeuralNetwork` with backpropagation, mini-batch SGD, and dropout
  (`src/models/neural_network.py`).
- Quantum-inspired particle swarm optimiser for hyperparameter search
  (`src/models/quantum_algo.py`).
- `ModelTrainer` (M1/M5/H1), early stopping, joblib export
  (`src/models/trainer.py`).
- `PatternDetector` — engulfing, doji, hammer, shooting-star
  (`src/analysis/pattern_detector.py`).
- `MarketAnalyzer` — trend, momentum, volatility regime classifier
  (`src/analysis/market_analyzer.py`).
- `SignalEngine` — composite signal scoring with regime adjustment
  (`src/analysis/signal_engine.py`).
- `DecisionEngine` — 0.65 confidence threshold, cooldown de-duplication
  (`src/trading/decision_engine.py`).
- `RiskManager` — spread filter, daily loss limit, drawdown enforcer, fractional
  Kelly position sizing (`src/trading/risk_manager.py`).
- `Execution` — position tracking, dry-run mode, per-account order targeting
  (`src/trading/execution.py`).
- `BaseAgent` abstract class with retry, timeout, exponential back-off, and
  LangSmith tracing (`src/agents/base_agent.py`).
- Groq agent implementation.
- `AgentOrchestrator` — Groq-only orchestration (`src/agents/orchestrator.py`).
- `TradingPayload` Pydantic schema, `PayloadBuilder` fluent API, HMAC-SHA256
  signing (`src/utils/payload.py`).
- Redis-backed market data cache (`src/utils/cache.py`).
- `CBotRunner` — full dual-account trading loop with SIGTERM/SIGINT shutdown
  (`calgo/cbot_runner.py`).
- `MultiSymbolTrainer` + `HyperLiquidFetcher` with pagination
  (`src/models/multi_symbol_trainer.py`, `src/data/hyperliquid_fetcher.py`).
- Comprehensive test suite (161 tests).

[Unreleased]: https://github.com/somat3k/cBocior_as/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/somat3k/cBocior_as/releases/tag/v0.1.0
