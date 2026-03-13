# TODO — cBocior_as Development Roadmap

This file tracks all planned work as **Epics A–Z** with detailed sub-tasks.
Status markers: `[ ]` pending · `[~]` in progress · `[x]` done

---

## Epic A — Project Bootstrap & Infrastructure

- [x] A1. Initialise repository structure (`src/`, `calgo/`, `data/`, `exports/`)
- [x] A2. Define all constants in `constants.py` with GitHub Actions secrets
- [x] A3. Create `.env.example` with every required variable documented
- [x] A4. Write `SECRETS.md` listing all GitHub Actions secrets with sources
- [x] A5. Create `AGENTS.md` with delegation rules and behavioral constraints
- [x] A6. Create `requirements.txt` (no TF/Torch/ONNX)
- [x] A7. Add `.gitignore` covering models, CSVs, venvs, secrets
- [x] A8. Set up GitHub Actions CI workflow (lint + test on push)
- [x] A9. Write initial `README.md` with architecture overview
- [x] A10. Configure `structlog` + `rich` logging pipeline
- [x] A11. Add `pyproject.toml` for linting (ruff/flake8) and formatting
- [ ] A12. Tag first release `v0.1.0-scaffold`

---

## Epic B — cTrader Open API Integration

- [x] B1. Implement `CTraderClient` wrapper (`src/data/ctrader_client.py`)
- [ ] B2. Implement OAuth2 token refresh flow
- [ ] B3. Subscribe to live trendbars for 1 m, 5 m, 1 H via Protobuf
- [ ] B4. Implement symbol lookup (name → symbol ID)
- [x] B5. Implement historical trendbar fetch (backfill)
- [x] B6. Write `DataFetcher` that orchestrates multi-timeframe data collection
- [x] B7. Export OHLCV data to CSV with consistent schema
- [ ] B8. Validate CSV output (no gaps, correct OHLCV types)
- [ ] B9. Implement reconnect / heartbeat logic (ping every 25 s)
- [x] B10. Add unit tests for `CTraderClient` (mock TCP responses)
- [ ] B11. Document Protobuf message types used
- [ ] B12. Instrument with LangSmith events for API call tracing

---

## Epic C — Model Training Pipeline

- [ ] C1. Design feature engineering pipeline (`FeatureBuilder`)
- [x] C2. Compute multiplex indicators for 1 m, 5 m, 1 H in `indicators.py`
- [x] C3. Implement numpy-based Neural Network class (`NeuralNetwork`)
- [x] C4. Implement backpropagation + mini-batch gradient descent
- [x] C5. Add dropout regularisation in the numpy NN
- [ ] C6. Implement quantum-inspired annealing optimiser (`QuantumAnnealer`)
- [x] C7. Implement quantum-inspired particle swarm for hyperparameter search
- [x] C8. Implement `ModelTrainer` with 1 m (10 000 bars, 200 epochs) schedule
- [x] C9. Implement 5 m training (5 000 bars, 200 epochs)
- [x] C10. Implement 1 H training (2 000 bars, 200 epochs)
- [x] C11. Early-stopping callback based on validation loss plateau
- [x] C12. Persist trained models with `joblib` to `MODEL_EXPORT_DIR`
- [x] C13. Persist feature scalers alongside models
- [ ] C14. Write integration test: train on synthetic data, assert artefacts exist
- [ ] C15. Add training progress bar via `rich`

---

## Epic D — Technical Indicators (Multiplex System)

- [x] D1. RSI (14-period) on all three timeframes
- [x] D2. MACD (12/26/9) on all three timeframes
- [x] D3. Bollinger Bands (20/2σ) on all three timeframes
- [x] D4. EMA family (9, 21, 50, 200) on all three timeframes
- [x] D5. ATR (14) for volatility normalisation
- [x] D6. Stochastic Oscillator (K14/D3/smooth3)
- [x] D7. Volume indicators (OBV, VWAP where available)
- [x] D8. Composite signal aggregator (weighted majority vote)
- [ ] D9. Cross-timeframe divergence detector
- [x] D10. Indicator normalisation to [0, 1] for model input
- [x] D11. Add indicator snapshot to JSON payload
- [ ] D12. Unit test each indicator against known reference values

---

## Epic E — AI Agent Framework

- [x] E1. Implement `BaseAgent` abstract class with retry + timeout logic
- [x] E2. Implement `OpenAIAgent` (GPT-4o, LangSmith-traced)
- [x] E3. Implement `GeminiAgent` (Gemini 1.5 Pro)
- [x] E4. Implement `GroqAgent` (LLaMA 3 70B, low-latency path)
- [x] E5. Implement `OpenRouterAgent` (consensus arbiter)
- [x] E6. Implement `AgentOrchestrator` (fan-out + majority vote)
- [ ] E7. Define LangSmith prompt templates for each agent role
- [ ] E8. Build payload formatter injecting indicators + model signals
- [x] E9. Implement agent degraded-mode fallback chain
- [x] E10. Add per-agent rate-limit back-off with exponential jitter
- [ ] E11. Unit tests: mock each provider, assert correct payload schema
- [ ] E12. Integration test: orchestrator with 4 mock agents, assert majority rule

---

## Epic F — JSON Payload System

- [x] F1. Design canonical payload schema (version, timestamp, source, action...)
- [ ] F2. Implement `PayloadBuilder` with builder pattern
- [x] F3. Implement `PayloadValidator` using Pydantic models
- [ ] F4. Serialise / deserialise with `ujson` for performance
- [ ] F5. Add payload versioning + migration stubs
- [ ] F6. Log every payload to LangSmith with tags
- [ ] F7. Add payload signing (HMAC-SHA256) for integrity
- [x] F8. Unit tests for builder, validator, serialisation round-trip
- [ ] F9. Document JSON schema in `docs/payload_schema.md`
- [ ] F10. Compression option (gzip) for large historical payloads

---

## Epic G — Market Analysis Engine

- [x] G1. Implement `PatternDetector` (candlestick + indicator patterns)
- [x] G2. Detect: engulfing, doji, hammer, shooting-star patterns
- [ ] G3. Detect: double-top/bottom, head-and-shoulders via price action
- [x] G4. Implement `MarketAnalyzer` (trend + momentum + volatility regime)
- [x] G5. Implement market regime classifier (trending / ranging / volatile)
- [x] G6. `SignalEngine`: combine pattern + indicator + model signals
- [x] G7. Signal strength scoring (0–100 composite)
- [ ] G8. Divergence alerts (price vs indicator)
- [ ] G9. Session filter (London / New York / overlap boost)
- [x] G10. Unit tests: assert correct regime classification on known data
- [x] G11. Back-test signal accuracy on historical CSV data
- [ ] G12. Add analysis output to canonical JSON payload

---

## Epic H — Decision Engine & Trade Execution

- [x] H1. Implement `DecisionEngine` consuming orchestrator payload
- [x] H2. Decision threshold: require ≥ 0.65 confidence to trade
- [x] H3. De-duplication: skip if identical signal within cooldown window
- [x] H4. Build order payload (`MARKET_ORDER`, `LIMIT_ORDER`)
- [x] H5. Submit order via `CTraderClient`
- [x] H6. Handle order acknowledgement + rejection
- [x] H7. Implement `Execution` class tracking open positions
- [ ] H8. Auto-adjust SL/TP based on ATR
- [ ] H9. Trailing stop logic
- [ ] H10. Position reconciliation on reconnect
- [x] H11. Unit tests: mock cTrader client, assert correct order params
- [ ] H12. Integration test: full cycle from signal to order submission

---

## Epic I — Risk Management

- [x] I1. Implement `RiskManager` class
- [x] I2. Daily loss limit enforcer (halt trading when exceeded)
- [x] I3. Max drawdown enforcer
- [x] I4. Position sizing calculator (fractional Kelly)
- [x] I5. Spread filter (reject orders above `TRADING_MAX_SPREAD_PIPS`)
- [ ] I6. Margin check before order submission
- [ ] I7. Correlation filter (limit multiple positions in correlated pairs)
- [ ] I8. News event blackout window (configurable)
- [ ] I9. Risk report payload appended to each decision payload
- [x] I10. Emergency stop: all positions closed if drawdown > 2× limit
- [x] I11. Unit tests covering every risk rule

---

## Epic J — Continuous Analysis Loop

- [x] J1. Implement `BotRunner` main event loop (`calgo/cbot_runner.py`)
- [x] J2. Tick-driven loop: subscribe to live trendbars, trigger analysis
- [x] J3. Cooldown logic: `BOT_ANALYSIS_COOLDOWN_SECONDS` between full analyses
- [x] J4. Integrate model inference into the loop
- [x] J5. Integrate orchestrator call into the loop
- [x] J6. Integrate `DecisionEngine` + `RiskManager` into the loop
- [x] J7. Graceful shutdown (SIGTERM / SIGINT handlers)
- [ ] J8. Automatic model re-load on file change (hot-reload)
- [ ] J9. Loop health-check endpoint (local HTTP `/health`)
- [ ] J10. Loop telemetry pushed to LangSmith every `BOT_LOOP_INTERVAL_SECONDS`
- [ ] J11. End-to-end smoke test with mock cTrader server

---

## Epic K — Model Export & Versioning

- [x] K1. Design model versioning scheme (`symbol_tf_v{n}.joblib`)
- [ ] K2. Keep last 3 versions; auto-archive older ones
- [x] K3. Model metadata JSON alongside each artefact
- [x] K4. Model registry (`exports/registry.json`) tracking all versions
- [ ] K5. A/B model comparison: track live performance per version
- [ ] K6. Automatic roll-back if new model underperforms baseline by > 5 %
- [ ] K7. Upload models to GitHub Release as artefacts via Actions
- [x] K8. Model integrity check (hash comparison) on load
- [x] K9. Unit tests for versioning and registry operations
- [ ] K10. Document model lifecycle in `docs/model_lifecycle.md`

---

## Epic L — LangSmith Prompt Engineering

- [ ] L1. Define base prompt templates for each agent role
- [ ] L2. Implement `PromptBuilder` using LangSmith's prompt hub
- [ ] L3. Groq speed-optimised prompt (< 500 tokens)
- [ ] L4. Gemini multi-timeframe analysis prompt
- [ ] L5. OpenAI trade justification prompt
- [ ] L6. OpenRouter consensus prompt with all agent inputs
- [ ] L7. Prompt versioning in LangSmith
- [ ] L8. Prompt performance metrics (win-rate by prompt version)
- [ ] L9. Feedback loop: human ratings pushed back to LangSmith
- [ ] L10. A/B test new prompt versions against baseline

---

## Epic M — GitHub Actions CI/CD

- [x] M1. Lint workflow (ruff) on pull requests
- [x] M2. Unit test workflow (pytest) on push to main + PRs
- [x] M3. Model training workflow (scheduled nightly)
- [ ] M4. Model export + GitHub Release upload workflow
- [ ] M5. Dependency vulnerability scan (pip-audit)
- [ ] M6. Secret rotation reminder workflow (monthly)
- [ ] M7. Deployment workflow: push cBot to production server
- [ ] M8. Rollback workflow: redeploy previous release
- [x] M9. Status badge in README
- [x] M10. Dependabot configuration for automatic dependency updates

---

## Epic N — Documentation

- [ ] N1. Architecture overview diagram (`docs/architecture.md`)
- [ ] N2. Quickstart guide (`docs/quickstart.md`)
- [ ] N3. API reference (`docs/api_reference.md`)
- [ ] N4. JSON payload schema reference (`docs/payload_schema.md`)
- [ ] N5. Model lifecycle guide (`docs/model_lifecycle.md`)
- [ ] N6. Agent configuration guide (`docs/agents_guide.md`)
- [ ] N7. cTrader setup guide (`docs/ctrader_setup.md`)
- [ ] N8. Troubleshooting guide (`docs/troubleshooting.md`)
- [ ] N9. CHANGELOG.md
- [ ] N10. Contributing guide (CONTRIBUTING.md)

---

## Epic O — Observability & Monitoring

- [x] O1. Structured logging with `structlog` (JSON lines)
- [ ] O2. Per-cycle metrics: latency, agent count, confidence, action
- [ ] O3. LangSmith dashboard: trace, latency, token cost views
- [ ] O4. Prometheus metrics endpoint (optional)
- [ ] O5. Alerts: email/Slack when daily loss limit approaches
- [ ] O6. Model drift detector: alert when prediction distribution shifts
- [ ] O7. Weekly automated performance report
- [ ] O8. Uptime monitor (ping `/health` endpoint)
- [ ] O9. Log rotation (retain 30 days)
- [ ] O10. Anomaly detection on loop timings

---

## Epic P — Back-testing Framework

- [x] P1. Implement `Backtester` class
- [ ] P2. Feed historical CSV through the full signal pipeline
- [x] P3. Simulate order fills with configurable spread + slippage
- [x] P4. Compute metrics: net PnL, win rate, max drawdown, Sharpe ratio
- [ ] P5. Walk-forward optimisation of indicator parameters
- [ ] P6. Monte Carlo simulation for robustness testing
- [ ] P7. Comparison report: multiple model versions
- [ ] P8. Export back-test results to CSV + HTML report
- [ ] P9. CI workflow: run back-test on every model retrain
- [ ] P10. Regression test: assert new model ≥ baseline Sharpe on held-out set

---

## Epic Q — Quantum-Inspired Optimisation

- [ ] Q1. Implement `QuantumAnnealer` (simulated quantum annealing with numpy)
- [x] Q2. Quantum-inspired particle swarm for hyperparameter search
- [ ] Q3. Variational quantum circuit approximation for feature weighting
- [ ] Q4. Phase estimation for signal frequency analysis
- [x] Q5. Integrate QA optimiser into `ModelTrainer` hyperparameter search
- [ ] Q6. Benchmark QA vs random search on training data
- [ ] Q7. Document quantum-inspired approach in `docs/quantum_approach.md`
- [x] Q8. Unit tests for `QuantumAnnealer` convergence properties
- [ ] Q9. Tune `QA_NUM_QUBITS`, `QA_ITERATIONS`, temperature schedule
- [ ] Q10. Publish results comparison in `docs/benchmarks.md`

---

## Epic R — Security & Compliance

- [ ] R1. Audit all external API calls (no PII transmitted)
- [ ] R2. Secret rotation procedures documented in `SECRETS.md`
- [ ] R3. Payload signing with HMAC-SHA256
- [ ] R4. TLS verification enforced on all HTTP clients
- [ ] R5. Rate-limit all outbound API calls (avoid abuse)
- [ ] R6. Input validation for all cTrader messages
- [x] R7. CodeQL scan integrated in CI
- [ ] R8. Dependency vulnerability scan (pip-audit) in CI
- [ ] R9. Principle of least privilege for GitHub Actions secrets
- [ ] R10. Penetration-test plan for local HTTP health endpoint
- [ ] R11. GDPR / data retention policy for trade logs

---

## Epic S — Scalability & Multi-Symbol Support

- [x] S1. Parameterise all symbol-specific code
- [ ] S2. Implement symbol portfolio manager
- [x] S3. Parallel data fetching for multiple symbols
- [x] S4. Separate models per symbol (EURUSD, GBPUSD, USDJPY, etc.)
- [ ] S5. Shared orchestrator with per-symbol agent context
- [ ] S6. Position correlation matrix across symbols
- [ ] S7. Portfolio-level risk limits
- [ ] S8. Performance attribution by symbol
- [ ] S9. Load test: 10 simultaneous symbols
- [ ] S10. Horizontal scaling design (multiple bot instances)

---

## Epic T — Testing & Quality

- [ ] T1. Unit test coverage ≥ 80 % enforced in CI
- [ ] T2. Integration tests with mock cTrader TCP server
- [ ] T3. Contract tests for every JSON payload schema version
- [ ] T4. Fuzz testing for payload parser
- [ ] T5. Property-based tests with Hypothesis for NN + indicators
- [ ] T6. Mutation testing (mutmut) to assess test quality
- [ ] T7. Load test: 1000 market events/second through analysis pipeline
- [ ] T8. Chaos tests: random agent failures, assert HOLD behaviour
- [ ] T9. End-to-end test in DEMO environment
- [ ] T10. Monthly manual acceptance test checklist

---

## Epic U — User Interface & Visualisation

- [ ] U1. CLI tool `cbocior` for training, status, manual override
- [ ] U2. `cbocior train --symbol EURUSD` command
- [ ] U3. `cbocior status` — print current model versions + loop health
- [ ] U4. `cbocior backtest --from 2023-01-01` command
- [ ] U5. Real-time chart of signals vs price (terminal/matplotlib)
- [ ] U6. Model performance dashboard (simple HTML)
- [ ] U7. Agent response explorer (list recent payloads)
- [ ] U8. Risk dashboard (current drawdown, positions)
- [ ] U9. LangSmith embedded view in documentation
- [ ] U10. `--dry-run` flag for all execution commands

---

## Epic V — Version Management & Releases

- [ ] V1. Semantic versioning (`vMAJOR.MINOR.PATCH`)
- [ ] V2. CHANGELOG.md maintained per release
- [ ] V3. GitHub Release with bundled model artefacts
- [ ] V4. Migration guide for breaking changes
- [ ] V5. Deprecation policy (2 minor versions notice)
- [ ] V6. Automatic release notes generated from commit messages
- [ ] V7. PyPI package (optional, for reuse)
- [ ] V8. Docker image for containerised deployment
- [ ] V9. Version pinning in `requirements.txt`
- [ ] V10. Compatibility matrix (Python versions, OS)

---

## Epic W — WebSocket / Real-Time Feeds (Advanced)

- [ ] W1. Evaluate cTrader FIX protocol as low-latency alternative
- [ ] W2. Implement WebSocket adapter for external price feeds (backup)
- [ ] W3. Multi-feed aggregator with conflict resolution
- [ ] W4. Millisecond timestamp alignment across feeds
- [ ] W5. Feed quality monitor (gaps, outliers)
- [ ] W6. Failover: switch feed source on degraded quality
- [ ] W7. Order book data integration (where available)
- [ ] W8. Latency benchmarks: TCP Protobuf vs WebSocket
- [ ] W9. Document feed architecture in `docs/feeds.md`
- [ ] W10. Alerting on feed latency spikes

---

## Epic X — Cross-Asset & Macro Integration

- [ ] X1. Integrate economic calendar API (news event filter)
- [ ] X2. Correlate FX moves with equity indices
- [ ] X3. Commodity correlation (Gold/Oil vs USD pairs)
- [ ] X4. Central bank decision calendar
- [ ] X5. Sentiment analysis from financial news headlines
- [ ] X6. Options market implied volatility as signal
- [ ] X7. Interest rate differential tracker
- [ ] X8. Macro regime classifier (risk-on / risk-off)
- [ ] X9. Inject macro context into agent payloads
- [ ] X10. Back-test: macro-filtered vs unfiltered strategy

---

## Epic Y — Advanced ML Research

- [ ] Y1. Evaluate Gaussian Process models for uncertainty quantification
- [ ] Y2. Implement LSTM substitute using numpy autoregressive model
- [ ] Y3. Evaluate XGBoost + LightGBM ensemble vs numpy NN
- [ ] Y4. Feature importance analysis (SHAP values)
- [ ] Y5. Bayesian hyperparameter optimisation
- [ ] Y6. Online learning: incremental model updates on live trades
- [ ] Y7. Anomaly detection model for outlier bar filtering
- [ ] Y8. Regime-adaptive model switching
- [ ] Y9. Kalman filter for signal smoothing
- [ ] Y10. Research log: document experiments and results

---

## Epic Z — Long-Term Vision & Roadmap

- [ ] Z1. Evolve cBot into a full autonomous trading system
- [ ] Z2. Support 20+ currency pairs simultaneously
- [ ] Z3. Build proprietary high-frequency indicator library
- [ ] Z4. Develop agent memory layer (persistent context across cycles)
- [ ] Z5. Implement self-healing: bot diagnoses and fixes its own failures
- [ ] Z6. Live paper trading competition framework
- [ ] Z7. Open-source community edition
- [ ] Z8. Institutional-grade audit trail
- [ ] Z9. Regulatory compliance module (MiFID II reporting)
- [ ] Z10. Partner integrations: prime brokers, liquidity providers
- [ ] Z11. Research publication on quantum-inspired trading optimisation
- [ ] Z12. Spin-off: generalised AI-trading framework package


---

## Epic B — cTrader Open API Integration

- [ ] B1. Implement `CTraderClient` wrapper (`src/data/ctrader_client.py`)
- [ ] B2. Implement OAuth2 token refresh flow
- [ ] B3. Subscribe to live trendbars for 1 m, 5 m, 1 H via Protobuf
- [ ] B4. Implement symbol lookup (name → symbol ID)
- [ ] B5. Implement historical trendbar fetch (backfill)
- [ ] B6. Write `DataFetcher` that orchestrates multi-timeframe data collection
- [ ] B7. Export OHLCV data to CSV with consistent schema
- [ ] B8. Validate CSV output (no gaps, correct OHLCV types)
- [ ] B9. Implement reconnect / heartbeat logic (ping every 25 s)
- [ ] B10. Add unit tests for `CTraderClient` (mock TCP responses)
- [ ] B11. Document Protobuf message types used
- [ ] B12. Instrument with LangSmith events for API call tracing

---

## Epic C — Model Training Pipeline

- [ ] C1. Design feature engineering pipeline (`FeatureBuilder`)
- [ ] C2. Compute multiplex indicators for 1 m, 5 m, 1 H in `indicators.py`
- [ ] C3. Implement numpy-based Neural Network class (`NeuralNetwork`)
- [ ] C4. Implement backpropagation + mini-batch gradient descent
- [ ] C5. Add dropout regularisation in the numpy NN
- [ ] C6. Implement quantum-inspired annealing optimiser (`QuantumAnnealer`)
- [ ] C7. Implement quantum-inspired particle swarm for hyperparameter search
- [ ] C8. Implement `ModelTrainer` with 1 m (2000 trades, 200 epochs) schedule
- [ ] C9. Implement 5 m training (1000 trades, 200 epochs)
- [ ] C10. Implement 1 H training (250 trades, 200 epochs)
- [ ] C11. Early-stopping callback based on validation loss plateau
- [ ] C12. Persist trained models with `joblib` to `MODEL_EXPORT_DIR`
- [ ] C13. Persist feature scalers alongside models
- [ ] C14. Write integration test: train on synthetic data, assert artefacts exist
- [ ] C15. Add training progress bar via `rich`

---

## Epic D — Technical Indicators (Multiplex System)

- [ ] D1. RSI (14-period) on all three timeframes
- [ ] D2. MACD (12/26/9) on all three timeframes
- [ ] D3. Bollinger Bands (20/2σ) on all three timeframes
- [ ] D4. EMA family (9, 21, 50, 200) on all three timeframes
- [ ] D5. ATR (14) for volatility normalisation
- [ ] D6. Stochastic Oscillator (K14/D3/smooth3)
- [ ] D7. Volume indicators (OBV, VWAP where available)
- [ ] D8. Composite signal aggregator (weighted majority vote)
- [ ] D9. Cross-timeframe divergence detector
- [ ] D10. Indicator normalisation to [0, 1] for model input
- [ ] D11. Add indicator snapshot to JSON payload
- [ ] D12. Unit test each indicator against known reference values

---

## Epic E — AI Agent Framework

- [ ] E1. Implement `BaseAgent` abstract class with retry + timeout logic
- [ ] E2. Implement `OpenAIAgent` (GPT-4o, LangSmith-traced)
- [ ] E3. Implement `GeminiAgent` (Gemini 1.5 Pro)
- [ ] E4. Implement `GroqAgent` (LLaMA 3 70B, low-latency path)
- [ ] E5. Implement `OpenRouterAgent` (consensus arbiter)
- [ ] E6. Implement `AgentOrchestrator` (fan-out + majority vote)
- [ ] E7. Define LangSmith prompt templates for each agent role
- [ ] E8. Build payload formatter injecting indicators + model signals
- [ ] E9. Implement agent degraded-mode fallback chain
- [ ] E10. Add per-agent rate-limit back-off with exponential jitter
- [ ] E11. Unit tests: mock each provider, assert correct payload schema
- [ ] E12. Integration test: orchestrator with 4 mock agents, assert majority rule

---

## Epic F — JSON Payload System

- [ ] F1. Design canonical payload schema (version, timestamp, source, action...)
- [ ] F2. Implement `PayloadBuilder` with builder pattern
- [ ] F3. Implement `PayloadValidator` using Pydantic models
- [ ] F4. Serialise / deserialise with `ujson` for performance
- [ ] F5. Add payload versioning + migration stubs
- [ ] F6. Log every payload to LangSmith with tags
- [ ] F7. Add payload signing (HMAC-SHA256) for integrity
- [ ] F8. Unit tests for builder, validator, serialisation round-trip
- [ ] F9. Document JSON schema in `docs/payload_schema.md`
- [ ] F10. Compression option (gzip) for large historical payloads

---

## Epic G — Market Analysis Engine

- [ ] G1. Implement `PatternDetector` (candlestick + indicator patterns)
- [ ] G2. Detect: engulfing, doji, hammer, shooting-star patterns
- [ ] G3. Detect: double-top/bottom, head-and-shoulders via price action
- [ ] G4. Implement `MarketAnalyzer` (trend + momentum + volatility regime)
- [ ] G5. Implement market regime classifier (trending / ranging / volatile)
- [ ] G6. `SignalEngine`: combine pattern + indicator + model signals
- [ ] G7. Signal strength scoring (0–100 composite)
- [ ] G8. Divergence alerts (price vs indicator)
- [ ] G9. Session filter (London / New York / overlap boost)
- [ ] G10. Unit tests: assert correct regime classification on known data
- [ ] G11. Back-test signal accuracy on historical CSV data
- [ ] G12. Add analysis output to canonical JSON payload

---

## Epic H — Decision Engine & Trade Execution

- [ ] H1. Implement `DecisionEngine` consuming orchestrator payload
- [ ] H2. Decision threshold: require ≥ 0.65 confidence to trade
- [ ] H3. De-duplication: skip if identical signal within cooldown window
- [ ] H4. Build order payload (`MARKET_ORDER`, `LIMIT_ORDER`)
- [ ] H5. Submit order via `CTraderClient`
- [ ] H6. Handle order acknowledgement + rejection
- [ ] H7. Implement `Execution` class tracking open positions
- [ ] H8. Auto-adjust SL/TP based on ATR
- [ ] H9. Trailing stop logic
- [ ] H10. Position reconciliation on reconnect
- [ ] H11. Unit tests: mock cTrader client, assert correct order params
- [ ] H12. Integration test: full cycle from signal to order submission

---

## Epic I — Risk Management

- [ ] I1. Implement `RiskManager` class
- [ ] I2. Daily loss limit enforcer (halt trading when exceeded)
- [ ] I3. Max drawdown enforcer
- [ ] I4. Position sizing calculator (fractional Kelly)
- [ ] I5. Spread filter (reject orders above `TRADING_MAX_SPREAD_PIPS`)
- [ ] I6. Margin check before order submission
- [ ] I7. Correlation filter (limit multiple positions in correlated pairs)
- [ ] I8. News event blackout window (configurable)
- [ ] I9. Risk report payload appended to each decision payload
- [ ] I10. Emergency stop: all positions closed if drawdown > 2× limit
- [ ] I11. Unit tests covering every risk rule

---

## Epic J — Continuous Analysis Loop

- [ ] J1. Implement `BotRunner` main event loop (`calgo/cbot_runner.py`)
- [ ] J2. Tick-driven loop: subscribe to live trendbars, trigger analysis
- [ ] J3. Cooldown logic: `BOT_ANALYSIS_COOLDOWN_SECONDS` between full analyses
- [ ] J4. Integrate model inference into the loop
- [ ] J5. Integrate orchestrator call into the loop
- [ ] J6. Integrate `DecisionEngine` + `RiskManager` into the loop
- [ ] J7. Graceful shutdown (SIGTERM / SIGINT handlers)
- [ ] J8. Automatic model re-load on file change (hot-reload)
- [ ] J9. Loop health-check endpoint (local HTTP `/health`)
- [ ] J10. Loop telemetry pushed to LangSmith every `BOT_LOOP_INTERVAL_SECONDS`
- [ ] J11. End-to-end smoke test with mock cTrader server

---

## Epic K — Model Export & Versioning

- [ ] K1. Design model versioning scheme (`symbol_tf_v{n}.joblib`)
- [ ] K2. Keep last 3 versions; auto-archive older ones
- [ ] K3. Model metadata JSON alongside each artefact
- [ ] K4. Model registry (`exports/registry.json`) tracking all versions
- [ ] K5. A/B model comparison: track live performance per version
- [ ] K6. Automatic roll-back if new model underperforms baseline by > 5 %
- [ ] K7. Upload models to GitHub Release as artefacts via Actions
- [ ] K8. Model integrity check (hash comparison) on load
- [ ] K9. Unit tests for versioning and registry operations
- [ ] K10. Document model lifecycle in `docs/model_lifecycle.md`

---

## Epic L — LangSmith Prompt Engineering

- [ ] L1. Define base prompt templates for each agent role
- [ ] L2. Implement `PromptBuilder` using LangSmith's prompt hub
- [ ] L3. Groq speed-optimised prompt (< 500 tokens)
- [ ] L4. Gemini multi-timeframe analysis prompt
- [ ] L5. OpenAI trade justification prompt
- [ ] L6. OpenRouter consensus prompt with all agent inputs
- [ ] L7. Prompt versioning in LangSmith
- [ ] L8. Prompt performance metrics (win-rate by prompt version)
- [ ] L9. Feedback loop: human ratings pushed back to LangSmith
- [ ] L10. A/B test new prompt versions against baseline

---

## Epic M — GitHub Actions CI/CD

- [ ] M1. Lint workflow (ruff) on pull requests
- [ ] M2. Unit test workflow (pytest) on push to main + PRs
- [ ] M3. Model training workflow (scheduled nightly)
- [ ] M4. Model export + GitHub Release upload workflow
- [ ] M5. Dependency vulnerability scan (pip-audit)
- [ ] M6. Secret rotation reminder workflow (monthly)
- [ ] M7. Deployment workflow: push cBot to production server
- [ ] M8. Rollback workflow: redeploy previous release
- [ ] M9. Status badge in README
- [ ] M10. Dependabot configuration for automatic dependency updates

---

## Epic N — Documentation

- [ ] N1. Architecture overview diagram (`docs/architecture.md`)
- [ ] N2. Quickstart guide (`docs/quickstart.md`)
- [ ] N3. API reference (`docs/api_reference.md`)
- [ ] N4. JSON payload schema reference (`docs/payload_schema.md`)
- [ ] N5. Model lifecycle guide (`docs/model_lifecycle.md`)
- [ ] N6. Agent configuration guide (`docs/agents_guide.md`)
- [ ] N7. cTrader setup guide (`docs/ctrader_setup.md`)
- [ ] N8. Troubleshooting guide (`docs/troubleshooting.md`)
- [ ] N9. CHANGELOG.md
- [ ] N10. Contributing guide (CONTRIBUTING.md)

---

## Epic O — Observability & Monitoring

- [ ] O1. Structured logging with `structlog` (JSON lines)
- [ ] O2. Per-cycle metrics: latency, agent count, confidence, action
- [ ] O3. LangSmith dashboard: trace, latency, token cost views
- [ ] O4. Prometheus metrics endpoint (optional)
- [ ] O5. Alerts: email/Slack when daily loss limit approaches
- [ ] O6. Model drift detector: alert when prediction distribution shifts
- [ ] O7. Weekly automated performance report
- [ ] O8. Uptime monitor (ping `/health` endpoint)
- [ ] O9. Log rotation (retain 30 days)
- [ ] O10. Anomaly detection on loop timings

---

## Epic P — Back-testing Framework

- [ ] P1. Implement `Backtester` class
- [ ] P2. Feed historical CSV through the full signal pipeline
- [ ] P3. Simulate order fills with configurable spread + slippage
- [ ] P4. Compute metrics: net PnL, win rate, max drawdown, Sharpe ratio
- [ ] P5. Walk-forward optimisation of indicator parameters
- [ ] P6. Monte Carlo simulation for robustness testing
- [ ] P7. Comparison report: multiple model versions
- [ ] P8. Export back-test results to CSV + HTML report
- [ ] P9. CI workflow: run back-test on every model retrain
- [ ] P10. Regression test: assert new model ≥ baseline Sharpe on held-out set

---

## Epic Q — Quantum-Inspired Optimisation

- [ ] Q1. Implement `QuantumAnnealer` (simulated quantum annealing with numpy)
- [ ] Q2. Quantum-inspired particle swarm for hyperparameter search
- [ ] Q3. Variational quantum circuit approximation for feature weighting
- [ ] Q4. Phase estimation for signal frequency analysis
- [ ] Q5. Integrate QA optimiser into `ModelTrainer` hyperparameter search
- [ ] Q6. Benchmark QA vs random search on training data
- [ ] Q7. Document quantum-inspired approach in `docs/quantum_approach.md`
- [ ] Q8. Unit tests for `QuantumAnnealer` convergence properties
- [ ] Q9. Tune `QA_NUM_QUBITS`, `QA_ITERATIONS`, temperature schedule
- [ ] Q10. Publish results comparison in `docs/benchmarks.md`

---

## Epic R — Security & Compliance

- [ ] R1. Audit all external API calls (no PII transmitted)
- [ ] R2. Secret rotation procedures documented in `SECRETS.md`
- [ ] R3. Payload signing with HMAC-SHA256
- [ ] R4. TLS verification enforced on all HTTP clients
- [ ] R5. Rate-limit all outbound API calls (avoid abuse)
- [ ] R6. Input validation for all cTrader messages
- [ ] R7. CodeQL scan integrated in CI
- [ ] R8. Dependency vulnerability scan (pip-audit) in CI
- [ ] R9. Principle of least privilege for GitHub Actions secrets
- [ ] R10. Penetration-test plan for local HTTP health endpoint
- [ ] R11. GDPR / data retention policy for trade logs

---

## Epic S — Scalability & Multi-Symbol Support

- [ ] S1. Parameterise all symbol-specific code
- [ ] S2. Implement symbol portfolio manager
- [ ] S3. Parallel data fetching for multiple symbols
- [ ] S4. Separate models per symbol (EURUSD, GBPUSD, USDJPY, etc.)
- [ ] S5. Shared orchestrator with per-symbol agent context
- [ ] S6. Position correlation matrix across symbols
- [ ] S7. Portfolio-level risk limits
- [ ] S8. Performance attribution by symbol
- [ ] S9. Load test: 10 simultaneous symbols
- [ ] S10. Horizontal scaling design (multiple bot instances)

---

## Epic T — Testing & Quality

- [ ] T1. Unit test coverage ≥ 80 % enforced in CI
- [ ] T2. Integration tests with mock cTrader TCP server
- [ ] T3. Contract tests for every JSON payload schema version
- [ ] T4. Fuzz testing for payload parser
- [ ] T5. Property-based tests with Hypothesis for NN + indicators
- [ ] T6. Mutation testing (mutmut) to assess test quality
- [ ] T7. Load test: 1000 market events/second through analysis pipeline
- [ ] T8. Chaos tests: random agent failures, assert HOLD behaviour
- [ ] T9. End-to-end test in DEMO environment
- [ ] T10. Monthly manual acceptance test checklist

---

## Epic U — User Interface & Visualisation

- [ ] U1. CLI tool `cbocior` for training, status, manual override
- [ ] U2. `cbocior train --symbol EURUSD` command
- [ ] U3. `cbocior status` — print current model versions + loop health
- [ ] U4. `cbocior backtest --from 2023-01-01` command
- [ ] U5. Real-time chart of signals vs price (terminal/matplotlib)
- [ ] U6. Model performance dashboard (simple HTML)
- [ ] U7. Agent response explorer (list recent payloads)
- [ ] U8. Risk dashboard (current drawdown, positions)
- [ ] U9. LangSmith embedded view in documentation
- [ ] U10. `--dry-run` flag for all execution commands

---

## Epic V — Version Management & Releases

- [ ] V1. Semantic versioning (`vMAJOR.MINOR.PATCH`)
- [ ] V2. CHANGELOG.md maintained per release
- [ ] V3. GitHub Release with bundled model artefacts
- [ ] V4. Migration guide for breaking changes
- [ ] V5. Deprecation policy (2 minor versions notice)
- [ ] V6. Automatic release notes generated from commit messages
- [ ] V7. PyPI package (optional, for reuse)
- [ ] V8. Docker image for containerised deployment
- [ ] V9. Version pinning in `requirements.txt`
- [ ] V10. Compatibility matrix (Python versions, OS)

---

## Epic W — WebSocket / Real-Time Feeds (Advanced)

- [ ] W1. Evaluate cTrader FIX protocol as low-latency alternative
- [ ] W2. Implement WebSocket adapter for external price feeds (backup)
- [ ] W3. Multi-feed aggregator with conflict resolution
- [ ] W4. Millisecond timestamp alignment across feeds
- [ ] W5. Feed quality monitor (gaps, outliers)
- [ ] W6. Failover: switch feed source on degraded quality
- [ ] W7. Order book data integration (where available)
- [ ] W8. Latency benchmarks: TCP Protobuf vs WebSocket
- [ ] W9. Document feed architecture in `docs/feeds.md`
- [ ] W10. Alerting on feed latency spikes

---

## Epic X — Cross-Asset & Macro Integration

- [ ] X1. Integrate economic calendar API (news event filter)
- [ ] X2. Correlate FX moves with equity indices
- [ ] X3. Commodity correlation (Gold/Oil vs USD pairs)
- [ ] X4. Central bank decision calendar
- [ ] X5. Sentiment analysis from financial news headlines
- [ ] X6. Options market implied volatility as signal
- [ ] X7. Interest rate differential tracker
- [ ] X8. Macro regime classifier (risk-on / risk-off)
- [ ] X9. Inject macro context into agent payloads
- [ ] X10. Back-test: macro-filtered vs unfiltered strategy

---

## Epic Y — Advanced ML Research

- [ ] Y1. Evaluate Gaussian Process models for uncertainty quantification
- [ ] Y2. Implement LSTM substitute using numpy autoregressive model
- [ ] Y3. Evaluate XGBoost + LightGBM ensemble vs numpy NN
- [ ] Y4. Feature importance analysis (SHAP values)
- [ ] Y5. Bayesian hyperparameter optimisation
- [ ] Y6. Online learning: incremental model updates on live trades
- [ ] Y7. Anomaly detection model for outlier bar filtering
- [ ] Y8. Regime-adaptive model switching
- [ ] Y9. Kalman filter for signal smoothing
- [ ] Y10. Research log: document experiments and results

---

## Epic Z — Long-Term Vision & Roadmap

- [ ] Z1. Evolve cBot into a full autonomous trading system
- [ ] Z2. Support 20+ currency pairs simultaneously
- [ ] Z3. Build proprietary high-frequency indicator library
- [ ] Z4. Develop agent memory layer (persistent context across cycles)
- [ ] Z5. Implement self-healing: bot diagnoses and fixes its own failures
- [ ] Z6. Live paper trading competition framework
- [ ] Z7. Open-source community edition
- [ ] Z8. Institutional-grade audit trail
- [ ] Z9. Regulatory compliance module (MiFID II reporting)
- [ ] Z10. Partner integrations: prime brokers, liquidity providers
- [ ] Z11. Research publication on quantum-inspired trading optimisation
- [ ] Z12. Spin-off: generalised AI-trading framework package
