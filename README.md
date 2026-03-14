# cBocior_as

[![CI](https://github.com/somat3k/cBocior_as/actions/workflows/ci.yml/badge.svg)](https://github.com/somat3k/cBocior_as/actions/workflows/ci.yml)

An autonomous AI-driven trading cBot for the cTrader platform.

Combines Neural Networks, Quantum-inspired optimisation, and a multi-agent
AI orchestration layer (Gemini, Groq, OpenRouter, OpenAI) to discover
market patterns, evaluate indicators across multiple timeframes, and execute
trades with comprehensive risk management.

---

## Architecture Overview

```
cTrader Open API (TCP/Protobuf)
         │
         ▼
  DataFetcher ──► CSV Export ──► ModelTrainer
         │                              │
         │             ┌────────────────┤
         │             │ NeuralNetwork  │  (numpy, no TF/Torch/ONNX)
         │             │ LightGBM       │
         │             │ QuantumPSO HP  │
         │             └────────────────┘
         │
         ▼
  Indicator Engine (M1 / M5 / H1 multiplex)
  ├── RSI, MACD, Bollinger Bands, EMA(9/21/50/200)
  ├── ATR, Stochastic, OBV
  └── Cross-TF divergence + PhaseEstimator (FFT quantum-inspired)
         │
         ▼  (JSON payload)
  AgentOrchestrator
  ├── GroqAgent        (LLaMA-3 70B — speed-first preliminary)
  ├── GeminiAgent      (Gemini 1.5 Pro — multi-TF synthesis)
  ├── OpenAIAgent      (GPT-4o — trade justification)
  └── OpenRouterAgent  (Claude 3.5 — consensus arbiter)
         │
         ▼  (majority-vote payload)
  DecisionEngine ──► RiskManager ──► Execution
         │
         ▼  (JSON trace)
  LangSmith (full tracing + prompt versioning)
```

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/somat3k/cBocior_as
cd cBocior_as
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env with your real API keys and cTrader credentials
```

See [SECRETS.md](SECRETS.md) for a full list of all required secrets.

### 3. Train models (fetch data + train before first launch)

```bash
python -m calgo.cbot_runner --train-only
```

Models are saved to `./exports/`.

### 4. Run the bot

```bash
# Dry-run (no real orders)
python -m calgo.cbot_runner --dry-run

# Live trading
python -m calgo.cbot_runner
```

---

## Training Schedule

| Timeframe | Bars fetched | Training epochs |
|-----------|-------------|-----------------|
| 1 m       | 2 000       | 200             |
| 5 m       | 1 000       | 200             |
| 1 H       | 250         | 200             |

Training uses Quantum Particle Swarm Optimisation (QPSO) to search for
optimal hidden-layer sizes before full training.

---

## Multiplex Indicator System

Indicators are computed independently for each timeframe (M1, M5, H1)
and merged into a unified feature vector for the models:

| Indicator              | M1 | M5 | H1 |
|------------------------|----|----|----|
| RSI (14)               | ✓  | ✓  | ✓  |
| MACD (12/26/9)         | ✓  | ✓  | ✓  |
| Bollinger Bands (20/2σ)| ✓  | ✓  | ✓  |
| EMA (9, 21, 50, 200)   | ✓  | ✓  | ✓  |
| ATR (14)               | ✓  | ✓  | ✓  |
| Stochastic K/D         | ✓  | ✓  | ✓  |
| OBV                    | ✓  | ✓  | ✓  |
| Cross-TF divergence    | -  | -  | ✓  |
| Phase estimator (FFT)  | -  | -  | ✓  |

---

## Multi-Agent Decision Protocol

1. **Groq** → rapid preliminary signal (< 10 s)
2. **Gemini** → multi-timeframe pattern synthesis
3. **OpenAI** → trade justification narrative
4. **OpenRouter** → consensus arbitration (receives all three outputs)
5. **DecisionEngine** → majority vote + confidence gate (>= 0.65)
6. **RiskManager** → spread, drawdown, daily loss checks

See [AGENTS.md](AGENTS.md) for full delegation rules.

---

## Project Structure

```
cBocior_as/
├── calgo/
│   └── cbot_runner.py          # Main entry point
├── src/
│   ├── data/
│   │   ├── ctrader_client.py   # cTrader Open API client (TCP/Protobuf)
│   │   └── data_fetcher.py     # OHLCV fetch + CSV export
│   ├── models/
│   │   ├── neural_network.py   # Numpy-based NN (no TF/Torch/ONNX)
│   │   ├── quantum_algo.py     # Quantum-inspired optimisers
│   │   ├── indicators.py       # Multiplex indicator computation
│   │   └── trainer.py          # Training pipeline
│   ├── agents/
│   │   ├── base_agent.py       # Abstract agent
│   │   ├── gemini_agent.py     # Google Gemini 1.5 Pro
│   │   ├── groq_agent.py       # Groq LLaMA-3 70B
│   │   ├── openai_agent.py     # OpenAI GPT-4o
│   │   ├── openrouter_agent.py # OpenRouter Claude 3.5
│   │   └── orchestrator.py     # Fan-out + majority-vote orchestrator
│   ├── analysis/
│   │   ├── pattern_detector.py # Candlestick + price-action patterns
│   │   ├── market_analyzer.py  # Regime classification + session filter
│   │   └── signal_engine.py    # Composite signal scoring
│   ├── trading/
│   │   ├── decision_engine.py  # Confidence-gated decision making
│   │   ├── execution.py        # cTrader order submission
│   │   └── risk_manager.py     # Risk controls
│   └── utils/
│       ├── logger.py           # structlog + rich logging
│       └── payload.py          # JSON payload builder/validator/signer
├── tests/                      # pytest test suite (50 tests)
├── data/                       # CSV data (git-ignored)
├── exports/                    # Trained model artefacts (git-ignored)
├── constants.py                # Centralised config (GitHub Actions + .env)
├── .env.example                # Environment variable template
├── SECRETS.md                  # All GitHub Actions secrets documented
├── AGENTS.md                   # Agent delegation rules (ROOT)
├── TODO.md                     # Epics A-Z with subtasks
└── requirements.txt            # Dependencies (no TF/Torch/ONNX)
```

---

## Required Secrets

See [SECRETS.md](SECRETS.md) for the complete list. Required secrets:

- `CTRADER_CLIENT_ID`, `CTRADER_CLIENT_SECRET`, `CTRADER_ACCESS_TOKEN`, `CTRADER_ACCOUNT_ID`
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`
- `LANGSMITH_API_KEY`

---

## Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

50 tests covering: NN training/inference/persistence, quantum optimisers,
multiplex indicators, payload builder/validator/signing, signal engine,
and decision engine.

---

## Key Design Decisions

| Constraint | Solution |
|---|---|
| No TensorFlow / PyTorch / ONNX | Numpy NN from scratch + LightGBM |
| cTrader integration | Official `ctrader-open-api` SDK (Twisted TCP/Protobuf) |
| Multi-agent communication | JSON payloads (Pydantic + ujson) |
| Prompt management | LangSmith prompt hub + tracing |
| Hyperparameter search | Quantum Particle Swarm Optimisation (QPSO) |
| Secret management | GitHub Actions secrets + `.env` fallback |

---

## Planned Work

See [TODO.md](TODO.md) for the full roadmap — Epics A through Z.
