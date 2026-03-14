"""
constants.py — Single source of truth for all configuration constants.

Priority order:
  1. Environment variables (set by CI/GitHub Actions secrets)
  2. .env file (local development fallback)
  3. Hardcoded defaults (safe/non-secret values only)

All GitHub Actions Secrets required by this project are documented in
SECRETS.md.  Never hard-code secret values here.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (no-op if running inside GitHub Actions where real env vars exist)
# ---------------------------------------------------------------------------
_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)


def _req(key: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(key, "")
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "See SECRETS.md and .env.example for setup instructions."
        )
    return value


def _opt(key: str, default: str = "") -> str:
    """Return an optional environment variable with a default."""
    return os.getenv(key, default)


# ─── cTrader Open API ──────────────────────────────────────────────────────
CTRADER_CLIENT_ID: str = _req("CTRADER_CLIENT_ID")
CTRADER_CLIENT_SECRET: str = _req("CTRADER_CLIENT_SECRET")
CTRADER_ACCESS_TOKEN: str = _req("CTRADER_ACCESS_TOKEN")
CTRADER_REFRESH_TOKEN: str = _opt("CTRADER_REFRESH_TOKEN")
# Primary account (Account 1, initial capital 10 000)
CTRADER_ACCOUNT_ID: int = int(_req("CTRADER_ACCOUNT_ID"))
# Secondary account (Account 2, initial capital 50).
# Defaults to the same ID as the primary so the bot works when only one
# account is configured; supply a real second account ID in production.
CTRADER_ACCOUNT_ID_ACC2: int = int(
    _opt("CTRADER_ACCOUNT_ID_ACC2", str(_req("CTRADER_ACCOUNT_ID")))
)
CTRADER_ENVIRONMENT: str = _opt("CTRADER_ENVIRONMENT", "DEMO").upper()

# Derived endpoints
_IS_LIVE = CTRADER_ENVIRONMENT == "LIVE"
CTRADER_HOST: str = (
    "live.ctraderapi.com" if _IS_LIVE else "demo.ctraderapi.com"
)
CTRADER_PORT: int = 5035

# ─── Groq ─────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = _req("GROQ_API_KEY")
GROQ_MODEL: str = _opt("GROQ_MODEL", "oss-120b")

# ─── LangSmith ────────────────────────────────────────────────────────────
LANGSMITH_API_KEY: str = _req("LANGSMITH_API_KEY")
LANGSMITH_PROJECT: str = _opt("LANGSMITH_PROJECT", "cBocior_as")
LANGCHAIN_TRACING_V2: str = _opt("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_ENDPOINT: str = _opt(
    "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
)

# ─── Trading Parameters ───────────────────────────────────────────────────
TRADING_SYMBOL: str = _opt("TRADING_SYMBOL", "EURUSD")
# TRADING_VOLUME is in centilots (lots × 100).
#   1 = 0.01 lot (micro-lot / 1 000 currency units) — minimum recommended
#  10 = 0.10 lot (mini-lot  / 10 000 currency units)
# 100 = 1.00 lot (standard  / 100 000 currency units)
TRADING_VOLUME: int = int(_opt("TRADING_VOLUME", "1"))
TRADING_MAX_SPREAD_PIPS: float = float(_opt("TRADING_MAX_SPREAD_PIPS", "2.0"))
TRADING_STOP_LOSS_PIPS: float = float(_opt("TRADING_STOP_LOSS_PIPS", "30"))
TRADING_TAKE_PROFIT_PIPS: float = float(_opt("TRADING_TAKE_PROFIT_PIPS", "60"))

# ─── Risk Management ──────────────────────────────────────────────────────
RISK_MAX_DRAWDOWN_PCT: float = float(_opt("RISK_MAX_DRAWDOWN_PCT", "5.0"))
RISK_MAX_POSITION_SIZE: int = int(_opt("RISK_MAX_POSITION_SIZE", "10000"))
RISK_DAILY_LOSS_LIMIT_USD: float = float(
    _opt("RISK_DAILY_LOSS_LIMIT_USD", "500.0")
)

# ─── Model / Training ─────────────────────────────────────────────────────
MODEL_EXPORT_DIR: Path = Path(_opt("MODEL_EXPORT_DIR", "./exports"))
DATA_DIR: Path = Path(_opt("DATA_DIR", "./data"))

# Training schedule (per timeframe)
# Candle counts set to the maximum practical range for each timeframe:
#   M1  — 10 000 bars ≈ 7 days of 1-minute data
#   M5  —  5 000 bars ≈ 17 days of 5-minute data
#   H1  —  2 000 bars ≈ 83 days (~3 months) of hourly data
# These defaults request the maximum useful history from HyperLiquid/cTrader.
TRAIN_1M_TRADES: int = int(_opt("TRAIN_1M_TRADES", "10000"))
TRAIN_1M_EPOCHS: int = int(_opt("TRAIN_1M_EPOCHS", "200"))
TRAIN_5M_TRADES: int = int(_opt("TRAIN_5M_TRADES", "5000"))
TRAIN_5M_EPOCHS: int = int(_opt("TRAIN_5M_EPOCHS", "200"))
TRAIN_1H_TRADES: int = int(_opt("TRAIN_1H_TRADES", "2000"))
TRAIN_1H_EPOCHS: int = int(_opt("TRAIN_1H_EPOCHS", "200"))

# ─── Timeframe constants ──────────────────────────────────────────────────
TF_1M: str = "M1"
TF_5M: str = "M5"
TF_1H: str = "H1"
SUPPORTED_TIMEFRAMES: tuple[str, ...] = (TF_1M, TF_5M, TF_1H)

# ─── Bot Behaviour ────────────────────────────────────────────────────────
BOT_LOOP_INTERVAL_SECONDS: int = int(_opt("BOT_LOOP_INTERVAL_SECONDS", "10"))
BOT_ANALYSIS_COOLDOWN_SECONDS: int = int(
    _opt("BOT_ANALYSIS_COOLDOWN_SECONDS", "60")
)
BOT_MAX_CONCURRENT_AGENTS: int = int(_opt("BOT_MAX_CONCURRENT_AGENTS", "1"))
LOG_LEVEL: str = _opt("LOG_LEVEL", "INFO").upper()

# ─── Training Symbols ─────────────────────────────────────────────────────
# All symbols for which models are trained on historical OHLCV data.
# cTrader symbol names may differ from exchange names — see cTrader docs.
TRAINING_SYMBOLS: tuple[str, ...] = (
    "XAUUSD",   # Gold (XAU vs USD)
    "GOOGL",    # Alphabet / Google
    "AMD",      # Advanced Micro Devices
    "US30",     # Dow Jones Industrial Average
    "US100",    # Nasdaq 100
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "FLOKIUSD", # Floki
    "BONKUSD",  # Bonk
    "SHIBUSD",  # Shiba Inu
    "ADAUSD",   # Cardano
    "LTCUSD",   # Litecoin
    "XRPUSD",   # XRP
    "SOLUSD",   # Solana
    "EURUSD",   # Euro / USD
    "GBPUSD",   # British Pound / USD
    "NZDCAD",   # New Zealand Dollar / Canadian Dollar
    "USDJPY",   # USD / Japanese Yen
)

# HyperLiquid symbol names corresponding to the TRAINING_SYMBOLS list above
# (used when the cTrader / cAlgo primary feed is unavailable).
# Note: HyperLiquid primarily supports crypto perpetuals.  For forex and
# equity-index symbols the mapping is best-effort; if HyperLiquid does not
# list the coin the fetcher returns an empty result and the pipeline falls
# back to previously saved CSV data.
HYPERLIQUID_SYMBOL_MAP: dict[str, str] = {
    "XAUUSD":   "XAU",
    "GOOGL":    "GOOGL",
    "AMD":      "AMD",
    "US30":     "DJI",     # TODO: verify HyperLiquid coin name for Dow Jones index
    "US100":    "NDX",     # TODO: verify HyperLiquid coin name for Nasdaq 100 index
    "BTCUSD":   "BTC",
    "ETHUSD":   "ETH",
    "FLOKIUSD": "FLOKI",
    "BONKUSD":  "BONK",
    "SHIBUSD":  "SHIB",
    "ADAUSD":   "ADA",
    "LTCUSD":   "LTC",
    "XRPUSD":   "XRP",
    "SOLUSD":   "SOL",
    "EURUSD":   "EURUSD",  # HyperLiquid forex pair identifier
    "GBPUSD":   "GBPUSD",  # HyperLiquid forex pair identifier
    "NZDCAD":   "NZDCAD",  # HyperLiquid forex pair identifier
    "USDJPY":   "USDJPY",  # HyperLiquid forex pair identifier
}

# ─── Account Capitals ─────────────────────────────────────────────────────
INITIAL_CAPITAL_ACC1: float = float(_opt("INITIAL_CAPITAL_ACC1", "10000"))
INITIAL_CAPITAL_ACC2: float = float(_opt("INITIAL_CAPITAL_ACC2", "50"))

# ─── Redis Cache ──────────────────────────────────────────────────────────
REDIS_URL: str = _opt("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_TTL_SECONDS: int = int(_opt("REDIS_CACHE_TTL_SECONDS", "3600"))
REDIS_ENABLED: bool = _opt("REDIS_ENABLED", "true").lower() == "true"

# ─── Indicator windows (defaults) ─────────────────────────────────────────
RSI_PERIOD: int = 14
MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9
BB_PERIOD: int = 20
BB_STD: float = 2.0
EMA_PERIODS: tuple[int, ...] = (9, 21, 50, 200)
ATR_PERIOD: int = 14
STOCH_K: int = 14
STOCH_D: int = 3
STOCH_SMOOTH: int = 3

# ─── Neural-Network hyper-parameters (numpy implementation) ───────────────
NN_HIDDEN_LAYERS: tuple[int, ...] = (128, 64, 32)
NN_LEARNING_RATE: float = 0.001
NN_DROPOUT_RATE: float = 0.2
NN_BATCH_SIZE: int = 32
NN_EARLY_STOP_PATIENCE: int = 20

# ─── Quantum-inspired optimiser ───────────────────────────────────────────
QA_NUM_QUBITS: int = 8          # logical qubits for annealing simulation
QA_ITERATIONS: int = 500
QA_TEMPERATURE_START: float = 10.0
QA_TEMPERATURE_END: float = 0.01

# ─── Payload / communication ──────────────────────────────────────────────
PAYLOAD_VERSION: str = "1.0"
PAYLOAD_ENCODING: str = "utf-8"

# ─── File name templates ──────────────────────────────────────────────────
CSV_TEMPLATE: str = "{symbol}_{timeframe}_{date}.csv"
MODEL_TEMPLATE: str = "{symbol}_{timeframe}_model.joblib"
SCALER_TEMPLATE: str = "{symbol}_{timeframe}_scaler.joblib"
