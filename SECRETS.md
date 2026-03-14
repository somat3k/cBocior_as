# SECRETS — GitHub Actions Repository Secrets Documentation

This document lists **every** GitHub Actions repository secret required by
`cBocior_as` and explains its purpose.  Set these in:

> **GitHub → Repository → Settings → Secrets and variables → Actions →
> Repository secrets**

When running locally, copy `.env.example` to `.env` and populate it.

---

## Required Secrets

### cTrader Open API

| Secret name | Description | Where to get it |
|---|---|---|
| `CTRADER_CLIENT_ID` | OAuth2 App Client ID from Spotware Connect | [connect.ctrader.com/apps](https://connect.ctrader.com/apps/) |
| `CTRADER_CLIENT_SECRET` | OAuth2 App Client Secret | Same as above |
| `CTRADER_ACCESS_TOKEN` | OAuth2 Bearer Access Token (account-level) | Obtained via the OAuth2 flow |
| `CTRADER_REFRESH_TOKEN` | OAuth2 Refresh Token (used to renew access) | Obtained via the OAuth2 flow |
| `CTRADER_ACCOUNT_ID` | Numeric cTrader account ID | cTrader platform → Accounts |
| `CTRADER_ENVIRONMENT` | `LIVE` or `DEMO` | User-defined |

### Groq

| Secret name | Description | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | Groq Cloud API key (`gsk_...`) | [console.groq.com/keys](https://console.groq.com/keys) |
| `GROQ_MODEL` | Model identifier (default `llama-3.1-120b`) | Optional override |

### LangSmith

| Secret name | Description | Where to get it |
|---|---|---|
| `LANGSMITH_API_KEY` | LangSmith tracing API key (`ls__...`) | [smith.langchain.com](https://smith.langchain.com/) |
| `LANGSMITH_PROJECT` | Project name (default `cBocior_as`) | Optional override |
| `LANGCHAIN_TRACING_V2` | Enable tracing (`true`/`false`) | Optional override |
| `LANGCHAIN_ENDPOINT` | LangSmith endpoint URL | Optional override |

### Trading Parameters (optional overrides)

| Secret name | Default | Description |
|---|---|---|
| `TRADING_SYMBOL` | `EURUSD` | Instrument symbol |
| `TRADING_VOLUME` | `1000` | Trade volume in units |
| `TRADING_MAX_SPREAD_PIPS` | `2.0` | Maximum allowed spread |
| `TRADING_STOP_LOSS_PIPS` | `30` | Stop-loss distance in pips |
| `TRADING_TAKE_PROFIT_PIPS` | `60` | Take-profit distance in pips |

### Risk Management (optional overrides)

| Secret name | Default | Description |
|---|---|---|
| `RISK_MAX_DRAWDOWN_PCT` | `5.0` | Maximum drawdown % before halt |
| `RISK_MAX_POSITION_SIZE` | `10000` | Max units per position |
| `RISK_DAILY_LOSS_LIMIT_USD` | `500.0` | Daily loss cap in USD |

### Model / Training (optional overrides)

| Secret name | Default | Description |
|---|---|---|
| `MODEL_EXPORT_DIR` | `./exports` | Directory for saved models |
| `DATA_DIR` | `./data` | Directory for CSV data |
| `TRAIN_1M_TRADES` | `2000` | Trades to load for 1 m training |
| `TRAIN_1M_EPOCHS` | `200` | Training epochs on 1 m data |
| `TRAIN_5M_TRADES` | `1000` | Trades to load for 5 m training |
| `TRAIN_5M_EPOCHS` | `200` | Training epochs on 5 m data |
| `TRAIN_1H_TRADES` | `250` | Trades to load for 1 H training |
| `TRAIN_1H_EPOCHS` | `200` | Training epochs on 1 H data |

### Bot Behaviour (optional overrides)

| Secret name | Default | Description |
|---|---|---|
| `BOT_LOOP_INTERVAL_SECONDS` | `10` | Main loop tick interval |
| `BOT_ANALYSIS_COOLDOWN_SECONDS` | `60` | Minimum gap between full agent analyses |
| `BOT_MAX_CONCURRENT_AGENTS` | `4` | Maximum parallel agent calls |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

---

## Notes

* Secrets with no default **must** be provided or the bot will raise an
  `EnvironmentError` at start-up (enforced in `constants.py`).
* Optional overrides use safe defaults and do not cause start-up failures.
* The `.env` file takes precedence for local development but is ignored when
  the real environment variables are already set (GitHub Actions injects them
  directly into the process environment).
