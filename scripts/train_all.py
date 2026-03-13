#!/usr/bin/env python3
"""
scripts/train_all.py — Train all models with maximum historical data and export.

Usage::

    python scripts/train_all.py [--no-qpso] [--symbols EURUSD BTCUSD ...]
    python -m scripts.train_all

What this script does
---------------------
1. Fetches the **maximum available** historical OHLCV data for every symbol in
   :data:`~constants.TRAINING_SYMBOLS` using HyperLiquid REST API (no cTrader
   connection required):

   * M1  — up to 10 000 bars  (~7 days of 1-minute data)
   * M5  — up to  5 000 bars  (~17 days of 5-minute data)
   * H1  — up to  2 000 bars  (~83 days of 1-hour data)

2. Trains a **NeuralNetwork + GBM ensemble** for each symbol × timeframe.

3. Runs a **backtest** on each model's validation set, targeting:

   - Win rate
   - Maximum Drawdown (%MDD)
   - Balance growth (total return %)
   - Trade-cancellation rate (signal-reversal exits)
   - Profit factor

4. Exports all model artefacts to ``exports/<SYMBOL>/``:

   * ``<SYMBOL>_<TF>_nn.npz``           — NeuralNetwork weights
   * ``<SYMBOL>_<TF>_model.joblib``     — GBM model
   * ``<SYMBOL>_<TF>_scaler.joblib``    — StandardScaler
   * ``<SYMBOL>_<TF>_features.joblib``  — feature column list

5. Prints a formatted summary table of all training and backtest metrics.

These files are committed to ``exports/<SYMBOL>/`` in the repository so the
cBot can load them at startup without a live training pass.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Ensure project root is on sys.path ───────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load environment (must happen before importing constants) ─────────────────
from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env", override=False)

# ── Minimal env defaults so constants.py doesn't raise on missing secrets ─────
_ENV_DEFAULTS: dict[str, str] = {
    "CTRADER_CLIENT_ID": "offline",
    "CTRADER_CLIENT_SECRET": "offline",
    "CTRADER_ACCESS_TOKEN": "offline",
    "CTRADER_ACCOUNT_ID": "0",
    "OPENAI_API_KEY": "offline",
    "GEMINI_API_KEY": "offline",
    "GROQ_API_KEY": "offline",
    "OPENROUTER_API_KEY": "offline",
    "LANGSMITH_API_KEY": "offline",
    "REDIS_ENABLED": "false",
    "LOG_LEVEL": "WARNING",
}
for _k, _v in _ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

# ── Now import project modules ────────────────────────────────────────────────
from constants import (
    MODEL_EXPORT_DIR,
    SUPPORTED_TIMEFRAMES,
    TRAINING_SYMBOLS,
)
from src.data.multi_symbol_fetcher import MultiSymbolFetcher
from src.models.multi_symbol_trainer import MultiSymbolTrainer
from src.utils.logger import configure_logging, get_logger

configure_logging()
logger = get_logger("train_all")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(v: float | None, decimals: int = 1) -> str:
    if v is None:
        return "  n/a "
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def _fmt_float(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return " n/a "
    return f"{v:.{decimals}f}"


def _fmt_int(v: int | None) -> str:
    if v is None:
        return " n/a"
    return str(v)


def _print_summary(results: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Print a formatted summary table of all training and backtest metrics."""
    divider = "═" * 105
    thin = "─" * 105

    print()
    print(divider)
    print("  cBocior_as — Full-Symbol Training & Backtest Report")
    print(divider)
    header = (
        f"  {'Symbol':<10} {'TF':<4} {'Rows':>6}  "
        f"{'NN Acc':>7} {'GBM Acc':>7}  "
        f"{'Trades':>6} {'WinRate':>8} {'%MDD':>6} "
        f"{'Return':>8} {'PF':>5} {'Cancelled':>10}"
    )
    print(header)
    print(thin)

    any_results = False
    for symbol, tf_results in results.items():
        for tf, metrics in tf_results.items():
            any_results = True
            nn_acc = _fmt_float(metrics.get("nn_val_acc"), 4)
            gbm_acc = _fmt_float(metrics.get("gbm_val_acc"), 4)
            trades = _fmt_int(metrics.get("bt_total_trades"))
            win_rate = _fmt_pct(metrics.get("bt_win_rate"))
            mdd = _fmt_pct(metrics.get("bt_max_drawdown_pct"))
            ret = _fmt_pct(metrics.get("bt_total_return_pct"))
            pf = _fmt_float(metrics.get("bt_profit_factor"))
            cancelled = _fmt_pct(metrics.get("bt_cancelled_rate"))

            print(
                f"  {symbol:<10} {tf:<4} {'n/a':>6}  "
                f"{nn_acc:>7} {gbm_acc:>7}  "
                f"{trades:>6} {win_rate:>8} {mdd:>6} "
                f"{ret:>8} {pf:>5} {cancelled:>10}"
            )

    if not any_results:
        print("  (no results — all symbols failed to fetch/train)")

    print(divider)
    print(f"  Models exported to: {MODEL_EXPORT_DIR}")
    print(divider)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    symbols: tuple[str, ...] = TRAINING_SYMBOLS,
    timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    export_dir: Path = MODEL_EXPORT_DIR,
    use_qpso: bool = True,
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Run the full training pipeline and return summary results.

    Parameters
    ----------
    symbols : tuple[str, ...]
        Symbols to train on.
    timeframes : tuple[str, ...]
        Timeframes to include.
    export_dir : Path
        Root directory for model exports.
    use_qpso : bool
        Whether to use QuantumParticleSwarm for NN hyperparameter search.

    Returns
    -------
    dict[symbol, dict[timeframe, metrics]]
    """
    logger.info(
        "Starting full-symbol training",
        symbols=len(symbols),
        timeframes=timeframes,
        export_dir=str(export_dir),
        qpso=use_qpso,
    )
    t0 = time.time()

    # ── Fetch maximum available historical data ───────────────────────
    print(f"\n{'─'*60}")
    print(f"  Fetching data for {len(symbols)} symbols via HyperLiquid ...")
    print(f"{'─'*60}")

    fetcher = MultiSymbolFetcher(
        ctrader_client=None,   # use HyperLiquid directly
        symbols=symbols,
        timeframes=timeframes,
    )
    all_data = fetcher.fetch_all()

    fetched = sum(
        1 for s in symbols if s in all_data and all_data[s]
    )
    print(f"  Data fetched for {fetched}/{len(symbols)} symbols.\n")

    # ── Train all models ──────────────────────────────────────────────
    print(f"{'─'*60}")
    print(f"  Training models (QPSO={'on' if use_qpso else 'off'}) ...")
    print(f"{'─'*60}")

    trainer = MultiSymbolTrainer(
        ctrader_client=None,
        export_dir=Path(export_dir),
        symbols=symbols,
        timeframes=timeframes,
        use_qpso=use_qpso,
    )

    # Train directly on pre-fetched data (avoids a redundant second fetch)
    results: dict[str, dict[str, dict[str, Any]]] = {}
    for sym in symbols:
        if sym not in all_data or not all_data[sym]:
            continue
        print(f"  Training {sym} ...", flush=True)
        # pylint: disable=protected-access
        symbol_result = trainer._train_symbol(sym, all_data[sym])
        if symbol_result:
            results[sym] = symbol_result

    logger.info(
        "MultiSymbolTrainer complete",
        symbols_trained=len(results),
    )

    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed:.1f}s.\n")

    # ── Print summary ─────────────────────────────────────────────────
    _print_summary(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all cBocior_as models with maximum historical data."
    )
    parser.add_argument(
        "--no-qpso",
        action="store_true",
        help="Disable QuantumParticleSwarm HP search (faster but less optimal).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(TRAINING_SYMBOLS),
        metavar="SYMBOL",
        help="Symbols to train. Defaults to all TRAINING_SYMBOLS.",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=list(SUPPORTED_TIMEFRAMES),
        metavar="TF",
        help="Timeframes to train. Defaults to all SUPPORTED_TIMEFRAMES.",
    )
    parser.add_argument(
        "--export-dir",
        default=str(MODEL_EXPORT_DIR),
        metavar="DIR",
        help=f"Model export directory. Defaults to {MODEL_EXPORT_DIR}.",
    )
    args = parser.parse_args()

    main(
        symbols=tuple(args.symbols),
        timeframes=tuple(args.timeframes),
        export_dir=Path(args.export_dir),
        use_qpso=not args.no_qpso,
    )
