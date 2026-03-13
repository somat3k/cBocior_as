"""
src/models/multi_symbol_trainer.py — Training pipeline for all symbols.

Drives :class:`~src.models.trainer.ModelTrainer` over the full
:data:`~constants.TRAINING_SYMBOLS` list, fetching OHLCV data via
:class:`~src.data.multi_symbol_fetcher.MultiSymbolFetcher`.

The target data source is the cAlgo / cTrader Open API.  HyperLiquid is
used transparently as a fallback if the cTrader feed fails for a symbol.

Initial account capitals:
  * Account 1 (``INITIAL_CAPITAL_ACC1``) — 10 000 USD (default)
  * Account 2 (``INITIAL_CAPITAL_ACC2``) —     50 USD (default)

Redis caching is enabled automatically via :func:`~src.utils.cache.get_cache`.

Usage::

    from src.models.multi_symbol_trainer import MultiSymbolTrainer

    trainer = MultiSymbolTrainer()
    results = trainer.run()
    # results["EURUSD"]["M1"]["nn_val_acc"] → float
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from constants import (
    INITIAL_CAPITAL_ACC1,
    INITIAL_CAPITAL_ACC2,
    MODEL_EXPORT_DIR,
    SUPPORTED_TIMEFRAMES,
    TRAINING_SYMBOLS,
)
from src.data.ctrader_client import CTraderClient
from src.data.multi_symbol_fetcher import MultiSymbolFetcher
from src.models.trainer import ModelTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MultiSymbolTrainer:
    """
    High-level orchestrator that trains models for all training symbols.

    Parameters
    ----------
    ctrader_client : CTraderClient, optional
        Connected cTrader client.  Pass ``None`` to skip the primary cTrader
        feed (HyperLiquid fallback and CSV cache will still be tried).
    export_dir : Path
        Root directory for model artefact exports.  Artefacts are placed in
        ``<export_dir>/<symbol>/``.
    symbols : tuple[str, ...], optional
        Symbols to train on.  Defaults to :data:`~constants.TRAINING_SYMBOLS`.
    timeframes : tuple[str, ...], optional
        Timeframes to use.  Defaults to
        :data:`~constants.SUPPORTED_TIMEFRAMES`.
    use_qpso : bool
        Whether to use QuantumParticleSwarm for NN hyperparameter search.
    """

    def __init__(
        self,
        ctrader_client: CTraderClient | None = None,
        export_dir: Path = MODEL_EXPORT_DIR,
        symbols: tuple[str, ...] = TRAINING_SYMBOLS,
        timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
        use_qpso: bool = True,
    ) -> None:
        self.ctrader_client = ctrader_client
        self.export_dir = Path(export_dir)
        self.symbols = symbols
        self.timeframes = timeframes
        self.use_qpso = use_qpso

        logger.info(
            "MultiSymbolTrainer initialised",
            symbols=len(symbols),
            timeframes=timeframes,
            initial_capital_acc1=INITIAL_CAPITAL_ACC1,
            initial_capital_acc2=INITIAL_CAPITAL_ACC2,
            export_dir=str(export_dir),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Fetch data and train models for every symbol.

        Returns
        -------
        dict[symbol, dict[timeframe, training_result]]
            ``training_result`` mirrors
            :meth:`~src.models.trainer.ModelTrainer._train_timeframe`'s
            return dict (``nn_val_acc``, ``gbm_val_acc``, etc.).
        """
        fetcher = MultiSymbolFetcher(
            ctrader_client=self.ctrader_client,
            timeframes=self.timeframes,
            symbols=self.symbols,
        )

        all_data = fetcher.fetch_all()
        results: dict[str, dict[str, dict[str, Any]]] = {}

        for symbol in self.symbols:
            symbol_data: dict[str, pd.DataFrame] = all_data.get(symbol, {})
            if not symbol_data:
                logger.warning("Skipping symbol — no data", symbol=symbol)
                continue

            results[symbol] = self._train_symbol(symbol, symbol_data)

        logger.info(
            "MultiSymbolTrainer complete",
            symbols_trained=len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Per-symbol training
    # ------------------------------------------------------------------

    def _train_symbol(
        self,
        symbol: str,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, Any]]:
        """Train all timeframe models for *symbol*."""
        symbol_export_dir = self.export_dir / symbol
        symbol_export_dir.mkdir(parents=True, exist_ok=True)

        trainer = ModelTrainer(
            symbol=symbol,
            export_dir=symbol_export_dir,
            use_qpso=self.use_qpso,
        )

        try:
            trained = trainer.train_all(data=data, timeframes=self.timeframes)
        except Exception as exc:
            logger.error(
                "Training failed for symbol",
                symbol=symbol,
                error=str(exc),
            )
            return {}

        # Flatten result: timeframe → metrics dict (drop heavy model objects)
        summary: dict[str, dict[str, Any]] = {}
        for tf, result in trained.items():
            summary[tf] = {
                "nn_val_acc": result.get("nn_val_acc"),
                "gbm_val_acc": result.get("gbm_val_acc"),
                "feature_cols": result.get("feature_cols", []),
            }
            logger.info(
                "Symbol training complete",
                symbol=symbol,
                timeframe=tf,
                nn_val_acc=result.get("nn_val_acc"),
                gbm_val_acc=result.get("gbm_val_acc"),
            )
        return summary
