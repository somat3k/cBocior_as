#!/usr/bin/env python3
"""
calgo/cbot_runner.py — cBocior_as cAlgo Python cBot Entry Point

This script is the main entry point that connects all subsystems:

  1. Connects to cTrader Open API (TCP/Protobuf)
  2. Subscribes to live OHLCV bars for M1, M5, H1
  3. Fetches historical data for training (if models not found)
  4. Trains NeuralNetwork + LightGBM models across all timeframes
  5. Starts the continuous analysis loop:
       a. Buffer incoming bars
       b. Compute multiplex indicators
       c. Detect candlestick patterns
       d. Analyse market regime
       e. Run model inference
       f. Build TradingPayload
       g. Invoke multi-agent orchestrator (async)
       h. Apply DecisionEngine + RiskManager
       i. Execute confirmed trade decisions
  6. Emits JSON payloads at every cycle for full traceability

Run with:
    python -m calgo.cbot_runner

Environment: requires all secrets from SECRETS.md (or .env file).
"""

from __future__ import annotations

import asyncio
import http.server
import json as _json
import signal
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Project root on PYTHONPATH ───────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load config (validates required secrets) ─────────────────────────────────
from constants import (
    BOT_LOOP_INTERVAL_SECONDS,
    CTRADER_ACCESS_TOKEN,
    CTRADER_ACCOUNT_ID,
    CTRADER_CLIENT_ID,
    CTRADER_CLIENT_SECRET,
    CTRADER_HOST,
    CTRADER_PORT,
    MODEL_EXPORT_DIR,
    SUPPORTED_TIMEFRAMES,
    TRADING_SYMBOL,
    TRADING_VOLUME,
    TRADING_STOP_LOSS_PIPS,
    TRADING_TAKE_PROFIT_PIPS,
    TF_1H,
    TF_1M,
    TF_5M,
)

# ── Source modules ───────────────────────────────────────────────────────────
from src.utils.logger import configure_logging, get_logger
from src.data.ctrader_client import CTraderClient, OHLCVBar
from src.data.data_fetcher import DataFetcher
from src.models.indicators import compute_indicators, snapshot_for_payload
from src.models.trainer import ModelTrainer
from src.models.quantum_algo import PhaseEstimator
from src.analysis.pattern_detector import PatternDetector
from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.signal_engine import SignalEngine
from src.trading.account_config import AccountConfig, get_account_configs
from src.trading.decision_engine import DecisionEngine
from src.trading.execution import Execution
from src.trading.risk_manager import RiskManager
from src.agents.orchestrator import AgentOrchestrator
from src.utils.payload import (
    IndicatorSnapshot,
    ModelSignals,
    PayloadBuilder,
    RiskFlags,
    TradingAction,
    TradingPayload,
)

# ── Logging ──────────────────────────────────────────────────────────────────
configure_logging(log_file=Path("logs/cbot.log"))
logger = get_logger("cbot_runner")

# ── Bar buffer size per timeframe ────────────────────────────────────────────
_BUFFER_SIZE: int = 500


# ---------------------------------------------------------------------------
# Per-account trading context
# ---------------------------------------------------------------------------

class _AccountContext:
    """
    Bundles the per-account trading subsystems.

    Each configured account gets its own :class:`~src.trading.risk_manager.RiskManager`
    (seeded with the account's initial capital for position sizing) and
    :class:`~src.trading.execution.Execution` (targeting the correct
    cTrader account ID).
    """

    def __init__(
        self,
        cfg: AccountConfig,
        ct_client: CTraderClient,
        dry_run: bool = False,
    ) -> None:
        self.cfg = cfg
        self.risk_manager = RiskManager(initial_capital=cfg.initial_capital)
        self.execution = Execution(
            client=ct_client,
            dry_run=dry_run,
            account_id=cfg.account_id,
        )

    def __repr__(self) -> str:
        return f"_AccountContext({self.cfg.label}, account_id={self.cfg.account_id})"


# ---------------------------------------------------------------------------
# CBot runner
# ---------------------------------------------------------------------------

class CBotRunner:
    """
    Main bot runner orchestrating all subsystems.

    The Twisted reactor runs in a background thread; the analysis loop
    runs on the main thread (or asyncio event loop).

    Dual-account trading
    --------------------
    The runner creates one :class:`_AccountContext` for each entry returned
    by :func:`~src.trading.account_config.get_account_configs`.  Both
    accounts receive the same trading signals; position sizes are scaled
    independently to each account's initial capital.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.symbol = TRADING_SYMBOL
        self._running = False

        # ── cTrader client ───────────────────────────────────────────────
        self._ct_client = CTraderClient(
            client_id=CTRADER_CLIENT_ID,
            client_secret=CTRADER_CLIENT_SECRET,
            access_token=CTRADER_ACCESS_TOKEN,
            account_id=CTRADER_ACCOUNT_ID,
            host=CTRADER_HOST,
            port=CTRADER_PORT,
        )
        self._ct_client.on_connected_callback = self._on_api_connected
        self._ct_client.on_error_callback = self._on_api_error

        # ── Data fetcher ─────────────────────────────────────────────────
        self._fetcher = DataFetcher(self._ct_client, self.symbol)

        # ── Model trainer ────────────────────────────────────────────────
        self._trainer = ModelTrainer(
            symbol=self.symbol,
            export_dir=MODEL_EXPORT_DIR,
            use_qpso=True,
        )
        self._models: dict[str, Any] = {}

        # ── Analysis subsystems ──────────────────────────────────────────
        self._pattern_detector = PatternDetector()
        self._market_analyzer = MarketAnalyzer()
        self._signal_engine = SignalEngine()
        self._phase_estimator = PhaseEstimator(top_k=5)

        # ── Dual-account trading contexts ────────────────────────────────
        account_cfgs = get_account_configs()
        self._account_contexts: list[_AccountContext] = [
            _AccountContext(cfg, self._ct_client, dry_run=dry_run)
            for cfg in account_cfgs
        ]
        logger.info(
            "Trading accounts configured",
            accounts=[str(ctx.cfg) for ctx in self._account_contexts],
        )

        # Shared decision engine (signal consensus is account-independent)
        self._decision_engine = DecisionEngine()

        # ── Agent orchestrator ───────────────────────────────────────────
        self._orchestrator = AgentOrchestrator()

        # ── Bar buffers ──────────────────────────────────────────────────
        self._bars: dict[str, deque[OHLCVBar]] = {
            tf: deque(maxlen=_BUFFER_SIZE) for tf in SUPPORTED_TIMEFRAMES
        }
        self._last_analysis_time: float = 0.0

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the bot (blocking)."""
        logger.info(
            "cBocior_as starting",
            symbol=self.symbol,
            dry_run=self.dry_run,
        )
        self._running = True
        self._start_time = datetime.now(timezone.utc)
        self._cycle_count = 0

        # Register graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_shutdown)

        # Start health-check HTTP server in a daemon thread  (J9)
        health_thread = threading.Thread(
            target=self._serve_health,
            daemon=True,
            name="health-server",
        )
        health_thread.start()

        # Start Twisted reactor in a daemon thread.
        # installSignalHandlers=False is required when reactor.run() is called
        # from a non-main thread (Twisted raises ValueError otherwise).
        reactor_thread = threading.Thread(
            target=lambda: self._ct_client.connect(install_signal_handlers=False),
            daemon=True,
            name="ctrader-reactor",
        )
        reactor_thread.start()

        # Main analysis loop (runs on main thread)
        asyncio.run(self._analysis_loop())

    # ------------------------------------------------------------------
    # cTrader callbacks
    # ------------------------------------------------------------------

    def _on_api_connected(self) -> None:
        """Called once the cTrader account is authorised."""
        logger.info("cTrader API connected and authorised")

        # Subscribe to live bars
        self._fetcher.subscribe_live(timeframes=SUPPORTED_TIMEFRAMES)

        # Also point the live bar callback to our buffer
        self._ct_client.on_bar_callback = self._on_live_bar

        # Bootstrap: load or train models
        threading.Thread(
            target=self._bootstrap_models,
            daemon=True,
            name="model-bootstrap",
        ).start()

    def _on_api_error(self, exc: Exception) -> None:
        logger.error("cTrader API error", error=str(exc))

    def _on_live_bar(self, bar: OHLCVBar) -> None:
        """Append incoming live bar to the appropriate buffer."""
        if bar.symbol != self.symbol:
            return
        tf = bar.timeframe
        if tf in self._bars:
            self._bars[tf].append(bar)
            logger.debug(
                "Bar received",
                timeframe=tf,
                close=bar.close,
                ts=bar.timestamp.isoformat(),
            )

    # ------------------------------------------------------------------
    # Model bootstrap
    # ------------------------------------------------------------------

    def _bootstrap_models(self) -> None:
        """Try to load existing models; train from scratch if not found."""
        existing = self._trainer.load_models(SUPPORTED_TIMEFRAMES)
        if existing:
            self._models = existing
            logger.info(
                "Pre-trained models loaded",
                timeframes=list(existing.keys()),
            )
            return

        logger.info(
            "No pre-trained models found — fetching data and training"
        )
        historical_data = self._fetcher.fetch_all_timeframes(SUPPORTED_TIMEFRAMES)
        if not historical_data:
            logger.error("Failed to fetch historical data for training")
            return

        self._models = self._trainer.train_all(
            data=historical_data,
            timeframes=SUPPORTED_TIMEFRAMES,
        )
        logger.info("Initial training complete", timeframes=list(self._models.keys()))

    # ------------------------------------------------------------------
    # Main analysis loop
    # ------------------------------------------------------------------

    async def _analysis_loop(self) -> None:
        """
        Continuous analysis loop.

        Wakes every BOT_LOOP_INTERVAL_SECONDS to:
        1. Check if enough bars are buffered.
        2. Run indicator computation.
        3. Invoke signal engine.
        4. Call agent orchestrator (async).
        5. Apply decision engine + risk manager.
        6. Execute if confirmed.
        """
        logger.info(
            "Analysis loop started",
            interval_s=BOT_LOOP_INTERVAL_SECONDS,
        )
        while self._running:
            await asyncio.sleep(BOT_LOOP_INTERVAL_SECONDS)
            try:
                await self._single_cycle()
                self._cycle_count += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Cycle error", error=str(exc), exc_info=True)

    async def _single_cycle(self) -> None:
        """Execute one complete analysis-decision-execution cycle."""
        if not self._models:
            logger.debug("Models not ready yet, skipping cycle")
            return

        import pandas as pd

        # Build DataFrames from bar buffers
        dfs: dict[str, pd.DataFrame] = {}
        for tf in SUPPORTED_TIMEFRAMES:
            bars = list(self._bars[tf])
            if len(bars) < 30:
                logger.debug(
                    "Not enough bars for analysis",
                    timeframe=tf,
                    buffered=len(bars),
                )
                continue
            data = [b.to_dict() for b in bars]
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
            dfs[tf] = df

        if not dfs:
            return

        # Compute indicators for all available timeframes
        indicator_dfs: dict[str, pd.DataFrame] = {}
        indicator_snapshots: list[IndicatorSnapshot] = []

        for tf, df in dfs.items():
            ind_df = compute_indicators(df, tf)
            indicator_dfs[tf] = ind_df
            snap_dict = snapshot_for_payload(ind_df, tf)
            indicator_snapshots.append(IndicatorSnapshot(**snap_dict))

        # Model inference
        model_signals: list[ModelSignals] = []
        for tf, ind_df in indicator_dfs.items():
            if tf not in self._models:
                continue
            try:
                probs = self._trainer.predict(
                    ind_df, tf, self._models[tf]
                )
                model_signals.append(ModelSignals(
                    timeframe=tf,
                    nn_prediction=probs["nn_prob"],
                    nn_confidence=probs["nn_prob"],
                    xgb_prediction=probs["gbm_prob"],
                    xgb_confidence=probs["gbm_prob"],
                    ensemble_prediction=probs["ensemble_prob"],
                    ensemble_confidence=probs["ensemble_prob"],
                ))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Model inference error",
                    timeframe=tf,
                    error=str(exc),
                )

        # Phase estimation (quantum-inspired frequency analysis)
        phase_info: dict[str, Any] = {}
        if TF_1H in dfs:
            close_1h = dfs[TF_1H]["close"].values
            phase_info = self._phase_estimator.estimate(close_1h)

        # Pattern detection
        pattern_results: list[dict[str, Any]] = []
        for tf, df in dfs.items():
            patterns = self._pattern_detector.detect_all(df, tf)
            pattern_results.append(patterns)

        # Market analysis
        market_analysis = self._market_analyzer.analyse(
            dfs=indicator_dfs,
            indicator_snapshots=[s.model_dump() for s in indicator_snapshots],
        )

        # Signal engine
        signal_data = self._signal_engine.compute(
            indicator_snapshots=indicator_snapshots,
            model_signals=model_signals,
            pattern_results=pattern_results,
            regime=market_analysis.get("regime", "UNKNOWN"),
        )

        # Build input payload for agents
        tf_div = signal_data.get("timeframe_divergence", False)
        input_payload = (
            PayloadBuilder()
            .source("signal_engine")
            .symbol(self.symbol)
            .action(signal_data["action"])
            .confidence(signal_data["confidence"])
            .reasoning(
                f"Signal score: {signal_data['score']}, "
                f"Regime: {market_analysis.get('regime')}"
            )
            .metadata(
                market_analysis=market_analysis,
                phase_info=phase_info,
                signal_components=signal_data.get("components", {}),
                timeframe_divergence=tf_div,
            )
            .build()
        )
        if tf_div:
            input_payload = input_payload.model_copy(
                update={"risk_flags": input_payload.risk_flags.model_copy(
                    update={"timeframe_divergence": True}
                )}
            )
        # Attach indicators + model signals
        for snap in indicator_snapshots:
            input_payload = input_payload.model_copy(
                update={"indicators": input_payload.indicators + [snap]}
            )
        for sig in model_signals:
            input_payload = input_payload.model_copy(
                update={"model_signals": input_payload.model_signals + [sig]}
            )

        # Risk check (preliminary, before agent call)
        # Use the first account's risk manager for the pre-agent check;
        # each account will be re-checked independently during execution.
        _risk_allowed, _risk_reason, risk_flags = (
            self._account_contexts[0].risk_manager.evaluate(
                input_payload, current_spread_pips=0.0
            )
        )

        # Agent orchestrator (async, potentially 4 API calls)
        agent_payload = await self._orchestrator.run(input_payload)

        # Final decision (shared across all accounts)
        decision = self._decision_engine.decide(
            agent_payload=agent_payload,
            signal_data=signal_data,
            risk_flags=risk_flags,
        )

        # Log the full cycle payload
        logger.info(
            "Cycle complete",
            action=decision.action.value,
            confidence=decision.confidence,
            symbol=self.symbol,
            cycle_id=decision.cycle_id,
        )

        # Execute on every configured account (if not HOLD)
        if decision.action != TradingAction.HOLD:
            for ctx in self._account_contexts:
                # Per-account risk re-check
                acc_allowed, acc_reason, acc_flags = ctx.risk_manager.evaluate(
                    decision, current_spread_pips=0.0
                )
                if not acc_allowed:
                    logger.info(
                        "Trade blocked for account",
                        account=ctx.cfg.label,
                        reason=acc_reason,
                    )
                    continue

                # Attach per-account risk report to decision metadata (I9)
                risk_report = ctx.risk_manager.build_risk_report()
                enriched_decision = decision.model_copy(
                    update={"metadata": {**decision.metadata, "risk_report": risk_report}}
                )

                # Size position relative to this account's capital
                volume = ctx.risk_manager.compute_position_size(
                    stop_loss_pips=TRADING_STOP_LOSS_PIPS,
                    risk_pct=1.0,
                )
                ctx.execution.execute(
                    payload=enriched_decision,
                    volume=volume,
                    stop_loss_pips=TRADING_STOP_LOSS_PIPS,
                    take_profit_pips=TRADING_TAKE_PROFIT_PIPS,
                )
                logger.info(
                    "Trade executed",
                    account=ctx.cfg.label,
                    account_id=ctx.cfg.account_id,
                    volume=volume,
                    action=decision.action.value,
                    symbol=self.symbol,
                )

    # ------------------------------------------------------------------
    # Health-check HTTP server  (J9)
    # ------------------------------------------------------------------

    def _serve_health(self, port: int = 8765) -> None:
        """
        Serve a minimal HTTP ``/health`` endpoint on *port*.

        Response example::

            {
              "status": "ok",
              "uptime_s": 3600,
              "cycles": 42,
              "symbol": "EURUSD",
              "accounts": 2,
              "running": true
            }

        The server is intentionally minimal (stdlib only) and runs in a
        daemon thread so it is automatically stopped when the bot exits.
        """
        runner = self  # capture for closure

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/health", "/health/"):
                    uptime = (
                        datetime.now(timezone.utc) - runner._start_time
                    ).total_seconds()
                    body = _json.dumps({
                        "status": "ok",
                        "uptime_s": round(uptime, 1),
                        "cycles": getattr(runner, "_cycle_count", 0),
                        "symbol": runner.symbol,
                        "accounts": len(runner._account_contexts),
                        "running": runner._running,
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, *args: Any) -> None:  # silence access log
                pass

        try:
            server = http.server.HTTPServer(("0.0.0.0", port), _Handler)
            logger.info("Health endpoint started", port=port, path="/health")
            server.serve_forever()
        except OSError as exc:
            logger.warning(
                "Could not start health server", port=port, error=str(exc)
            )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        logger.info("Shutdown signal received", signal=signum)
        self._running = False
        try:
            from twisted.internet import reactor
            if reactor.running:
                reactor.callFromThread(reactor.stop)
        except Exception:
            pass
        sys.exit(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="cBocior_as — AI-driven cTrader cBot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without placing real orders (paper trading mode).",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Download data and train models, then exit.",
    )
    args = parser.parse_args()

    if args.train_only:
        # Standalone training mode
        logger.info("Train-only mode activated")
        ct = CTraderClient(
            client_id=CTRADER_CLIENT_ID,
            client_secret=CTRADER_CLIENT_SECRET,
            access_token=CTRADER_ACCESS_TOKEN,
            account_id=CTRADER_ACCOUNT_ID,
            host=CTRADER_HOST,
            port=CTRADER_PORT,
        )
        fetcher = DataFetcher(ct, TRADING_SYMBOL)
        trainer = ModelTrainer(TRADING_SYMBOL, MODEL_EXPORT_DIR)

        def _train_worker() -> None:
            """Run blocking data-fetch + training in a separate thread.

            fetch_historical_bars() uses time.sleep() internally which must
            not block the Twisted reactor thread.  Starting it in its own
            thread keeps the reactor responsive so that API responses arrive.
            """
            try:
                data = fetcher.fetch_all_timeframes(SUPPORTED_TIMEFRAMES)
                trainer.train_all(data, SUPPORTED_TIMEFRAMES)
                logger.info("Training complete — stopping reactor")
            except Exception as exc:
                logger.error("Training failed", error=str(exc))
            finally:
                from twisted.internet import reactor as _reactor
                _reactor.callFromThread(_reactor.stop)

        def _on_connected_start_training() -> None:
            """Reactor callback — schedule the worker thread (non-blocking)."""
            threading.Thread(
                target=_train_worker,
                daemon=True,
                name="train-worker",
            ).start()

        ct.on_connected_callback = _on_connected_start_training
        # The reactor must run on the main thread in train-only mode so that
        # signal handlers (SIGINT/SIGTERM) work correctly.
        ct.connect(install_signal_handlers=True)
    else:
        runner = CBotRunner(dry_run=args.dry_run)
        runner.run()
