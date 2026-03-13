"""
src/models/trainer.py — Model training pipeline.

Orchestrates training of the NeuralNetwork and XGBoost/LightGBM ensemble
across three timeframes with the specified trade counts and epoch budgets:

  | Timeframe | Trades | Epochs |
  |-----------|--------|--------|
  | 1 m       | 2000   | 200    |
  | 5 m       | 1000   | 200    |
  | 1 H       | 250    | 200    |

Training steps:
  1. Load CSV data for each timeframe.
  2. Compute multiplex indicators.
  3. Build feature matrix + target (next-bar direction: 1=up, 0=down).
  4. Split train/validation (80/20).
  5. Scale features with StandardScaler.
  6. Use QuantumParticleSwarm to search for NN hyperparameters.
  7. Train NeuralNetwork.
  8. Train XGBoost / LightGBM models.
  9. Export models + scalers to MODEL_EXPORT_DIR.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from constants import (
    MODEL_EXPORT_DIR,
    MODEL_TEMPLATE,
    NN_BATCH_SIZE,
    NN_DROPOUT_RATE,
    NN_EARLY_STOP_PATIENCE,
    NN_HIDDEN_LAYERS,
    NN_LEARNING_RATE,
    SCALER_TEMPLATE,
    SUPPORTED_TIMEFRAMES,
    TRAIN_1H_EPOCHS,
    TRAIN_1H_TRADES,
    TRAIN_1M_EPOCHS,
    TRAIN_1M_TRADES,
    TRAIN_5M_EPOCHS,
    TRAIN_5M_TRADES,
    TF_1H,
    TF_1M,
    TF_5M,
)
from src.models.indicators import compute_indicators, get_feature_columns
from src.models.neural_network import NeuralNetwork
from src.models.quantum_algo import QuantumParticleSwarm
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Per-timeframe training schedule
_TF_SCHEDULE: dict[str, dict[str, int]] = {
    TF_1M: {"trades": TRAIN_1M_TRADES, "epochs": TRAIN_1M_EPOCHS},
    TF_5M: {"trades": TRAIN_5M_TRADES, "epochs": TRAIN_5M_EPOCHS},
    TF_1H: {"trades": TRAIN_1H_TRADES, "epochs": TRAIN_1H_EPOCHS},
}


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    End-to-end training pipeline for cBocior_as.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. "EURUSD").
    export_dir : Path
        Directory where trained model artefacts are saved.
    use_qpso : bool
        Whether to use QuantumParticleSwarm for NN hyperparameter search.
    """

    def __init__(
        self,
        symbol: str,
        export_dir: Path = MODEL_EXPORT_DIR,
        use_qpso: bool = True,
    ) -> None:
        self.symbol = symbol
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.use_qpso = use_qpso
        self.trained_models: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train_all(
        self,
        data: dict[str, pd.DataFrame],
        timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES,
    ) -> dict[str, dict[str, Any]]:
        """
        Train models for all specified timeframes.

        Parameters
        ----------
        data : dict[str, pd.DataFrame]
            Maps timeframe → OHLCV DataFrame.

        Returns
        -------
        dict mapping timeframe → {'nn': NeuralNetwork, 'gbm': model, 'scaler': scaler}
        """
        for tf in timeframes:
            if tf not in data:
                logger.warning("No data for timeframe, skipping", timeframe=tf)
                continue
            logger.info("Starting training", symbol=self.symbol, timeframe=tf)
            result = self._train_timeframe(data[tf], tf)
            self.trained_models[tf] = result
            self._export(result, tf)

        return self.trained_models

    # ------------------------------------------------------------------
    # Per-timeframe training
    # ------------------------------------------------------------------

    def _train_timeframe(
        self, df: pd.DataFrame, timeframe: str
    ) -> dict[str, Any]:
        schedule = _TF_SCHEDULE.get(timeframe, {"trades": 500, "epochs": 100})
        max_rows = schedule["trades"]
        epochs = schedule["epochs"]

        # Trim to the required number of most-recent bars
        df = df.tail(max_rows).reset_index(drop=True)
        logger.info(
            "Using bars for training",
            timeframe=timeframe,
            rows=len(df),
            epochs=epochs,
        )

        # Compute indicators
        df = compute_indicators(df, timeframe)

        # Build features + target
        feat_cols = get_feature_columns(timeframe)
        available = [c for c in feat_cols if c in df.columns]
        if len(available) < 3:
            raise ValueError(
                f"Too few feature columns for {timeframe}: {available}"
            )

        X_raw = df[available].values.astype(np.float64)
        # Target: 1 if next bar closes higher than current
        close = df["close"].values
        y = (np.roll(close, -1) > close).astype(int)
        # Drop last row (no future bar)
        X_raw = X_raw[:-1]
        y = y[:-1]

        # Remove NaN rows
        valid_mask = ~np.isnan(X_raw).any(axis=1)
        X_raw, y = X_raw[valid_mask], y[valid_mask]

        if len(X_raw) < 10:
            raise ValueError(f"Insufficient clean samples for {timeframe}")

        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X_raw, y, test_size=0.2, shuffle=False
        )

        # Scale
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)

        # ── NN hyperparameter search via QPSO ────────────────────────
        n_features = X_train_sc.shape[1]
        best_layer_sizes = self._qpso_hp_search(
            X_train_sc, y_train, X_val_sc, y_val, n_features
        ) if self.use_qpso else NN_HIDDEN_LAYERS

        # ── Train NeuralNetwork ──────────────────────────────────────
        layer_sizes = (n_features,) + tuple(best_layer_sizes) + (1,)
        nn = NeuralNetwork(
            layer_sizes=layer_sizes,
            learning_rate=NN_LEARNING_RATE,
            dropout_rate=NN_DROPOUT_RATE,
        )
        nn.fit(
            X_train_sc, y_train,
            epochs=epochs,
            batch_size=NN_BATCH_SIZE,
            X_val=X_val_sc, y_val=y_val,
            patience=NN_EARLY_STOP_PATIENCE,
        )

        # ── Train GBM (no TF/Torch) ───────────────────────────────────
        try:
            import lightgbm as lgb
            gbm = lgb.LGBMClassifier(
                n_estimators=min(200, epochs),
                learning_rate=0.05,
                max_depth=6,
                n_jobs=1,
                verbose=-1,
            )
        except ImportError:
            gbm = GradientBoostingClassifier(
                n_estimators=min(100, epochs),
                learning_rate=0.05,
                max_depth=4,
            )

        gbm.fit(X_train_sc, y_train)
        gbm_val_acc = (gbm.predict(X_val_sc) == y_val).mean()
        nn_val_acc = (nn.predict(X_val_sc) == y_val).mean()

        logger.info(
            "Training complete",
            timeframe=timeframe,
            nn_val_acc=round(float(nn_val_acc), 4),
            gbm_val_acc=round(float(gbm_val_acc), 4),
            features=len(available),
        )

        return {
            "nn": nn,
            "gbm": gbm,
            "scaler": scaler,
            "feature_cols": available,
            "nn_val_acc": float(nn_val_acc),
            "gbm_val_acc": float(gbm_val_acc),
        }

    # ------------------------------------------------------------------
    # QPSO hyperparameter search
    # ------------------------------------------------------------------

    def _qpso_hp_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_features: int,
    ) -> tuple[int, ...]:
        """
        Use QPSO to search for optimal hidden layer sizes.

        Search space: 2 hidden layers, each between 16 and 256 neurons.
        """
        qpso = QuantumParticleSwarm(n_particles=8, max_iterations=20, seed=42)

        def objective(params: np.ndarray) -> float:
            h1 = max(8, int(params[0]))
            h2 = max(8, int(params[1]))
            layer_sizes = (n_features, h1, h2, 1)
            net = NeuralNetwork(
                layer_sizes=layer_sizes,
                learning_rate=NN_LEARNING_RATE,
                dropout_rate=NN_DROPOUT_RATE,
                seed=0,
            )
            net.fit(
                X_train, y_train,
                epochs=20,  # quick search
                batch_size=NN_BATCH_SIZE,
                X_val=X_val, y_val=y_val,
                patience=5,
                verbose=False,
            )
            val_loss = net.val_loss_history[-1] if net.val_loss_history else 1.0
            return val_loss

        bounds = np.array([
            [16, 256],
            [16, 128],
        ], dtype=float)

        result = qpso.minimise(objective, bounds)
        h1 = max(8, int(result.best_params[0]))
        h2 = max(8, int(result.best_params[1]))
        logger.info(
            "QPSO HP search done",
            best_hidden=(h1, h2),
            best_val_loss=round(result.best_fitness, 5),
        )
        return (h1, h2)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self, result: dict[str, Any], timeframe: str) -> None:
        # Save NN
        nn_path = self.export_dir / f"{self.symbol}_{timeframe}_nn"
        result["nn"].save(nn_path)

        # Save GBM
        gbm_path = self.export_dir / MODEL_TEMPLATE.format(
            symbol=self.symbol, timeframe=timeframe
        )
        joblib.dump(result["gbm"], gbm_path)

        # Save scaler
        scaler_path = self.export_dir / SCALER_TEMPLATE.format(
            symbol=self.symbol, timeframe=timeframe
        )
        joblib.dump(result["scaler"], scaler_path)

        # Save feature column list so load_models() can reconstruct predict()
        feat_path = self.export_dir / f"{self.symbol}_{timeframe}_features.joblib"
        joblib.dump(result["feature_cols"], feat_path)

        logger.info(
            "Model artefacts exported",
            timeframe=timeframe,
            nn_path=str(nn_path),
            gbm_path=str(gbm_path),
            scaler_path=str(scaler_path),
        )

    # ------------------------------------------------------------------
    # Load previously trained models
    # ------------------------------------------------------------------

    def load_models(
        self, timeframes: tuple[str, ...] = SUPPORTED_TIMEFRAMES
    ) -> dict[str, dict[str, Any]]:
        """Load all exported models from MODEL_EXPORT_DIR."""
        loaded: dict[str, dict[str, Any]] = {}
        for tf in timeframes:
            nn_path = self.export_dir / f"{self.symbol}_{tf}_nn.npz"
            gbm_path = self.export_dir / MODEL_TEMPLATE.format(
                symbol=self.symbol, timeframe=tf
            )
            scaler_path = self.export_dir / SCALER_TEMPLATE.format(
                symbol=self.symbol, timeframe=tf
            )
            feat_path = self.export_dir / f"{self.symbol}_{tf}_features.joblib"
            if not (nn_path.exists() and gbm_path.exists() and scaler_path.exists()):
                logger.warning(
                    "Model artefacts not found, skipping", timeframe=tf
                )
                continue
            # Load feature columns from file if available, fall back to computed defaults
            if feat_path.exists():
                feature_cols = joblib.load(feat_path)
            else:
                feature_cols = get_feature_columns(tf)
                logger.warning(
                    "Feature columns file missing, using defaults", timeframe=tf
                )
            loaded[tf] = {
                "nn": NeuralNetwork.load(nn_path),
                "gbm": joblib.load(gbm_path),
                "scaler": joblib.load(scaler_path),
                "feature_cols": feature_cols,
            }
            logger.info("Models loaded", timeframe=tf)
        return loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        timeframe: str,
        models: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """
        Run inference on the latest bar and return signal probabilities.

        Returns
        -------
        dict with keys: nn_prob, gbm_prob, ensemble_prob
        """
        models = models or self.trained_models.get(timeframe)
        if not models:
            raise RuntimeError(f"No models loaded for {timeframe}")

        df_ind = compute_indicators(df.tail(300), timeframe)
        feat_cols = models["feature_cols"]
        available = [c for c in feat_cols if c in df_ind.columns]
        X = df_ind[available].iloc[-1:].values.astype(np.float64)

        if np.isnan(X).any():
            logger.warning("NaN in features, returning HOLD signal")
            return {"nn_prob": 0.5, "gbm_prob": 0.5, "ensemble_prob": 0.5}

        X_sc = models["scaler"].transform(X)
        nn_prob = float(models["nn"].predict_proba(X_sc)[0])
        gbm_prob = float(models["gbm"].predict_proba(X_sc)[0, 1])
        ensemble_prob = 0.6 * nn_prob + 0.4 * gbm_prob

        return {
            "nn_prob": round(nn_prob, 4),
            "gbm_prob": round(gbm_prob, 4),
            "ensemble_prob": round(ensemble_prob, 4),
        }
