"""
tests/test_trainer_integration.py — Integration test for the ModelTrainer.

Trains on a minimal synthetic OHLCV dataset and verifies that all expected
artefact files are created in a temporary export directory.

This covers TODO item C14.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Set required env vars before importing any project module
os.environ.setdefault("CTRADER_CLIENT_ID", "test")
os.environ.setdefault("CTRADER_CLIENT_SECRET", "test")
os.environ.setdefault("CTRADER_ACCESS_TOKEN", "test")
os.environ.setdefault("CTRADER_ACCOUNT_ID", "12345678")
os.environ.setdefault("CTRADER_ENVIRONMENT", "DEMO")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("GEMINI_API_KEY", "AIza-test")
os.environ.setdefault("GROQ_API_KEY", "gsk_test")
os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-test")
os.environ.setdefault("LANGSMITH_API_KEY", "ls__test")
os.environ.setdefault("LANGSMITH_PROJECT", "test")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from constants import TF_1M
from src.models.trainer import ModelTrainer


def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate a minimal synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, n))
    high = close + rng.uniform(0, 0.001, n)
    low = close - rng.uniform(0, 0.001, n)
    open_ = close + rng.normal(0, 0.0002, n)
    volume = rng.integers(100, 1000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


@pytest.mark.slow
class TestModelTrainerIntegration:
    def test_train_all_creates_artefacts(self, tmp_path: Path) -> None:
        """Train on M1 synthetic data; assert all artefact files exist."""
        trainer = ModelTrainer(symbol="TESTSYM", export_dir=tmp_path, use_qpso=False)
        df = _make_ohlcv(n=400)

        results = trainer.train_all({TF_1M: df}, timeframes=(TF_1M,))

        assert TF_1M in results, "Expected training result for M1"

        # NN artefact (.npz)
        nn_path = tmp_path / "TESTSYM_M1_nn.npz"
        assert nn_path.exists(), f"NN artefact not found: {nn_path}"

        # GBM artefact (.joblib)
        gbm_path = tmp_path / "TESTSYM_M1_model.joblib"
        assert gbm_path.exists(), f"GBM artefact not found: {gbm_path}"

        # Scaler artefact (.joblib)
        scaler_path = tmp_path / "TESTSYM_M1_scaler.joblib"
        assert scaler_path.exists(), f"Scaler artefact not found: {scaler_path}"

        # Feature column list
        feat_path = tmp_path / "TESTSYM_M1_features.joblib"
        assert feat_path.exists(), f"Features artefact not found: {feat_path}"

    def test_train_result_has_expected_keys(self, tmp_path: Path) -> None:
        """Training result dict must include nn, gbm, scaler, feature_cols."""
        trainer = ModelTrainer(symbol="TESTSYM", export_dir=tmp_path, use_qpso=False)
        df = _make_ohlcv(n=400)
        results = trainer.train_all({TF_1M: df}, timeframes=(TF_1M,))
        result = results[TF_1M]

        for key in ("nn", "gbm", "scaler", "feature_cols"):
            assert key in result, f"Missing key '{key}' in training result"

    def test_registry_json_created(self, tmp_path: Path) -> None:
        """A registry.json must be created in the export directory."""
        trainer = ModelTrainer(symbol="TESTSYM", export_dir=tmp_path, use_qpso=False)
        df = _make_ohlcv(n=400)
        trainer.train_all({TF_1M: df}, timeframes=(TF_1M,))

        registry_path = tmp_path / "registry.json"
        assert registry_path.exists(), "registry.json not found"

    def test_metadata_json_created(self, tmp_path: Path) -> None:
        """A per-model metadata JSON file must be created."""
        trainer = ModelTrainer(symbol="TESTSYM", export_dir=tmp_path, use_qpso=False)
        df = _make_ohlcv(n=400)
        trainer.train_all({TF_1M: df}, timeframes=(TF_1M,))

        meta_path = tmp_path / "TESTSYM" / "TESTSYM_M1_meta.json"
        assert meta_path.exists(), f"Metadata JSON not found: {meta_path}"

    def test_load_models_after_train(self, tmp_path: Path) -> None:
        """Models must be loadable (and integrity-checked) after training."""
        trainer = ModelTrainer(symbol="TESTSYM", export_dir=tmp_path, use_qpso=False)
        df = _make_ohlcv(n=400)
        trainer.train_all({TF_1M: df}, timeframes=(TF_1M,))

        loaded = trainer.load_models(timeframes=(TF_1M,))
        assert TF_1M in loaded, "Could not reload trained models"

    def test_predict_after_train(self, tmp_path: Path) -> None:
        """Model must produce a prediction after training."""
        trainer = ModelTrainer(symbol="TESTSYM", export_dir=tmp_path, use_qpso=False)
        df = _make_ohlcv(n=400)
        trainer.train_all({TF_1M: df}, timeframes=(TF_1M,))

        pred = trainer.predict(df, TF_1M)
        assert "ensemble_prob" in pred
        assert 0.0 <= pred["ensemble_prob"] <= 1.0

    def test_skips_timeframe_without_data(self, tmp_path: Path) -> None:
        """train_all() must not raise when a timeframe has no data."""
        trainer = ModelTrainer(symbol="TESTSYM", export_dir=tmp_path, use_qpso=False)
        # Pass H1 data but ask for M1 — M1 has no data
        df = _make_ohlcv(n=400)
        results = trainer.train_all({"H1": df}, timeframes=(TF_1M,))
        assert TF_1M not in results, "Should not have results for M1 with no data"
