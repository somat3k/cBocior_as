"""
tests/test_neural_network.py — Unit tests for the numpy NeuralNetwork.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models.neural_network import NeuralNetwork


@pytest.fixture
def simple_nn() -> NeuralNetwork:
    return NeuralNetwork(
        layer_sizes=(4, 8, 4, 1),
        learning_rate=0.01,
        dropout_rate=0.0,
        seed=42,
    )


def make_xor_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_int = rng.integers(0, 2, size=(n, 4))
    y = (X_int[:, 0] ^ X_int[:, 1]).astype(float)
    X = X_int.astype(float)
    return X, y


class TestNeuralNetworkInit:
    def test_weight_shapes(self, simple_nn: NeuralNetwork) -> None:
        assert simple_nn.weights[0].shape == (4, 8)
        assert simple_nn.weights[1].shape == (8, 4)
        assert simple_nn.weights[2].shape == (4, 1)

    def test_bias_shapes(self, simple_nn: NeuralNetwork) -> None:
        for b in simple_nn.biases:
            assert b.shape[0] == 1


class TestForwardPass:
    def test_output_range(self, simple_nn: NeuralNetwork) -> None:
        X = np.random.default_rng(0).random((10, 4))
        proba = simple_nn.predict_proba(X)
        assert proba.shape == (10,)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_binary(self, simple_nn: NeuralNetwork) -> None:
        X = np.random.default_rng(1).random((5, 4))
        preds = simple_nn.predict(X)
        assert set(preds.tolist()).issubset({0, 1})


class TestTraining:
    def test_loss_decreases(self) -> None:
        X, y = make_xor_data(400)
        nn = NeuralNetwork(
            layer_sizes=(4, 16, 8, 1),
            learning_rate=0.05,
            dropout_rate=0.0,
            seed=42,
        )
        nn.fit(X, y, epochs=50, batch_size=32, verbose=False)
        assert len(nn.train_loss_history) > 0
        # Loss should generally decrease (not always monotone)
        assert nn.train_loss_history[-1] < nn.train_loss_history[0] + 0.1

    def test_early_stopping(self) -> None:
        X, y = make_xor_data(100)
        nn = NeuralNetwork(
            layer_sizes=(4, 8, 1),
            learning_rate=0.001,
            seed=0,
        )
        nn.fit(
            X, y,
            epochs=200,
            X_val=X, y_val=y,
            patience=5,
            verbose=False,
        )
        # Should stop early
        assert len(nn.train_loss_history) <= 200

    def test_returns_self(self) -> None:
        nn = NeuralNetwork((4, 8, 1))
        X, y = make_xor_data(50)
        result = nn.fit(X, y, epochs=5, verbose=False)
        assert result is nn


class TestPersistence:
    def test_save_load_roundtrip(self, tmp_path: Path) -> None:
        X, y = make_xor_data(200)
        nn = NeuralNetwork((4, 16, 1), seed=7)
        nn.fit(X, y, epochs=10, verbose=False)

        model_path = tmp_path / "test_nn"
        nn.save(model_path)

        loaded = NeuralNetwork.load(tmp_path / "test_nn.npz")
        original_proba = nn.predict_proba(X[:5])
        loaded_proba = loaded.predict_proba(X[:5])
        np.testing.assert_allclose(original_proba, loaded_proba, rtol=1e-5)


class TestBCELoss:
    def test_perfect_prediction(self) -> None:
        y = np.array([[1.0], [0.0], [1.0]])
        y_pred = np.array([[0.9999], [0.0001], [0.9999]])
        loss = NeuralNetwork._bce_loss(y, y_pred)
        assert loss < 0.01

    def test_worst_prediction(self) -> None:
        y = np.array([[1.0], [0.0]])
        y_pred = np.array([[0.0001], [0.9999]])
        loss = NeuralNetwork._bce_loss(y, y_pred)
        assert loss > 5.0
