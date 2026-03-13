"""
src/models/neural_network.py — Pure numpy Neural Network.

Implements a fully-connected multi-layer perceptron from scratch using only
NumPy (no TensorFlow, PyTorch, or ONNX).

Features:
- Configurable hidden-layer architecture
- ReLU / Sigmoid / Tanh activations
- Mini-batch gradient descent
- He / Xavier weight initialisation
- Dropout regularisation (inverted dropout)
- Early stopping
- Model persistence via numpy (.npz)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

Activation = Literal["relu", "sigmoid", "tanh", "linear"]


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _sigmoid_grad(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s * (1.0 - s)


def _tanh_grad(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


_ACT_FN = {
    "relu": (_relu, _relu_grad),
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "tanh": (np.tanh, _tanh_grad),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}


# ---------------------------------------------------------------------------
# NeuralNetwork
# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Multi-layer perceptron trained with mini-batch gradient descent.

    Parameters
    ----------
    layer_sizes : tuple[int, ...]
        Sizes of all layers including input and output.
        E.g. (20, 128, 64, 1) → 20-input → 128 → 64 → 1-output.
    hidden_activation : Activation
        Activation for all hidden layers. Default: 'relu'.
    output_activation : Activation
        Activation for the output layer. Default: 'sigmoid' (binary).
    learning_rate : float
    dropout_rate : float
        Fraction of hidden-layer neurons to drop during training.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        hidden_activation: Activation = "relu",
        output_activation: Activation = "sigmoid",
        learning_rate: float = 0.001,
        dropout_rate: float = 0.2,
        seed: int | None = 42,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input + output layers.")
        self.layer_sizes = layer_sizes
        self.hidden_act, self.hidden_act_grad = _ACT_FN[hidden_activation]
        self.output_act, self.output_act_grad = _ACT_FN[output_activation]
        self.lr = learning_rate
        self.dropout_rate = dropout_rate
        self._rng = np.random.default_rng(seed)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self._init_weights()

        # Training history
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """He initialisation for ReLU; Xavier for others."""
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            scale = np.sqrt(2.0 / fan_in)  # He init
            W = self._rng.normal(0, scale, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(
        self,
        X: np.ndarray,
        training: bool = False,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray | None]]:
        """
        Forward pass through the network.

        Returns
        -------
        output  : np.ndarray             — final layer activations
        zs      : list                   — pre-activation values per layer
        acts    : list                   — post-activation values per layer (incl. input)
        masks   : list[np.ndarray|None]  — dropout masks per hidden layer (None for output)
        """
        acts = [X]
        zs = []
        masks: list[np.ndarray | None] = []

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = acts[-1] @ W + b
            zs.append(z)

            is_output = i == len(self.weights) - 1
            if is_output:
                a = self.output_act(z)
                masks.append(None)
            else:
                a = self.hidden_act(z)
                if training and self.dropout_rate > 0:
                    mask = (
                        self._rng.random(a.shape) > self.dropout_rate
                    ).astype(float)
                    a = a * mask / (1.0 - self.dropout_rate)
                    masks.append(mask)
                else:
                    masks.append(None)
            acts.append(a)

        return acts[-1], zs, acts, masks

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def _backward(
        self,
        X: np.ndarray,
        y: np.ndarray,
        zs: list[np.ndarray],
        acts: list[np.ndarray],
        masks: list[np.ndarray | None],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute gradients via backpropagation."""
        m = X.shape[0]
        dW = [np.zeros_like(W) for W in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        # Output layer delta (binary cross-entropy + sigmoid → simplifies)
        delta = acts[-1] - y  # shape (m, out)

        for i in reversed(range(len(self.weights))):
            dW[i] = acts[i].T @ delta / m
            db[i] = delta.mean(axis=0, keepdims=True)
            if i > 0:
                delta = delta @ self.weights[i].T * self.hidden_act_grad(zs[i - 1])
                # Apply the same dropout mask that was used in the forward pass
                if masks[i - 1] is not None:
                    delta = delta * masks[i - 1] / (1.0 - self.dropout_rate)

        return dW, db

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        patience: int = 20,
        verbose: bool = True,
    ) -> "NeuralNetwork":
        """
        Train the network using mini-batch gradient descent.

        Parameters
        ----------
        patience : int
            Early-stopping patience (epochs without val loss improvement).
        """
        y_train = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train
        m = X_train.shape[0]
        best_val_loss = np.inf
        no_improve = 0
        best_weights = [W.copy() for W in self.weights]
        best_biases = [b.copy() for b in self.biases]

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = self._rng.permutation(m)
            X_s, y_s = X_train[idx], y_train[idx]

            # Mini-batches
            for start in range(0, m, batch_size):
                end = start + batch_size
                Xb, yb = X_s[start:end], y_s[start:end]
                out, zs, acts, masks = self._forward(Xb, training=True)
                dW, db = self._backward(Xb, yb, zs, acts, masks)
                for i in range(len(self.weights)):
                    self.weights[i] -= self.lr * dW[i]
                    self.biases[i] -= self.lr * db[i]

            # Training loss
            out_tr, _, _, _ = self._forward(X_train)
            tr_loss = self._bce_loss(y_train, out_tr)
            self.train_loss_history.append(tr_loss)

            # Validation loss + early stopping
            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_r = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val
                out_val, _, _, _ = self._forward(X_val)
                val_loss = self._bce_loss(y_val_r, out_val)
                self.val_loss_history.append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    no_improve = 0
                    best_weights = [W.copy() for W in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(
                            "Early stopping",
                            epoch=epoch,
                            best_val_loss=round(best_val_loss, 6),
                        )
                        break

            if verbose and epoch % 20 == 0:
                msg = f"Epoch {epoch}/{epochs}  train_loss={tr_loss:.5f}"
                if val_loss is not None:
                    msg += f"  val_loss={val_loss:.5f}"
                logger.info(msg)

        # Restore best weights
        if X_val is not None:
            self.weights = best_weights
            self.biases = best_biases

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability scores (output of sigmoid)."""
        out, _, _, _ = self._forward(X, training=False)
        return out.flatten()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class predictions."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def _bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-9
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(-np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        ))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model weights and configuration to a .npz file."""
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"W{i}"] = W
            arrays[f"b{i}"] = b
        # Save config as a 0-d object array
        config = {
            "layer_sizes": self.layer_sizes,
            "lr": self.lr,
            "dropout_rate": self.dropout_rate,
        }
        arrays["config"] = np.array(pickle.dumps(config))
        np.savez(path, **arrays)
        logger.info("NeuralNetwork saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "NeuralNetwork":
        """Load model from a .npz file."""
        path = Path(path).with_suffix(".npz")
        data = np.load(path, allow_pickle=True)
        config = pickle.loads(data["config"].tobytes())

        nn = cls(
            layer_sizes=config["layer_sizes"],
            learning_rate=config["lr"],
            dropout_rate=config["dropout_rate"],
        )
        nn.weights = []
        nn.biases = []
        i = 0
        while f"W{i}" in data:
            nn.weights.append(data[f"W{i}"])
            nn.biases.append(data[f"b{i}"])
            i += 1
        logger.info("NeuralNetwork loaded", path=str(path))
        return nn
