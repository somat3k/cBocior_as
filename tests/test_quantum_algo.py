"""
tests/test_quantum_algo.py — Unit tests for quantum-inspired algorithms.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.quantum_algo import PhaseEstimator, QuantumAnnealer, QuantumParticleSwarm


class TestQuantumAnnealer:
    def test_minimises_sphere_function(self) -> None:
        """Sphere function f(x) = sum(x^2) has minimum at origin."""
        qa = QuantumAnnealer(num_qubits=3, iterations=300, seed=42)
        bounds = np.array([[-5.0, 5.0]] * 3)
        result = qa.minimise(lambda x: float(np.sum(x ** 2)), bounds)
        assert result.best_energy < 2.0
        assert len(result.energy_history) > 0
        assert result.iterations == 300

    def test_respects_bounds(self) -> None:
        qa = QuantumAnnealer(num_qubits=2, iterations=100, seed=0)
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        result = qa.minimise(lambda x: float(np.sum(x)), bounds)
        assert np.all(result.best_params >= 0.0)
        assert np.all(result.best_params <= 1.0)

    def test_energy_decreases(self) -> None:
        qa = QuantumAnnealer(num_qubits=2, iterations=200, seed=5)
        bounds = np.array([[-10.0, 10.0]] * 2)
        result = qa.minimise(lambda x: float(np.sum(x ** 2)), bounds)
        # Final energy should be ≤ initial
        assert result.energy_history[-1] <= result.energy_history[0] + 0.1


class TestQuantumParticleSwarm:
    def test_minimises_rosenbrock(self) -> None:
        """Rosenbrock function: global min at (1,1) = 0."""
        def rosenbrock(x: np.ndarray) -> float:
            return float(
                100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
            )

        qpso = QuantumParticleSwarm(n_particles=15, max_iterations=100, seed=42)
        bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
        result = qpso.minimise(rosenbrock, bounds)
        assert result.best_fitness < 10.0  # within reasonable range

    def test_fitness_history_non_increasing(self) -> None:
        qpso = QuantumParticleSwarm(n_particles=10, max_iterations=50, seed=1)
        bounds = np.array([[-5.0, 5.0]] * 2)
        result = qpso.minimise(lambda x: float(np.sum(x ** 2)), bounds)
        history = result.fitness_history
        # History should be non-increasing (best never gets worse)
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1] + 1e-9


class TestPhaseEstimator:
    def test_known_frequency(self) -> None:
        """Estimate dominant frequency of a pure sine wave."""
        n = 256
        freq = 0.1  # 1 cycle per 10 samples
        t = np.arange(n)
        signal = np.sin(2 * np.pi * freq * t)

        estimator = PhaseEstimator(top_k=3)
        result = estimator.estimate(signal)

        assert "phases" in result
        assert "amplitudes" in result
        assert "dominant_period" in result
        assert len(result["amplitudes"]) > 0
        # Dominant amplitude should be the largest
        assert result["amplitudes"][0] == pytest.approx(1.0, abs=1e-3)

    def test_short_series(self) -> None:
        estimator = PhaseEstimator(top_k=3)
        result = estimator.estimate(np.array([1.0, 2.0, 3.0]))
        assert result["phases"] == []
        assert result["amplitudes"] == []
        assert result["dominant_period"] is None

    def test_normalised_amplitudes(self) -> None:
        estimator = PhaseEstimator(top_k=5)
        series = np.random.default_rng(42).random(128)
        result = estimator.estimate(series)
        if result["amplitudes"]:
            assert max(result["amplitudes"]) == pytest.approx(1.0, abs=1e-6)
