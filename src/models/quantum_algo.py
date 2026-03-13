"""
src/models/quantum_algo.py — Quantum-inspired optimisation algorithms.

Implements quantum-inspired heuristics using only NumPy:

1. **QuantumAnnealer** — simulates quantum tunnelling behaviour via a
   transverse-field Ising model approximation.  Used for hyperparameter
   optimisation and feature selection.

2. **QuantumParticleSwarm** — a quantum-behaved PSO variant where each
   particle moves according to a quantum potential well centred on the
   personal best.

3. **PhaseEstimator** — approximates dominant frequency components in a
   price/indicator time-series using the quantum phase estimation concept
   (FFT-based).

None of these require Qiskit, Cirq, or any external quantum library.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from constants import (
    QA_ITERATIONS,
    QA_NUM_QUBITS,
    QA_TEMPERATURE_END,
    QA_TEMPERATURE_START,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

ObjectiveFn = Callable[[np.ndarray], float]


# ---------------------------------------------------------------------------
# 1. Quantum Annealer
# ---------------------------------------------------------------------------

@dataclass
class AnnealerResult:
    best_params: np.ndarray
    best_energy: float
    energy_history: list[float] = field(default_factory=list)
    iterations: int = 0


class QuantumAnnealer:
    """
    Simulated quantum annealing (SQA) without external quantum hardware.

    The algorithm maintains a set of ``num_replicas`` classical copies of
    the solution ("Trotter slices") that interact via a transverse field
    Gamma(T).  As temperature T decreases, the quantum tunnelling effect
    diminishes and the system settles into a low-energy state.

    Parameters
    ----------
    num_qubits : int
        Dimension of the parameter space (logical qubits).
    num_replicas : int
        Number of Trotter replicas.
    iterations : int
        Total annealing steps.
    t_start, t_end : float
        Start and end temperature.
    seed : int | None
    """

    def __init__(
        self,
        num_qubits: int = QA_NUM_QUBITS,
        num_replicas: int = 4,
        iterations: int = QA_ITERATIONS,
        t_start: float = QA_TEMPERATURE_START,
        t_end: float = QA_TEMPERATURE_END,
        seed: int | None = 42,
    ) -> None:
        self.num_qubits = num_qubits
        self.num_replicas = num_replicas
        self.iterations = iterations
        self.t_start = t_start
        self.t_end = t_end
        self._rng = np.random.default_rng(seed)

    def minimise(
        self,
        objective: ObjectiveFn,
        bounds: np.ndarray,
    ) -> AnnealerResult:
        """
        Minimise ``objective(x)`` subject to ``bounds``.

        Parameters
        ----------
        objective : callable
            Function mapping np.ndarray (shape: num_qubits,) → float.
        bounds : np.ndarray
            Shape (num_qubits, 2) with [low, high] for each dimension.

        Returns
        -------
        AnnealerResult
        """
        n = self.num_qubits
        low, high = bounds[:, 0], bounds[:, 1]
        span = high - low

        # Initialise replicas uniformly within bounds
        replicas = self._rng.random((self.num_replicas, n)) * span + low
        energies = np.array([objective(r) for r in replicas])

        best_idx = int(np.argmin(energies))
        best_params = replicas[best_idx].copy()
        best_energy = float(energies[best_idx])
        energy_history: list[float] = [best_energy]

        for step in range(1, self.iterations + 1):
            # Temperature schedule (geometric)
            T = self.t_start * (self.t_end / self.t_start) ** (
                step / self.iterations
            )
            # Transverse field Γ (weakens as T → 0)
            gamma = T * np.log(1.0 + self.iterations / (step + 1e-9))

            for r in range(self.num_replicas):
                # Propose perturbation proportional to Γ
                perturbation = self._rng.normal(0, gamma * span * 0.1, n)
                candidate = np.clip(replicas[r] + perturbation, low, high)
                e_new = objective(candidate)
                delta_e = e_new - energies[r]

                # Metropolis acceptance
                accept = (delta_e < 0) or (
                    self._rng.random() < np.exp(-delta_e / (T + 1e-12))
                )
                if accept:
                    replicas[r] = candidate
                    energies[r] = e_new
                    if e_new < best_energy:
                        best_energy = e_new
                        best_params = candidate.copy()

            # Replica coupling (quantum tunnelling between Trotter slices)
            if self.num_replicas > 1:
                for r in range(self.num_replicas):
                    r_next = (r + 1) % self.num_replicas
                    coupling = gamma * np.sum(
                        (replicas[r] - replicas[r_next]) ** 2
                    )
                    if self._rng.random() < np.exp(-coupling / (T + 1e-12)):
                        # Exchange replicas
                        replicas[r], replicas[r_next] = (
                            replicas[r_next].copy(),
                            replicas[r].copy(),
                        )
                        energies[r], energies[r_next] = (
                            energies[r_next],
                            energies[r],
                        )

            energy_history.append(best_energy)

        logger.info(
            "QuantumAnnealer finished",
            best_energy=round(best_energy, 6),
            iterations=self.iterations,
        )
        return AnnealerResult(
            best_params=best_params,
            best_energy=best_energy,
            energy_history=energy_history,
            iterations=self.iterations,
        )


# ---------------------------------------------------------------------------
# 2. Quantum Particle Swarm Optimiser
# ---------------------------------------------------------------------------

@dataclass
class QPSOResult:
    best_params: np.ndarray
    best_fitness: float
    fitness_history: list[float] = field(default_factory=list)


class QuantumParticleSwarm:
    """
    Quantum-behaved Particle Swarm Optimisation (QPSO).

    Particles move according to a quantum potential well centred on the
    attractor (mean of personal bests).  The ``beta`` parameter controls
    the contraction-expansion coefficient.

    Ref: Sun, Feng, Xu & Li (2012) — QPSO algorithm
    """

    def __init__(
        self,
        n_particles: int = 20,
        max_iterations: int = 200,
        beta: float = 0.75,
        seed: int | None = 42,
    ) -> None:
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.beta = beta
        self._rng = np.random.default_rng(seed)

    def minimise(
        self,
        objective: ObjectiveFn,
        bounds: np.ndarray,
    ) -> QPSOResult:
        n = bounds.shape[0]
        low, high = bounds[:, 0], bounds[:, 1]
        span = high - low

        # Initialise particles
        pos = self._rng.random((self.n_particles, n)) * span + low
        pbest = pos.copy()
        pbest_fitness = np.array([objective(p) for p in pbest])

        gbest_idx = int(np.argmin(pbest_fitness))
        gbest = pbest[gbest_idx].copy()
        gbest_fitness = float(pbest_fitness[gbest_idx])
        fitness_history: list[float] = [gbest_fitness]

        for t in range(1, self.max_iterations + 1):
            # Mean best position (global attractor)
            mbest = pbest.mean(axis=0)

            for i in range(self.n_particles):
                phi = self._rng.random(n)
                # Local attractor between personal and global best
                attractor = phi * pbest[i] + (1 - phi) * gbest
                # Quantum delta potential: exponential distribution
                u = self._rng.random(n)
                L = (1.0 / self.beta) * np.abs(mbest - pos[i])
                sign = np.where(self._rng.random(n) < 0.5, 1, -1)
                pos[i] = attractor + sign * L * np.log(1.0 / (u + 1e-12))
                pos[i] = np.clip(pos[i], low, high)

                fitness = objective(pos[i])
                if fitness < pbest_fitness[i]:
                    pbest[i] = pos[i].copy()
                    pbest_fitness[i] = fitness
                    if fitness < gbest_fitness:
                        gbest = pos[i].copy()
                        gbest_fitness = fitness

            fitness_history.append(gbest_fitness)

        logger.info(
            "QPSO finished",
            best_fitness=round(gbest_fitness, 6),
            iterations=t,
        )
        return QPSOResult(
            best_params=gbest,
            best_fitness=gbest_fitness,
            fitness_history=fitness_history,
        )


# ---------------------------------------------------------------------------
# 3. Phase Estimator (FFT-based quantum-inspired frequency analysis)
# ---------------------------------------------------------------------------

class PhaseEstimator:
    """
    Approximates the quantum phase estimation algorithm using the Fast
    Fourier Transform to identify dominant frequency components in a
    time-series.

    In a quantum phase estimator, the eigenvalues of a unitary operator
    encode phase angles corresponding to oscillation frequencies.  Here we
    simulate this with the DFT.

    Returns the top-k frequencies and their amplitudes normalised to [0, 1].
    """

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def estimate(self, series: np.ndarray) -> dict[str, Any]:
        """
        Parameters
        ----------
        series : np.ndarray, shape (N,)
            1-D price or indicator time-series (evenly sampled).

        Returns
        -------
        dict with keys:
            - ``phases``: array of normalised phase angles [0, π]
            - ``amplitudes``: normalised amplitudes [0, 1]
            - ``dominant_period``: estimated period of the strongest cycle
        """
        n = len(series)
        if n < 8:
            return {"phases": [], "amplitudes": [], "dominant_period": None}

        # Detrend
        detrended = series - np.polyval(np.polyfit(np.arange(n), series, 1), np.arange(n))

        # FFT
        spectrum = np.fft.rfft(detrended * np.hanning(n))
        magnitudes = np.abs(spectrum)
        phases = np.angle(spectrum)  # [-π, π]

        # Top-k frequencies (excluding DC)
        freqs = np.fft.rfftfreq(n)
        mag_no_dc = magnitudes[1:]
        top_idx = np.argsort(mag_no_dc)[-self.top_k:][::-1] + 1  # offset DC

        top_freqs = freqs[top_idx]
        top_phases = (phases[top_idx] % np.pi)  # normalise to [0, π]
        top_amps = mag_no_dc[top_idx - 1]

        # Normalise amplitudes
        max_amp = top_amps.max() if top_amps.max() > 0 else 1.0
        top_amps_norm = top_amps / max_amp

        dominant_freq = top_freqs[0] if len(top_freqs) > 0 else None
        dominant_period = (
            int(round(1.0 / dominant_freq)) if dominant_freq and dominant_freq > 0 else None
        )

        return {
            "phases": top_phases.tolist(),
            "amplitudes": top_amps_norm.tolist(),
            "dominant_period": dominant_period,
            "frequencies": top_freqs.tolist(),
        }
