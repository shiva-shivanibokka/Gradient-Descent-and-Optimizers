"""
sgd.py
======
Pure-NumPy implementations of gradient descent variants.

Classes
-------
BatchGD         — Full-batch gradient descent
StochasticGD    — Stochastic gradient descent (one sample per step)
MiniBatchGD     — Mini-batch gradient descent (industry standard)
MomentumSGD     — SGD with classical momentum (Polyak, 1964)

All classes inherit from ``Optimizer`` and record their trajectory
so optimization paths can be visualized on a 2D loss landscape.

Notes
-----
These are *update-rule* implementations — they receive pre-computed
gradients and return updated parameters.  The caller is responsible
for computing the gradient (either analytically or via autograd).
This separation keeps the math explicit and testable.
"""

from __future__ import annotations

import logging

import numpy as np

from gdo.optimizers.base import Optimizer, OptimizerState

logger = logging.getLogger(__name__)


class BatchGD(Optimizer):
    """
    Full-batch gradient descent.

    Update rule::

        θ ← θ - lr * ∇L(θ)

    The gradient is computed over the entire dataset each step.
    Stable convergence, but impractically slow on large datasets.

    Parameters
    ----------
    lr:
        Learning rate (step size).
    """

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr=lr, name="Batch GD")

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one full-batch gradient update."""
        self._validate_inputs(params, grads)
        new_params = params - self._lr * grads
        self._record(new_params)
        logger.debug("[BatchGD] step %d | grad_norm=%.6f", self._t, float(np.linalg.norm(grads)))
        return new_params

    def reset(self) -> None:
        """Reset step counter and trajectory."""
        self._t = 0
        self._trajectory = []
        self._loss_history = []

    @property
    def state(self) -> OptimizerState:
        return OptimizerState(step=self._t, lr=self._lr)


class StochasticGD(Optimizer):
    """
    Stochastic gradient descent (SGD).

    Update rule::

        θ ← θ - lr * ∇L(θ; xᵢ, yᵢ)

    The gradient is computed from a single randomly sampled data point.
    Noisy updates help escape local minima but cause oscillation near
    the optimum.  The ``noise_scale`` parameter adds artificial noise
    to simulate the stochasticity when a pre-computed gradient is passed.

    Parameters
    ----------
    lr:
        Learning rate.
    noise_scale:
        Standard deviation of Gaussian noise added to the gradient to
        simulate sample-level stochasticity.  Set to 0 for deterministic
        mode (useful when the caller already passes a stochastic gradient).
    seed:
        Random seed for reproducibility.
    """

    def __init__(self, lr: float = 0.01, noise_scale: float = 0.1, seed: int = 42) -> None:
        super().__init__(lr=lr, name="SGD")
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one stochastic gradient update."""
        self._validate_inputs(params, grads)
        noisy_grads = grads + self._rng.normal(0, self._noise_scale, size=grads.shape)
        new_params = params - self._lr * noisy_grads
        self._record(new_params)
        logger.debug(
            "[SGD] step %d | grad_norm=%.6f | noisy_grad_norm=%.6f",
            self._t,
            float(np.linalg.norm(grads)),
            float(np.linalg.norm(noisy_grads)),
        )
        return new_params

    def reset(self) -> None:
        self._t = 0
        self._trajectory = []
        self._loss_history = []

    @property
    def state(self) -> OptimizerState:
        return OptimizerState(
            step=self._t,
            lr=self._lr,
            extra={"noise_scale": self._noise_scale},
        )


class MiniBatchGD(Optimizer):
    """
    Mini-batch gradient descent — the industry standard.

    Update rule::

        θ ← θ - lr * (1/B) * Σ ∇L(θ; xᵢ, yᵢ)   for i in batch B

    The gradient is averaged over a small batch of B samples.
    Balances the stability of full-batch GD with the speed of SGD.
    The ``noise_scale`` parameter simulates batch-level gradient noise.

    Parameters
    ----------
    lr:
        Learning rate.
    batch_size:
        Number of samples per gradient estimate (informational only —
        actual batching is done by the caller's DataLoader).
    noise_scale:
        Gradient noise scale.  Smaller than SGD because averaging
        reduces variance by sqrt(B).
    seed:
        Random seed.
    """

    def __init__(
        self,
        lr: float = 0.01,
        batch_size: int = 32,
        noise_scale: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__(lr=lr, name=f"Mini-Batch GD (B={batch_size})")
        self._batch_size = batch_size
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one mini-batch gradient update."""
        self._validate_inputs(params, grads)
        noise = self._rng.normal(0, self._noise_scale, size=grads.shape)
        effective_grads = grads + noise
        new_params = params - self._lr * effective_grads
        self._record(new_params)
        logger.debug(
            "[MiniBatchGD] step %d | B=%d | grad_norm=%.6f",
            self._t,
            self._batch_size,
            float(np.linalg.norm(grads)),
        )
        return new_params

    def reset(self) -> None:
        self._t = 0
        self._trajectory = []
        self._loss_history = []

    @property
    def state(self) -> OptimizerState:
        return OptimizerState(
            step=self._t,
            lr=self._lr,
            extra={"batch_size": self._batch_size, "noise_scale": self._noise_scale},
        )


class MomentumSGD(Optimizer):
    """
    SGD with classical momentum (Polyak, 1964).

    Update rules::

        v ← β * v - lr * ∇L(θ)      # velocity accumulation
        θ ← θ + v                    # parameter update

    Momentum accumulates a velocity vector that dampens oscillations
    and accelerates convergence in the consistent gradient direction.
    This is equivalent to an exponential moving average of gradients.

    Parameters
    ----------
    lr:
        Learning rate.
    momentum:
        Momentum coefficient β (typically 0.9).  Higher values give
        more inertia; values ≥ 1.0 cause divergence.
    noise_scale:
        Gradient noise to simulate stochasticity.
    seed:
        Random seed.

    Notes
    -----
    This implementation uses the "classical" formulation where the
    velocity is initialized to zero.  PyTorch uses the same convention.
    Nesterov momentum is NOT included here — it changes the gradient
    evaluation point and is covered in Notebook 2.
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        noise_scale: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__(lr=lr, name=f"SGD + Momentum (β={momentum})")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self._momentum = momentum
        self._noise_scale = noise_scale
        self._rng = np.random.default_rng(seed)
        self._velocity: np.ndarray | None = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one momentum SGD update."""
        self._validate_inputs(params, grads)

        # Initialize velocity to zero on first step
        if self._velocity is None:
            self._velocity = np.zeros_like(params)

        # Add noise to simulate stochasticity
        noisy_grads = grads + self._rng.normal(0, self._noise_scale, size=grads.shape)

        # Velocity update: v ← β * v - lr * g
        velocity = self._momentum * self._velocity - self._lr * noisy_grads
        self._velocity = velocity

        # Parameter update: θ ← θ + v
        new_params = params + velocity

        self._record(new_params)
        logger.debug(
            "[MomentumSGD] step %d | β=%.2f | velocity_norm=%.6f | grad_norm=%.6f",
            self._t,
            self._momentum,
            float(np.linalg.norm(velocity)),
            float(np.linalg.norm(grads)),
        )
        return new_params

    def reset(self) -> None:
        self._t = 0
        self._trajectory = []
        self._loss_history = []
        self._velocity = None

    @property
    def state(self) -> OptimizerState:
        vel_norm = float(np.linalg.norm(self._velocity)) if self._velocity is not None else 0.0
        return OptimizerState(
            step=self._t,
            lr=self._lr,
            extra={"momentum": self._momentum, "velocity_norm": vel_norm},
        )
