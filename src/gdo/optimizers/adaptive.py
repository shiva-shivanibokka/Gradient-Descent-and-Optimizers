"""
adaptive.py
===========
Pure-NumPy implementations of adaptive learning rate optimizers.

Classes
-------
RMSProp   — Root Mean Square Propagation (Hinton, 2012)
Adam      — Adaptive Moment Estimation (Kingma & Ba, 2014)
AdamW     — Adam with decoupled weight decay (Loshchilov & Hutter, 2019)
Lion      — EvoLved Sign Momentum (Chen et al., Google Brain, 2023)

Design principle
----------------
Each class implements the exact mathematical update rules from the
original paper — no shortcuts.  Bias correction, epsilon placement,
and weight decay handling all match the canonical formulations so that
unit tests can verify against the paper's pseudocode line by line.

Key distinction: AdamW vs Adam
-------------------------------
Adam applies weight decay as L2 regularization (adds λθ to the gradient
before computing moments).  This means the effective weight decay is
scaled by the adaptive learning rate, which is not what most practitioners
intend.  AdamW decouples weight decay from the gradient update:

    Adam:   θ ← θ - α * m̂ / (√v̂ + ε) - α * λ * θ   (L2 reg)
    AdamW:  θ ← (1 - α*λ) * θ - α * m̂ / (√v̂ + ε)  (decoupled)

AdamW is the default for all Transformer models (BERT, GPT, etc.).
"""

from __future__ import annotations

import logging

import numpy as np

from gdo.optimizers.base import Optimizer, OptimizerState

logger = logging.getLogger(__name__)


class RMSProp(Optimizer):
    """
    RMSProp — Root Mean Square Propagation.

    Update rules (Hinton's unpublished lecture notes, 2012)::

        v ← α * v + (1 - α) * g²          # EMA of squared gradients
        θ ← θ - lr / √(v + ε) * g         # adaptive update

    Divides the learning rate by a running average of recent gradient
    magnitudes.  Prevents learning rate from growing too large in
    dimensions where gradients are consistently large, and helps
    in dimensions where gradients are small.

    Parameters
    ----------
    lr:
        Base learning rate.
    alpha:
        Smoothing constant for the squared gradient EMA (default 0.99).
    epsilon:
        Numerical stability term added inside the square root.
    """

    def __init__(self, lr: float = 0.001, alpha: float = 0.99, epsilon: float = 1e-8) -> None:
        super().__init__(lr=lr, name="RMSProp")
        self._alpha = alpha
        self._epsilon = epsilon
        self._v: np.ndarray | None = None  # EMA of squared gradients

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one RMSProp update."""
        self._validate_inputs(params, grads)

        if self._v is None:
            self._v = np.zeros_like(params)

        # v ← α * v + (1 - α) * g²
        self._v = self._alpha * self._v + (1.0 - self._alpha) * grads**2

        # θ ← θ - lr / sqrt(v + ε) * g
        new_params = params - (self._lr / np.sqrt(self._v + self._epsilon)) * grads

        self._record(new_params)
        logger.debug(
            "[RMSProp] step %d | grad_norm=%.6f | rms=%.6f",
            self._t,
            float(np.linalg.norm(grads)),
            float(np.sqrt(np.mean(self._v))),
        )
        return new_params

    def reset(self) -> None:
        self._t = 0
        self._trajectory = []
        self._loss_history = []
        self._v = None

    @property
    def state(self) -> OptimizerState:
        rms = float(np.sqrt(np.mean(self._v))) if self._v is not None else 0.0
        return OptimizerState(
            step=self._t,
            lr=self._lr,
            extra={"alpha": self._alpha, "rms": rms},
        )


class Adam(Optimizer):
    """
    Adam — Adaptive Moment Estimation (Kingma & Ba, 2014).

    Update rules::

        m ← β₁ * m + (1 - β₁) * g           # first moment (mean)
        v ← β₂ * v + (1 - β₂) * g²          # second moment (variance)
        m̂ = m / (1 - β₁ᵗ)                    # bias-corrected mean
        v̂ = v / (1 - β₂ᵗ)                    # bias-corrected variance
        θ ← θ - lr * m̂ / (√v̂ + ε)           # parameter update

    Adam combines RMSProp (adaptive scale) with momentum (velocity).
    It is the most widely used optimizer in deep learning as of 2024.

    Parameters
    ----------
    lr:
        Base learning rate (default 0.001 — the paper's recommendation).
    beta1:
        First moment decay rate (default 0.9).
    beta2:
        Second moment decay rate (default 0.999).
    epsilon:
        Numerical stability term (default 1e-8).

    Notes
    -----
    Bias correction is REQUIRED for correct behavior.  Without it,
    the first few updates are too small because m and v are initialized
    to zero and are biased toward zero.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(lr=lr, name="Adam")
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._m: np.ndarray | None = None  # first moment
        self._v: np.ndarray | None = None  # second moment

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one Adam update."""
        self._validate_inputs(params, grads)

        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)

        assert self._v is not None  # guaranteed by the if-block above

        # Increment step BEFORE computing bias correction (t starts at 1)
        t = self._t + 1

        # m ← β₁ * m + (1 - β₁) * g
        self._m = self._beta1 * self._m + (1.0 - self._beta1) * grads

        # v ← β₂ * v + (1 - β₂) * g²
        self._v = self._beta2 * self._v + (1.0 - self._beta2) * grads**2

        # Bias-corrected estimates
        m_hat = self._m / (1.0 - self._beta1**t)
        v_hat = self._v / (1.0 - self._beta2**t)

        # θ ← θ - lr * m̂ / (√v̂ + ε)
        new_params = params - self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)

        self._record(new_params)
        logger.debug(
            "[Adam] step %d | grad_norm=%.6f | m_hat_norm=%.6f",
            self._t,
            float(np.linalg.norm(grads)),
            float(np.linalg.norm(m_hat)),
        )
        return new_params

    def reset(self) -> None:
        self._t = 0
        self._trajectory = []
        self._loss_history = []
        self._m = None
        self._v = None

    @property
    def state(self) -> OptimizerState:
        m_norm = float(np.linalg.norm(self._m)) if self._m is not None else 0.0
        v_norm = float(np.linalg.norm(self._v)) if self._v is not None else 0.0
        return OptimizerState(
            step=self._t,
            lr=self._lr,
            extra={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "m_norm": m_norm,
                "v_norm": v_norm,
            },
        )


class AdamW(Optimizer):
    """
    AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Update rules::

        m ← β₁ * m + (1 - β₁) * g           # first moment
        v ← β₂ * v + (1 - β₂) * g²          # second moment
        m̂ = m / (1 - β₁ᵗ)
        v̂ = v / (1 - β₂ᵗ)
        θ ← (1 - lr * λ) * θ                 # weight decay FIRST
        θ ← θ - lr * m̂ / (√v̂ + ε)           # then gradient step

    The critical difference from Adam: weight decay is applied directly
    to the parameters BEFORE the gradient update, NOT as part of the
    gradient.  This prevents the adaptive learning rate from scaling
    down the regularization effect, which is the correct interpretation
    of L2 regularization for neural networks.

    Parameters
    ----------
    lr:
        Base learning rate.
    beta1:
        First moment decay (default 0.9).
    beta2:
        Second moment decay (default 0.999).
    epsilon:
        Numerical stability term.
    weight_decay:
        Decoupled weight decay coefficient λ (default 0.01).
        Values in [0.01, 0.1] are typical for Transformers.
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__(lr=lr, name="AdamW")
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._m: np.ndarray | None = None
        self._v: np.ndarray | None = None

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one AdamW update."""
        self._validate_inputs(params, grads)

        if self._m is None:
            self._m = np.zeros_like(params)
            self._v = np.zeros_like(params)

        assert self._v is not None

        t = self._t + 1

        # Moment updates (on original gradients, NOT including weight decay term)
        self._m = self._beta1 * self._m + (1.0 - self._beta1) * grads
        self._v = self._beta2 * self._v + (1.0 - self._beta2) * grads**2

        m_hat = self._m / (1.0 - self._beta1**t)
        v_hat = self._v / (1.0 - self._beta2**t)

        # Decoupled weight decay: θ ← (1 - lr * λ) * θ
        params_wd = (1.0 - self._lr * self._weight_decay) * params

        # Gradient step on weight-decayed params
        new_params = params_wd - self._lr * m_hat / (np.sqrt(v_hat) + self._epsilon)

        self._record(new_params)
        logger.debug(
            "[AdamW] step %d | grad_norm=%.6f | wd=%.4f",
            self._t,
            float(np.linalg.norm(grads)),
            self._weight_decay,
        )
        return new_params

    def reset(self) -> None:
        self._t = 0
        self._trajectory = []
        self._loss_history = []
        self._m = None
        self._v = None

    @property
    def state(self) -> OptimizerState:
        m_norm = float(np.linalg.norm(self._m)) if self._m is not None else 0.0
        return OptimizerState(
            step=self._t,
            lr=self._lr,
            extra={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "weight_decay": self._weight_decay,
                "m_norm": m_norm,
            },
        )


class Lion(Optimizer):
    """
    Lion — EvoLved Sign Momentum (Chen et al., Google Brain, 2023).

    Update rules::

        c ← β₁ * m + (1 - β₁) * g           # interpolated update candidate
        θ ← (1 - lr * λ) * θ - lr * sign(c) # weight decay + sign update
        m ← β₂ * m + (1 - β₂) * g           # momentum update

    Key properties vs Adam:
      - Uses only the *sign* of the update — constant magnitude per dimension.
      - Memory-efficient: stores only one moment vector (vs two in Adam).
      - Originally discovered by Google Brain's evolutionary search over
        optimizer programs.
      - Default LR is ~3-10x smaller than Adam (typically 1e-4).
      - Works best with a large batch size and strong weight decay.

    Parameters
    ----------
    lr:
        Learning rate (default 1e-4 — 10x smaller than Adam default).
    beta1:
        Update interpolation coefficient (default 0.9).
    beta2:
        Momentum decay for the stored moment (default 0.99).
    weight_decay:
        Decoupled weight decay coefficient.

    References
    ----------
    Chen et al. (2023). "Symbolic Discovery of Optimization Algorithms."
    arXiv:2302.06675.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(lr=lr, name="Lion")
        self._beta1 = beta1
        self._beta2 = beta2
        self._weight_decay = weight_decay
        self._m: np.ndarray | None = None  # momentum buffer

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """Apply one Lion update."""
        self._validate_inputs(params, grads)

        if self._m is None:
            self._m = np.zeros_like(params)

        # Interpolated candidate for the update direction
        c = self._beta1 * self._m + (1.0 - self._beta1) * grads

        # Decoupled weight decay + sign-based update
        new_params = (1.0 - self._lr * self._weight_decay) * params - self._lr * np.sign(c)

        # Update momentum buffer AFTER the parameter update
        self._m = self._beta2 * self._m + (1.0 - self._beta2) * grads

        self._record(new_params)
        logger.debug(
            "[Lion] step %d | grad_norm=%.6f | sign_fraction=%.3f",
            self._t,
            float(np.linalg.norm(grads)),
            float(np.mean(np.abs(np.sign(c)))),
        )
        return new_params

    def reset(self) -> None:
        self._t = 0
        self._trajectory = []
        self._loss_history = []
        self._m = None

    @property
    def state(self) -> OptimizerState:
        m_norm = float(np.linalg.norm(self._m)) if self._m is not None else 0.0
        return OptimizerState(
            step=self._t,
            lr=self._lr,
            extra={
                "beta1": self._beta1,
                "beta2": self._beta2,
                "weight_decay": self._weight_decay,
                "m_norm": m_norm,
            },
        )
