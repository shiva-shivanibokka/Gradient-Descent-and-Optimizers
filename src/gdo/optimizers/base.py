"""
base.py
=======
Abstract base class for all NumPy optimizers in this library.

Design
------
Every optimizer in ``gdo`` is a pure-NumPy implementation that operates
on flat parameter arrays.  This keeps the math explicit and inspectable
— the primary goal of this project.

The PyTorch optimizers (``torch.optim``) are used separately in the
Trainer class for actual model training.  These NumPy implementations
are used for:
  - Loss landscape trajectory visualization (Notebooks 1 & 2)
  - The interactive web demo (landscape tab)
  - Unit tests that verify update-rule math

Interface contract
------------------
Every subclass must implement:
  - ``step(params, grads)``  → updated params
  - ``state``                → current internal state (moments, etc.)
  - ``reset()``              → reset state to initial values

The optimizer does NOT own the parameters — it only transforms gradients
into parameter updates.  This matches the PyTorch convention.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizerState:
    """
    Snapshot of an optimizer's internal state at a single step.

    Used for serialization, comparison, and logging.
    """

    step: int = 0
    lr: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        extras = ", ".join(f"{k}={v:.6g}" for k, v in self.extra.items())
        return (
            f"OptimizerState(step={self.step}, lr={self.lr:.6g}"
            + (f", {extras}" if extras else "")
            + ")"
        )


class Optimizer(ABC):
    """
    Abstract base class for NumPy-based gradient descent optimizers.

    Parameters
    ----------
    lr:
        Learning rate (must be > 0).
    name:
        Human-readable name used in logs and plot legends.

    Notes
    -----
    All subclasses must implement ``step`` and ``reset``.
    The ``trajectory`` attribute accumulates parameter vectors after
    each call to ``step`` — pass it directly to ``LandscapePlotter``
    to visualize the optimization path.
    """

    def __init__(self, lr: float, name: str) -> None:
        if lr <= 0.0:
            raise ValueError(f"Learning rate must be > 0, got {lr}")
        self._lr: float = lr
        self._name: str = name
        self._t: int = 0
        self._trajectory: list[np.ndarray] = []
        self._loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """
        Apply one gradient update and return the new parameters.

        Parameters
        ----------
        params:
            Current parameter vector, shape ``(n,)``.
        grads:
            Gradient of the loss w.r.t. ``params``, shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Updated parameter vector, shape ``(n,)``.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state (moments, step counter, trajectory)."""
        ...

    @property
    @abstractmethod
    def state(self) -> OptimizerState:
        """Return a snapshot of the current optimizer state."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers (available to all subclasses)
    # ------------------------------------------------------------------

    def _record(self, params: np.ndarray, loss: float | None = None) -> None:
        """
        Append current params to the trajectory buffer.

        Called at the end of every ``step()`` implementation.
        """
        self._trajectory.append(params.copy())
        if loss is not None:
            self._loss_history.append(loss)
        self._t += 1

    def _validate_inputs(self, params: np.ndarray, grads: np.ndarray) -> None:
        """Assert shape consistency between params and grads."""
        if params.shape != grads.shape:
            raise ValueError(f"params shape {params.shape} != grads shape {grads.shape}")
        if not np.isfinite(grads).all():
            logger.warning(
                "[%s] step %d: non-finite gradients detected (NaN/Inf). "
                "Consider gradient clipping or a lower learning rate.",
                self._name,
                self._t,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Optimizer display name."""
        return self._name

    @property
    def lr(self) -> float:
        """Current learning rate."""
        return self._lr

    @lr.setter
    def lr(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(f"Learning rate must be > 0, got {value}")
        self._lr = value

    @property
    def trajectory(self) -> list[np.ndarray]:
        """
        List of parameter vectors recorded after each step.

        Index 0 is the initial point (before any steps).
        Index k is the point after k steps.
        """
        return self._trajectory

    @property
    def loss_history(self) -> list[float]:
        """List of loss values recorded if passed to ``_record``."""
        return self._loss_history

    @property
    def n_steps(self) -> int:
        """Number of gradient steps taken so far."""
        return self._t

    def set_initial_point(self, params: np.ndarray) -> None:
        """
        Record the starting point before the first step.

        Call this once before the optimization loop so the trajectory
        includes the initial parameter position.
        """
        self._trajectory = [params.copy()]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lr={self._lr}, steps={self._t})"
