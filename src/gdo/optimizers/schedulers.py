"""
schedulers.py
=============
Learning rate schedulers — pure-Python implementations that work with
both the NumPy optimizers and as a reference alongside PyTorch's
``torch.optim.lr_scheduler`` module.

Every scheduler follows the same interface:
  - ``step(epoch, metric=None)`` → new learning rate
  - ``get_lr()``                 → current learning rate
  - ``plot()``                   → return LR curve array for visualization

Classes
-------
StepLR            — Reduce LR by a factor every N epochs
CosineAnnealingLR — Smooth cosine decay (used in HAN, TFT, BERT)
OneCycleLR        — Warmup then decay (fast.ai, modern training recipes)
CyclicalLR        — Cyclical Learning Rates (Leslie Smith, 2017)
WarmupScheduler   — Linear warmup then cosine decay (Transformer standard)
ReduceLROnPlateau — Reduce when validation metric stops improving
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class LRScheduler(ABC):
    """
    Abstract base class for all learning rate schedulers.

    Parameters
    ----------
    optimizer_lr:
        Initial (base) learning rate of the optimizer.
    """

    def __init__(self, optimizer_lr: float) -> None:
        if optimizer_lr <= 0.0:
            raise ValueError(f"optimizer_lr must be > 0, got {optimizer_lr}")
        self._base_lr = optimizer_lr
        self._current_lr = optimizer_lr
        self._epoch = 0

    @abstractmethod
    def step(self, epoch: int | None = None, metric: float | None = None) -> float:
        """
        Advance the scheduler by one epoch and return the new LR.

        Parameters
        ----------
        epoch:
            Current epoch index (0-based).  If None, auto-increments.
        metric:
            Validation metric (used only by ReduceLROnPlateau).

        Returns
        -------
        float
            The learning rate for the next epoch.
        """
        ...

    def get_lr(self) -> float:
        """Return the current learning rate."""
        return self._current_lr

    def get_lr_curve(self, total_epochs: int) -> np.ndarray:
        """
        Simulate the LR schedule over ``total_epochs`` and return the curve.

        Useful for visualization without modifying the scheduler's state.

        Parameters
        ----------
        total_epochs:
            Number of epochs to simulate.

        Returns
        -------
        np.ndarray
            Array of shape ``(total_epochs,)`` containing LR at each epoch.
        """
        # Save state
        saved_lr = self._current_lr
        saved_epoch = self._epoch

        lrs = []
        for e in range(total_epochs):
            lr = self.step(epoch=e)
            lrs.append(lr)

        # Restore state
        self._current_lr = saved_lr
        self._epoch = saved_epoch
        return np.array(lrs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(base_lr={self._base_lr}, current_lr={self._current_lr:.6g})"
        )


class StepLR(LRScheduler):
    """
    Reduce learning rate by ``gamma`` every ``step_size`` epochs.

    Schedule::

        lr = base_lr * gamma^(floor(epoch / step_size))

    Simple and interpretable.  The most common scheduler before
    cosine annealing became the default.

    Parameters
    ----------
    optimizer_lr:
        Initial learning rate.
    step_size:
        Number of epochs between reductions.
    gamma:
        Multiplicative factor (default 0.1 = 10x reduction per step).
    """

    def __init__(self, optimizer_lr: float, step_size: int = 10, gamma: float = 0.1) -> None:
        super().__init__(optimizer_lr)
        if step_size <= 0:
            raise ValueError(f"step_size must be > 0, got {step_size}")
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        self._step_size = step_size
        self._gamma = gamma

    def step(self, epoch: int | None = None, metric: float | None = None) -> float:
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch
        self._current_lr = self._base_lr * (self._gamma ** (self._epoch // self._step_size))
        logger.debug("[StepLR] epoch %d | lr=%.6g", self._epoch, self._current_lr)
        return self._current_lr


class CosineAnnealingLR(LRScheduler):
    """
    Smooth cosine decay (Loshchilov & Hutter, 2017).

    Schedule::

        lr(t) = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(π * t / T_max))

    The learning rate follows a half-cosine curve from ``base_lr`` to
    ``eta_min`` over ``t_max`` epochs.  Used in:
    - BERT pretraining (with warmup)
    - HAN super-resolution (this portfolio)
    - TFT forecasting (this portfolio)
    - Most modern vision models

    Parameters
    ----------
    optimizer_lr:
        Initial (maximum) learning rate.
    t_max:
        Half-period (epochs to reach eta_min).
    eta_min:
        Minimum learning rate at the trough (default 0).
    """

    def __init__(self, optimizer_lr: float, t_max: int = 50, eta_min: float = 0.0) -> None:
        super().__init__(optimizer_lr)
        if t_max <= 0:
            raise ValueError(f"t_max must be > 0, got {t_max}")
        self._t_max = t_max
        self._eta_min = eta_min

    def step(self, epoch: int | None = None, metric: float | None = None) -> float:
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch
        t = self._epoch % self._t_max
        self._current_lr = self._eta_min + 0.5 * (self._base_lr - self._eta_min) * (
            1.0 + math.cos(math.pi * t / self._t_max)
        )
        logger.debug("[CosineAnnealingLR] epoch %d | lr=%.6g", self._epoch, self._current_lr)
        return self._current_lr


class OneCycleLR(LRScheduler):
    """
    1-Cycle Learning Rate Policy (Smith & Touvron, 2019).

    Phase 1 — Warmup (``pct_start`` fraction of total epochs):
        LR increases linearly from ``base_lr`` to ``max_lr``.

    Phase 2 — Annealing (remaining epochs):
        LR decreases from ``max_lr`` to a final value (base_lr / 1e4).

    Used in fast.ai training recipes and increasingly in modern
    vision and NLP training loops.  Allows training with a much
    higher peak LR than constant-LR training.

    Parameters
    ----------
    optimizer_lr:
        Starting (minimum) learning rate for warmup phase.
    max_lr:
        Peak learning rate at the top of the cycle.
    total_epochs:
        Total number of training epochs.
    pct_start:
        Fraction of total epochs used for warmup (default 0.3).
    """

    def __init__(
        self,
        optimizer_lr: float,
        max_lr: float,
        total_epochs: int,
        pct_start: float = 0.3,
    ) -> None:
        super().__init__(optimizer_lr)
        if max_lr <= optimizer_lr:
            raise ValueError(f"max_lr ({max_lr}) must be > optimizer_lr ({optimizer_lr})")
        if not 0.0 < pct_start < 1.0:
            raise ValueError(f"pct_start must be in (0, 1), got {pct_start}")
        self._max_lr = max_lr
        self._total_epochs = total_epochs
        self._pct_start = pct_start
        self._warmup_epochs = int(pct_start * total_epochs)
        self._final_lr = optimizer_lr / 1e4

    def step(self, epoch: int | None = None, metric: float | None = None) -> float:
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch

        e = self._epoch
        if e <= self._warmup_epochs:
            # Linear warmup
            self._current_lr = self._base_lr + (self._max_lr - self._base_lr) * e / max(
                1, self._warmup_epochs
            )
        else:
            # Cosine annealing from max_lr to final_lr
            progress = (e - self._warmup_epochs) / max(1, self._total_epochs - self._warmup_epochs)
            self._current_lr = self._final_lr + 0.5 * (self._max_lr - self._final_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

        logger.debug("[OneCycleLR] epoch %d | lr=%.6g", self._epoch, self._current_lr)
        return self._current_lr


class CyclicalLR(LRScheduler):
    """
    Cyclical Learning Rates (Leslie Smith, 2017).

    The learning rate oscillates between ``base_lr`` and ``max_lr``
    using a triangular wave (linear up + linear down).  Each half-cycle
    takes ``step_size_up`` epochs.

    Key insight from Smith (2017): allowing the LR to increase
    periodically helps escape saddle points and sharp local minima,
    often achieving better generalization than monotonically decaying LR.

    Parameters
    ----------
    optimizer_lr:
        Minimum (base) LR at the bottom of each cycle.
    max_lr:
        Maximum LR at the peak of each cycle.
    step_size_up:
        Number of epochs for the increasing half of the cycle.
    mode:
        ``"triangular"`` (constant amplitude) or
        ``"triangular2"`` (amplitude halved every full cycle).
    """

    def __init__(
        self,
        optimizer_lr: float,
        max_lr: float,
        step_size_up: int = 4,
        mode: str = "triangular",
    ) -> None:
        super().__init__(optimizer_lr)
        if max_lr <= optimizer_lr:
            raise ValueError("max_lr must be > base_lr")
        if mode not in ("triangular", "triangular2"):
            raise ValueError(f"mode must be 'triangular' or 'triangular2', got {mode}")
        self._max_lr = max_lr
        self._step_size_up = step_size_up
        self._mode = mode

    def step(self, epoch: int | None = None, metric: float | None = None) -> float:
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch

        cycle_length = 2 * self._step_size_up
        cycle = math.floor(1 + self._epoch / cycle_length)
        x = abs(self._epoch / self._step_size_up - 2 * cycle + 1)
        scale = 1.0 if self._mode == "triangular" else 0.5 ** (cycle - 1)
        self._current_lr = (
            self._base_lr + (self._max_lr - self._base_lr) * max(0.0, 1.0 - x) * scale
        )

        logger.debug(
            "[CyclicalLR] epoch %d | lr=%.6g | cycle=%d", self._epoch, self._current_lr, cycle
        )
        return self._current_lr


class WarmupScheduler(LRScheduler):
    """
    Linear warmup followed by cosine annealing decay.

    This is the standard schedule used for Transformer pretraining
    (BERT, GPT, T5, ViT).  Cold-start with a large LR causes instability
    in the first few steps because the loss landscape is steep and
    the gradient magnitude is large; warmup avoids this.

    Schedule::

        Phase 1 (epoch < warmup_steps):
            lr = base_lr * epoch / warmup_steps          # linear warmup

        Phase 2 (epoch >= warmup_steps):
            lr = eta_min + 0.5 * (base_lr - eta_min)
                 * (1 + cos(π * (e - warmup) / (total - warmup)))  # cosine decay

    Parameters
    ----------
    optimizer_lr:
        Peak learning rate (reached at end of warmup).
    total_epochs:
        Total training epochs.
    warmup_steps:
        Number of warmup epochs (not steps — simplified for notebook use).
    eta_min:
        Minimum LR after cosine decay completes.
    """

    def __init__(
        self,
        optimizer_lr: float,
        total_epochs: int,
        warmup_steps: int = 5,
        eta_min: float = 0.0,
    ) -> None:
        super().__init__(optimizer_lr)
        self._total_epochs = total_epochs
        self._warmup_steps = warmup_steps
        self._eta_min = eta_min

    def step(self, epoch: int | None = None, metric: float | None = None) -> float:
        if epoch is None:
            self._epoch += 1
        else:
            self._epoch = epoch

        e = self._epoch
        if e < self._warmup_steps:
            # Linear warmup from 0 to base_lr
            self._current_lr = self._base_lr * (e + 1) / self._warmup_steps
        else:
            # Cosine annealing from base_lr to eta_min
            progress = (e - self._warmup_steps) / max(1, self._total_epochs - self._warmup_steps)
            self._current_lr = self._eta_min + 0.5 * (self._base_lr - self._eta_min) * (
                1.0 + math.cos(math.pi * progress)
            )

        logger.debug("[WarmupScheduler] epoch %d | lr=%.6g", self._epoch, self._current_lr)
        return self._current_lr


class ReduceLROnPlateau(LRScheduler):
    """
    Reduce LR when a validation metric stops improving.

    Monitors a scalar metric (e.g., validation loss) and multiplies
    the current LR by ``factor`` if no improvement is seen for
    ``patience`` consecutive epochs.

    Parameters
    ----------
    optimizer_lr:
        Initial learning rate.
    factor:
        LR reduction multiplier when triggered (default 0.1).
    patience:
        Epochs with no improvement before triggering (default 5).
    threshold:
        Minimum meaningful improvement (default 1e-4).
    mode:
        ``"min"`` (metric should decrease) or ``"max"`` (should increase).
    min_lr:
        Floor on the learning rate.
    """

    def __init__(
        self,
        optimizer_lr: float,
        factor: float = 0.1,
        patience: int = 5,
        threshold: float = 1e-4,
        mode: str = "min",
        min_lr: float = 1e-8,
    ) -> None:
        super().__init__(optimizer_lr)
        if not 0.0 < factor < 1.0:
            raise ValueError(f"factor must be in (0, 1), got {factor}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self._factor = factor
        self._patience = patience
        self._threshold = threshold
        self._mode = mode
        self._min_lr = min_lr
        self._best: float = math.inf if mode == "min" else -math.inf
        self._wait = 0
        self._num_reductions = 0

    def step(self, epoch: int | None = None, metric: float | None = None) -> float:
        if epoch is not None:
            self._epoch = epoch
        else:
            self._epoch += 1

        if metric is None:
            logger.warning("[ReduceLROnPlateau] No metric provided — LR unchanged.")
            return self._current_lr

        improved = (
            metric < self._best - self._threshold
            if self._mode == "min"
            else metric > self._best + self._threshold
        )

        if improved:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                new_lr = max(self._current_lr * self._factor, self._min_lr)
                if new_lr < self._current_lr:
                    self._num_reductions += 1
                    logger.info(
                        "[ReduceLROnPlateau] No improvement for %d epochs. "
                        "Reducing LR: %.6g → %.6g",
                        self._patience,
                        self._current_lr,
                        new_lr,
                    )
                    self._current_lr = new_lr
                self._wait = 0

        return self._current_lr

    @property
    def num_reductions(self) -> int:
        """Number of times the LR has been reduced."""
        return self._num_reductions
