"""
metrics.py
==========
Training metrics tracking and monitoring utilities.

Classes
-------
EpochMetrics        — Dataclass holding per-epoch train/val stats
ConvergenceTracker  — Tracks convergence, early stopping, best checkpoint
GradientNormMonitor — Monitors gradient norms per layer and globally

These classes are used by the Trainer to:
  1. Build structured metric records logged to MLflow
  2. Detect early stopping conditions
  3. Detect gradient explosion (warns when norm exceeds threshold)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Metrics recorded at the end of a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    lr: float
    grad_norm: float
    epoch_time_s: float

    def to_dict(self) -> dict[str, float]:
        """Return a flat dict for MLflow metric logging."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
            "lr": self.lr,
            "grad_norm": self.grad_norm,
            "epoch_time_s": self.epoch_time_s,
        }

    def __str__(self) -> str:
        return (
            f"Epoch {self.epoch:03d} | "
            f"train_loss={self.train_loss:.4f} | val_loss={self.val_loss:.4f} | "
            f"train_acc={self.train_acc:.3f} | val_acc={self.val_acc:.3f} | "
            f"lr={self.lr:.2e} | grad_norm={self.grad_norm:.4f}"
        )


class ConvergenceTracker:
    """
    Tracks validation loss for early stopping and best-model detection.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
        Set to None to disable early stopping.
    min_delta:
        Minimum improvement to count as "improvement".
    mode:
        ``"min"`` (lower val_loss is better) or ``"max"`` (higher is better).
    warmup_epochs:
        Do not apply early stopping in the first N epochs.

    Example
    -------
    >>> tracker = ConvergenceTracker(patience=5)
    >>> for epoch in range(100):
    ...     metrics = trainer.run_epoch()
    ...     if tracker.update(metrics.val_loss, epoch):
    ...         print("Early stopping triggered")
    ...         break
    """

    def __init__(
        self,
        patience: int | None = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        warmup_epochs: int = 0,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._warmup_epochs = warmup_epochs
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._best_epoch: int = 0
        self._wait: int = 0
        self._history: list[EpochMetrics] = []

    def update(self, metrics: EpochMetrics) -> bool:
        """
        Update the tracker with new epoch metrics.

        Parameters
        ----------
        metrics:
            Metrics from the current epoch.

        Returns
        -------
        bool
            True if early stopping should be triggered.
        """
        self._history.append(metrics)
        val = metrics.val_loss
        epoch = metrics.epoch

        improved = (
            val < self._best - self._min_delta
            if self._mode == "min"
            else val > self._best + self._min_delta
        )

        if improved:
            self._best = val
            self._best_epoch = epoch
            self._wait = 0
            logger.debug(
                "[ConvergenceTracker] New best val_loss=%.6f at epoch %d",
                self._best,
                epoch,
            )
        else:
            self._wait += 1
            logger.debug(
                "[ConvergenceTracker] No improvement for %d/%s epochs.",
                self._wait,
                str(self._patience) if self._patience else "∞",
            )

        if self._patience is None or epoch < self._warmup_epochs:
            return False

        should_stop = self._wait >= self._patience
        if should_stop:
            logger.info(
                "[ConvergenceTracker] Early stopping at epoch %d. Best val_loss=%.6f at epoch %d.",
                epoch,
                self._best,
                self._best_epoch,
            )
        return should_stop

    @property
    def best_value(self) -> float:
        """Best validation metric seen so far."""
        return self._best

    @property
    def best_epoch(self) -> int:
        """Epoch index where the best value was achieved."""
        return self._best_epoch

    @property
    def epochs_without_improvement(self) -> int:
        """How many consecutive epochs without improvement."""
        return self._wait

    @property
    def history(self) -> list[EpochMetrics]:
        """Full list of EpochMetrics recorded so far."""
        return self._history

    def train_loss_curve(self) -> np.ndarray:
        return np.array([m.train_loss for m in self._history])

    def val_loss_curve(self) -> np.ndarray:
        return np.array([m.val_loss for m in self._history])

    def val_acc_curve(self) -> np.ndarray:
        return np.array([m.val_acc for m in self._history])

    def convergence_step(self, threshold: float = 0.01) -> int | None:
        """
        Return the first epoch where val_loss dropped below ``threshold``.

        Returns None if the threshold was never reached.
        """
        for m in self._history:
            if (self._mode == "min" and m.val_loss <= threshold) or (
                self._mode == "max" and m.val_loss >= threshold
            ):
                return m.epoch
        return None


class GradientNormMonitor:
    """
    Monitors per-layer and global gradient norms during training.

    Detects gradient explosion and logs warnings when the global
    gradient norm exceeds ``explosion_threshold``.

    Parameters
    ----------
    explosion_threshold:
        Log a WARNING when global grad norm exceeds this value.
        Default 10.0 (typical threshold for gradient clipping).
    window:
        Rolling window size for computing moving average of grad norms.

    Usage
    -----
    Called inside the training loop after ``loss.backward()``:

    >>> monitor = GradientNormMonitor()
    >>> for batch in dataloader:
    ...     loss.backward()
    ...     monitor.record(model.get_grad_norms(), model.get_total_grad_norm())
    ...     optimizer.step()
    >>> print(monitor.summary())
    """

    def __init__(self, explosion_threshold: float = 10.0, window: int = 50) -> None:
        self._threshold = explosion_threshold
        self._window = window
        self._global_norms: list[float] = []
        self._layer_norms: dict[str, list[float]] = {}
        self._explosion_count: int = 0
        self._rolling: deque[float] = deque(maxlen=window)

    def record(self, layer_norms: dict[str, float], global_norm: float) -> None:
        """
        Record gradient norms for the current step.

        Parameters
        ----------
        layer_norms:
            Per-layer gradient norms from ``model.get_grad_norms()``.
        global_norm:
            Total gradient norm from ``model.get_total_grad_norm()``.
        """
        self._global_norms.append(global_norm)
        self._rolling.append(global_norm)

        for name, norm in layer_norms.items():
            self._layer_norms.setdefault(name, []).append(norm)

        if global_norm > self._threshold:
            self._explosion_count += 1
            logger.warning(
                "[GradientNormMonitor] Gradient explosion detected! "
                "global_norm=%.4f (threshold=%.1f). Step=%d. "
                "Consider reducing LR or enabling gradient clipping.",
                global_norm,
                self._threshold,
                len(self._global_norms),
            )

    def rolling_mean(self) -> float:
        """Moving average of gradient norm over the last ``window`` steps."""
        return float(np.mean(self._rolling)) if self._rolling else 0.0

    def global_norm_history(self) -> np.ndarray:
        """All global gradient norms recorded so far."""
        return np.array(self._global_norms)

    def layer_norm_history(self, layer_name: str) -> np.ndarray | None:
        """Gradient norm history for a specific layer."""
        return np.array(self._layer_norms[layer_name]) if layer_name in self._layer_norms else None

    def summary(self) -> dict[str, float]:
        """Return a summary dict suitable for MLflow logging."""
        if not self._global_norms:
            return {}
        arr = np.array(self._global_norms)
        return {
            "grad_norm_mean": float(arr.mean()),
            "grad_norm_max": float(arr.max()),
            "grad_norm_min": float(arr.min()),
            "grad_norm_std": float(arr.std()),
            "explosion_count": float(self._explosion_count),
        }

    def reset(self) -> None:
        """Clear all recorded norms."""
        self._global_norms = []
        self._layer_norms = {}
        self._explosion_count = 0
        self._rolling.clear()
