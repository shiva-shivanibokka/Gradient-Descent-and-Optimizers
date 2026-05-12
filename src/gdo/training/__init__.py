"""PyTorch training infrastructure — Trainer, models, and metrics."""

from gdo.training.trainer import Trainer
from gdo.training.models import MLP, CNN
from gdo.training.metrics import ConvergenceTracker, GradientNormMonitor

__all__ = [
    "Trainer",
    "MLP",
    "CNN",
    "ConvergenceTracker",
    "GradientNormMonitor",
]
