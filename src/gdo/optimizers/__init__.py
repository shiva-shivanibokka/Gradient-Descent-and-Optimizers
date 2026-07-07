"""Optimizer implementations — NumPy from-scratch and PyTorch wrappers."""

from gdo.optimizers.adaptive import Adam, AdamW, Lion, RMSProp
from gdo.optimizers.base import Optimizer, OptimizerState
from gdo.optimizers.schedulers import (
    CosineAnnealingLR,
    CyclicalLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    WarmupScheduler,
)
from gdo.optimizers.sgd import BatchGD, MiniBatchGD, MomentumSGD, StochasticGD

__all__ = [
    "Optimizer",
    "OptimizerState",
    "BatchGD",
    "StochasticGD",
    "MiniBatchGD",
    "MomentumSGD",
    "RMSProp",
    "Adam",
    "AdamW",
    "Lion",
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "CyclicalLR",
    "WarmupScheduler",
    "ReduceLROnPlateau",
]
