"""Optimizer implementations — NumPy from-scratch and PyTorch wrappers."""

from gdo.optimizers.base import Optimizer, OptimizerState
from gdo.optimizers.sgd import BatchGD, StochasticGD, MiniBatchGD, MomentumSGD
from gdo.optimizers.adaptive import RMSProp, Adam, AdamW, Lion
from gdo.optimizers.schedulers import (
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    OneCycleLR,
    CyclicalLR,
    WarmupScheduler,
    ReduceLROnPlateau,
)

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
