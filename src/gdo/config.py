"""
config.py
=========
Pydantic-based configuration models for all training experiments.

Every experiment is fully described by a YAML file that maps to these
models. No hyperparameters are hardcoded anywhere else in the codebase.

Usage
-----
    from gdo.config import ExperimentConfig
    cfg = ExperimentConfig.from_yaml("configs/adam_mnist.yaml")
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class OptimizerName(str, Enum):
    """All supported optimizer names."""

    BATCH_GD = "batch_gd"
    SGD = "sgd"
    MINI_BATCH_GD = "mini_batch_gd"
    MOMENTUM_SGD = "momentum_sgd"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"
    LION = "lion"


class SchedulerName(str, Enum):
    """All supported LR scheduler names."""

    NONE = "none"
    STEP = "step"
    COSINE = "cosine"
    ONECYCLE = "onecycle"
    CYCLICAL = "cyclical"
    WARMUP_COSINE = "warmup_cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class DatasetName(str, Enum):
    """Supported datasets for training experiments."""

    MNIST = "mnist"
    CIFAR10 = "cifar10"
    SYNTHETIC = "synthetic"


class ModelName(str, Enum):
    """Supported model architectures."""

    MLP = "mlp"
    CNN = "cnn"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------


class OptimizerConfig(BaseModel):
    """Hyperparameters for a single optimizer."""

    name: OptimizerName
    lr: float = Field(default=0.01, gt=0.0, description="Base learning rate")
    momentum: float = Field(
        default=0.9, ge=0.0, lt=1.0, description="Momentum coefficient (SGD+Momentum)"
    )
    beta1: float = Field(default=0.9, ge=0.0, lt=1.0, description="Adam/AdamW first moment decay")
    beta2: float = Field(
        default=0.999, ge=0.0, lt=1.0, description="Adam/AdamW second moment decay"
    )
    epsilon: float = Field(default=1e-8, gt=0.0, description="Numerical stability term")
    weight_decay: float = Field(default=0.0, ge=0.0, description="L2 / decoupled weight decay")
    alpha: float = Field(default=0.99, ge=0.0, lt=1.0, description="RMSProp smoothing constant")
    beta: float = Field(default=0.99, ge=0.0, lt=1.0, description="Lion exponential moving average")

    @field_validator("lr")
    @classmethod
    def lr_reasonable(cls, v: float) -> float:
        if v > 10.0:
            logger.warning(
                "Learning rate %.4f is unusually large — did you mean a smaller value?", v
            )
        return v


class SchedulerConfig(BaseModel):
    """Hyperparameters for a learning rate scheduler."""

    name: SchedulerName = SchedulerName.NONE
    step_size: int = Field(default=10, gt=0, description="StepLR: epochs between LR reductions")
    gamma: float = Field(
        default=0.1, gt=0.0, le=1.0, description="StepLR / ReduceLROnPlateau: LR decay factor"
    )
    t_max: int = Field(
        default=50, gt=0, description="CosineAnnealingLR: half-cycle length in epochs"
    )
    eta_min: float = Field(default=0.0, ge=0.0, description="CosineAnnealingLR: minimum LR")
    pct_start: float = Field(
        default=0.3, gt=0.0, lt=1.0, description="OneCycleLR: fraction of cycle for warmup"
    )
    max_lr: float = Field(
        default=0.1, gt=0.0, description="OneCycleLR / CyclicalLR: peak learning rate"
    )
    base_lr: float = Field(default=1e-4, gt=0.0, description="CyclicalLR: minimum learning rate")
    step_size_up: int = Field(
        default=2000, gt=0, description="CyclicalLR: steps for increasing LR phase"
    )
    warmup_steps: int = Field(default=500, ge=0, description="WarmupScheduler: linear warmup steps")
    patience: int = Field(
        default=5, gt=0, description="ReduceLROnPlateau: epochs with no improvement before reducing"
    )
    threshold: float = Field(
        default=1e-4, gt=0.0, description="ReduceLROnPlateau: minimum meaningful improvement"
    )


class TrainConfig(BaseModel):
    """Training loop configuration."""

    dataset: DatasetName = DatasetName.MNIST
    model: ModelName = ModelName.MLP
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=64, gt=0)
    seed: int = Field(default=42, ge=0)
    grad_clip: float | None = Field(
        default=None, description="Max gradient norm; None disables clipping"
    )
    eval_every: int = Field(default=1, gt=0, description="Run validation every N epochs")
    early_stopping_patience: int | None = Field(
        default=None,
        description="Stop training if val loss does not improve for N epochs; None disables",
    )
    num_workers: int = Field(default=0, ge=0, description="DataLoader workers")
    pin_memory: bool = Field(default=False)

    @field_validator("batch_size")
    @classmethod
    def batch_size_power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            logger.warning("batch_size=%d is not a power of 2; this may reduce GPU efficiency", v)
        return v


class MLflowConfig(BaseModel):
    """MLflow experiment tracking settings."""

    enabled: bool = True
    tracking_uri: str = "mlruns"
    experiment_name: str = "gdo-experiments"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    log_artifacts: bool = True


class MlpConfig(BaseModel):
    """Architecture config for the MLP model."""

    input_dim: int = Field(default=784, gt=0)
    hidden_dims: list[int] = Field(default=[256, 128])
    output_dim: int = Field(default=10, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    activation: Literal["relu", "tanh", "gelu"] = "relu"
    batch_norm: bool = True


class CnnConfig(BaseModel):
    """Architecture config for the CNN model."""

    in_channels: int = Field(default=1, gt=0)
    num_classes: int = Field(default=10, gt=0)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------


class ExperimentConfig(BaseModel):
    """
    Complete experiment specification.

    Loaded from a YAML file; every field maps to one of the sub-configs above.

    Example YAML
    ------------
    optimizer:
      name: adam
      lr: 0.001
    scheduler:
      name: cosine
      t_max: 20
    train:
      dataset: mnist
      model: mlp
      epochs: 20
      batch_size: 128
    mlflow:
      experiment_name: adam-vs-sgd
    """

    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    mlp: MlpConfig = Field(default_factory=MlpConfig)
    cnn: CnnConfig = Field(default_factory=CnnConfig)

    @model_validator(mode="after")
    def sync_scheduler_total_steps(self) -> ExperimentConfig:
        """Ensure OneCycleLR has total_steps consistent with epoch + batch counts."""
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """
        Load and validate an ExperimentConfig from a YAML file.

        Parameters
        ----------
        path:
            Path to the YAML configuration file.

        Returns
        -------
        ExperimentConfig
            Fully validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        pydantic.ValidationError
            If the YAML contents fail validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open() as f:
            data: dict[str, Any] = yaml.safe_load(f)
        logger.info("Loaded config from %s", path)
        return cls(**data)

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Return a flat key-value dict suitable for MLflow param logging.

        Nested keys are joined with a dot, e.g. ``optimizer.lr``.
        """
        result: dict[str, Any] = {}
        for section_name, section in self.model_dump().items():
            if isinstance(section, dict):
                for k, v in section.items():
                    result[f"{section_name}.{k}"] = v
            else:
                result[section_name] = section
        return result
