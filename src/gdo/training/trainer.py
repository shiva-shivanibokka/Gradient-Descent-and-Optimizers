"""
trainer.py
==========
Generic Trainer class that runs the full PyTorch train/eval loop.

The Trainer is the central object in every experiment:
  - Accepts any ``torch.optim.Optimizer`` and ``torch.optim.lr_scheduler``
  - Logs every epoch to MLflow via ``ExperimentLogger``
  - Tracks convergence via ``ConvergenceTracker``
  - Monitors gradient norms via ``GradientNormMonitor``
  - Supports early stopping and gradient clipping
  - Returns a ``TrainingResult`` with all metrics for downstream analysis

Usage
-----
    from gdo.training import Trainer
    from gdo.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml("configs/adam_mnist.yaml")
    result = Trainer.from_config(cfg).fit()
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gdo.config import ExperimentConfig, TrainConfig
from gdo.training.metrics import ConvergenceTracker, EpochMetrics, GradientNormMonitor

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """
    Complete output of a training run.

    Returned by ``Trainer.fit()`` and passed to ``ExperimentLogger``
    for MLflow artifact logging.
    """

    optimizer_name: str
    scheduler_name: str
    config: TrainConfig
    history: list[EpochMetrics] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    best_epoch: int = 0
    total_train_time_s: float = 0.0
    convergence_epoch: int | None = None
    grad_norm_summary: dict[str, float] = field(default_factory=dict)

    @property
    def train_losses(self) -> list[float]:
        return [m.train_loss for m in self.history]

    @property
    def val_losses(self) -> list[float]:
        return [m.val_loss for m in self.history]

    @property
    def val_accs(self) -> list[float]:
        return [m.val_acc for m in self.history]

    @property
    def lr_history(self) -> list[float]:
        return [m.lr for m in self.history]


class Trainer:
    """
    Generic PyTorch Trainer with MLflow integration.

    Parameters
    ----------
    model:
        Any ``nn.Module`` (MLP or CNN from ``gdo.training.models``).
    optimizer:
        Any ``torch.optim.Optimizer``.
    train_loader:
        Training DataLoader.
    val_loader:
        Validation DataLoader.
    config:
        ``TrainConfig`` from ``gdo.config``.
    scheduler:
        Optional PyTorch LR scheduler.
    experiment_logger:
        Optional ``ExperimentLogger`` for MLflow logging.
    device:
        Torch device. Auto-detected if None.
    on_epoch_end:
        Optional callback ``(epoch, metrics) → None`` — used by the
        Gradio app to stream live metrics to the UI.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        config: TrainConfig,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        experiment_logger: object | None = None,
        device: torch.device | None = None,
        on_epoch_end: Callable[[int, EpochMetrics], None] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.scheduler = scheduler
        self._logger = experiment_logger
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.on_epoch_end = on_epoch_end

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self._convergence = ConvergenceTracker(
            patience=config.early_stopping_patience,
            mode="min",
        )
        self._grad_monitor = GradientNormMonitor()

        logger.info(
            "[Trainer] Initialized | device=%s | optimizer=%s | epochs=%d",
            self.device,
            type(optimizer).__name__,
            config.epochs,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> TrainingResult:
        """
        Run the full training loop.

        Returns
        -------
        TrainingResult
            Complete training history and summary metrics.
        """
        optimizer_name = type(self.optimizer).__name__
        scheduler_name = type(self.scheduler).__name__ if self.scheduler else "None"

        result = TrainingResult(
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            config=self.config,
        )

        t_start = time.perf_counter()

        for epoch in range(self.config.epochs):
            epoch_start = time.perf_counter()

            train_loss, train_acc = self._train_epoch()

            if epoch % self.config.eval_every == 0:
                val_loss, val_acc = self._eval_epoch()
            else:
                # Carry forward last val metrics
                if result.history:
                    val_loss = result.history[-1].val_loss
                    val_acc = result.history[-1].val_acc
                else:
                    val_loss, val_acc = float("nan"), float("nan")

            # Get current LR
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Get gradient norm (recorded during _train_epoch)
            grad_norm = self._grad_monitor.rolling_mean()

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=current_lr,
                grad_norm=grad_norm,
                epoch_time_s=time.perf_counter() - epoch_start,
            )

            result.history.append(metrics)
            logger.info(str(metrics))

            # Update best
            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                result.best_val_acc = val_acc
                result.best_epoch = epoch

            # Log to MLflow
            if self._logger is not None:
                self._logger.log_epoch(epoch, metrics.to_dict())  # type: ignore[union-attr]

            # Scheduler step
            if self.scheduler is not None:
                if hasattr(self.scheduler, "step"):
                    if isinstance(
                        self.scheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                    ):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()

            # Callback for Gradio live streaming
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, metrics)

            # Early stopping
            if self._convergence.update(metrics):
                logger.info("[Trainer] Early stopping at epoch %d.", epoch)
                break

        result.total_train_time_s = time.perf_counter() - t_start
        result.grad_norm_summary = self._grad_monitor.summary()
        result.convergence_epoch = self._convergence.convergence_step(threshold=0.5)

        logger.info(
            "[Trainer] Training complete | best_val_loss=%.4f at epoch %d | total_time=%.1fs",
            result.best_val_loss,
            result.best_epoch,
            result.total_train_time_s,
        )
        return result

    def fit_streaming(self) -> Generator[EpochMetrics, None, TrainingResult]:
        """
        Generator-based fit that yields EpochMetrics after each epoch.

        Used by the Gradio app to stream live updates without blocking.

        Yields
        ------
        EpochMetrics
            Metrics after each epoch, immediately as they complete.

        Returns
        -------
        TrainingResult
            Full result after training is complete (StopIteration value).
        """
        optimizer_name = type(self.optimizer).__name__
        scheduler_name = type(self.scheduler).__name__ if self.scheduler else "None"

        result = TrainingResult(
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            config=self.config,
        )

        t_start = time.perf_counter()

        for epoch in range(self.config.epochs):
            epoch_start = time.perf_counter()
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._eval_epoch()
            current_lr = self.optimizer.param_groups[0]["lr"]
            grad_norm = self._grad_monitor.rolling_mean()

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=current_lr,
                grad_norm=grad_norm,
                epoch_time_s=time.perf_counter() - epoch_start,
            )
            result.history.append(metrics)

            if val_loss < result.best_val_loss:
                result.best_val_loss = val_loss
                result.best_val_acc = val_acc
                result.best_epoch = epoch

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            yield metrics

            if self._convergence.update(metrics):
                break

        result.total_train_time_s = time.perf_counter() - t_start
        result.grad_norm_summary = self._grad_monitor.summary()
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self) -> tuple[float, float]:
        """Run one training epoch. Returns (mean_loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Record gradient norms AFTER clipping
            if hasattr(self.model, "get_grad_norms"):
                self._grad_monitor.record(
                    self.model.get_grad_norms(),  # type: ignore[union-attr]
                    self.model.get_total_grad_norm(),  # type: ignore[union-attr]
                )

            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

        return total_loss / total, correct / total

    def _eval_epoch(self) -> tuple[float, float]:
        """Run one validation epoch. Returns (mean_loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                total_loss += loss.item() * inputs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += inputs.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: ExperimentConfig,
        on_epoch_end: Callable[[int, EpochMetrics], None] | None = None,
    ) -> Trainer:
        """
        Build a fully configured Trainer from an ExperimentConfig.

        This method:
        1. Sets the random seed for reproducibility
        2. Loads the dataset and creates DataLoaders
        3. Instantiates the correct model
        4. Builds the PyTorch optimizer and scheduler
        5. Creates the ExperimentLogger

        Parameters
        ----------
        cfg:
            Fully validated ExperimentConfig from YAML.
        on_epoch_end:
            Optional callback for live metric streaming (Gradio app).
        """
        import random

        import numpy as np

        from gdo.config import ModelName
        from gdo.experiment.logger import ExperimentLogger
        from gdo.training.models import CNN, MLP

        # Seed everything
        torch.manual_seed(cfg.train.seed)
        np.random.seed(cfg.train.seed)
        random.seed(cfg.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.train.seed)

        # Build dataset
        train_loader, val_loader = _build_dataloaders(cfg)

        # Build model
        if cfg.train.model == ModelName.MLP:
            model: nn.Module = MLP(cfg.mlp)
        else:
            model = CNN(cfg.cnn)

        # Build optimizer
        opt = _build_torch_optimizer(cfg, model)

        # Build scheduler
        sched = _build_torch_scheduler(cfg, opt, len(train_loader))

        # Build experiment logger
        exp_logger = ExperimentLogger(cfg.mlflow) if cfg.mlflow.enabled else None

        return cls(
            model=model,
            optimizer=opt,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg.train,
            scheduler=sched,
            experiment_logger=exp_logger,
            on_epoch_end=on_epoch_end,
        )


# ---------------------------------------------------------------------------
# Private factory helpers
# ---------------------------------------------------------------------------


def _build_dataloaders(cfg: ExperimentConfig) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """Build train and val DataLoaders from config."""
    from torch.utils.data import random_split
    from torchvision import datasets, transforms

    from gdo.config import DatasetName

    if cfg.train.dataset == DatasetName.MNIST:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        full = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        n_val = int(0.15 * len(full))
        train_ds, val_ds = random_split(
            full,
            [len(full) - n_val, n_val],
            generator=torch.Generator().manual_seed(cfg.train.seed),
        )

    elif cfg.train.dataset == DatasetName.CIFAR10:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_ds = datasets.CIFAR10(
            root="data", train=True, download=True, transform=train_transform
        )
        val_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=val_transform)

    else:
        raise ValueError(f"Unsupported dataset: {cfg.train.dataset}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size * 2,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    return train_loader, val_loader


def _build_torch_optimizer(cfg: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Instantiate the correct torch.optim optimizer from config."""
    from gdo.config import OptimizerName

    oc = cfg.optimizer
    params = model.parameters()

    name = oc.name
    if name == OptimizerName.BATCH_GD:
        return torch.optim.SGD(params, lr=oc.lr)
    elif name == OptimizerName.SGD:
        return torch.optim.SGD(params, lr=oc.lr)
    elif name == OptimizerName.MINI_BATCH_GD:
        return torch.optim.SGD(params, lr=oc.lr)
    elif name == OptimizerName.MOMENTUM_SGD:
        return torch.optim.SGD(params, lr=oc.lr, momentum=oc.momentum)
    elif name == OptimizerName.RMSPROP:
        return torch.optim.RMSprop(params, lr=oc.lr, alpha=oc.alpha, eps=oc.epsilon)
    elif name == OptimizerName.ADAM:
        return torch.optim.Adam(params, lr=oc.lr, betas=(oc.beta1, oc.beta2), eps=oc.epsilon)
    elif name == OptimizerName.ADAMW:
        return torch.optim.AdamW(
            params,
            lr=oc.lr,
            betas=(oc.beta1, oc.beta2),
            eps=oc.epsilon,
            weight_decay=oc.weight_decay,
        )
    elif name == OptimizerName.LION:
        try:
            from lion_pytorch import Lion  # type: ignore[import-not-found]

            return Lion(params, lr=oc.lr, betas=(oc.beta1, oc.beta), weight_decay=oc.weight_decay)
        except ImportError:
            logger.warning(
                "lion_pytorch not installed — falling back to AdamW for PyTorch training."
            )
            return torch.optim.AdamW(params, lr=oc.lr, weight_decay=oc.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _build_torch_scheduler(
    cfg: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Instantiate the correct torch LR scheduler from config."""
    from gdo.config import SchedulerName

    sc = cfg.scheduler
    tc = cfg.train

    if sc.name == SchedulerName.NONE:
        return None
    elif sc.name == SchedulerName.STEP:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc.step_size, gamma=sc.gamma)
    elif sc.name == SchedulerName.COSINE:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sc.t_max, eta_min=sc.eta_min
        )
    elif sc.name == SchedulerName.ONECYCLE:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=sc.max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=tc.epochs,
            pct_start=sc.pct_start,
        )
    elif sc.name == SchedulerName.CYCLICAL:
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=sc.base_lr,
            max_lr=sc.max_lr,
            step_size_up=sc.step_size_up,
            mode="triangular2",
        )
    elif sc.name == SchedulerName.WARMUP_COSINE:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # scheduler.step() is called once per EPOCH by the Trainer, so warmup must
        # be measured in epochs. Derive it from warmup_steps, clamped so both the
        # warmup and cosine phases are always >= 1 epoch (avoids negative T_max).
        warmup_epochs = max(1, min(sc.warmup_steps // max(steps_per_epoch, 1), tc.epochs - 1))
        cosine_epochs = max(1, tc.epochs - warmup_epochs)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=sc.eta_min)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    elif sc.name == SchedulerName.REDUCE_ON_PLATEAU:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=sc.patience, factor=sc.gamma, threshold=sc.threshold
        )
    else:
        raise ValueError(f"Unknown scheduler: {sc.name}")
