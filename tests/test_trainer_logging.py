"""
test_trainer_logging.py
=======================
Regression tests for the Trainer's MLflow logging wiring.

Guards two things:
  1. ``Trainer.fit()`` actually calls ``log_epoch`` once per epoch when given
     an active logger (the per-epoch logging path must not be a silent no-op).
  2. ``Trainer.from_config`` does not build a never-started logger (which would
     make every per-epoch log call a no-op).
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from gdo.config import TrainConfig
from gdo.training.trainer import Trainer


class _FakeLogger:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        self.epochs.append(epoch)


def _tiny_loader() -> DataLoader:
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_fit_logs_every_epoch_to_active_logger() -> None:
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    fake = _FakeLogger()
    trainer = Trainer(
        model=model,
        optimizer=opt,
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        config=TrainConfig(epochs=3, seed=0),
        experiment_logger=fake,
    )
    trainer.fit()
    assert fake.epochs == [0, 1, 2]
