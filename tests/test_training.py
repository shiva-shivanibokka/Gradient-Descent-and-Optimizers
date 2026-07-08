"""
test_training.py
================
Coverage for the training-side modules that the parity/optimizer suites don't
touch: models (MLP/CNN), metrics, the Trainer's optimizer/scheduler factories,
the MLflow ExperimentLogger, and the matplotlib LandscapePlotter.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

from gdo.config import (
    CnnConfig,
    ExperimentConfig,
    MLflowConfig,
    MlpConfig,
    OptimizerConfig,
    OptimizerName,
    SchedulerConfig,
    SchedulerName,
    TrainConfig,
)
from gdo.experiment.logger import ExperimentLogger
from gdo.landscapes.plotter import LandscapePlotter, _get_color
from gdo.landscapes.surfaces import QuadraticSurface
from gdo.optimizers.sgd import BatchGD, MomentumSGD
from gdo.training.metrics import ConvergenceTracker, EpochMetrics, GradientNormMonitor
from gdo.training.models import CNN, MLP
from gdo.training.trainer import (
    TrainingResult,
    _build_torch_optimizer,
    _build_torch_scheduler,
)

matplotlib.use("Agg")  # headless backend for CI (figures are only created in tests)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModels:
    def test_mlp_forward_shape(self) -> None:
        model = MLP(MlpConfig())
        out = model(torch.randn(4, 784))
        assert out.shape == (4, 10)

    def test_mlp_flattens_image_input(self) -> None:
        model = MLP(MlpConfig())
        out = model(torch.randn(4, 1, 28, 28))  # (B, C, H, W) → flattened
        assert out.shape == (4, 10)

    def test_mlp_tanh_uses_xavier_init(self) -> None:
        # Exercises the non-ReLU init branch.
        model = MLP(MlpConfig(activation="tanh"))
        assert model(torch.randn(4, 784)).shape == (4, 10)

    def test_mlp_grad_norms(self) -> None:
        model = MLP(MlpConfig())
        loss = model(torch.randn(4, 784)).sum()
        loss.backward()
        norms = model.get_grad_norms()
        assert norms and all(v >= 0 for v in norms.values())
        assert model.get_total_grad_norm() > 0

    def test_cnn_forward_and_grad_norms(self) -> None:
        model = CNN(CnnConfig(in_channels=3))
        out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)
        out.sum().backward()
        assert model.get_grad_norms()
        assert model.get_total_grad_norm() > 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _epoch(e: int, val_loss: float) -> EpochMetrics:
    return EpochMetrics(
        epoch=e,
        train_loss=val_loss,
        val_loss=val_loss,
        train_acc=0.5,
        val_acc=0.5,
        lr=0.01,
        grad_norm=1.0,
        epoch_time_s=0.1,
    )


class TestMetrics:
    def test_epoch_metrics_to_dict_and_str(self) -> None:
        m = _epoch(3, 0.25)
        assert set(m.to_dict()) == {
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
            "lr",
            "grad_norm",
            "epoch_time_s",
        }
        assert "Epoch 003" in str(m)

    def test_convergence_tracker_early_stops(self) -> None:
        tracker = ConvergenceTracker(patience=2)
        # improve, then plateau
        assert tracker.update(_epoch(0, 1.0)) is False
        assert tracker.update(_epoch(1, 0.5)) is False  # improved
        assert tracker.update(_epoch(2, 0.5)) is False  # no improve, wait=1
        assert tracker.update(_epoch(3, 0.5)) is True  # wait=2 >= patience
        assert tracker.best_value == 0.5
        assert tracker.best_epoch == 1
        assert tracker.epochs_without_improvement == 2
        assert len(tracker.history) == 4
        assert tracker.val_loss_curve().tolist() == [1.0, 0.5, 0.5, 0.5]
        assert tracker.train_loss_curve().shape == (4,)
        assert tracker.val_acc_curve().shape == (4,)

    def test_convergence_tracker_disabled_patience(self) -> None:
        tracker = ConvergenceTracker(patience=None)
        for e in range(10):
            assert tracker.update(_epoch(e, 1.0)) is False

    def test_convergence_step(self) -> None:
        tracker = ConvergenceTracker(patience=5)
        for e, v in enumerate([1.0, 0.6, 0.2]):
            tracker.update(_epoch(e, v))
        assert tracker.convergence_step(threshold=0.5) == 2
        assert tracker.convergence_step(threshold=0.01) is None

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            ConvergenceTracker(mode="sideways")

    def test_gradient_norm_monitor(self) -> None:
        mon = GradientNormMonitor(explosion_threshold=5.0, window=3)
        mon.record({"layer": 1.0}, 1.0)
        mon.record({"layer": 12.0}, 12.0)  # explosion
        s = mon.summary()
        assert s["explosion_count"] == 1.0
        assert s["grad_norm_max"] == 12.0
        assert mon.rolling_mean() > 0
        assert mon.global_norm_history().tolist() == [1.0, 12.0]
        assert mon.layer_norm_history("layer") is not None
        assert mon.layer_norm_history("missing") is None
        mon.reset()
        assert mon.summary() == {}
        assert mon.rolling_mean() == 0.0


# ---------------------------------------------------------------------------
# Trainer factories
# ---------------------------------------------------------------------------


def _cfg(opt: OptimizerName, sched: SchedulerName = SchedulerName.NONE) -> ExperimentConfig:
    return ExperimentConfig(
        optimizer=OptimizerConfig(name=opt, lr=0.01),
        scheduler=SchedulerConfig(name=sched),
        train=TrainConfig(epochs=10),
    )


class TestTrainerFactories:
    @pytest.mark.parametrize("name", list(OptimizerName))
    def test_build_optimizer(self, name: OptimizerName) -> None:
        model = torch.nn.Linear(4, 2)
        opt = _build_torch_optimizer(_cfg(name), model)
        assert isinstance(opt, torch.optim.Optimizer)

    @pytest.mark.parametrize("name", list(SchedulerName))
    def test_build_scheduler(self, name: SchedulerName) -> None:
        model = torch.nn.Linear(4, 2)
        cfg = _cfg(OptimizerName.ADAM, name)
        opt = _build_torch_optimizer(cfg, model)
        sched = _build_torch_scheduler(cfg, opt, steps_per_epoch=100)
        if name == SchedulerName.NONE:
            assert sched is None
        else:
            assert sched is not None

    def test_onecycle_schedule_runs(self) -> None:
        # OneCycle must rise toward max_lr then anneal when stepped ONCE PER EPOCH,
        # the way Trainer.fit() drives it. Guards the per-batch/per-epoch mismatch:
        # if built with steps_per_epoch*epochs total_steps, per-epoch stepping only
        # advances a fraction of the cycle and the LR never leaves its low start.
        epochs = 10
        cfg = ExperimentConfig(
            optimizer=OptimizerConfig(name=OptimizerName.ADAM, lr=0.001),
            scheduler=SchedulerConfig(name=SchedulerName.ONECYCLE, max_lr=0.1, pct_start=0.3),
            train=TrainConfig(epochs=epochs),
        )
        model = torch.nn.Linear(4, 2)
        opt = _build_torch_optimizer(cfg, model)
        sched = _build_torch_scheduler(cfg, opt, steps_per_epoch=100)
        assert sched is not None
        lrs = []
        for _ in range(epochs):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        # Peak approaches max_lr (proves warmup happened); final falls back below the
        # peak (proves annealing happened).
        assert max(lrs) > 0.05, f"OneCycle never warmed up: peak LR was {max(lrs):.4g}"
        assert lrs[-1] < max(lrs), "OneCycle never annealed back down"


# ---------------------------------------------------------------------------
# ExperimentLogger
# ---------------------------------------------------------------------------


def _result() -> TrainingResult:
    r = TrainingResult(optimizer_name="Adam", scheduler_name="None", config=TrainConfig())
    r.history = [_epoch(0, 1.0), _epoch(1, 0.5)]
    r.best_val_loss = 0.5
    r.best_val_acc = 0.5
    r.best_epoch = 1
    r.grad_norm_summary = {"mean": 1.0}
    return r


class TestExperimentLogger:
    def test_disabled_logger_is_noop(self) -> None:
        log = ExperimentLogger(MLflowConfig(enabled=False))
        log.start()
        log.log_params({"a": 1})
        log.log_epoch(0, {"loss": 1.0})
        log.log_metric("x", 1.0)
        log.log_run(_result())
        log.end()
        assert log.run_id is None

    def test_full_mlflow_cycle(self, tmp_path) -> None:
        # file:// URI so the path is valid on Windows and POSIX alike.
        uri = (tmp_path / "mlruns").as_uri()
        cfg = MLflowConfig(enabled=True, tracking_uri=uri, experiment_name="unit-test")
        with ExperimentLogger(cfg) as log:
            log.log_params({"lr": 0.01})
            log.log_epoch(0, {"loss": 1.0})
            log.log_run(_result(), None)
            run_id = log.run_id
        assert run_id is not None
        metrics = ExperimentLogger.load_run_metrics(run_id, uri)
        assert isinstance(metrics, dict) and metrics


# ---------------------------------------------------------------------------
# LandscapePlotter
# ---------------------------------------------------------------------------


@pytest.fixture
def optimizers_with_trajectory() -> list[BatchGD]:
    surface = QuadraticSurface()
    opts: list[BatchGD] = []
    for factory in (lambda: BatchGD(lr=0.05), lambda: MomentumSGD(lr=0.05, noise_scale=0.0)):
        opt = factory()
        surface.run_optimizer(opt, start=(-2.5, 2.5), n_steps=30)
        opts.append(opt)
    return opts


class TestPlotter:
    def test_get_color(self) -> None:
        assert _get_color("Adam") == "#8c564b"
        assert _get_color("unknown-name", idx=1)  # falls back to default cycle

    def test_contour(self, optimizers_with_trajectory) -> None:
        fig = LandscapePlotter.contour(
            QuadraticSurface(), optimizers_with_trajectory, resolution=60
        )
        assert fig is not None
        plt.close(fig)

    def test_trajectory_comparison(self, optimizers_with_trajectory) -> None:
        fig = LandscapePlotter.trajectory_comparison(
            QuadraticSurface(), optimizers_with_trajectory, resolution=60
        )
        assert fig is not None
        plt.close(fig)

    def test_convergence_curves(self, optimizers_with_trajectory) -> None:
        fig = LandscapePlotter.convergence_curves(optimizers_with_trajectory)
        assert fig is not None
        plt.close(fig)

    def test_lr_schedule_plot(self) -> None:
        import numpy as np

        fig = LandscapePlotter.lr_schedule_plot({"cosine": np.linspace(0.1, 0.0, 30)})
        assert fig is not None
        plt.close(fig)

    def test_gradient_norm_plot_smoothed_and_raw(self) -> None:
        fig = LandscapePlotter.gradient_norm_plot(
            {"long": [1.0] * 30, "short": [1.0] * 5}  # >20 → smoothing, <20 → raw
        )
        assert fig is not None
        plt.close(fig)
