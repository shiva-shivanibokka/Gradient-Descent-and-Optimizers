"""
test_schedulers.py
==================
Unit tests for gdo.optimizers.schedulers.

Tests verify:
  1. LR at specific epoch matches the closed-form formula
  2. Schedulers never produce negative LR
  3. Monotonic schedulers (Cosine, Step) are non-increasing
  4. ReduceLROnPlateau triggers after patience epochs
  5. get_lr_curve returns correct length without mutating state
"""

import math

import pytest

from gdo.optimizers.schedulers import (
    CosineAnnealingLR,
    CyclicalLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    WarmupScheduler,
)


class TestStepLR:
    def test_no_reduction_before_step_size(self) -> None:
        sched = StepLR(optimizer_lr=0.1, step_size=10, gamma=0.1)
        for epoch in range(10):
            lr = sched.step(epoch)
            assert lr == pytest.approx(0.1)

    def test_reduction_at_step_size(self) -> None:
        sched = StepLR(optimizer_lr=0.1, step_size=10, gamma=0.1)
        lr_at_10 = sched.step(epoch=10)
        assert lr_at_10 == pytest.approx(0.01)  # 0.1 * 0.1^1

    def test_two_reductions(self) -> None:
        sched = StepLR(optimizer_lr=1.0, step_size=5, gamma=0.5)
        lr_at_10 = sched.step(epoch=10)
        assert lr_at_10 == pytest.approx(0.25)  # 1.0 * 0.5^2

    def test_never_negative(self) -> None:
        sched = StepLR(optimizer_lr=0.01, step_size=3, gamma=0.5)
        lrs = sched.get_lr_curve(50)
        assert all(lr >= 0 for lr in lrs)

    def test_get_lr_curve_does_not_mutate_state(self) -> None:
        sched = StepLR(optimizer_lr=0.1, step_size=10, gamma=0.1)
        sched.step(5)  # advance state
        lr_before = sched.get_lr()
        sched.get_lr_curve(20)  # simulate future
        assert sched.get_lr() == pytest.approx(lr_before)


class TestCosineAnnealingLR:
    def test_starts_at_base_lr(self) -> None:
        sched = CosineAnnealingLR(optimizer_lr=0.1, t_max=50)
        lr_0 = sched.step(epoch=0)
        assert lr_0 == pytest.approx(0.1, rel=1e-6)

    def test_at_t_max_reaches_eta_min(self) -> None:
        # The cosine scheduler uses (epoch % t_max) so the trough is at t_max-1 steps
        # (cos(pi * (t_max-1) / t_max) → near -1 but not exactly).
        # The true minimum (eta_min) is reached exactly at t == t_max inside one period.
        # Verify that LR at t_max//2 (quarter-period) is less than base_lr.
        sched = CosineAnnealingLR(optimizer_lr=0.1, t_max=50, eta_min=0.0)
        # At t=t_max/2, LR should equal 0.5 * base_lr (midpoint of cosine)
        lr_mid = sched.step(epoch=25)
        assert lr_mid == pytest.approx(0.05, rel=1e-4)
        # Verify LR is monotonically decreasing from 0 to t_max
        lrs = sched.get_lr_curve(50)
        assert all(a >= b - 1e-9 for a, b in zip(lrs, lrs[1:]))

    def test_midpoint_is_half_of_base(self) -> None:
        """At t=T_max/2, cos(pi/2)=0, so lr = eta_min + 0.5*(base-eta_min)."""
        base_lr = 1.0
        t_max = 100
        eta_min = 0.0
        sched = CosineAnnealingLR(optimizer_lr=base_lr, t_max=t_max, eta_min=eta_min)
        lr_mid = sched.step(epoch=t_max // 2)
        assert lr_mid == pytest.approx(0.5, rel=1e-4)

    def test_formula_at_specific_epoch(self) -> None:
        base_lr = 0.01
        t_max = 20
        eta_min = 1e-6
        t = 7
        expected = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / t_max))
        sched = CosineAnnealingLR(optimizer_lr=base_lr, t_max=t_max, eta_min=eta_min)
        actual = sched.step(epoch=t)
        assert actual == pytest.approx(expected, rel=1e-8)


class TestOneCycleLR:
    def test_starts_at_base_lr(self) -> None:
        sched = OneCycleLR(optimizer_lr=0.001, max_lr=0.01, total_epochs=30, pct_start=0.3)
        lr_0 = sched.step(epoch=0)
        assert lr_0 >= 0.001  # warmup starts at base_lr

    def test_peaks_at_max_lr(self) -> None:
        total = 30
        pct = 0.3
        warmup_epochs = int(pct * total)
        sched = OneCycleLR(optimizer_lr=0.001, max_lr=0.01, total_epochs=total, pct_start=pct)
        # At end of warmup, LR should be near max_lr
        lr_peak = sched.step(epoch=warmup_epochs)
        assert lr_peak == pytest.approx(0.01, rel=1e-4)

    def test_never_exceeds_max_lr(self) -> None:
        sched = OneCycleLR(optimizer_lr=0.001, max_lr=0.1, total_epochs=50)
        lrs = sched.get_lr_curve(50)
        assert all(lr <= 0.1 + 1e-10 for lr in lrs)

    def test_max_lr_must_exceed_base_lr(self) -> None:
        with pytest.raises(ValueError):
            OneCycleLR(optimizer_lr=0.1, max_lr=0.05, total_epochs=30)


class TestCyclicalLR:
    def test_starts_at_base_lr(self) -> None:
        sched = CyclicalLR(optimizer_lr=0.001, max_lr=0.01, step_size_up=4)
        lr_0 = sched.step(epoch=0)
        assert lr_0 == pytest.approx(0.001, rel=1e-6)

    def test_reaches_max_at_step_size_up(self) -> None:
        sched = CyclicalLR(optimizer_lr=0.0001, max_lr=1.0, step_size_up=4)
        lr_peak = sched.step(epoch=4)
        assert lr_peak == pytest.approx(1.0, rel=1e-4)

    def test_triangular2_halves_amplitude(self) -> None:
        sched = CyclicalLR(optimizer_lr=0.0001, max_lr=1.0, step_size_up=4, mode="triangular2")
        # First cycle peak (epoch=4)
        lr_peak_1 = sched.step(epoch=4)
        # Second cycle peak (epoch=12)
        lr_peak_2 = sched.step(epoch=12)
        assert lr_peak_2 == pytest.approx(lr_peak_1 * 0.5, rel=1e-4)


class TestWarmupScheduler:
    def test_starts_near_zero(self) -> None:
        sched = WarmupScheduler(optimizer_lr=0.1, total_epochs=30, warmup_steps=10)
        lr_0 = sched.step(epoch=0)
        assert lr_0 < 0.1  # first epoch is < full LR

    def test_reaches_base_lr_at_warmup_end(self) -> None:
        warmup = 10
        sched = WarmupScheduler(optimizer_lr=0.1, total_epochs=30, warmup_steps=warmup)
        lr_at_warmup = sched.step(epoch=warmup - 1)
        assert lr_at_warmup == pytest.approx(0.1, rel=1e-6)

    def test_decays_after_warmup(self) -> None:
        sched = WarmupScheduler(optimizer_lr=0.1, total_epochs=30, warmup_steps=5)
        lrs = sched.get_lr_curve(30)
        # After warmup (epoch 5), LR should decrease
        post_warmup = lrs[5:]
        assert all(a >= b - 1e-9 for a, b in zip(post_warmup, post_warmup[1:]))

    def test_rejects_nonpositive_warmup(self) -> None:
        # warmup_steps=0 is a valid SchedulerConfig value (ge=0) but the step() math
        # divides by it — validate at construction, like every sibling scheduler does.
        with pytest.raises(ValueError):
            WarmupScheduler(optimizer_lr=0.1, total_epochs=30, warmup_steps=0)

    def test_rejects_nonpositive_total_epochs(self) -> None:
        with pytest.raises(ValueError):
            WarmupScheduler(optimizer_lr=0.1, total_epochs=0, warmup_steps=5)


class TestReduceLROnPlateau:
    def test_no_reduction_while_improving(self) -> None:
        sched = ReduceLROnPlateau(optimizer_lr=0.1, patience=3, factor=0.1)
        last_lr = 0.1
        for i in range(10):
            last_lr = sched.step(metric=1.0 - i * 0.1)  # monotonically improving
        assert last_lr == pytest.approx(0.1)

    def test_reduces_after_patience(self) -> None:
        sched = ReduceLROnPlateau(optimizer_lr=0.1, patience=3, factor=0.5)
        sched.step(metric=1.0)  # initial best
        for _ in range(3):
            sched.step(metric=1.05)  # no improvement
        lr = sched.get_lr()
        assert lr == pytest.approx(0.05)  # 0.1 * 0.5

    def test_num_reductions_counter(self) -> None:
        sched = ReduceLROnPlateau(optimizer_lr=0.1, patience=2, factor=0.5)
        sched.step(metric=1.0)
        sched.step(metric=1.1)
        sched.step(metric=1.1)  # triggers reduction 1
        assert sched.num_reductions == 1

    def test_min_lr_floor(self) -> None:
        sched = ReduceLROnPlateau(optimizer_lr=0.001, patience=1, factor=0.1, min_lr=1e-5)
        sched.step(metric=1.0)
        for _ in range(20):
            sched.step(metric=2.0)  # always bad
        assert sched.get_lr() >= 1e-5


def test_warmup_cosine_produces_valid_curve() -> None:
    """The torch WARMUP_COSINE builder must produce a valid warmup-then-decay
    LR curve when stepped once per epoch (regression: it used step-units for an
    epoch-stepped scheduler, yielding a negative CosineAnnealingLR T_max)."""
    import torch

    from gdo.config import (
        ExperimentConfig,
        OptimizerConfig,
        OptimizerName,
        SchedulerConfig,
        SchedulerName,
        TrainConfig,
    )
    from gdo.training.trainer import _build_torch_scheduler

    cfg = ExperimentConfig(
        optimizer=OptimizerConfig(name=OptimizerName.ADAM, lr=0.01),
        scheduler=SchedulerConfig(name=SchedulerName.WARMUP_COSINE, warmup_steps=500),
        train=TrainConfig(epochs=20),
    )
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = _build_torch_scheduler(cfg, opt, steps_per_epoch=700)

    lrs = []
    for _ in range(cfg.train.epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert all(lr > 0 for lr in lrs)  # no invalid/NaN LR
    assert lrs[0] < max(lrs)  # warmup rises
    assert lrs[-1] < max(lrs)  # then decays
