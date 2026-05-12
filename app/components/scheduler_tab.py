"""
scheduler_tab.py
================
Gradio component for the Scheduler Explorer tab.

Shows the LR schedule curve for any combination of base optimizer +
scheduler, and demonstrates the effect on training loss on the synthetic
2-spiral task.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from gdo.optimizers.schedulers import (
    CosineAnnealingLR,
    CyclicalLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    WarmupScheduler,
)

SCHEDULER_MAP: dict[str, type[LRScheduler]] = {
    "None (constant)": None,  # type: ignore[dict-item]
    "StepLR": StepLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "OneCycleLR": OneCycleLR,
    "CyclicalLR": CyclicalLR,
    "Warmup + Cosine": WarmupScheduler,
    "ReduceLROnPlateau": ReduceLROnPlateau,
}


def build_scheduler(name: str, base_lr: float, total_epochs: int) -> LRScheduler | None:
    """Instantiate the named scheduler with sensible defaults."""
    if name == "None (constant)" or SCHEDULER_MAP[name] is None:
        return None
    elif name == "StepLR":
        return StepLR(base_lr, step_size=max(1, total_epochs // 3), gamma=0.1)
    elif name == "CosineAnnealingLR":
        return CosineAnnealingLR(base_lr, t_max=total_epochs)
    elif name == "OneCycleLR":
        return OneCycleLR(base_lr, max_lr=base_lr * 10, total_epochs=total_epochs, pct_start=0.3)
    elif name == "CyclicalLR":
        return CyclicalLR(base_lr, max_lr=base_lr * 5, step_size_up=max(2, total_epochs // 6))
    elif name == "Warmup + Cosine":
        return WarmupScheduler(
            base_lr, total_epochs=total_epochs, warmup_steps=max(1, total_epochs // 10)
        )
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(base_lr, factor=0.5, patience=3)
    return None


def plot_scheduler_curve(
    scheduler_names: list[str],
    base_lr: float,
    total_epochs: int,
) -> object:
    """
    Plot the LR curves for all selected schedulers over total_epochs.

    Parameters
    ----------
    scheduler_names:
        List of scheduler names from SCHEDULER_MAP keys.
    base_lr:
        Starting learning rate.
    total_epochs:
        Total number of training epochs to simulate.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, name in enumerate(scheduler_names):
        sched = build_scheduler(name, base_lr, total_epochs)
        if sched is None:
            # Constant LR
            lrs = np.full(total_epochs, base_lr)
        else:
            lrs = sched.get_lr_curve(total_epochs)
        ax.plot(lrs, label=name, color=colors[i % len(colors)], linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(f"LR Schedules — {total_epochs} Epochs (base_lr={base_lr})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
