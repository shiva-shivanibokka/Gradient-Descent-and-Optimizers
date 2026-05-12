"""
landscape_tab.py
================
Gradio component for the Loss Landscape tab.

Renders a 2D contour plot of the selected surface and animates the
optimizer trajectory as the user changes optimizer or LR.

All computation uses gdo.optimizers (NumPy) and gdo.landscapes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is on the path when running from app/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from gdo.landscapes import Beale, Himmelblau, LandscapePlotter, QuadraticSurface, Rosenbrock
from gdo.optimizers import (
    Adam,
    AdamW,
    BatchGD,
    Lion,
    MiniBatchGD,
    MomentumSGD,
    RMSProp,
    StochasticGD,
)

SURFACE_MAP = {
    "Quadratic (ill-conditioned)": lambda: QuadraticSurface(a=1.0, b=10.0),
    "Rosenbrock (banana)": Rosenbrock,
    "Beale": Beale,
    "Himmelblau (4 minima)": Himmelblau,
}

OPTIMIZER_MAP = {
    "Batch GD": lambda lr: BatchGD(lr=lr),
    "SGD (stochastic)": lambda lr: StochasticGD(lr=lr, noise_scale=0.05, seed=42),
    "Mini-Batch GD": lambda lr: MiniBatchGD(lr=lr, batch_size=32, noise_scale=0.03, seed=42),
    "SGD + Momentum": lambda lr: MomentumSGD(lr=lr, momentum=0.9, noise_scale=0.03, seed=42),
    "RMSProp": lambda lr: RMSProp(lr=lr),
    "Adam": lambda lr: Adam(lr=lr),
    "AdamW": lambda lr: AdamW(lr=lr, weight_decay=0.01),
    "Lion": lambda lr: Lion(lr=lr),
}

DEFAULT_STARTS = {
    "Quadratic (ill-conditioned)": (-2.5, 2.5),
    "Rosenbrock (banana)": (-1.5, 1.0),
    "Beale": (1.0, 1.0),
    "Himmelblau (4 minima)": (0.0, 0.0),
}


def run_landscape(
    surface_name: str,
    optimizer_name: str,
    lr: float,
    n_steps: int,
) -> object:
    """
    Run the selected optimizer on the selected surface and return
    a matplotlib figure of the trajectory.

    Parameters
    ----------
    surface_name:
        One of the keys in SURFACE_MAP.
    optimizer_name:
        One of the keys in OPTIMIZER_MAP.
    lr:
        Learning rate.
    n_steps:
        Number of gradient steps.

    Returns
    -------
    matplotlib.figure.Figure
    """
    surface = SURFACE_MAP[surface_name]()
    optimizer = OPTIMIZER_MAP[optimizer_name](lr)
    start = DEFAULT_STARTS[surface_name]

    surface.run_optimizer(optimizer, start=start, n_steps=n_steps)

    fig = LandscapePlotter.contour(
        surface=surface,
        optimizers=[optimizer],
        log_scale=True,
        figsize=(8, 6),
        title=f"{surface_name} — {optimizer_name} (lr={lr}, steps={optimizer.n_steps})",
    )
    return fig


def run_landscape_comparison(
    surface_name: str,
    optimizer_names: list[str],
    lr: float,
    n_steps: int,
) -> object:
    """
    Run multiple optimizers on the same surface and return a comparison figure.
    """
    surface = SURFACE_MAP[surface_name]()
    start = DEFAULT_STARTS[surface_name]
    optimizers = []
    for name in optimizer_names:
        opt = OPTIMIZER_MAP[name](lr)
        surface.run_optimizer(opt, start=start, n_steps=n_steps)
        optimizers.append(opt)

    fig = LandscapePlotter.contour(
        surface=surface,
        optimizers=optimizers,
        log_scale=True,
        figsize=(9, 7),
        title=f"{surface_name} — Optimizer Comparison",
    )
    return fig
