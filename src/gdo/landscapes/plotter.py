"""
plotter.py
==========
Visualization utilities for loss landscapes and optimizer trajectories.

LandscapePlotter produces:
  - 2D contour plots with optimizer trajectory overlays
  - Side-by-side multi-optimizer trajectory comparison
  - Convergence curves (loss vs step) for any number of optimizers
  - Plotly versions of all of the above (for the Gradio app)

Design note
-----------
All plot methods return the figure object rather than calling
``plt.show()`` directly.  This makes them composable in notebooks
(cell output) and usable in the Gradio app (``gr.Plot`` component)
without side effects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from gdo.landscapes.surfaces import LossSurface
    from gdo.optimizers.base import Optimizer

logger = logging.getLogger(__name__)

# ── Color palette consistent across all plots ──────────────────────────────
OPTIMIZER_COLORS: dict[str, str] = {
    "Batch GD": "#1f77b4",
    "SGD": "#ff7f0e",
    "Mini-Batch GD": "#2ca02c",
    "SGD + Momentum": "#d62728",
    "RMSProp": "#9467bd",
    "Adam": "#8c564b",
    "AdamW": "#e377c2",
    "Lion": "#bcbd22",
}
_DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _get_color(name: str, idx: int = 0) -> str:
    """Return a deterministic color for an optimizer by name or index."""
    for key, color in OPTIMIZER_COLORS.items():
        if key.lower() in name.lower():
            return color
    return _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)]


class LandscapePlotter:
    """
    Factory class for loss landscape and convergence visualizations.

    All methods are class methods — no instance state needed.

    Example
    -------
    >>> surface = Rosenbrock()
    >>> optimizers = [BatchGD(lr=0.001), MomentumSGD(lr=0.001)]
    >>> for opt in optimizers:
    ...     surface.run_optimizer(opt, start=(-1.5, 1.0), n_steps=500)
    >>> fig = LandscapePlotter.trajectory_comparison(surface, optimizers)
    >>> fig.savefig("assets/rosenbrock_comparison.png", dpi=150)
    """

    @classmethod
    def contour(
        cls,
        surface: "LossSurface",
        optimizers: list["Optimizer"] | None = None,
        resolution: int = 300,
        log_scale: bool = True,
        figsize: tuple[float, float] = (8, 6),
        title: str | None = None,
        mark_optimum: bool = True,
    ) -> Figure:
        """
        Draw a 2D contour plot of the loss surface with optional
        optimizer trajectories overlaid.

        Parameters
        ----------
        surface:
            The loss surface to visualize.
        optimizers:
            List of optimizers whose ``trajectory`` will be drawn.
            If None, only the surface is drawn.
        resolution:
            Meshgrid resolution (higher = sharper contours, slower).
        log_scale:
            Apply log(1 + Z) before drawing contours — improves
            contrast for surfaces with very large ranges (e.g. Rosenbrock).
        figsize:
            Matplotlib figure size.
        title:
            Plot title. Defaults to the surface name.
        mark_optimum:
            Mark the global minimum with a gold star.

        Returns
        -------
        matplotlib.figure.Figure
        """
        X, Y, Z = surface.meshgrid(resolution=resolution)
        Z_plot = np.log1p(Z) if log_scale else Z

        fig, ax = plt.subplots(figsize=figsize)
        ax.contourf(X, Y, Z_plot, levels=40, cmap="viridis", alpha=0.85)
        ax.contour(X, Y, Z_plot, levels=20, colors="white", linewidths=0.4, alpha=0.4)

        if mark_optimum:
            ox, oy = surface.optimum
            ax.plot(ox, oy, "*", color="gold", markersize=14, zorder=10, label="Global minimum")

        if optimizers:
            for i, opt in enumerate(optimizers):
                traj = opt.trajectory
                if len(traj) < 2:
                    logger.warning(
                        "[LandscapePlotter] Optimizer '%s' has < 2 trajectory points.", opt.name
                    )
                    continue
                coords = np.array(traj)
                color = _get_color(opt.name, i)
                ax.plot(
                    coords[:, 0],
                    coords[:, 1],
                    "-",
                    color=color,
                    linewidth=1.5,
                    alpha=0.9,
                    label=opt.name,
                )
                # Mark start
                ax.plot(coords[0, 0], coords[0, 1], "o", color=color, markersize=7)
                # Mark end
                ax.plot(coords[-1, 0], coords[-1, 1], "s", color=color, markersize=7)

        b = surface.bounds
        ax.set_xlim(b.x_min, b.x_max)
        ax.set_ylim(b.y_min, b.y_max)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(title or f"{surface.name} — Optimizer Trajectories", fontsize=14)
        if optimizers or mark_optimum:
            ax.legend(fontsize=10, loc="upper right")

        fig.tight_layout()
        logger.debug("[LandscapePlotter] contour plot generated for surface '%s'", surface.name)
        return fig

    @classmethod
    def trajectory_comparison(
        cls,
        surface: "LossSurface",
        optimizers: list["Optimizer"],
        resolution: int = 250,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """
        Draw one subplot per optimizer, all on the same surface.

        Useful for directly comparing trajectories side by side.
        """
        n = len(optimizers)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig_w = figsize[0] if figsize else 5.5 * cols
        fig_h = figsize[1] if figsize else 4.5 * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

        X, Y, Z = surface.meshgrid(resolution=resolution)
        Z_plot = np.log1p(Z)
        ox, oy = surface.optimum

        for idx, opt in enumerate(optimizers):
            row, col = divmod(idx, cols)
            ax: Axes = axes[row][col]
            ax.contourf(X, Y, Z_plot, levels=30, cmap="viridis", alpha=0.85)
            ax.plot(ox, oy, "*", color="gold", markersize=12, zorder=10)

            traj = opt.trajectory
            if len(traj) >= 2:
                coords = np.array(traj)
                color = _get_color(opt.name, idx)
                ax.plot(coords[:, 0], coords[:, 1], "-", color=color, linewidth=1.5)
                ax.plot(coords[0, 0], coords[0, 1], "o", color=color, markersize=8)
                ax.plot(coords[-1, 0], coords[-1, 1], "s", color=color, markersize=8)

            b = surface.bounds
            ax.set_xlim(b.x_min, b.x_max)
            ax.set_ylim(b.y_min, b.y_max)
            ax.set_title(f"{opt.name}\n(steps={opt.n_steps})", fontsize=11)
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("y", fontsize=10)

        # Hide empty subplots
        for idx in range(n, rows * cols):
            row, col = divmod(idx, cols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"Trajectory Comparison — {surface.name}", fontsize=14, y=1.01)
        fig.tight_layout()
        return fig

    @classmethod
    def convergence_curves(
        cls,
        optimizers: list["Optimizer"],
        figsize: tuple[float, float] = (9, 5),
        log_scale: bool = True,
        title: str = "Convergence Curves — Loss vs Step",
    ) -> Figure:
        """
        Plot loss history for each optimizer on the same axes.

        Parameters
        ----------
        optimizers:
            List of optimizers with a populated ``loss_history``.
        log_scale:
            Use log scale on the y-axis.
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, opt in enumerate(optimizers):
            history = opt.loss_history
            if not history:
                logger.warning("[LandscapePlotter] Optimizer '%s' has no loss history.", opt.name)
                continue
            color = _get_color(opt.name, i)
            ax.plot(history, label=opt.name, color=color, linewidth=2)

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Loss" + (" (log scale)" if log_scale else ""), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @classmethod
    def lr_schedule_plot(
        cls,
        schedules: dict[str, np.ndarray],
        figsize: tuple[float, float] = (9, 4),
        title: str = "Learning Rate Schedules",
    ) -> Figure:
        """
        Plot multiple LR schedules on the same axes.

        Parameters
        ----------
        schedules:
            Dict mapping scheduler name → array of LR values per epoch.
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, (name, lrs) in enumerate(schedules.items()):
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            ax.plot(lrs, label=name, color=color, linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Learning Rate", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @classmethod
    def gradient_norm_plot(
        cls,
        norm_histories: dict[str, list[float]],
        figsize: tuple[float, float] = (9, 4),
        title: str = "Gradient Norm During Training",
    ) -> Figure:
        """
        Plot gradient norm over training steps for multiple runs.

        Parameters
        ----------
        norm_histories:
            Dict mapping run name → list of gradient norms per step.
        """
        fig, ax = plt.subplots(figsize=figsize)

        for i, (name, norms) in enumerate(norm_histories.items()):
            color = _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
            # Smooth with a moving average for readability
            if len(norms) > 20:
                kernel = np.ones(10) / 10
                smoothed = np.convolve(norms, kernel, mode="valid")
                ax.plot(smoothed, label=name, color=color, linewidth=2)
                ax.plot(norms, color=color, linewidth=0.5, alpha=0.3)
            else:
                ax.plot(norms, label=name, color=color, linewidth=2)

        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Gradient Norm", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    @classmethod
    def to_plotly(cls, fig: Figure) -> object:
        """
        Convert a matplotlib Figure to a Plotly Figure for Gradio.

        Requires ``plotly`` to be installed.
        """
        try:
            import plotly.tools as tls

            return tls.mpl_to_plotly(fig)
        except ImportError:
            logger.warning("plotly not installed — returning matplotlib figure.")
            return fig
