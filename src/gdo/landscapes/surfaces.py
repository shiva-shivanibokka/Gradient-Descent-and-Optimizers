"""
surfaces.py
===========
Analytical loss surface definitions for optimizer trajectory visualization.

Each surface provides:
  - ``__call__(x, y)``   — scalar or array loss value
  - ``gradient(x, y)``   — analytical gradient [∂f/∂x, ∂f/∂y]
  - ``bounds``           — recommended (x_min, x_max, y_min, y_max)
  - ``optimum``          — known global minimum (x*, y*)

These surfaces are used to run the NumPy optimizers and visualize
their trajectories on a 2D contour plot — the core visualization
in Notebook 1 and the web landscape tab.

Why analytical surfaces?
------------------------
Computing exact gradients allows perfect trajectory visualization
without the noise of data-driven gradients.  The optimizer math is
isolated from data preprocessing concerns.

Surfaces included
-----------------
QuadraticSurface  — Convex bowl (easy, good for first demos)
Rosenbrock        — Non-convex banana valley (gradient descent's nemesis)
Beale             — Three local minima, one global minimum
Himmelblau        — Four equal global minima (shows multi-modal landscape)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


class Bounds(NamedTuple):
    """Axis-aligned bounding box for a 2D surface."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class LossSurface(ABC):
    """
    Abstract base class for 2D analytical loss surfaces.

    All surfaces operate on 2D parameter vectors [x, y] so they can
    be directly visualized as contour plots.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name for plot titles and legends."""
        ...

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        """Recommended axis limits for visualization."""
        ...

    @property
    @abstractmethod
    def optimum(self) -> tuple[float, float]:
        """Global minimum location (x*, y*)."""
        ...

    @abstractmethod
    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """
        Compute the loss value at (x, y).

        Accepts both scalars and NumPy arrays (for meshgrid evaluation).
        """
        ...

    @abstractmethod
    def gradient(self, x: float, y: float) -> np.ndarray:
        """
        Compute the analytical gradient [∂f/∂x, ∂f/∂y] at (x, y).

        Returns
        -------
        np.ndarray
            Shape (2,) gradient vector.
        """
        ...

    def meshgrid(self, resolution: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a meshgrid of loss values for contour plotting.

        Parameters
        ----------
        resolution:
            Number of grid points per axis.

        Returns
        -------
        X, Y, Z:
            Meshgrid arrays of shape (resolution, resolution).
        """
        b = self.bounds
        xs = np.linspace(b.x_min, b.x_max, resolution)
        ys = np.linspace(b.y_min, b.y_max, resolution)
        X, Y = np.meshgrid(xs, ys)
        Z = self(X, Y)
        logger.debug("[%s] meshgrid generated at resolution %d", self.name, resolution)
        return X, Y, Z  # type: ignore[return-value]

    def run_optimizer(
        self,
        optimizer: object,
        start: tuple[float, float],
        n_steps: int = 200,
    ) -> list[np.ndarray]:
        """
        Run a ``gdo.optimizers.Optimizer`` on this surface and return
        the full trajectory.

        Parameters
        ----------
        optimizer:
            Any ``gdo.optimizers.Optimizer`` instance.
        start:
            Starting parameter position (x₀, y₀).
        n_steps:
            Number of gradient steps.

        Returns
        -------
        list[np.ndarray]
            List of parameter arrays, including the starting point.
        """
        from gdo.optimizers.base import Optimizer  # avoid circular at module level

        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"Expected gdo Optimizer, got {type(optimizer)}")

        params = np.array(start, dtype=float)
        optimizer.set_initial_point(params)

        for step in range(n_steps):
            grads = self.gradient(params[0], params[1])
            params = optimizer.step(params, grads)
            loss = float(self(params[0], params[1]))
            optimizer._loss_history.append(loss)

            # Early stop if we're at the optimum
            dist = np.linalg.norm(params - np.array(self.optimum))
            if dist < 1e-6:
                logger.info(
                    "[%s] Optimizer '%s' converged at step %d (dist=%.2e)",
                    self.name,
                    optimizer.name,
                    step,
                    dist,
                )
                break

        return optimizer.trajectory


class QuadraticSurface(LossSurface):
    """
    Axis-aligned quadratic bowl.

    f(x, y) = a * x² + b * y²

    The simplest convex surface.  When a ≠ b (ill-conditioned),
    gradient descent oscillates across the steep axis — this is
    exactly why momentum and adaptive optimizers help.

    Parameters
    ----------
    a:
        Curvature along x-axis (default 1.0).
    b:
        Curvature along y-axis (default 10.0 — creates an elongated valley
        that clearly shows the difference between GD variants).
    """

    def __init__(self, a: float = 1.0, b: float = 10.0) -> None:
        self._a = a
        self._b = b

    @property
    def name(self) -> str:
        return f"Quadratic (a={self._a}, b={self._b})"

    @property
    def bounds(self) -> Bounds:
        return Bounds(-3.0, 3.0, -3.0, 3.0)

    @property
    def optimum(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return self._a * x**2 + self._b * y**2

    def gradient(self, x: float, y: float) -> np.ndarray:
        return np.array([2.0 * self._a * x, 2.0 * self._b * y])


class Rosenbrock(LossSurface):
    """
    Rosenbrock function — the classic optimizer benchmark.

    f(x, y) = (a - x)² + b * (y - x²)²

    The global minimum lies inside a long, narrow, parabolic-shaped
    valley.  Reaching the minimum is easy; finding it is hard because
    the floor of the valley is nearly flat.  This surface:
    - Exposes the difference between momentum and no-momentum GD
    - Shows why adaptive optimizers need many steps to navigate the valley
    - Is the most commonly used surface in optimizer comparison papers

    Default: a=1, b=100.  Global minimum at (1, 1) with f=0.

    Parameters
    ----------
    a:
        Shift of the minimum (default 1.0).
    b:
        Banana shape parameter (default 100.0).
    """

    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        self._a = a
        self._b = b

    @property
    def name(self) -> str:
        return "Rosenbrock"

    @property
    def bounds(self) -> Bounds:
        return Bounds(-2.0, 2.0, -1.0, 3.0)

    @property
    def optimum(self) -> tuple[float, float]:
        return (float(self._a), float(self._a**2))

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (self._a - x) ** 2 + self._b * (y - x**2) ** 2

    def gradient(self, x: float, y: float) -> np.ndarray:
        dfdx = -2.0 * (self._a - x) - 4.0 * self._b * x * (y - x**2)
        dfdy = 2.0 * self._b * (y - x**2)
        return np.array([dfdx, dfdy])


class Beale(LossSurface):
    """
    Beale function — multi-local-minima surface.

    f(x, y) = (1.5 - x + xy)²
            + (2.25 - x + xy²)²
            + (2.625 - x + xy³)²

    Global minimum: f(3, 0.5) = 0.

    Contains several local minima and a steep outer region, making it
    a good test for whether an optimizer can escape bad initializations.

    Domain: x, y ∈ [-4.5, 4.5]
    """

    @property
    def name(self) -> str:
        return "Beale"

    @property
    def bounds(self) -> Bounds:
        return Bounds(-4.5, 4.5, -4.5, 4.5)

    @property
    def optimum(self) -> tuple[float, float]:
        return (3.0, 0.5)

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        term1 = (1.5 - x + x * y) ** 2
        term2 = (2.25 - x + x * y**2) ** 2
        term3 = (2.625 - x + x * y**3) ** 2
        return term1 + term2 + term3

    def gradient(self, x: float, y: float) -> np.ndarray:
        t1 = 1.5 - x + x * y
        t2 = 2.25 - x + x * y**2
        t3 = 2.625 - x + x * y**3

        dfdx = 2 * t1 * (-1 + y) + 2 * t2 * (-1 + y**2) + 2 * t3 * (-1 + y**3)
        dfdy = 2 * t1 * x + 2 * t2 * (2 * x * y) + 2 * t3 * (3 * x * y**2)
        return np.array([dfdx, dfdy])


class Himmelblau(LossSurface):
    """
    Himmelblau's function — four global minima.

    f(x, y) = (x² + y - 11)² + (x + y² - 7)²

    Four global minima, all with f = 0:
      (3.0,       2.0)
      (-2.805118, 3.131312)
      (-3.779310, -3.283186)
      (3.584428,  -1.848126)

    Excellent for showing how the starting point determines which
    minimum an optimizer converges to — a key concept in non-convex
    optimization.

    Domain: x, y ∈ [-5, 5]
    """

    MINIMA: list[tuple[float, float]] = [
        (3.0, 2.0),
        (-2.805118, 3.131312),
        (-3.779310, -3.283186),
        (3.584428, -1.848126),
    ]

    @property
    def name(self) -> str:
        return "Himmelblau"

    @property
    def bounds(self) -> Bounds:
        return Bounds(-5.0, 5.0, -5.0, 5.0)

    @property
    def optimum(self) -> tuple[float, float]:
        return self.MINIMA[0]  # return the canonical first minimum

    def __call__(self, x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    def gradient(self, x: float, y: float) -> np.ndarray:
        dfdx = 4 * x * (x**2 + y - 11) + 2 * (x + y**2 - 7)
        dfdy = 2 * (x**2 + y - 11) + 4 * y * (x + y**2 - 7)
        return np.array([dfdx, dfdy])
