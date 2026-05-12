"""
test_landscapes.py
==================
Unit tests for gdo.landscapes.surfaces.

Tests verify:
  1. All surfaces evaluate to 0 at their known global minimum
  2. Gradient is zero (or near-zero) at the global minimum
  3. Meshgrid returns correct shapes
  4. run_optimizer records a trajectory
"""

import numpy as np
import pytest

from gdo.landscapes import Beale, Himmelblau, QuadraticSurface, Rosenbrock
from gdo.optimizers import BatchGD


class TestQuadraticSurface:
    def test_minimum_is_zero(self) -> None:
        surface = QuadraticSurface(a=1.0, b=5.0)
        ox, oy = surface.optimum
        assert surface(ox, oy) == pytest.approx(0.0)

    def test_gradient_at_minimum_is_zero(self) -> None:
        surface = QuadraticSurface()
        grad = surface.gradient(0.0, 0.0)
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_meshgrid_shapes(self) -> None:
        surface = QuadraticSurface()
        X, Y, Z = surface.meshgrid(resolution=50)
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert Z.shape == (50, 50)

    def test_run_optimizer_records_trajectory(self) -> None:
        surface = QuadraticSurface()
        opt = BatchGD(lr=0.05)
        traj = surface.run_optimizer(opt, start=(-2.0, 2.0), n_steps=50)
        assert len(traj) >= 2
        assert len(traj[0]) == 2  # 2D surface


class TestRosenbrock:
    def test_minimum_at_1_1(self) -> None:
        surface = Rosenbrock()
        assert surface(1.0, 1.0) == pytest.approx(0.0)
        assert surface.optimum == (1.0, 1.0)

    def test_gradient_at_minimum_is_zero(self) -> None:
        surface = Rosenbrock()
        grad = surface.gradient(1.0, 1.0)
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_far_from_minimum_large_value(self) -> None:
        surface = Rosenbrock()
        assert surface(-2.0, -2.0) > 100


class TestBeale:
    def test_minimum_value_is_zero(self) -> None:
        surface = Beale()
        ox, oy = surface.optimum
        val = float(surface(ox, oy))
        assert val == pytest.approx(0.0, abs=1e-8)

    def test_gradient_at_minimum_is_near_zero(self) -> None:
        surface = Beale()
        ox, oy = surface.optimum
        grad = surface.gradient(ox, oy)
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-5)


class TestHimmelblau:
    def test_all_four_minima_are_zero(self) -> None:
        surface = Himmelblau()
        for x, y in surface.MINIMA:
            val = float(surface(x, y))
            assert val == pytest.approx(0.0, abs=1e-5), f"f({x}, {y}) = {val}"

    def test_gradient_at_canonical_minimum(self) -> None:
        surface = Himmelblau()
        grad = surface.gradient(3.0, 2.0)
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-5)

    def test_meshgrid_returns_positive_values(self) -> None:
        surface = Himmelblau()
        _, _, Z = surface.meshgrid(resolution=50)
        # Himmelblau is non-negative everywhere
        assert float(Z.min()) >= -1e-5
