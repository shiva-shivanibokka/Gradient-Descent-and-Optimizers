"""
test_landscapes.py
==================
Unit tests for gdo.landscapes.surfaces.

Tests verify:
  1. All surfaces evaluate to 0 at their known global minimum
  2. Gradient is zero (or near-zero) at the global minimum
  3. Meshgrid returns correct shapes
  4. run_optimizer records a trajectory
  5. All optimizers CONVERGE to the QuadraticSurface minimum (functional test)
"""

import numpy as np
import pytest
from gdo.landscapes import Beale, Himmelblau, QuadraticSurface, Rosenbrock
from gdo.optimizers import Adam, AdamW, BatchGD, MomentumSGD, RMSProp


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


# ---------------------------------------------------------------------------
# Convergence tests — the most important functional property of any optimizer
# ---------------------------------------------------------------------------


class TestOptimizerConvergence:
    """
    Verify that every optimizer actually converges to the QuadraticSurface
    minimum within a fixed number of steps.

    Why QuadraticSurface(a=1, b=1) (symmetric, not ill-conditioned)?
      We want to test that the UPDATE RULE is correct, not that the
      optimizer handles ill-conditioning.  Ill-conditioned tests belong
      in integration tests.  A symmetric bowl is the minimum surface
      that any correct gradient descent must converge on.

    Tolerance: 0.05 Euclidean distance from (0, 0).
    Steps: 500 — generous enough for all variants to converge.
    """

    SURFACE = QuadraticSurface(a=1.0, b=1.0)
    START: tuple[float, float] = (-2.0, 2.0)
    TOLERANCE = 0.05
    N_STEPS = 500

    def _run_and_check(self, opt: object) -> None:
        """Run optimizer and assert it reaches the minimum."""
        self.SURFACE.run_optimizer(opt, start=self.START, n_steps=self.N_STEPS)  # type: ignore[arg-type]
        final = np.array(opt.trajectory[-1])  # type: ignore[union-attr]
        optimum = np.array(self.SURFACE.optimum)
        dist = float(np.linalg.norm(final - optimum))
        assert dist < self.TOLERANCE, (
            f"{opt.name} did not converge: "  # type: ignore[union-attr]
            f"final={final}, dist={dist:.4f} > tol={self.TOLERANCE}"
        )

    def test_batch_gd_converges(self) -> None:
        """Full-batch gradient descent on a symmetric quadratic must converge."""
        self._run_and_check(BatchGD(lr=0.1))

    def test_momentum_sgd_converges(self) -> None:
        """Momentum SGD with noise_scale=0 (deterministic) must converge."""
        self._run_and_check(MomentumSGD(lr=0.1, momentum=0.9, noise_scale=0.0))

    def test_rmsprop_converges(self) -> None:
        """RMSProp must converge on a simple convex surface."""
        self._run_and_check(RMSProp(lr=0.01))

    def test_adam_converges(self) -> None:
        """Adam must converge — this is the most important optimizer to verify."""
        self._run_and_check(Adam(lr=0.05))

    def test_adamw_zero_wd_converges(self) -> None:
        """AdamW with weight_decay=0 must converge identically to Adam."""
        self._run_and_check(AdamW(lr=0.05, weight_decay=0.0))

    def test_adamw_with_wd_converges(self) -> None:
        """
        AdamW with weight_decay > 0 must still converge.
        Weight decay adds a pull toward zero which helps on a quadratic
        centred at the origin (same direction as the gradient).
        """
        self._run_and_check(AdamW(lr=0.05, weight_decay=0.01))

    def test_final_loss_is_near_zero(self) -> None:
        """
        Beyond just reaching the parameter optimum, verify the loss value
        at convergence is near zero — catches bugs where the optimizer
        reaches a wrong point that happens to be close in parameter space.
        """
        opt = Adam(lr=0.05)
        self.SURFACE.run_optimizer(opt, start=self.START, n_steps=self.N_STEPS)
        final = opt.trajectory[-1]
        final_loss = float(self.SURFACE(final[0], final[1]))
        assert final_loss < 0.01, f"Adam converged to a point with non-zero loss: {final_loss:.6f}"
