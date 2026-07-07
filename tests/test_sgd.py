"""
test_sgd.py
===========
Unit tests for gdo.optimizers.sgd — verifying the mathematical
correctness of each update rule.

Tests verify:
  1. Exact numerical correctness against hand-computed update formulas
  2. Correct trajectory recording (n points after n steps)
  3. Reset clears all state
  4. Shape validation raises on mismatch
"""

import numpy as np
import pytest
from gdo.optimizers.sgd import BatchGD, MomentumSGD


class TestBatchGD:
    def test_single_step_exact(self) -> None:
        """θ ← θ - lr * g  (exact numerical check)."""
        lr = 0.1
        params = np.array([3.0, -2.0])
        grads = np.array([1.0, -0.5])
        opt = BatchGD(lr=lr)

        new_params = opt.step(params, grads)
        expected = params - lr * grads

        np.testing.assert_allclose(new_params, expected, rtol=1e-10)

    def test_two_steps_independent(self) -> None:
        """Each step is independent of the previous (no state)."""
        lr = 0.05
        params = np.array([1.0, 1.0])
        grads = np.array([2.0, 2.0])
        opt = BatchGD(lr=lr)

        p1 = opt.step(params.copy(), grads)
        p2 = opt.step(p1, grads)

        np.testing.assert_allclose(p1, [0.9, 0.9], rtol=1e-10)
        np.testing.assert_allclose(p2, [0.8, 0.8], rtol=1e-10)

    def test_trajectory_length(self) -> None:
        """Trajectory accumulates one point per step."""
        opt = BatchGD(lr=0.1)
        params = np.zeros(3)
        opt.set_initial_point(params)
        for _ in range(5):
            params = opt.step(params, np.ones(3))
        assert len(opt.trajectory) == 6  # initial + 5 steps

    def test_n_steps_counter(self) -> None:
        opt = BatchGD(lr=0.1)
        params = np.zeros(2)
        for _ in range(10):
            params = opt.step(params, np.ones(2))
        assert opt.n_steps == 10

    def test_reset_clears_state(self) -> None:
        opt = BatchGD(lr=0.1)
        params = np.zeros(2)
        for _ in range(5):
            params = opt.step(params, np.ones(2))
        opt.reset()
        assert opt.n_steps == 0
        assert len(opt.trajectory) == 0

    def test_invalid_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="Learning rate"):
            BatchGD(lr=0.0)

    def test_shape_mismatch_raises(self) -> None:
        opt = BatchGD(lr=0.1)
        with pytest.raises(ValueError, match="shape"):
            opt.step(np.zeros(3), np.zeros(2))

    def test_does_not_mutate_input_params(self) -> None:
        """step() must return a new array, not modify params in-place."""
        opt = BatchGD(lr=0.1)
        params = np.array([1.0, 2.0])
        original = params.copy()
        opt.step(params, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(params, original)


class TestMomentumSGD:
    def test_first_step_matches_sgd(self) -> None:
        """On the first step, velocity=0 so Momentum SGD = SGD (deterministic mode)."""
        lr = 0.1
        momentum = 0.9
        params = np.array([2.0, -1.0])
        grads = np.array([1.0, -0.5])

        opt = MomentumSGD(lr=lr, momentum=momentum, noise_scale=0.0)
        new_params = opt.step(params.copy(), grads)

        # velocity after first step = 0 * 0.9 - lr * grads = -lr * grads
        # new_params = params + velocity = params - lr * grads
        expected = params - lr * grads
        np.testing.assert_allclose(new_params, expected, rtol=1e-10)

    def test_velocity_accumulates(self) -> None:
        """After two steps, velocity carries information from step 1."""
        lr = 0.1
        beta = 0.9
        grads = np.array([1.0, 0.0])
        params = np.zeros(2)

        opt = MomentumSGD(lr=lr, momentum=beta, noise_scale=0.0)

        # Step 1: v1 = 0*beta - lr*g = -0.1; params1 = params + v1 = -0.1
        p1 = opt.step(params.copy(), grads)
        # Step 2: v2 = v1*beta - lr*g = -0.09 - 0.1 = -0.19; params2 = p1 + v2
        p2 = opt.step(p1, grads)

        v2 = beta * (-lr * grads[0]) - lr * grads[0]
        expected_p2_x = p1[0] + v2
        np.testing.assert_allclose(p2[0], expected_p2_x, rtol=1e-10)

    def test_invalid_momentum_raises(self) -> None:
        with pytest.raises(ValueError, match="momentum"):
            MomentumSGD(lr=0.1, momentum=1.0)

    def test_reset_clears_velocity(self) -> None:
        opt = MomentumSGD(lr=0.1, momentum=0.9, noise_scale=0.0)
        params = np.zeros(2)
        params = opt.step(params, np.ones(2))
        opt.reset()
        assert opt._velocity is None
        assert opt.n_steps == 0
