"""
test_adaptive.py
================
Unit tests for gdo.optimizers.adaptive — verifying the mathematical
correctness of RMSProp, Adam, AdamW, and Lion update rules.

Tests verify:
  1. Exact numerical correctness against the paper's pseudocode
  2. Bias correction is applied correctly in Adam/AdamW
  3. AdamW applies weight decay BEFORE the gradient step (decoupled)
  4. Lion uses sign of the interpolated candidate
  5. All optimizers reset cleanly
"""


import numpy as np
from gdo.optimizers.adaptive import Adam, AdamW, Lion, RMSProp


class TestRMSProp:
    def test_first_step_exact(self) -> None:
        """
        Manual computation for the first step:
          v = (1-alpha) * g^2
          theta = theta - lr / sqrt(v + eps) * g
        """
        lr = 0.001
        alpha = 0.99
        eps = 1e-8
        params = np.array([1.0, -1.0])
        grads = np.array([2.0, -3.0])

        opt = RMSProp(lr=lr, alpha=alpha, epsilon=eps)
        new_params = opt.step(params.copy(), grads)

        v = (1.0 - alpha) * grads**2
        expected = params - lr / np.sqrt(v + eps) * grads
        np.testing.assert_allclose(new_params, expected, rtol=1e-7)

    def test_second_step_ema_update(self) -> None:
        """After two steps, EMA should be correctly accumulated."""
        lr = 0.001
        alpha = 0.9
        eps = 1e-8
        params = np.array([1.0])
        g1 = np.array([2.0])
        g2 = np.array([1.0])

        opt = RMSProp(lr=lr, alpha=alpha, epsilon=eps)
        p1 = opt.step(params.copy(), g1)

        v1 = (1.0 - alpha) * g1**2
        v2 = alpha * v1 + (1.0 - alpha) * g2**2
        expected_p2 = p1 - lr / np.sqrt(v2 + eps) * g2

        p2 = opt.step(p1, g2)
        np.testing.assert_allclose(p2, expected_p2, rtol=1e-7)

    def test_reset(self) -> None:
        opt = RMSProp(lr=0.001)
        opt.step(np.zeros(2), np.ones(2))
        opt.reset()
        assert opt._v is None
        assert opt.n_steps == 0


class TestAdam:
    def test_first_step_bias_correction(self) -> None:
        """
        Verify bias correction at t=1:
          m = (1-b1)*g, v = (1-b2)*g^2
          m_hat = m / (1-b1), v_hat = v / (1-b2)
          theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
        """
        lr = 0.001
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        params = np.array([0.5, -0.5])
        grads = np.array([1.0, -1.0])

        opt = Adam(lr=lr, beta1=b1, beta2=b2, epsilon=eps)
        new_params = opt.step(params.copy(), grads)

        m = (1 - b1) * grads
        v = (1 - b2) * grads**2
        m_hat = m / (1 - b1**1)
        v_hat = v / (1 - b2**1)
        expected = params - lr * m_hat / (np.sqrt(v_hat) + eps)

        np.testing.assert_allclose(new_params, expected, rtol=1e-7)

    def test_bias_correction_numerical(self) -> None:
        """
        Without bias correction, early steps would be too small.
        Verify that m_hat > m at t=1.
        """
        b1 = 0.9
        g = np.array([1.0])
        m = (1 - b1) * g  # = 0.1
        m_hat = m / (1 - b1**1)  # = 0.1 / 0.1 = 1.0 = g

        # m_hat should equal g exactly at t=1 (bias correction restores the true estimate)
        np.testing.assert_allclose(m_hat, g, rtol=1e-10)

    def test_moments_not_none_after_step(self) -> None:
        opt = Adam(lr=0.001)
        opt.step(np.zeros(3), np.ones(3))
        assert opt._m is not None
        assert opt._v is not None

    def test_reset_clears_moments(self) -> None:
        opt = Adam(lr=0.001)
        opt.step(np.zeros(2), np.ones(2))
        opt.reset()
        assert opt._m is None
        assert opt._v is None
        assert opt.n_steps == 0


class TestAdamW:
    def test_weight_decay_applied_before_gradient(self) -> None:
        """
        AdamW must apply weight decay BEFORE the gradient step.
        At t=1 with small grads:
          theta_wd = (1 - lr*wd) * theta
          theta_new = theta_wd - lr * m_hat / (sqrt(v_hat) + eps)
        """
        lr = 0.001
        wd = 0.1
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        params = np.array([2.0])
        grads = np.array([0.1])

        opt = AdamW(lr=lr, beta1=b1, beta2=b2, epsilon=eps, weight_decay=wd)
        new_params = opt.step(params.copy(), grads)

        m = (1 - b1) * grads
        v = (1 - b2) * grads**2
        m_hat = m / (1 - b1**1)
        v_hat = v / (1 - b2**1)
        params_wd = (1 - lr * wd) * params
        expected = params_wd - lr * m_hat / (np.sqrt(v_hat) + eps)

        np.testing.assert_allclose(new_params, expected, rtol=1e-7)

    def test_adamw_not_equal_adam_with_l2(self) -> None:
        """
        AdamW applies weight decay BEFORE the gradient step (multiplicative shrink).
        Adam+L2 adds weight decay to the gradient (additive), which then gets
        scaled by the adaptive denominator.

        After many steps the divergence becomes large. Verify with 10 steps.
        """
        lr = 0.01
        wd = 1.0  # extreme wd to make the divergence visible quickly
        params = np.array([3.0, -2.0])
        grads = np.array([0.5, -0.5])

        adamw = AdamW(lr=lr, weight_decay=wd)
        adam_l2 = Adam(lr=lr)

        p_adamw = params.copy()
        p_adam_l2 = params.copy()
        for _ in range(10):
            p_adamw = adamw.step(p_adamw, grads)
            grads_l2 = grads + wd * p_adam_l2
            p_adam_l2 = adam_l2.step(p_adam_l2, grads_l2)

        # After 10 steps with wd=1.0 the results must differ meaningfully
        assert not np.allclose(p_adamw, p_adam_l2, atol=0.01), (
            f"AdamW and Adam+L2 should diverge over 10 steps; got {p_adamw} vs {p_adam_l2}"
        )

    def test_zero_weight_decay_matches_adam(self) -> None:
        """AdamW with weight_decay=0 should match Adam exactly."""
        lr = 0.001
        b1, b2 = 0.9, 0.999
        eps = 1e-8
        params = np.array([1.0, -1.0])
        grads = np.array([0.5, 0.5])

        adam = Adam(lr=lr, beta1=b1, beta2=b2, epsilon=eps)
        adamw = AdamW(lr=lr, beta1=b1, beta2=b2, epsilon=eps, weight_decay=0.0)

        p_adam = adam.step(params.copy(), grads)
        p_adamw = adamw.step(params.copy(), grads)

        np.testing.assert_allclose(p_adam, p_adamw, rtol=1e-10)


class TestLion:
    def test_first_step_sign_update(self) -> None:
        """
        At t=1, m=0, so c = (1-b1)*g.
        Update = sign(c) = sign((1-b1)*g) = sign(g).
        theta_new = (1 - lr*wd)*theta - lr*sign(g)
        """
        lr = 1e-4
        b1 = 0.9
        wd = 0.0
        params = np.array([1.0, -2.0])
        grads = np.array([3.0, -1.0])

        opt = Lion(lr=lr, beta1=b1, weight_decay=wd)
        new_params = opt.step(params.copy(), grads)

        # c = beta1 * 0 + (1-beta1)*g = (1-0.9)*g = 0.1*g
        c = (1 - b1) * grads
        expected = params - lr * np.sign(c)
        np.testing.assert_allclose(new_params, expected, rtol=1e-10)

    def test_update_is_always_unit_magnitude(self) -> None:
        """Every parameter update has magnitude exactly lr (sign of c)."""
        lr = 1e-4
        params = np.array([5.0, -10.0, 0.001])
        grads = np.array([100.0, -50.0, 0.001])

        opt = Lion(lr=lr, weight_decay=0.0)
        new_params = opt.step(params.copy(), grads)

        # |theta_new - theta| should be exactly lr for each param (wd=0)
        update_magnitudes = np.abs(new_params - params)
        np.testing.assert_allclose(update_magnitudes, lr, rtol=1e-10)

    def test_weight_decay_shrinks_params(self) -> None:
        """With weight decay, parameters move toward zero independently of gradients."""
        lr = 1e-4
        wd = 0.01
        params = np.array([10.0])
        grads = np.array([0.001])  # tiny grad — weight decay dominates

        opt = Lion(lr=lr, weight_decay=wd)
        new_params = opt.step(params.copy(), grads)

        # Parameter should be smaller in magnitude than original
        assert abs(new_params[0]) < abs(params[0])

    def test_reset(self) -> None:
        opt = Lion(lr=1e-4)
        opt.step(np.zeros(2), np.ones(2))
        opt.reset()
        assert opt._m is None
        assert opt.n_steps == 0
