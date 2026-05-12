"""
training_tab.py
===============
Gradio component for the Optimizer Comparison tab.

Runs a lightweight MLP on a synthetic classification task
(no MNIST download needed in the app) and streams live loss curves
for two user-selected optimizers side by side.

Uses the gdo.optimizers NumPy implementations for speed —
no PyTorch DataLoader overhead for the demo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from gdo.optimizers import Adam, AdamW, BatchGD, Lion, MomentumSGD, RMSProp, StochasticGD


# ---------------------------------------------------------------------------
# Tiny synthetic MLP in NumPy for fast in-app demo
# ---------------------------------------------------------------------------


class TinyMLP:
    """
    Minimal 2-layer NumPy MLP for the Gradio app demo.
    Fast enough to train 500 steps in under 1 second on CPU.
    """

    def __init__(
        self, input_dim: int = 2, hidden: int = 32, output_dim: int = 2, seed: int = 42
    ) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(0, scale1, (input_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, scale2, (hidden, output_dim))
        self.b2 = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        # Softmax
        exp_z = np.exp(z2 - z2.max(axis=1, keepdims=True))
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        cache = {"X": X, "z1": z1, "a1": a1, "probs": probs}
        return probs, cache

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        """Cross-entropy loss + gradients flattened into a single vector."""
        probs, cache = self.forward(X)
        n = X.shape[0]
        # Cross-entropy loss
        log_probs = np.log(probs[np.arange(n), y] + 1e-8)
        loss = -log_probs.mean()

        # Backprop
        dz2 = probs.copy()
        dz2[np.arange(n), y] -= 1
        dz2 /= n

        dW2 = cache["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (cache["z1"] > 0)
        dW1 = cache["X"].T @ dz1
        db1 = dz1.sum(axis=0)

        grads = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])
        return loss, grads

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_params(self, params: np.ndarray) -> None:
        i = 0
        s = self.W1.size
        self.W1 = params[i : i + s].reshape(self.W1.shape)
        i += s
        s = self.b1.size
        self.b1 = params[i : i + s]
        i += s
        s = self.W2.size
        self.W2 = params[i : i + s].reshape(self.W2.shape)
        i += s
        s = self.b2.size
        self.b2 = params[i : i + s]

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        probs, _ = self.forward(X)
        return float((probs.argmax(axis=1) == y).mean())


def _make_dataset(n: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2D two-spiral classification dataset."""
    rng = np.random.default_rng(seed)
    n_half = n // 2

    def spiral(n_pts: int, label: int) -> tuple[np.ndarray, np.ndarray]:
        t = np.linspace(0, 4 * np.pi, n_pts)
        r = t / (4 * np.pi)
        x = r * np.cos(t + label * np.pi) + rng.normal(0, 0.1, n_pts)
        y = r * np.sin(t + label * np.pi) + rng.normal(0, 0.1, n_pts)
        return np.stack([x, y], axis=1), np.full(n_pts, label)

    X0, y0 = spiral(n_half, 0)
    X1, y1 = spiral(n_half, 1)
    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.concatenate([y0, y1])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


OPTIMIZER_BUILDERS = {
    "Batch GD": lambda lr: BatchGD(lr=lr),
    "SGD": lambda lr: StochasticGD(lr=lr, noise_scale=0.1, seed=42),
    "SGD + Momentum": lambda lr: MomentumSGD(lr=lr, momentum=0.9, seed=42),
    "RMSProp": lambda lr: RMSProp(lr=lr),
    "Adam": lambda lr: Adam(lr=lr),
    "AdamW": lambda lr: AdamW(lr=lr, weight_decay=0.01),
    "Lion": lambda lr: Lion(lr=lr * 0.1),  # Lion needs ~10x smaller LR
}


def run_training_comparison(
    opt1_name: str,
    opt2_name: str,
    lr: float,
    n_steps: int,
) -> object:
    """
    Train two optimizers on the same 2D classification task.

    Returns a matplotlib figure with loss curves for both.
    """
    import matplotlib.pyplot as plt

    X, y = _make_dataset(n=600, seed=42)

    results = {}
    for name in [opt1_name, opt2_name]:
        model = TinyMLP(seed=42)
        opt = OPTIMIZER_BUILDERS[name](lr)
        params = model.get_params()
        opt.set_initial_point(params)

        losses = []
        for _ in range(n_steps):
            loss, grads = model.loss_and_grads(X, y)
            params = opt.step(params, grads)
            model.set_params(params)
            losses.append(loss)
        results[name] = losses

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#d62728"]
    for (name, losses), color in zip(results.items(), colors):
        ax.plot(losses, label=name, color=color, linewidth=2)

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_title(
        f"Optimizer Comparison — 2-Spiral Classification\n({n_steps} steps, lr={lr})", fontsize=13
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
