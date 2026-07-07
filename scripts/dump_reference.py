"""
dump_reference.py
=================
Dump reference values from the Python ``gdo`` package to JSON fixtures that the
TypeScript web app's Vitest suite asserts against. This is the parity harness:
it proves the in-browser TS math is the same math verified by the Python tests.

Covers the DETERMINISTIC pieces only:
  - surface value + gradient at a grid of test points (all four surfaces)
  - optimizer trajectories for the noise-free optimizers (BatchGD, RMSProp,
    Adam, AdamW, Lion) on well-conditioned cases
  - LR-schedule curves for the pure epoch->lr schedulers

The three noisy SGD variants (SGD, MiniBatchGD, MomentumSGD) use NumPy's RNG and
cannot be reproduced bit-for-bit in TS, so they are intentionally excluded.

Run:  python scripts/dump_reference.py
Out:  web/tests/fixtures/reference.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gdo.landscapes.surfaces import Beale, Himmelblau, QuadraticSurface, Rosenbrock
from gdo.optimizers.adaptive import RMSProp, Adam, AdamW, Lion
from gdo.optimizers.sgd import BatchGD
from gdo.optimizers.schedulers import (
    CosineAnnealingLR,
    CyclicalLR,
    OneCycleLR,
    StepLR,
    WarmupScheduler,
)

OUT = Path(__file__).resolve().parent.parent / "web" / "tests" / "fixtures" / "reference.json"

# Surface constructors mirror the TS defaults exactly (Quadratic a=1,b=10; Rosenbrock a=1,b=100).
SURFACES = {
    "quadratic": QuadraticSurface(),
    "rosenbrock": Rosenbrock(),
    "beale": Beale(),
    "himmelblau": Himmelblau(),
}

# Deterministic optimizers only. Factory takes the lr from each trajectory case.
OPTIMIZERS = {
    "batch_gd": lambda lr: BatchGD(lr=lr),
    "rmsprop": lambda lr: RMSProp(lr=lr),
    "adam": lambda lr: Adam(lr=lr),
    "adamw": lambda lr: AdamW(lr=lr),
    "lion": lambda lr: Lion(lr=lr),
}

# Well-conditioned trajectory cases (finite, non-diverging) so parity is meaningful.
# Quadratic bowl is stable for every optimizer; add one Rosenbrock+Adam case.
TRAJECTORY_CASES = [
    {"surface": "quadratic", "optimizer": opt, "lr": lr, "start": [-2.5, 2.5], "steps": 60}
    for opt, lr in [
        ("batch_gd", 0.05),
        ("rmsprop", 0.05),
        ("adam", 0.10),
        ("adamw", 0.10),
        ("lion", 0.05),
    ]
] + [
    {"surface": "rosenbrock", "optimizer": "adam", "lr": 0.005, "start": [-1.5, 2.5], "steps": 200},
]

# Scheduler cases. Each scheduler is a pure function of epoch (no metric).
SCHEDULER_CASES = [
    {"scheduler": "step", "base_lr": 0.1, "epochs": 50, "ctor": lambda b: StepLR(b, step_size=10, gamma=0.5)},
    {"scheduler": "cosine", "base_lr": 0.1, "epochs": 50, "ctor": lambda b: CosineAnnealingLR(b, t_max=50)},
    {"scheduler": "onecycle", "base_lr": 0.01, "epochs": 50, "ctor": lambda b: OneCycleLR(b, max_lr=0.1, total_epochs=50)},
    {"scheduler": "cyclical", "base_lr": 0.001, "epochs": 50, "ctor": lambda b: CyclicalLR(b, max_lr=0.05, step_size_up=5)},
    {"scheduler": "warmup_cosine", "base_lr": 0.1, "epochs": 50, "ctor": lambda b: WarmupScheduler(b, total_epochs=50, warmup_steps=5)},
]


def dump_surfaces() -> dict:
    out = {}
    for key, surf in SURFACES.items():
        b = surf.bounds
        xs = np.linspace(b.x_min + 0.3, b.x_max - 0.3, 5)
        ys = np.linspace(b.y_min + 0.3, b.y_max - 0.3, 5)
        points, values, grads = [], [], []
        for x in xs:
            for y in ys:
                points.append([float(x), float(y)])
                values.append(float(surf(x, y)))
                g = surf.gradient(float(x), float(y))
                grads.append([float(g[0]), float(g[1])])
        out[key] = {"points": points, "values": values, "grads": grads}
    return out


def dump_trajectories() -> list:
    out = []
    for case in TRAJECTORY_CASES:
        surf = SURFACES[case["surface"]]
        opt = OPTIMIZERS[case["optimizer"]](case["lr"])
        p = np.array(case["start"], dtype=float)
        opt.set_initial_point(p)
        for _ in range(case["steps"]):
            g = surf.gradient(p[0], p[1])
            p = opt.step(p, g)
        path = [[float(v[0]), float(v[1])] for v in opt.trajectory]
        out.append({**{k: case[k] for k in ("surface", "optimizer", "lr", "start", "steps")}, "path": path})
    return out


def dump_schedules() -> list:
    out = []
    for case in SCHEDULER_CASES:
        sched = case["ctor"](case["base_lr"])
        curve = [float(v) for v in sched.get_lr_curve(case["epochs"])]
        out.append(
            {"scheduler": case["scheduler"], "base_lr": case["base_lr"], "epochs": case["epochs"], "curve": curve}
        )
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "surfaces": dump_surfaces(),
        "trajectories": dump_trajectories(),
        "schedules": dump_schedules(),
    }
    OUT.write_text(json.dumps(data, indent=2))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
