# Production-Grade Revamp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the `gdo` optimizer library from "claims production-grade" to actually production-grade for a portfolio/interview audience — green CI, correct code, and the developer-workflow tooling big companies actually use.

**Architecture:** No architectural change. The package/notebook/app-share-one-tested-package design is already sound. This plan fixes correctness bugs, makes the CI gates real (lint + type + coverage all block), corrects identity/docs, and adds the standard Python-shop tooling layer (pre-commit, Makefile, Dependabot).

**Tech Stack:** Python 3.10–3.12, NumPy, PyTorch, Pydantic v2, MLflow, Gradio, pytest, ruff, black, mypy, GitHub Actions, pre-commit.

## Global Constraints

- **Free tooling only** — no paid services. Coverage enforced via `pytest-cov --cov-fail-under` (no Codecov account); CI badge via GitHub Actions' built-in status badge.
- **Deployment target is Hugging Face Spaces (free CPU Basic)** — keep Gradio, do not rewrite the app.
- **Author identity:** name = `Shivani Bokka`, GitHub = `shiva-shivanibokka`, repo = `Gradient-Descent-and-Optimizers`. Every URL/name must use these — never `siddharth-bokka` / `Siddharth Bokka`.
- **No new runtime dependencies** for bug fixes. New dev-only tools (pre-commit) go in the `dev` extra / `requirements-dev.txt` only.
- Every task ends green: `pytest tests/ -q` and `ruff check src/ tests/` both pass before commit.

---

### Task 1: Green the lint baseline

**Files:**
- Modify: `tests/test_sgd.py:17` (unused imports) and any other ruff hits across `src/`, `tests/`

**Interfaces:**
- Produces: a repo where `ruff check src/ tests/` exits 0. All later tasks assume this.

- [ ] **Step 1: See the full ruff failure set**

Run: `python -m ruff check src/ tests/`
Expected: ~40 errors (F401 unused imports, etc.).

- [ ] **Step 2: Auto-fix the safe ones**

Run: `python -m ruff check --fix src/ tests/`
Expected: 38 fixed, ~2 remain.

- [ ] **Step 3: Hand-fix the remainder**

For `tests/test_sgd.py:17` remove `MiniBatchGD, StochasticGD` from the import if still unused, or add a use. For any remaining hit, resolve it directly (do not add blanket `# noqa`).

- [ ] **Step 4: Verify clean**

Run: `python -m ruff check src/ tests/`
Expected: `All checks passed!`

- [ ] **Step 5: Verify tests still pass**

Run: `python -m pytest tests/ -q`
Expected: `68 passed`.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "style: fix all ruff lint errors so CI lint gate passes"
```

---

### Task 2: Fix author identity everywhere

**Files:**
- Modify: `README.md` (author line + both GitHub URLs)
- Modify: `pyproject.toml:12` (author name)
- Modify: `app/app.py:260` (GitHub link in the app header)

**Interfaces:**
- Produces: zero occurrences of `siddharth` or `Siddharth` in the repo.

- [ ] **Step 1: Find every wrong-identity occurrence**

Run: `git grep -in "siddharth"`
Expected: hits in `README.md`, `app/app.py`.

- [ ] **Step 2: Replace name and URLs**

- `README.md`: `**Author:** ... Siddharth Bokka` → `Shivani Bokka`; `github.com/siddharth-bokka/Gradient-Descent-and-Optimizers` → `github.com/shiva-shivanibokka/Gradient-Descent-and-Optimizers` (clone command + any prose link).
- `app/app.py:260`: same URL replacement.
- `pyproject.toml:12`: `authors = [{ name = "Shivani Bokka" }]`.

- [ ] **Step 3: Verify no wrong identity remains**

Run: `git grep -in "siddharth"`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add README.md app/app.py pyproject.toml
git commit -m "docs: correct author name and GitHub URLs to Shivani Bokka / shiva-shivanibokka"
```

---

### Task 3: Fix the broken WARMUP_COSINE scheduler (TDD)

**Problem:** `_build_torch_scheduler` builds warmup_cosine with `T_max = tc.epochs - sc.warmup_steps` and `milestones=[sc.warmup_steps]`. `warmup_steps` defaults to 500 while `epochs` defaults to 20 → `T_max = -480` (invalid) and the warmup→cosine handoff never fires within the run. The scheduler steps **per epoch** in the Trainer, so warmup must be expressed in epochs.

**Files:**
- Modify: `src/gdo/training/trainer.py:588-593` (`_build_torch_scheduler`, WARMUP_COSINE branch)
- Test: `tests/test_schedulers.py` (add a test that steps the built scheduler across epochs and asserts a valid warmup-then-decay LR curve)

**Interfaces:**
- Consumes: `ExperimentConfig`, `torch.optim.Optimizer`, `steps_per_epoch: int` (existing signature of `_build_torch_scheduler`).
- Produces: for warmup_cosine, an LR sequence that (a) starts below base LR, (b) rises for `warmup_epochs` epochs, (c) then decays, with all `T_max`/`milestones` values ≥ 1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schedulers.py
def test_warmup_cosine_produces_valid_curve():
    import torch
    from gdo.config import ExperimentConfig, OptimizerConfig, SchedulerConfig, TrainConfig
    from gdo.config import OptimizerName, SchedulerName
    from gdo.training.trainer import _build_torch_scheduler

    cfg = ExperimentConfig(
        optimizer=OptimizerConfig(name=OptimizerName.ADAM, lr=0.01),
        scheduler=SchedulerConfig(name=SchedulerName.WARMUP_COSINE, warmup_steps=500),
        train=TrainConfig(epochs=20),
    )
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    sched = _build_torch_scheduler(cfg, opt, steps_per_epoch=700)

    lrs = []
    for _ in range(cfg.train.epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert all(lr > 0 for lr in lrs)          # no invalid/NaN LR
    assert lrs[0] < max(lrs)                    # warmup rises
    assert lrs[-1] < max(lrs)                   # then decays
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schedulers.py::test_warmup_cosine_produces_valid_curve -v`
Expected: FAIL (ValueError on negative T_max, or assertion failure).

- [ ] **Step 3: Fix the branch to work in epoch units**

```python
    elif sc.name == SchedulerName.WARMUP_COSINE:
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # scheduler.step() is called once per EPOCH by the Trainer, so warmup
        # must be measured in epochs. Derive it from warmup_steps, clamped so
        # both phases are always >= 1 epoch.
        warmup_epochs = max(1, min(sc.warmup_steps // max(steps_per_epoch, 1), tc.epochs - 1))
        cosine_epochs = max(1, tc.epochs - warmup_epochs)
        warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=sc.eta_min)
        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
```

- [ ] **Step 4: Run the new test + the whole scheduler suite**

Run: `python -m pytest tests/test_schedulers.py -v`
Expected: all pass, including the new test.

- [ ] **Step 5: Commit**

```bash
git add src/gdo/training/trainer.py tests/test_schedulers.py
git commit -m "fix: WARMUP_COSINE scheduler used step-units for an epoch-stepped scheduler (negative T_max)"
```

---

### Task 4: Remove the dead MLflow logger + prove per-epoch logging works (TDD)

**Problem:** `Trainer.from_config` constructs an `ExperimentLogger` but never calls `.start()`, so every `self._logger.log_epoch(...)` inside `fit()` is a silent no-op (guarded by `not self._active`). Meanwhile `__main__` context-manages a *second* logger and logs everything via `log_run()`. The internal logger is dead code, and the per-epoch logging path is never exercised.

**Fix (root cause):** `__main__` stays the single MLflow owner via its context-managed logger + `log_run()`. `from_config` passes `experiment_logger=None` (delete the dead construction). Add a unit test that drives `fit()` with a fake *started* logger to lock in that the per-epoch wiring actually fires.

**Files:**
- Modify: `src/gdo/training/trainer.py:429-441` (`from_config`: drop the dead logger)
- Test: `tests/test_trainer_logging.py` (create)

**Interfaces:**
- Consumes: `Trainer.__init__(..., experiment_logger: object | None)` (existing).
- Produces: `Trainer.from_config(cfg)._logger is None`; `fit()` calls `log_epoch(epoch, dict)` once per epoch when given a truthy logger with `log_epoch`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trainer_logging.py
import torch
from torch.utils.data import DataLoader, TensorDataset

from gdo.config import TrainConfig
from gdo.training.trainer import Trainer


class _FakeLogger:
    def __init__(self):
        self.epochs = []

    def log_epoch(self, epoch, metrics):
        self.epochs.append(epoch)


def _tiny_loader():
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_fit_logs_every_epoch_to_active_logger():
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    fake = _FakeLogger()
    trainer = Trainer(
        model=model,
        optimizer=opt,
        train_loader=_tiny_loader(),
        val_loader=_tiny_loader(),
        config=TrainConfig(epochs=3, seed=0),
        experiment_logger=fake,
    )
    trainer.fit()
    assert fake.epochs == [0, 1, 2]
```

- [ ] **Step 2: Run test to verify it passes already** (this path already works when a logger is passed in — the test locks it in)

Run: `python -m pytest tests/test_trainer_logging.py -v`
Expected: PASS. (If it fails, the per-epoch wiring is broken — fix `fit()` before proceeding.)

- [ ] **Step 3: Delete the dead logger construction in `from_config`**

In `src/gdo/training/trainer.py`, replace:

```python
        # Build experiment logger
        exp_logger = ExperimentLogger(cfg.mlflow) if cfg.mlflow.enabled else None
```

with:

```python
        # MLflow logging is owned by the caller (see __main__), which context-manages
        # a single ExperimentLogger and calls log_run(). Creating one here without
        # start()ing it would make every per-epoch log a silent no-op.
        exp_logger = None
```

Remove the now-unused `ExperimentLogger` import from the local imports in `from_config` if it is no longer referenced there.

- [ ] **Step 4: Run full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass (69 now).

- [ ] **Step 5: Commit**

```bash
git add src/gdo/training/trainer.py tests/test_trainer_logging.py
git commit -m "fix: remove never-started MLflow logger in from_config that silently dropped per-epoch logs"
```

---

### Task 5: Delete dead code + fix CLI docstring

**Files:**
- Modify: `src/gdo/config.py:228-231` (remove no-op `sync_scheduler_total_steps` validator)
- Modify: `src/gdo/__main__.py:8-10` (docstring says `python -m gdo.train`; correct to `python -m gdo`)

**Interfaces:**
- Produces: no behavioral change; `ExperimentConfig` still validates identically.

- [ ] **Step 1: Remove the no-op validator**

Delete the `@model_validator(mode="after")` `sync_scheduler_total_steps` method (it only does `return self`). If `model_validator` is now unused in the file, drop it from the import on `config.py:23`.

- [ ] **Step 2: Fix the CLI docstring**

In `src/gdo/__main__.py`, change the three usage lines from `python -m gdo.train --config ...` to `python -m gdo --config ...`.

- [ ] **Step 3: Verify config + lint still green**

Run: `python -m pytest tests/ -q && python -m ruff check src/ tests/`
Expected: tests pass, `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add src/gdo/config.py src/gdo/__main__.py
git commit -m "chore: remove no-op scheduler validator and fix CLI usage docstring"
```

---

### Task 6: Make mypy a real CI gate

**Problem:** CI runs mypy with `continue-on-error: true` — type errors are reported but never block. "Type checked" is currently cosmetic.

**Files:**
- Modify: `.github/workflows/ci.yml:41-44` (remove `continue-on-error`)
- Modify: `pyproject.toml:53-56` (`[tool.mypy]`) — relax only where third-party stubs are the blocker, not the project's own code
- Possibly modify: source files with cheap, real type fixes

**Interfaces:**
- Produces: `mypy src/gdo/` exits 0, and CI fails if it ever doesn't.

- [ ] **Step 1: Measure the actual error count**

Run: `python -m mypy src/gdo/ --ignore-missing-imports`
Note the count. (mypy imports torch, so this is slow — allow a few minutes.)

- [ ] **Step 2: Fix the cheap, legitimate errors**

Add missing return-type / parameter annotations, fix genuinely wrong types. Do **not** silence real bugs with `# type: ignore`.

- [ ] **Step 3: Scope strictness so third-party gaps don't block**

If `strict = true` produces unfixable noise from torch/mlflow internals, keep `ignore_missing_imports = true` and, if needed, narrow with per-module overrides rather than disabling checking globally, e.g.:

```toml
[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = ["mlflow.*", "torchvision.*"]
ignore_errors = true
```

- [ ] **Step 4: Confirm mypy is clean**

Run: `python -m mypy src/gdo/`
Expected: `Success: no issues found`.

- [ ] **Step 5: Remove `continue-on-error` from CI**

In `.github/workflows/ci.yml`, delete the `continue-on-error: true` line (and its comment) under the mypy step.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml src/gdo/
git commit -m "ci: make mypy a blocking gate and fix type errors"
```

---

### Task 7: Enforce coverage as a gate + CI status badge

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]` addopts — add `--cov-fail-under`)
- Modify: `.github/workflows/ci.yml` (coverage already computed; ensure the fail-under applies)
- Modify: `README.md` (add CI status badge at top)

**Interfaces:**
- Produces: CI fails if coverage drops below the chosen floor; README shows live CI status.

- [ ] **Step 1: Measure current coverage**

Run: `python -m pytest tests/ -q --cov=gdo --cov-report=term-missing`
Note the `TOTAL` percentage.

- [ ] **Step 2: Set the floor just below current**

In `pyproject.toml`, extend addopts (pick a round number a few points below the measured TOTAL, e.g. if TOTAL is 88% set 85):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --cov=gdo --cov-report=term-missing --cov-fail-under=85"
```

- [ ] **Step 3: Verify the gate holds**

Run: `python -m pytest tests/ -q`
Expected: passes and prints coverage; exit code 0.

- [ ] **Step 4: Add the CI badge to README**

At the very top of `README.md`, under the title:

```markdown
[![CI](https://github.com/shiva-shivanibokka/Gradient-Descent-and-Optimizers/actions/workflows/ci.yml/badge.svg)](https://github.com/shiva-shivanibokka/Gradient-Descent-and-Optimizers/actions/workflows/ci.yml)
```

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml README.md
git commit -m "ci: enforce coverage floor and add CI status badge to README"
```

---

### Task 8: Add the standard Python-shop dev tooling (pre-commit, Makefile, Dependabot)

**Why:** These are the things a reviewer at a real company expects to see and takes as a signal of engineering maturity. All free, all config-only, no runtime deps.

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `Makefile`
- Create: `.github/dependabot.yml`
- Modify: `requirements-dev.txt` (add `pre-commit`)
- Modify: `README.md` (document `make` targets + pre-commit setup in Quick Start)

**Interfaces:**
- Produces: `pre-commit run --all-files` passes; `make test`, `make lint`, `make format`, `make type`, `make app` work.

- [ ] **Step 1: pre-commit config (pins mirror the versions already in dev extra)**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

- [ ] **Step 2: Makefile**

```makefile
.PHONY: install test lint format type app clean

install:
	pip install -e ".[dev,app]"
	pre-commit install

test:
	pytest tests/ -q

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/ app/

type:
	mypy src/gdo/

app:
	python app/app.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
```

- [ ] **Step 3: Dependabot**

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: pip
    directory: "/"
    schedule:
      interval: weekly
  - package-ecosystem: github-actions
    directory: "/"
    schedule:
      interval: weekly
```

- [ ] **Step 4: Add `pre-commit` to dev deps**

Append `pre-commit>=3.7` to `requirements-dev.txt` and to the `dev` extra in `pyproject.toml`.

- [ ] **Step 5: Install hooks and run them**

Run: `pip install pre-commit && pre-commit install && pre-commit run --all-files`
Expected: hooks run; any auto-fixes get applied. Re-run until green.

- [ ] **Step 6: Document in README Quick Start**

Add under Quick Start: `make install` (sets up deps + hooks), and a one-line mention that pre-commit runs ruff/format on every commit.

- [ ] **Step 7: Verify everything still green**

Run: `python -m pytest tests/ -q && python -m ruff check src/ tests/`
Expected: tests pass, lint clean.

- [ ] **Step 8: Commit**

```bash
git add .pre-commit-config.yaml Makefile .github/dependabot.yml requirements-dev.txt pyproject.toml README.md
git commit -m "chore: add pre-commit, Makefile, and Dependabot dev tooling"
```

---

### Task 9: Reconcile README claims + docker-compose resource limits

**Files:**
- Modify: `README.md` (soften/verify claims that are not yet true; add a short Architecture section pointer)
- Modify: `docker-compose.yml` (add mem/CPU limits to the `app` service)

**Interfaces:**
- Produces: a README where every stated fact is currently true, and a compose file with explicit resource limits.

- [ ] **Step 1: Truth-check README claims**

Verify each claim against the repo. Specifically: the "deployed as a live Gradio app on Hugging Face Spaces" line — if not yet deployed, change to "deployable to Hugging Face Spaces (see Deployment)" until Phase 4 makes it literally true. Confirm the AdamW test name referenced in README still exists (`git grep test_adamw_not_equal_adam_with_l2`).

- [ ] **Step 2: Add compose resource limits**

Under the `app` service in `docker-compose.yml`, add:

```yaml
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
```

- [ ] **Step 3: Verify compose parses**

Run: `docker compose config >/dev/null && echo OK`
Expected: `OK` (skip if Docker is unavailable locally; note it for CI/deploy phase).

- [ ] **Step 4: Commit**

```bash
git add README.md docker-compose.yml
git commit -m "docs: reconcile README claims with reality; add compose resource limits"
```

---

## Deferred to Phase 4 (deployment — separate checkpoint, not in this plan)

- Hugging Face Space config: HF Spaces needs a Space repo with a README metadata header (`sdk: gradio`, `app_file: app/app.py`) or a root `app.py`. Create the Space, push, verify the live URL, then flip the README deployment claim to present-tense. **Requires explicit go-ahead before pushing anything external.**

## Deliberately skipped (YAGNI — state, don't silently drop)

- **`.env.example`** — the app takes no secrets/env vars; a file documenting "nothing required" is noise. Add only if config ever grows secrets.
- **Full dependency lockfile / pinned torch** — torch pinning across CPU/GPU/OS wheels is brittle and painful for a demo; Dependabot + floor constraints are the pragmatic middle. Add a real lock (uv/pip-tools) only if build reproducibility ever bites.
- **Codecov / coverage-% badge** — needs a third-party account; `--cov-fail-under` gives the enforcement for free. The CI status badge covers the "is it green" signal.

---

## Self-Review

- **Spec coverage:** All 10 audit findings map to tasks — 1(ruff), 2(author), 3(WARMUP_COSINE), 4(dead logger), 5(dead validator + CLI docstring), 6(mypy gate), 7(coverage+badge), 9(README claims + compose limits). Findings 8 (coverage reported) → Task 7; 9 (.env) → explicitly skipped with reason; 10 (compose limits) → Task 9. Big-company layer → Task 8.
- **Placeholder scan:** No TBDs; every code step shows real code.
- **Type consistency:** `_build_torch_scheduler(cfg, optimizer, steps_per_epoch)` signature reused verbatim in Task 3; `Trainer(...experiment_logger=...)` matches existing `__init__`; `from_config` return type unchanged.
