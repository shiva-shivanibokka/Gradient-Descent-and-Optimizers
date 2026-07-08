# Repo Audit Report — Gradient-Descent-and-Optimizers

**Date:** 2026-07-07
**Stack detected:** Python 3.11 (NumPy / PyTorch / Pydantic / MLflow) package `gdo` + TypeScript/React (Next.js 16, Recharts, Vitest) web app in `web/`
**Scope:** Full `src/gdo/**`, `web/lib/**`, `web/components/**`, `tests/**`, `configs/**`, CI, Dockerfile. Excluded: `web/node_modules`, generated `__pycache__`, notebooks (glue not executed — dataset download hangs offline; the `gdo` code they call is covered).

## Summary

- Total findings: **6** (1 Major, 2 Minor, 3 Notes)
- Critical: 0 | Major: 1 | Minor: 2 | Notes: 3
- **Status: all 3 bugs (1 Major + 2 Minor) FIXED and verified — 2026-07-07.** See "Resolution" below. The 3 Notes are maintainability observations, left as-is by design.

## Resolution (2026-07-07)

All actionable findings were fixed test-first (write the failing test → fix → green):

| Finding | Fix | Verifying test |
|---|---|---|
| 1 · OneCycleLR frozen (Major) | `trainer.py`: build with `total_steps=tc.epochs` instead of `steps_per_epoch × epochs` | `test_training.py::TestTrainerFactories::test_onecycle_schedule_runs` |
| 2 · WarmupScheduler no validation (Minor) | `schedulers.py`: raise `ValueError` on `warmup_steps <= 0` / `total_epochs <= 0` | `test_schedulers.py::TestWarmupScheduler::test_rejects_nonpositive_{warmup,total_epochs}` |
| 3 · Scheduler test asserts existence only (Minor) | Added the behavioral OneCycle test above | (same) |

Full suite after fixes: **107 Python tests pass, 18 web tests pass.**

The core math is clean. The optimizer update rules (both the NumPy `gdo` versions and the TS ports) match their papers and each other; the surface gradients are correct; the schedulers' pure-function curves are correct. The one real bug is in the **PyTorch training path**, not the web demo or the optimizer math: `OneCycleLR` is built for per-batch stepping but the Trainer steps it per-epoch, so its schedule silently never runs.

## Production-readiness scorecard

| Category | Status | Notes |
|---|---|---|
| Correctness | ⚠️ | Optimizer/surface/scheduler-curve math correct. One scheduler wired wrong in the torch training path (Finding 1). |
| Silent failures | ⚠️ | Finding 1 fails silently (no error, just a dead schedule). MLflow wrapper's broad `except`/log-and-continue is intentional and appropriate for tracking-server outages. |
| Security | ✅ | No secrets, no injection surface. Web app is static/client-side with no user-supplied strings; Python is a local CLI. Nothing to flag. |
| Concurrency | ✅ | No threads/async/shared mutable state. Each optimizer/scheduler owns its state; DataLoader workers are stdlib-managed. |
| Performance | ✅ | Appropriate for scale. Full-batch NumPy loops and the 520² heatmap are recomputed only on input change (memoized). No N+1, no unbounded growth. |
| Architecture | ✅ | Clean layering: config → model/optimizer/scheduler → trainer → logger. `get_grad_norms`/`get_total_grad_norm` are duplicated across MLP and CNN (Note 1). |
| Production-readiness | ⚠️ | Dockerfile has a healthcheck but runs as root (Note 2). `WarmupScheduler` missing the input validation its siblings all have (Finding 2). |
| Test coverage | ⚠️ | 81% line coverage, but the scheduler tests only assert `is not None` — they never *step* a scheduler, which is exactly why Finding 1 slipped through (Finding 3). |

(✅ no issues found · ⚠️ minor/notes only or one localized issue · ❌ at least one systemic major/critical)

## Auto-fixed (trivial-safe)

None. No finding qualified for silent auto-fix — every real issue changes behavior or adds code, so all went to PLAN.md for review.

## Findings requiring review

### Correctness

**`src/gdo/training/trainer.py:578-585` — Major — OneCycleLR is stepped once per epoch but built for per-batch stepping**

`_build_torch_scheduler` constructs `torch.optim.lr_scheduler.OneCycleLR(..., steps_per_epoch=len(train_loader), epochs=tc.epochs)`. PyTorch derives `total_steps = epochs × steps_per_epoch` internally. But `Trainer.fit()` calls `self.scheduler.step()` **once per epoch** (line 226), so the scheduler only advances `epochs` steps out of `epochs × steps_per_epoch`.

*Why it matters in production:* Running `python -m gdo --config <cfg with scheduler.name: onecycle>` trains at a nearly-frozen LR pinned to OneCycle's low initial value (`max_lr / div_factor`). The warmup-then-anneal that OneCycle exists to provide never happens — the run looks fine, logs fine, and produces quietly worse results. It fails silently. Note the `warmup_cosine` branch right below it (line 594) was already rewritten to step per-epoch; OneCycle wasn't — the inconsistency is the tell. Suggested fix: build it with `total_steps=tc.epochs` (one full cycle per epoch-step). Full patch in PLAN.md Task 1.

### Missing code

**`src/gdo/optimizers/schedulers.py:353-383` — Minor — `WarmupScheduler` accepts `warmup_steps=0` and then divides by it**

Every other scheduler in this file validates its constructor args (`StepLR` checks `step_size > 0`, `CosineAnnealingLR` checks `t_max > 0`, `OneCycleLR`/`CyclicalLR` check `max_lr`). `WarmupScheduler.__init__` validates nothing, and `step()` computes `base_lr * (e + 1) / warmup_steps`. `SchedulerConfig.warmup_steps` is `ge=0`, so a config with `warmup_steps: 0` is valid and yields a `ZeroDivisionError` on the first epoch of `get_lr_curve()` / notebook use.

*Why it matters in production:* An inconsistency a reader can't predict — the same "0 warmup" input that's harmless for the torch path (which clamps via `max(1, ...)`) crashes the pure-Python scheduler. Suggested fix: validate `warmup_steps > 0` and `total_epochs > 0` in `__init__`, matching the sibling classes. Full patch in PLAN.md Task 2.

### Test coverage gaps

**`tests/test_training.py:179-188` — Minor — scheduler test asserts existence, not behavior**

`test_build_scheduler` only checks `sched is not None`. It never calls `sched.step()` across epochs and never inspects the resulting LR trajectory, so a scheduler that builds successfully but produces the wrong curve (exactly Finding 1) passes. Suggested fix: add a test that steps OneCycle `epochs` times and asserts the LR actually rises to near `max_lr` and back down. Full spec in PLAN.md Task 3 (write it before applying Task 1's fix).

## Clean areas (verified, no findings)

- **Optimizer update rules** — `src/gdo/optimizers/{sgd,adaptive}.py` and their TS ports in `web/lib/optimizers.ts`: bias correction, epsilon placement, AdamW's decoupled decay, and Lion's sign/momentum ordering all match the papers and match each other (and are parity-tested to 1e-6).
- **Loss surfaces** — `web/lib/surfaces.ts` and `src/gdo/landscapes/surfaces.py`: gradients are analytically correct (previously spot-checked against finite differences ~1e-8); Himmelblau's four minima are right.
- **Scheduler curves (pure functions)** — `web/lib/schedulers.ts` matches `schedulers.py` epoch-by-epoch for the six curve types; boundary handling (`e <= warmup`) is consistent across both.
- **Web app** — `LineChart` handles the empty-series case (`Math.max(0, ...[])`), canvas trajectory clipping and `prefers-reduced-motion` are handled, and there is no user-input trust boundary.
- **Config / security** — Pydantic validates every field with sane bounds; no hardcoded secrets anywhere; MLflow tracking URI is externalized.

## Notes (maintainability — not bugs)

1. **Duplicate grad-norm methods** — `get_grad_norms`/`get_total_grad_norm` are byte-identical in `MLP` and `CNN` (`models.py`). A shared mixin would DRY it, but it's a behavior-neutral refactor, low priority.
2. **Dockerfile runs as root** — no `USER` directive. Fine for a throwaway training container; add a non-root user if it's ever exposed.
3. **`Trainer.fit_streaming` has no caller** and diverges from `fit` (ignores `eval_every` and `on_epoch_end`). It's public API kept for a future streaming UI; left in place, flagged so the divergence is known.
