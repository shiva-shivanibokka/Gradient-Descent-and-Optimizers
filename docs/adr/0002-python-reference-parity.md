# ADR 0002: Verify the TypeScript port against Python with a parity suite

- **Status:** Accepted
- **Date:** 2026-07-07

## Context

[ADR 0001](0001-client-side-web-app.md) reimplements the optimizer/scheduler/
surface math in TypeScript so it can run in the browser. Duplicated numerical
code is a classic source of silent drift: a subtly wrong Adam update in the TS
port would produce plausible-but-wrong trajectories that no one notices.

The Python `gdo` package is already unit-tested (it verifies, for example, that
AdamW applies decoupled weight decay rather than L2). We want that same
correctness guarantee to cover what actually runs in the browser.

## Decision

Treat the Python package as the **single source of truth** and hold the TS port
to it with an automated parity test:

1. `scripts/dump_reference.py` runs the real `gdo` optimizers/schedulers/surfaces
   and writes reference values to `web/tests/fixtures/reference.json`.
2. A **Vitest** suite in `web/tests/` asserts the TS output matches those fixtures
   to **1e-6** (surface value + gradient, optimizer trajectories, LR curves).
3. The web CI job runs the parity suite on every push.

Parity is scoped to the **deterministic** code paths. The three noise-based SGD
variants use NumPy's RNG, which cannot be reproduced bit-for-bit in TS, so they
are intentionally excluded and use seeded TS noise for cosmetic jitter only.

## Consequences

**Positive**

- The `Adam` update rule verified by `pytest` is provably the same one running in
  the browser — the duplication is safe, not a liability.
- Regressions in the TS math fail CI immediately.
- The fixtures are regenerable from Python, so the reference can't silently rot.

**Negative / mitigations**

- Fixtures must be regenerated when the Python math intentionally changes.
  → Accepted: it is one command (`python scripts/dump_reference.py`) and a
  changed fixture in a diff is a useful signal that behavior changed.
