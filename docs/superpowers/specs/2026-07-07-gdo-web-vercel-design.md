# gdo-web — Client-Side Optimizer Playground on Vercel

**Date:** 2026-07-07
**Status:** Approved

## Goal

Replace the Gradio app with a production-grade, good-looking **Next.js + TypeScript** web app,
deployed on **Vercel** (free Hobby tier), where **all computation runs in the browser** — no
backend, no API, no serverless functions. Feature-parity with the current three-tab Gradio demo,
but a dark technical-dashboard aesthetic and always-instant load.

## Why client-side (not a Python API)

PyTorch (~800 MB) cannot fit in Vercel serverless functions (~250 MB cap), and a free Python API
host (HF Spaces) sleeps and cold-starts 30–60 s — fatal for a portfolio link that must load
instantly on first click. All the demo math is lightweight (2D optimizer trajectories, LR curves,
a tiny spiral MLP), so porting it to TypeScript gives a static, always-on, free-forever, zero-ops
site. The Python `gdo` package remains as the tested reference implementation (retains Docker/CLI
as backend evidence).

## Architecture & Stack

- **Framework:** Next.js (App Router) + TypeScript, in a new `web/` directory (monorepo alongside
  `src/gdo/`). Vercel project root = `web/`. Auto-deploy on push; PR preview URLs.
- **Styling:** Tailwind CSS + shadcn/ui primitives (button, slider, tabs, select, checkbox).
  Dark theme: near-black background, single accent color, monospace numerics, thin gridlines.
- **Loss-surface rendering:** custom HTML Canvas — compute Z over a grid, map to a dark colormap,
  overlay glowing optimizer trajectories.
- **Line charts** (loss curves, LR schedules): Recharts, themed for dark mode.

## TypeScript port (`web/lib/`)

Pure, dependency-free functions mirroring `src/gdo/`:

- `surfaces.ts` — Quadratic (ill-conditioned), Rosenbrock, Beale, Himmelblau: value + analytic gradient.
- `optimizers.ts` — BatchGD, SGD, Momentum, RMSProp, Adam, AdamW, Lion: step functions over a
  parameter vector (used both for 2D landscape points and for MLP params).
- `schedulers.ts` — Step, Cosine, OneCycle, Cyclical, Warmup+Cosine, ReduceOnPlateau: LR curves.
- `mlp.ts` — 2-layer MLP with hand-coded forward/backward; deterministic seeded 2-class spiral
  dataset generator.

## Correctness / parity

A Python script (`scripts/dump_reference.py`) dumps reference trajectories and curves from the
actual `gdo` package to JSON fixtures under `web/tests/fixtures/`. A Vitest parity test asserts the
TS output matches the Python reference to ~6 decimal places, for the **deterministic** pieces
(optimizers on surfaces, scheduler curves). The stochastic MLP tab is **not** parity-tested — stated
honestly. This proves the in-browser math is the same math verified by the Python test suite.

## UI — three tabs (feature parity with current Gradio app)

1. **Loss Landscape** — surface + optimizer dropdowns, LR + steps sliders, single trajectory;
   plus a multi-optimizer "race" (checkbox group) on a shared surface.
2. **Optimizer Comparison** — pick two optimizers, train the tiny spiral MLP with each (same seed),
   live loss curves.
3. **Scheduler Explorer** — multi-select schedulers, base-LR + epochs inputs, LR-curve comparison.

## Repo changes

- **Remove** the Gradio app (`app/`), the `gradio` dependency, and the `app` optional-dependency
  extra. Repoint the Dockerfile `CMD` and `docker-compose` to the training CLI (Docker stays as
  backend/training evidence). Update README (architecture, quick start, stack table, deploy section).
- **CI:** add a Node job (install, lint, typecheck, Vitest) for `web/`, alongside the existing
  Python job. Add a `web/` Dependabot npm ecosystem entry.

## Deployment

- Vercel free Hobby tier, GitHub-connected, project root `web/`. Framework preset: Next.js.
- Build: `next build`. No environment variables, no secrets (nothing to configure).
- Executing the actual Vercel deploy requires explicit user confirmation (external action).

## Deliberately skipped (YAGNI)

- No backend, API, auth, DB, or state persistence.
- No SSR/data-fetching — it is compute-in-browser; pages are effectively static.
- No in-browser MNIST/CIFAR (too heavy) — those stay CLI-only in the Python package.
- No parity test for the stochastic MLP tab.

## Open items to revisit after deploy (user's call)

- Gradio removal is final per this spec (can be restored from git history if ever wanted).
- Recharts vs hand-rolled SVG for line charts — locked to Recharts; revisit if bundle size matters.
