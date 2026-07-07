# ADR 0001: Client-side TypeScript web app over a Python API backend

- **Status:** Accepted
- **Date:** 2026-07-07

## Context

The project needed an interactive web demo of the optimizers, schedulers, and
loss landscapes. The reference implementation is the Python `gdo` package
(NumPy + PyTorch). Two obvious ways to put it on the web:

1. **Python API backend** — expose `gdo` behind FastAPI, call it from a frontend.
2. **Client-side port** — reimplement the (small, pure) math in TypeScript and run
   it entirely in the browser.

Constraints that mattered:

- **Free hosting only.** PyTorch is ~800 MB and does not fit in a Vercel
  serverless function (~250 MB cap). A free Python API host (e.g. Hugging Face
  Spaces) **sleeps and cold-starts 30–60 s** on the first request.
- The demo is the first thing a reviewer clicks. It must load **instantly, every
  time**, with no cold start.
- The math the demo needs is lightweight: 2D optimizer trajectories, LR curves,
  and a tiny spiral MLP — all trivially portable.

## Decision

Build the demo as a **client-side Next.js + TypeScript app** with all
computation in the browser. Port the surfaces/optimizers/schedulers/MLP to TS.
Keep the Python `gdo` package as the tested reference implementation.

## Consequences

**Positive**

- Static site → always-on, instant load, zero cold start, free forever, nothing
  to monitor or keep running.
- Deployable to any static host, not just Vercel (de-risks hosting).
- Full-stack breadth in one repo: a tested Python library *and* a React frontend.

**Negative / mitigations**

- The math is now implemented twice (Python + TS) and could drift.
  → **Mitigation:** a Vitest parity suite asserts the TS output matches the
  Python reference to 1e-6 on the deterministic paths (see [ADR 0002](0002-python-reference-parity.md)).
- No live server means no server-side model training in the demo.
  → Accepted: heavy training stays in the Python CLI, which is the right home for it.

## Alternatives considered

- **FastAPI on Hugging Face Spaces** — rejected: cold starts make the portfolio
  link look broken on first load, and it adds a second host to maintain.
- **Pyodide (run the Python in-browser via WebAssembly)** — rejected: multi-MB
  wasm payload hurts load time, and PyTorch is not available in Pyodide.
