# gdo-web (Vercel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace the Gradio app with a client-side Next.js + TypeScript optimizer playground, deployed on Vercel free tier, feature-parity with the current three tabs, dark technical-dashboard aesthetic, all math running in-browser and verified against the Python `gdo` package.

**Architecture:** New `web/` Next.js (App Router) app in the monorepo. Pure TS ports of surfaces/optimizers/schedulers/MLP in `web/lib/`. Loss surfaces rendered on Canvas; curves via Recharts. Vitest parity tests assert TS matches Python-dumped JSON fixtures to 6 decimals. No backend.

**Tech Stack:** Next.js 15, TypeScript, Tailwind CSS, shadcn/ui, Recharts, Vitest, Canvas 2D, Vercel.

## Global Constraints

- **All computation client-side.** No API routes, no serverless functions, no `"use server"`.
- **Free tooling only.** Vercel Hobby, no paid deps.
- **Author:** any author metadata uses `Shivani Bokka` / `shiva-shivanibokka`. Never `siddharth`.
- **Node package manager:** npm (lockfile committed).
- Every task ends green: `npm run lint && npm run typecheck && npm test` in `web/` pass before commit.
- TS math must match the Python `gdo` reference to `1e-6` on deterministic paths.

## File Structure

```
web/
├── app/
│   ├── layout.tsx            # dark theme shell, fonts
│   ├── page.tsx              # tabbed container
│   └── globals.css           # tailwind + theme tokens
├── components/
│   ├── tabs/LandscapeTab.tsx
│   ├── tabs/CompareTab.tsx
│   ├── tabs/SchedulerTab.tsx
│   ├── LossSurfaceCanvas.tsx # canvas contour + trajectory
│   ├── LineChart.tsx         # Recharts wrapper (loss/LR curves)
│   └── ui/                   # shadcn primitives
├── lib/
│   ├── surfaces.ts
│   ├── optimizers.ts
│   ├── schedulers.ts
│   ├── mlp.ts
│   └── types.ts
├── tests/
│   ├── fixtures/*.json       # dumped from Python
│   ├── surfaces.test.ts
│   ├── optimizers.test.ts
│   └── schedulers.test.ts
├── package.json
├── vitest.config.ts
└── next.config.ts
scripts/dump_reference.py     # Python → JSON fixtures (repo root, uses gdo)
```

---

### Task 1: Scaffold the Next.js app

**Files:** Create `web/` via create-next-app; add Vitest.

- [ ] **Step 1:** From repo root: `npx create-next-app@latest web --typescript --tailwind --app --eslint --src-dir=false --import-alias "@/*" --no-turbopack --use-npm --yes`
- [ ] **Step 2:** Add dev/test deps: `cd web && npm i recharts && npm i -D vitest @vitejs/plugin-react jsdom @testing-library/react`
- [ ] **Step 3:** Create `web/vitest.config.ts`:

```ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", globals: true },
});
```

- [ ] **Step 4:** Add scripts to `web/package.json`: `"typecheck": "tsc --noEmit"`, `"test": "vitest run"`.
- [ ] **Step 5:** Verify: `npm run build` succeeds, `npm run lint` clean.
- [ ] **Step 6:** Commit: `feat(web): scaffold Next.js + TS + Tailwind + Vitest app`.

---

### Task 2: Types + surfaces port + parity harness

**Files:** Create `web/lib/types.ts`, `web/lib/surfaces.ts`, `scripts/dump_reference.py`, `web/tests/fixtures/surfaces.json`, `web/tests/surfaces.test.ts`.

**Interfaces:**
- Produces: `type Vec = number[]`; `interface Surface { value(p: Vec): number; grad(p: Vec): Vec; domain: [number, number, number, number]; optimum: Vec }`; `const SURFACES: Record<string, Surface>`.

- [ ] **Step 1: Write `web/lib/types.ts`**

```ts
export type Vec = number[];

export interface Surface {
  name: string;
  value(p: Vec): number;
  grad(p: Vec): Vec;
  domain: [number, number, number, number]; // xmin, xmax, ymin, ymax
  optimum: Vec;
  start: Vec;
}
```

- [ ] **Step 2: Write `web/lib/surfaces.ts`** — Quadratic, Rosenbrock, Beale, Himmelblau with analytic gradients matching `src/gdo/landscapes/surfaces.py`. Rosenbrock example:

```ts
import type { Surface, Vec } from "./types";

const rosenbrock = (a = 1, b = 100): Surface => ({
  name: "Rosenbrock",
  value: ([x, y]) => (a - x) ** 2 + b * (y - x * x) ** 2,
  grad: ([x, y]) => [
    -2 * (a - x) - 4 * b * x * (y - x * x),
    2 * b * (y - x * x),
  ],
  domain: [-2, 2, -1, 3],
  optimum: [a, a * a],
  start: [-1.5, 2.5],
});
// ...quadratic, beale, himmelblau similarly (gradients ported from surfaces.py)

export const SURFACES: Record<string, Surface> = { /* ... */ };
```

- [ ] **Step 3: Write `scripts/dump_reference.py`** — import `gdo.landscapes.surfaces`, evaluate value+grad at a fixed grid of test points, write `web/tests/fixtures/surfaces.json` as `{surface: {points: [[x,y]...], values: [...], grads: [[gx,gy]...]}}`.
- [ ] **Step 4: Run it:** `python scripts/dump_reference.py`. Expected: fixture file written.
- [ ] **Step 5: Write `web/tests/surfaces.test.ts`** — load fixture, for each surface+point assert `Math.abs(ts - py) < 1e-6` for value and both grad components.
- [ ] **Step 6: Run:** `npm test`. Fix TS until parity holds.
- [ ] **Step 7: Commit:** `feat(web): port loss surfaces to TS with Python parity tests`.

---

### Task 3: Optimizers port + parity

**Files:** `web/lib/optimizers.ts`, extend `scripts/dump_reference.py`, `web/tests/fixtures/trajectories.json`, `web/tests/optimizers.test.ts`.

**Interfaces:**
- Produces: `interface Optimizer { step(p: Vec, g: Vec): Vec }`; factory `makeOptimizer(name, opts): Optimizer`; helper `descend(surface, optName, lr, steps, opts): Vec[]` returning the trajectory.

- [ ] **Step 1:** Write `web/lib/optimizers.ts` porting `src/gdo/optimizers/{sgd,adaptive}.py`: BatchGD, SGD, Momentum, RMSProp, Adam, AdamW, Lion. Each keeps internal state (velocity, m/v moments, step count). Adam template:

```ts
export function adam(lr: number, b1 = 0.9, b2 = 0.999, eps = 1e-8): Optimizer {
  let m: Vec = [], v: Vec = [], t = 0;
  return {
    step(p, g) {
      if (!m.length) { m = p.map(() => 0); v = p.map(() => 0); }
      t++;
      return p.map((pi, i) => {
        m[i] = b1 * m[i] + (1 - b1) * g[i];
        v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
        const mh = m[i] / (1 - b1 ** t), vh = v[i] / (1 - b2 ** t);
        return pi - (lr * mh) / (Math.sqrt(vh) + eps);
      });
    },
  };
}
```

- [ ] **Step 2:** Add `descend(surface, optName, lr, steps, opts)` iterating `p = opt.step(p, surface.grad(p))`.
- [ ] **Step 3:** Extend `dump_reference.py` to run each NumPy `gdo` optimizer on each surface for N steps (fixed seed/start) and dump full trajectories to `trajectories.json`.
- [ ] **Step 4:** Write `optimizers.test.ts` asserting TS trajectory matches Python to `1e-6` per step.
- [ ] **Step 5:** Run `npm test`; fix until parity. **Commit:** `feat(web): port optimizers to TS with trajectory parity tests`.

---

### Task 4: Schedulers port + parity

**Files:** `web/lib/schedulers.ts`, extend dump script, `web/tests/fixtures/schedules.json`, `web/tests/schedulers.test.ts`.

**Interfaces:** Produces `lrCurve(name, baseLr, epochs, opts): number[]`.

- [ ] **Step 1:** Port `src/gdo/optimizers/schedulers.py` LR-curve generation to TS (Step, Cosine, OneCycle, Cyclical, Warmup+Cosine, ReduceOnPlateau-with-fixed-metric-trace-or-omit).
- [ ] **Step 2:** Extend dump script → `schedules.json`.
- [ ] **Step 3:** `schedulers.test.ts` parity to `1e-6`.
- [ ] **Step 4:** Run, fix, **commit:** `feat(web): port LR schedulers to TS with parity tests`.

---

### Task 5: Spiral MLP (unit-tested for convergence, not parity)

**Files:** `web/lib/mlp.ts`, `web/tests/mlp.test.ts`.

**Interfaces:** Produces `makeSpiral(n, seed): {X: Vec[], y: number[]}`; `class MLP { forward(x); trainStep(X, y, opt): loss }`.

- [ ] **Step 1:** Seeded RNG (mulberry32), 2-class spiral generator, 2-layer MLP (hidden ReLU, softmax + cross-entropy) with hand-coded backprop; accepts an `Optimizer` from `optimizers.ts`.
- [ ] **Step 2:** `mlp.test.ts`: training Adam for ~300 steps drives loss below its initial value (assert monotone-ish decrease / final < 0.5 * initial). No Python parity (stochastic).
- [ ] **Step 3:** Run, **commit:** `feat(web): add spiral dataset + 2-layer MLP with backprop`.

---

### Task 6: Loss-surface Canvas component

**Files:** `web/components/LossSurfaceCanvas.tsx`.

- [ ] **Step 1:** Component takes `surface`, `trajectories: {label, color, path: Vec[]}[]`. Render: sample Z over a grid → map to dark colormap → draw heatmap to an offscreen canvas → draw glowing trajectory polylines + start/end markers. Map domain coords → pixels.
- [ ] **Step 2:** Manual verify in a scratch route that a Rosenbrock + Adam trajectory renders. (No unit test — visual.)
- [ ] **Step 3:** **Commit:** `feat(web): canvas loss-surface renderer with trajectory overlay`.

---

### Task 7: App shell + dark theme + shadcn primitives

**Files:** `web/app/layout.tsx`, `web/app/page.tsx`, `web/app/globals.css`, `web/components/ui/*`.

- [ ] **Step 1:** Use the frontend-design skill for the dark technical-dashboard look (tokens, type scale, spacing). Add shadcn primitives: tabs, select, slider, checkbox, button.
- [ ] **Step 2:** `page.tsx` renders a header + three-tab container (empty tab bodies for now).
- [ ] **Step 3:** `npm run build` clean. **Commit:** `feat(web): app shell, dark dashboard theme, shadcn primitives`.

---

### Task 8: Landscape tab

**Files:** `web/components/tabs/LandscapeTab.tsx`.

- [ ] **Step 1:** Controls: surface select, optimizer select, LR slider, steps slider → single trajectory via `descend()` → `LossSurfaceCanvas`. Plus multi-optimizer checkbox group → multiple colored trajectories on a shared surface.
- [ ] **Step 2:** Verify interactions in dev. **Commit:** `feat(web): loss-landscape tab`.

---

### Task 9: Optimizer Comparison tab

**Files:** `web/components/tabs/CompareTab.tsx`, `web/components/LineChart.tsx`.

- [ ] **Step 1:** Recharts dark-themed `LineChart`. Controls: two optimizer selects, LR, steps → train spiral MLP with each (same seed) → overlay loss curves.
- [ ] **Step 2:** Verify. **Commit:** `feat(web): optimizer comparison tab with loss curves`.

---

### Task 10: Scheduler Explorer tab

**Files:** `web/components/tabs/SchedulerTab.tsx`.

- [ ] **Step 1:** Multi-select schedulers, base-LR + epochs inputs → `lrCurve()` per scheduler → Recharts overlay.
- [ ] **Step 2:** Verify. **Commit:** `feat(web): scheduler explorer tab`.

---

### Task 11: Remove Gradio + repoint Docker + update docs

**Files:** delete `app/`; modify `pyproject.toml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `README.md`.

- [ ] **Step 1:** `git rm -r app/`. Remove `gradio` from `requirements.txt` and the `app` extra in `pyproject.toml`.
- [ ] **Step 2:** Dockerfile: drop `COPY app/`, change `CMD` to run the training CLI (e.g. `["python", "-m", "gdo", "--help"]` as default) or an entrypoint note; healthcheck stays `import gdo`. docker-compose: drop the gradio port/service specifics, keep MLflow.
- [ ] **Step 3:** README: replace Gradio references with the Vercel web app (architecture, quick start `cd web && npm run dev`, stack table Web = Next.js/TS, deployment = Vercel). Keep Python usage.
- [ ] **Step 4:** `pytest tests/ -q` still green (Python untouched). **Commit:** `refactor: remove Gradio app in favor of the Vercel web app`.

---

### Task 12: CI + Dependabot for web

**Files:** `.github/workflows/ci.yml`, `.github/dependabot.yml`.

- [ ] **Step 1:** Add a `web` job to `ci.yml`: `runs-on: ubuntu-latest`, `setup-node@v4` (node 20, cache npm at `web/package-lock.json`), `working-directory: web`, run `npm ci`, `npm run lint`, `npm run typecheck`, `npm test`, `npm run build`.
- [ ] **Step 2:** Add npm ecosystem (`directory: /web`) to `dependabot.yml`.
- [ ] **Step 3:** **Commit:** `ci: build, lint, typecheck, and test the web app`.

---

### Task 13: Deploy to Vercel (CHECKPOINT — explicit confirmation required)

- [ ] **Step 1:** Confirm with user before any external deploy action.
- [ ] **Step 2:** Deploy via the Vercel MCP / CLI with project root `web/`, framework Next.js. No env vars.
- [ ] **Step 3:** Verify the live URL loads all three tabs and a trajectory renders.
- [ ] **Step 4:** Update README with the live URL. **Commit:** `docs: add live Vercel URL`.

---

## Self-Review

- **Spec coverage:** surfaces/optimizers/schedulers/MLP ports (T2–5), parity harness (T2–4), canvas + Recharts (T6, T9), three tabs (T8–10), theme (T7), Gradio removal (T11), CI (T12), deploy (T13). All spec sections covered.
- **Placeholder scan:** UI tasks intentionally describe components rather than full JSX (built with frontend-design at execution); algorithmic primitives include real code.
- **Type consistency:** `Vec`, `Surface`, `Optimizer`, `descend()`, `lrCurve()` used consistently across tasks.
