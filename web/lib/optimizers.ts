// NumPy optimizers ported to TS, from src/gdo/optimizers/{sgd,adaptive}.py.
// Each factory returns a stateful Optimizer whose step() matches the Python
// update rule. The five deterministic optimizers (batch_gd, rmsprop, adam,
// adamw, lion) are parity-tested against Python; the noise-based SGD family
// uses a seeded TS Gaussian for cosmetic jitter (not parity-tested).

import { gaussian } from "./rng";
import type { Optimizer, Surface, Vec } from "./types";

export interface OptOpts {
  beta1?: number;
  beta2?: number;
  epsilon?: number;
  alpha?: number;
  weightDecay?: number;
  momentum?: number;
  noiseScale?: number;
  seed?: number;
}

const zeros = (n: number): Vec => new Array(n).fill(0);

function batchGD(lr: number): Optimizer {
  return { step: (p, g) => p.map((pi, i) => pi - lr * g[i]) };
}

function noisySGD(lr: number, noiseScale: number, seed: number): Optimizer {
  const noise = gaussian(seed);
  return {
    step: (p, g) => p.map((pi, i) => pi - lr * (g[i] + noiseScale * noise())),
  };
}

function momentumSGD(lr: number, momentum: number, noiseScale: number, seed: number): Optimizer {
  const noise = gaussian(seed);
  let v: Vec = [];
  return {
    step(p, g) {
      if (!v.length) v = zeros(p.length);
      v = p.map((_, i) => momentum * v[i] - lr * (g[i] + noiseScale * noise()));
      return p.map((pi, i) => pi + v[i]);
    },
  };
}

function rmsprop(lr: number, alpha: number, eps: number): Optimizer {
  let v: Vec = [];
  return {
    step(p, g) {
      if (!v.length) v = zeros(p.length);
      v = p.map((_, i) => alpha * v[i] + (1 - alpha) * g[i] * g[i]);
      return p.map((pi, i) => pi - (lr / Math.sqrt(v[i] + eps)) * g[i]);
    },
  };
}

function adam(lr: number, b1: number, b2: number, eps: number): Optimizer {
  let m: Vec = [];
  let v: Vec = [];
  let t = 0;
  return {
    step(p, g) {
      if (!m.length) {
        m = zeros(p.length);
        v = zeros(p.length);
      }
      t += 1;
      m = p.map((_, i) => b1 * m[i] + (1 - b1) * g[i]);
      v = p.map((_, i) => b2 * v[i] + (1 - b2) * g[i] * g[i]);
      return p.map((pi, i) => {
        const mHat = m[i] / (1 - b1 ** t);
        const vHat = v[i] / (1 - b2 ** t);
        return pi - (lr * mHat) / (Math.sqrt(vHat) + eps);
      });
    },
  };
}

function adamw(lr: number, b1: number, b2: number, eps: number, wd: number): Optimizer {
  let m: Vec = [];
  let v: Vec = [];
  let t = 0;
  return {
    step(p, g) {
      if (!m.length) {
        m = zeros(p.length);
        v = zeros(p.length);
      }
      t += 1;
      m = p.map((_, i) => b1 * m[i] + (1 - b1) * g[i]);
      v = p.map((_, i) => b2 * v[i] + (1 - b2) * g[i] * g[i]);
      return p.map((pi, i) => {
        const mHat = m[i] / (1 - b1 ** t);
        const vHat = v[i] / (1 - b2 ** t);
        const pWd = (1 - lr * wd) * pi;
        return pWd - (lr * mHat) / (Math.sqrt(vHat) + eps);
      });
    },
  };
}

function lion(lr: number, b1: number, b2: number, wd: number): Optimizer {
  let m: Vec = [];
  return {
    step(p, g) {
      if (!m.length) m = zeros(p.length);
      const c = p.map((_, i) => b1 * m[i] + (1 - b1) * g[i]);
      const next = p.map((pi, i) => (1 - lr * wd) * pi - lr * Math.sign(c[i]));
      m = p.map((_, i) => b2 * m[i] + (1 - b2) * g[i]);
      return next;
    },
  };
}

/** Build an optimizer by slug (matches the Python dump script keys). */
export function makeOptimizer(slug: string, lr: number, o: OptOpts = {}): Optimizer {
  const seed = o.seed ?? 42;
  switch (slug) {
    case "batch_gd":
      return batchGD(lr);
    case "sgd":
      return noisySGD(lr, o.noiseScale ?? 0.1, seed);
    case "mini_batch_gd":
      return noisySGD(lr, o.noiseScale ?? 0.05, seed);
    case "momentum_sgd":
      return momentumSGD(lr, o.momentum ?? 0.9, o.noiseScale ?? 0.05, seed);
    case "rmsprop":
      return rmsprop(lr, o.alpha ?? 0.99, o.epsilon ?? 1e-8);
    case "adam":
      return adam(lr, o.beta1 ?? 0.9, o.beta2 ?? 0.999, o.epsilon ?? 1e-8);
    case "adamw":
      return adamw(lr, o.beta1 ?? 0.9, o.beta2 ?? 0.999, o.epsilon ?? 1e-8, o.weightDecay ?? 0.01);
    case "lion":
      return lion(lr, o.beta1 ?? 0.9, o.beta2 ?? 0.99, o.weightDecay ?? 0.0);
    default:
      throw new Error(`Unknown optimizer: ${slug}`);
  }
}

/** Run an optimizer on a surface from `start`, returning the trajectory [start, ...points]. */
export function descend(
  surface: Surface,
  slug: string,
  lr: number,
  steps: number,
  start: Vec = surface.start,
  o: OptOpts = {},
): Vec[] {
  const opt = makeOptimizer(slug, lr, o);
  let p = [...start];
  const path: Vec[] = [[...p]];
  for (let i = 0; i < steps; i++) {
    p = opt.step(p, surface.grad(p));
    path.push([...p]);
  }
  return path;
}

// Display-name → slug for the UI dropdowns (mirrors the Gradio OPTIMIZER_MAP).
export const OPTIMIZER_LABELS: Record<string, string> = {
  "Batch GD": "batch_gd",
  SGD: "sgd",
  "Mini-Batch GD": "mini_batch_gd",
  "SGD + Momentum": "momentum_sgd",
  RMSProp: "rmsprop",
  Adam: "adam",
  AdamW: "adamw",
  Lion: "lion",
};
