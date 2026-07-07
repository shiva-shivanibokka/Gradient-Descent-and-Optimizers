// A tiny 2-layer MLP with hand-coded backprop, trained on a 2-class spiral.
// Powers the "Optimizer Comparison" tab: same net, same data, same seed — only
// the optimizer differs. Uses the ported optimizers from ./optimizers.
//
// Not parity-tested against Python (the Python comparison uses PyTorch with a
// different RNG); instead unit-tested for convergence.

import { makeOptimizer, type OptOpts } from "./optimizers";
import { mulberry32 } from "./rng";
import type { Vec } from "./types";

export interface Dataset {
  X: Vec[]; // [n][2]
  y: number[]; // class 0/1
}

/** Deterministic 2-class spiral. */
export function makeSpiral(nPerClass = 100, seed = 0): Dataset {
  const rand = mulberry32(seed);
  const X: Vec[] = [];
  const y: number[] = [];
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < nPerClass; i++) {
      const r = i / nPerClass;
      const t = c * Math.PI + 4 * r + (rand() - 0.5) * 0.4;
      X.push([r * Math.sin(t), r * Math.cos(t)]);
      y.push(c);
    }
  }
  return { X, y };
}

const IN = 2;
const OUT = 2;

/** 2-layer MLP: input(2) → ReLU hidden(H) → logits(2), softmax + cross-entropy. */
export class MLP {
  readonly hidden: number;
  private params: Vec; // flat: [W1(H*2), b1(H), W2(2*H), b2(2)]
  private readonly oW1 = 0;
  private readonly oB1: number;
  private readonly oW2: number;
  private readonly oB2: number;

  constructor(hidden = 16, seed = 0) {
    this.hidden = hidden;
    this.oB1 = this.oW1 + hidden * IN;
    this.oW2 = this.oB1 + hidden;
    this.oB2 = this.oW2 + OUT * hidden;
    const size = this.oB2 + OUT;
    const rand = mulberry32(seed + 1);
    // Small random init (He-ish for the hidden layer).
    this.params = new Array(size).fill(0).map((_, i) => {
      if (i < this.oB1) return (rand() * 2 - 1) * Math.sqrt(2 / IN);
      if (i >= this.oW2 && i < this.oB2) return (rand() * 2 - 1) * Math.sqrt(2 / hidden);
      return 0;
    });
  }

  getParams(): Vec {
    return [...this.params];
  }

  /** Forward pass for one sample → class probabilities. */
  predict(x: Vec): Vec {
    const { probs } = this.forwardOne(x, this.params);
    return probs;
  }

  private forwardOne(x: Vec, p: Vec) {
    const H = this.hidden;
    const z1 = new Array(H).fill(0);
    const a1 = new Array(H).fill(0);
    for (let j = 0; j < H; j++) {
      let s = p[this.oB1 + j];
      for (let k = 0; k < IN; k++) s += p[this.oW1 + j * IN + k] * x[k];
      z1[j] = s;
      a1[j] = Math.max(0, s);
    }
    const logits = new Array(OUT).fill(0);
    for (let c = 0; c < OUT; c++) {
      let s = p[this.oB2 + c];
      for (let j = 0; j < H; j++) s += p[this.oW2 + c * H + j] * a1[j];
      logits[c] = s;
    }
    const m = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sum);
    return { z1, a1, probs };
  }

  /** Mean cross-entropy loss + flat gradient over the whole dataset. */
  lossAndGrad(data: Dataset): { loss: number; grad: Vec } {
    const H = this.hidden;
    const grad = new Array(this.params.length).fill(0);
    let loss = 0;
    const n = data.X.length;
    for (let s = 0; s < n; s++) {
      const x = data.X[s];
      const label = data.y[s];
      const { z1, a1, probs } = this.forwardOne(x, this.params);
      loss += -Math.log(Math.max(probs[label], 1e-12));

      const dz2 = probs.map((pc, c) => pc - (c === label ? 1 : 0));
      const da1 = new Array(H).fill(0);
      for (let c = 0; c < OUT; c++) {
        for (let j = 0; j < H; j++) {
          grad[this.oW2 + c * H + j] += dz2[c] * a1[j];
          da1[j] += this.params[this.oW2 + c * H + j] * dz2[c];
        }
        grad[this.oB2 + c] += dz2[c];
      }
      for (let j = 0; j < H; j++) {
        const dz1 = z1[j] > 0 ? da1[j] : 0;
        for (let k = 0; k < IN; k++) grad[this.oW1 + j * IN + k] += dz1 * x[k];
        grad[this.oB1 + j] += dz1;
      }
    }
    for (let i = 0; i < grad.length; i++) grad[i] /= n;
    return { loss: loss / n, grad };
  }

  /** One full-batch optimizer step; returns the loss before the update. */
  trainStep(data: Dataset, opt: ReturnType<typeof makeOptimizer>): number {
    const { loss, grad } = this.lossAndGrad(data);
    this.params = opt.step(this.params, grad);
    return loss;
  }
}

/** Train the spiral MLP with one optimizer; returns the loss curve. */
export function trainCurve(
  optSlug: string,
  lr: number,
  steps: number,
  opts: { hidden?: number; seed?: number; nPerClass?: number; optOpts?: OptOpts } = {},
): number[] {
  const seed = opts.seed ?? 0;
  const data = makeSpiral(opts.nPerClass ?? 100, seed);
  const net = new MLP(opts.hidden ?? 16, seed);
  const opt = makeOptimizer(optSlug, lr, opts.optOpts ?? {});
  const losses: number[] = [];
  for (let i = 0; i < steps; i++) losses.push(net.trainStep(data, opt));
  return losses;
}
