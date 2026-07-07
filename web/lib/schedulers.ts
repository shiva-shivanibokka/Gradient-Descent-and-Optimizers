// LR schedulers ported from src/gdo/optimizers/schedulers.py.
// lrCurve() replays the schedule over `epochs` (calling the per-epoch rule for
// e = 0..epochs-1), exactly like the Python get_lr_curve(). Parity-tested.
// ReduceLROnPlateau is intentionally excluded (it is metric-driven, not a pure
// function of epoch).

export interface SchedOpts {
  stepSize?: number;
  gamma?: number;
  tMax?: number;
  etaMin?: number;
  maxLr?: number;
  totalEpochs?: number;
  pctStart?: number;
  stepSizeUp?: number;
  mode?: "triangular" | "triangular2";
  warmupSteps?: number;
}

export function lrCurve(slug: string, baseLr: number, epochs: number, o: SchedOpts = {}): number[] {
  const etaMin = o.etaMin ?? 0.0;
  const total = o.totalEpochs ?? epochs;
  const curve: number[] = [];

  for (let e = 0; e < epochs; e++) {
    let lr: number;
    switch (slug) {
      case "none": {
        lr = baseLr;
        break;
      }
      case "step": {
        const stepSize = o.stepSize ?? 10;
        const gamma = o.gamma ?? 0.5;
        lr = baseLr * gamma ** Math.floor(e / stepSize);
        break;
      }
      case "cosine": {
        const tMax = o.tMax ?? epochs;
        const t = e % tMax;
        lr = etaMin + 0.5 * (baseLr - etaMin) * (1 + Math.cos((Math.PI * t) / tMax));
        break;
      }
      case "onecycle": {
        const maxLr = o.maxLr ?? baseLr * 10;
        const pctStart = o.pctStart ?? 0.3;
        const warmup = Math.floor(pctStart * total);
        const finalLr = baseLr / 1e4;
        if (e <= warmup) {
          lr = baseLr + ((maxLr - baseLr) * e) / Math.max(1, warmup);
        } else {
          const progress = (e - warmup) / Math.max(1, total - warmup);
          lr = finalLr + 0.5 * (maxLr - finalLr) * (1 + Math.cos(Math.PI * progress));
        }
        break;
      }
      case "cyclical": {
        const maxLr = o.maxLr ?? baseLr * 10;
        const stepUp = o.stepSizeUp ?? 5;
        const mode = o.mode ?? "triangular";
        const cycleLength = 2 * stepUp;
        const cycle = Math.floor(1 + e / cycleLength);
        const x = Math.abs(e / stepUp - 2 * cycle + 1);
        const scale = mode === "triangular" ? 1.0 : 0.5 ** (cycle - 1);
        lr = baseLr + (maxLr - baseLr) * Math.max(0.0, 1.0 - x) * scale;
        break;
      }
      case "warmup_cosine": {
        const warmup = o.warmupSteps ?? 5;
        if (e < warmup) {
          lr = (baseLr * (e + 1)) / warmup;
        } else {
          const progress = (e - warmup) / Math.max(1, total - warmup);
          lr = etaMin + 0.5 * (baseLr - etaMin) * (1 + Math.cos(Math.PI * progress));
        }
        break;
      }
      default:
        throw new Error(`Unknown scheduler: ${slug}`);
    }
    curve.push(lr);
  }
  return curve;
}

// Display-name → slug for UI (mirrors the Gradio SCHEDULER_MAP).
export const SCHEDULER_LABELS: Record<string, string> = {
  "None (constant)": "none",
  StepLR: "step",
  CosineAnnealingLR: "cosine",
  OneCycleLR: "onecycle",
  CyclicalLR: "cyclical",
  "Warmup + Cosine": "warmup_cosine",
};
