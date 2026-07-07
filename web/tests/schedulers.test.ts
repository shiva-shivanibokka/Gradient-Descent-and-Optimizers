import { describe, expect, it } from "vitest";

import { lrCurve, type SchedOpts } from "@/lib/schedulers";
import ref from "./fixtures/reference.json";

// The same ctor options used by scripts/dump_reference.py, keyed by scheduler.
const OPTS: Record<string, SchedOpts> = {
  step: { stepSize: 10, gamma: 0.5 },
  cosine: { tMax: 50 },
  onecycle: { maxLr: 0.1, totalEpochs: 50, pctStart: 0.3 },
  cyclical: { maxLr: 0.05, stepSizeUp: 5, mode: "triangular" },
  warmup_cosine: { warmupSteps: 5, totalEpochs: 50 },
};

// Parity: TS LR curves must match the Python `gdo` schedulers to 1e-6.
describe("scheduler LR curves parity vs Python gdo", () => {
  for (const c of ref.schedules) {
    it(`${c.scheduler} (baseLr=${c.base_lr}, ${c.epochs} epochs)`, () => {
      const curve = lrCurve(c.scheduler, c.base_lr, c.epochs, OPTS[c.scheduler]);
      expect(curve.length).toBe(c.curve.length);
      curve.forEach((lr, i) => expect(lr).toBeCloseTo(c.curve[i], 6));
    });
  }
});
