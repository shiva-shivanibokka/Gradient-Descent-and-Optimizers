import { describe, expect, it } from "vitest";

import { descend } from "@/lib/optimizers";
import { SURFACES } from "@/lib/surfaces";
import ref from "./fixtures/reference.json";

// Parity: TS optimizer trajectories must match the Python `gdo` package step
// for step, to 1e-6, on the deterministic (noise-free) optimizers.
describe("optimizer trajectories parity vs Python gdo", () => {
  for (const c of ref.trajectories) {
    it(`${c.optimizer} on ${c.surface} (lr=${c.lr}, ${c.steps} steps)`, () => {
      const surface = SURFACES[c.surface];
      const path = descend(surface, c.optimizer, c.lr, c.steps, c.start);
      expect(path.length).toBe(c.path.length);
      path.forEach((pt, i) => {
        expect(pt[0]).toBeCloseTo(c.path[i][0], 6);
        expect(pt[1]).toBeCloseTo(c.path[i][1], 6);
      });
    });
  }
});
