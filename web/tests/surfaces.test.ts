import { describe, expect, it } from "vitest";

import { SURFACES } from "@/lib/surfaces";
import ref from "./fixtures/reference.json";

// Parity: TS surface value + gradient must match the Python `gdo` package to 1e-6.
describe("surfaces parity vs Python gdo", () => {
  for (const [key, data] of Object.entries(ref.surfaces)) {
    it(`${key}: value + gradient match at every test point`, () => {
      const surface = SURFACES[key];
      expect(surface).toBeDefined();
      data.points.forEach((p, i) => {
        expect(surface.value(p)).toBeCloseTo(data.values[i], 6);
        const g = surface.grad(p);
        expect(g[0]).toBeCloseTo(data.grads[i][0], 6);
        expect(g[1]).toBeCloseTo(data.grads[i][1], 6);
      });
    });
  }
});
