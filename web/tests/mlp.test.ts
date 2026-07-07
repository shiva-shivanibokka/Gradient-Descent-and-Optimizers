import { describe, expect, it } from "vitest";

import { makeSpiral, MLP, trainCurve } from "@/lib/mlp";

describe("spiral MLP", () => {
  it("generates a balanced 2-class spiral deterministically", () => {
    const a = makeSpiral(50, 7);
    const b = makeSpiral(50, 7);
    expect(a.X.length).toBe(100);
    expect(a.y.filter((c) => c === 0).length).toBe(50);
    expect(a.X).toEqual(b.X); // same seed → identical data
  });

  it("predict returns a valid probability distribution", () => {
    const net = new MLP(16, 0);
    const p = net.predict([0.3, -0.2]);
    expect(p.length).toBe(2);
    expect(p[0] + p[1]).toBeCloseTo(1, 6);
    expect(p.every((v) => v >= 0)).toBe(true);
  });

  it("Adam drives the loss well below its starting value", () => {
    const losses = trainCurve("adam", 0.05, 300, { seed: 0 });
    expect(losses[losses.length - 1]).toBeLessThan(0.5 * losses[0]);
  });
});
