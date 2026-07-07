// Tiny seeded RNG (mulberry32) + Box–Muller Gaussian.
// Used only for the *cosmetic* noise in the SGD-family optimizers and for the
// spiral dataset. Not used by the deterministic optimizers that are parity-tested.

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Returns a function drawing standard-normal samples from a seeded stream. */
export function gaussian(seed: number): () => number {
  const rand = mulberry32(seed);
  return function () {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };
}
