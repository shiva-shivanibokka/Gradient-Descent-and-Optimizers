// 2D analytical loss surfaces, ported from src/gdo/landscapes/surfaces.py.
// Constructor defaults match the Python defaults exactly so the parity
// fixtures (dumped from Python) agree to ~1e-6.

import type { Surface } from "./types";

function quadratic(a = 1.0, b = 10.0): Surface {
  return {
    name: "Quadratic (ill-conditioned)",
    value: ([x, y]) => a * x ** 2 + b * y ** 2,
    grad: ([x, y]) => [2.0 * a * x, 2.0 * b * y],
    domain: [-3, 3, -3, 3],
    optimum: [0, 0],
    start: [-2.5, 2.5],
  };
}

function rosenbrock(a = 1.0, b = 100.0): Surface {
  return {
    name: "Rosenbrock",
    value: ([x, y]) => (a - x) ** 2 + b * (y - x * x) ** 2,
    grad: ([x, y]) => [
      -2.0 * (a - x) - 4.0 * b * x * (y - x * x),
      2.0 * b * (y - x * x),
    ],
    domain: [-2, 2, -1, 3],
    optimum: [a, a * a],
    start: [-1.5, 2.5],
  };
}

function beale(): Surface {
  return {
    name: "Beale",
    value: ([x, y]) =>
      (1.5 - x + x * y) ** 2 +
      (2.25 - x + x * y ** 2) ** 2 +
      (2.625 - x + x * y ** 3) ** 2,
    grad: ([x, y]) => {
      const t1 = 1.5 - x + x * y;
      const t2 = 2.25 - x + x * y ** 2;
      const t3 = 2.625 - x + x * y ** 3;
      return [
        2 * t1 * (-1 + y) + 2 * t2 * (-1 + y ** 2) + 2 * t3 * (-1 + y ** 3),
        2 * t1 * x + 2 * t2 * (2 * x * y) + 2 * t3 * (3 * x * y ** 2),
      ];
    },
    domain: [-4.5, 4.5, -4.5, 4.5],
    optimum: [3.0, 0.5],
    start: [-3.5, -3.5],
  };
}

function himmelblau(): Surface {
  return {
    name: "Himmelblau",
    value: ([x, y]) => (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2,
    grad: ([x, y]) => [
      4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7),
      2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7),
    ],
    domain: [-5, 5, -5, 5],
    optimum: [3.0, 2.0],
    start: [-4.0, -4.0],
  };
}

// Keyed by the same slugs used in the Python dump script.
export const SURFACES: Record<string, Surface> = {
  quadratic: quadratic(),
  rosenbrock: rosenbrock(),
  beale: beale(),
  himmelblau: himmelblau(),
};

// Display-name → slug, for UI dropdowns.
export const SURFACE_LABELS: Record<string, string> = {
  "Quadratic (ill-conditioned)": "quadratic",
  Rosenbrock: "rosenbrock",
  Beale: "beale",
  Himmelblau: "himmelblau",
};
