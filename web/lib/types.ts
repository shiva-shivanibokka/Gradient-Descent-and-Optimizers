// Core numeric types shared across the optimizer/landscape math.
// Mirrors the interfaces of the Python `gdo` package so trajectories match.

export type Vec = number[];

export interface Surface {
  name: string;
  /** Loss value at a 2D point. */
  value(p: Vec): number;
  /** Analytical gradient [∂f/∂x, ∂f/∂y] at a 2D point. */
  grad(p: Vec): Vec;
  /** Recommended axis limits: [xMin, xMax, yMin, yMax]. */
  domain: [number, number, number, number];
  /** Known global minimum (marked with a crosshair). */
  optimum: Vec;
  /** All global minima, if there are several (e.g. Himmelblau). Defaults to [optimum]. */
  minima?: Vec[];
  /** Default starting point for trajectory visualization. */
  start: Vec;
}

export interface Optimizer {
  /** Apply one update: returns new params given current params + gradient. */
  step(params: Vec, grad: Vec): Vec;
}
