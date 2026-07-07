// Categorical series colors (colorblind-aware, tuned for the blue-black bg) and
// the loss-surface colormap. Comparing multiple optimizers is the core job, so
// the series palette is a first-class part of the design.

export const SERIES = [
  "#38e1c6", // teal
  "#f5b544", // amber
  "#e85d9c", // magenta
  "#7be06b", // green
  "#ff6b5a", // coral
  "#9b8cff", // violet
  "#5aa9ff", // blue
  "#ff9fb2", // rose
];

export function seriesColor(i: number): string {
  return SERIES[i % SERIES.length];
}

// Loss-surface colormap: a desaturated slate-blue ramp. Kept low-saturation on
// purpose so every saturated trajectory color (teal, violet, green, coral, …)
// stays legible against it — the surface is context, the trajectories are data.
// Vibrancy comes from the page glows, the bright basin, and the animation.
// t in [0, 1].
const STOPS: [number, [number, number, number]][] = [
  [0.0, [10, 12, 20]], // near-black (high loss)
  [0.4, [26, 32, 50]], // dark slate
  [0.7, [48, 60, 90]], // slate
  [0.88, [82, 100, 142]], // light slate-blue
  [1.0, [134, 154, 198]], // bright basin (low loss)
];

export function colormap(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  for (let i = 1; i < STOPS.length; i++) {
    if (x <= STOPS[i][0]) {
      const [t0, c0] = STOPS[i - 1];
      const [t1, c1] = STOPS[i];
      const f = (x - t0) / (t1 - t0);
      return [
        Math.round(c0[0] + f * (c1[0] - c0[0])),
        Math.round(c0[1] + f * (c1[1] - c0[1])),
        Math.round(c0[2] + f * (c1[2] - c0[2])),
      ];
    }
  }
  return STOPS[STOPS.length - 1][1];
}
