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

// Loss-surface colormap: deep navy → teal → pale mint. t in [0, 1].
const STOPS: [number, [number, number, number]][] = [
  [0.0, [10, 13, 19]], // bg navy
  [0.35, [18, 46, 58]], // deep teal
  [0.7, [30, 120, 120]], // teal
  [1.0, [150, 230, 210]], // pale mint
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
