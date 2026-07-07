// The signature element: a loss surface rendered as a topographic instrument
// readout, with optimizer trajectories that ANIMATE — tracing out step by step
// so you watch the optimizers descend. Dark outlines + white-cored heads keep
// every trajectory color legible over any part of the surface.

"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { colormap } from "@/lib/palette";
import type { Surface, Vec } from "@/lib/types";

export interface Trajectory {
  label: string;
  color: string;
  path: Vec[];
}

const RES = 520; // heatmap grid + canvas logical size
const MARGIN = 14; // px beyond the frame before a diverging path is clipped

export function LossSurfaceCanvas({
  surface,
  trajectories,
}: {
  surface: Surface;
  trajectories: Trajectory[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number | undefined>(undefined);
  const [replay, setReplay] = useState(0);

  // Heatmap RGBA — recomputed only when the surface changes. Plain typed array
  // (server-safe); the ImageData wrapper is built in the browser-only effect.
  const rgba = useMemo(() => {
    const [xMin, xMax, yMin, yMax] = surface.domain;
    const vals = new Float64Array(RES * RES);
    let lo = Infinity;
    let hi = -Infinity;
    for (let j = 0; j < RES; j++) {
      const y = yMax - (j / (RES - 1)) * (yMax - yMin);
      for (let i = 0; i < RES; i++) {
        const x = xMin + (i / (RES - 1)) * (xMax - xMin);
        const v = Math.log1p(Math.max(0, surface.value([x, y])));
        vals[j * RES + i] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const out = new Uint8ClampedArray(RES * RES * 4);
    const span = hi - lo || 1;
    for (let k = 0; k < vals.length; k++) {
      const [r, g, b] = colormap(1 - (vals[k] - lo) / span); // low loss = brighter
      out[k * 4] = r;
      out[k * 4 + 1] = g;
      out[k * 4 + 2] = b;
      out[k * 4 + 3] = 255;
    }
    return out;
  }, [surface]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const [xMin, xMax, yMin, yMax] = surface.domain;
    const toPx = ([x, y]: Vec): [number, number] => [
      ((x - xMin) / (xMax - xMin)) * RES,
      (1 - (y - yMin) / (yMax - yMin)) * RES,
    ];
    const inFrame = ([px, py]: [number, number]) =>
      px >= -MARGIN && px <= RES + MARGIN && py >= -MARGIN && py <= RES + MARGIN;

    // Precompute clipped pixel paths once per change.
    const paths = trajectories
      .map((t) => {
        const pts: [number, number][] = [];
        for (const p of t.path) {
          const px = toPx(p);
          pts.push(px);
          if (!inFrame(px)) break; // keep the first escaping point, then stop
        }
        return { color: t.color, pts };
      })
      .filter((p) => p.pts.length >= 2);

    const heat = new ImageData(rgba, RES, RES);
    const [ox, oy] = toPx(surface.optimum);
    const maxLen = Math.max(2, ...paths.map((p) => p.pts.length));
    const durationMs = Math.min(2200, Math.max(700, maxLen * 6));
    const reduceMotion =
      typeof window !== "undefined" &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    let start = 0;

    const drawFrame = (progress: number) => {
      ctx.putImageData(heat, 0, 0);

      // Optimum crosshair.
      ctx.strokeStyle = "rgba(240,244,250,0.85)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ox - 6, oy);
      ctx.lineTo(ox + 6, oy);
      ctx.moveTo(ox, oy - 6);
      ctx.lineTo(ox, oy + 6);
      ctx.stroke();

      for (const { color, pts } of paths) {
        const head = Math.max(1, Math.floor(progress * (pts.length - 1)));
        const shown = pts.slice(0, head + 1);

        // Dark outline first → keeps the bright line legible over light regions.
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.strokeStyle = "rgba(0,0,0,0.55)";
        ctx.lineWidth = 4.5;
        ctx.beginPath();
        shown.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
        ctx.stroke();

        // Glowing colored line.
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        shown.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
        ctx.stroke();
        ctx.restore();

        // Start ring.
        const [sx, sy] = pts[0];
        ctx.strokeStyle = "rgba(255,255,255,0.8)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.stroke();

        // Head marker with white core → visible on any background.
        const [hx, hy] = shown[shown.length - 1];
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(hx, hy, 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(hx, hy, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    if (reduceMotion || paths.length === 0) {
      drawFrame(1);
      return;
    }

    const step = (ts: number) => {
      if (!start) start = ts;
      const progress = Math.min(1, (ts - start) / durationMs);
      drawFrame(progress);
      if (progress < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [surface, rgba, trajectories, replay]);

  return (
    <div className="relative w-full">
      <canvas
        ref={canvasRef}
        width={RES}
        height={RES}
        className="aspect-square w-full rounded-xl border border-border-bright shadow-[0_0_0_1px_var(--border-bright),0_0_30px_-8px_var(--accent)]"
      />

      {trajectories.length > 0 && (
        <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1.5 rounded-lg border border-border-bright bg-bg/80 px-3 py-2 backdrop-blur-md">
          {trajectories.map((t) => (
            <div key={t.label} className="flex items-center gap-2 text-xs">
              <span
                className="h-2.5 w-4 rounded-full"
                style={{ background: t.color, boxShadow: `0 0 8px ${t.color}` }}
              />
              <span className="text-fg">{t.label}</span>
            </div>
          ))}
        </div>
      )}

      <button
        type="button"
        onClick={() => setReplay((r) => r + 1)}
        className="absolute bottom-3 right-3 flex items-center gap-1.5 rounded-lg border border-border-bright bg-bg/80 px-3 py-1.5 text-xs text-fg backdrop-blur-md transition-colors hover:border-accent hover:text-accent"
      >
        ▸ Replay
      </button>
    </div>
  );
}
