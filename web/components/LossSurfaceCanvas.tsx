// The signature element: a loss surface rendered as a topographic instrument
// readout (log-scaled heatmap) with luminous optimizer trajectories traced on it.

"use client";

import { useEffect, useMemo, useRef } from "react";

import { colormap } from "@/lib/palette";
import type { Surface, Vec } from "@/lib/types";

export interface Trajectory {
  label: string;
  color: string;
  path: Vec[];
}

const RES = 460; // heatmap grid + canvas logical size

export function LossSurfaceCanvas({
  surface,
  trajectories,
}: {
  surface: Surface;
  trajectories: Trajectory[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Compute the heatmap RGBA bytes when the surface changes. Kept as a plain
  // Uint8ClampedArray (server-safe) — the ImageData wrapper is built in the
  // effect below, since ImageData is a browser-only API.
  const rgba = useMemo(() => {
    const [xMin, xMax, yMin, yMax] = surface.domain;
    const vals = new Float64Array(RES * RES);
    let lo = Infinity;
    let hi = -Infinity;
    for (let j = 0; j < RES; j++) {
      const y = yMax - (j / (RES - 1)) * (yMax - yMin); // top = yMax
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
      const t = 1 - (vals[k] - lo) / span; // low loss = bright
      const [r, g, b] = colormap(t);
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

    ctx.putImageData(new ImageData(rgba, RES, RES), 0, 0);

    // Optimum marker.
    const [ox, oy] = toPx(surface.optimum);
    ctx.strokeStyle = "rgba(230,234,240,0.7)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ox - 5, oy);
    ctx.lineTo(ox + 5, oy);
    ctx.moveTo(ox, oy - 5);
    ctx.lineTo(ox, oy + 5);
    ctx.stroke();

    // Trajectories with glow. A diverging path is clipped at the point it
    // first leaves the frame, so divergence reads as "escaped" not as a glitch.
    const M = 12; // px margin beyond the canvas before we stop drawing
    const inFrame = ([px, py]: [number, number]) =>
      px >= -M && px <= RES + M && py >= -M && py <= RES + M;

    for (const traj of trajectories) {
      if (traj.path.length < 2) continue;
      const pts: [number, number][] = [];
      for (const p of traj.path) {
        const px = toPx(p);
        pts.push(px);
        if (!inFrame(px)) break; // keep the first out-of-frame point, then stop
      }
      if (pts.length < 2) continue;

      ctx.save();
      ctx.shadowColor = traj.color;
      ctx.shadowBlur = 8;
      ctx.strokeStyle = traj.color;
      ctx.lineWidth = 1.75;
      ctx.lineJoin = "round";
      ctx.beginPath();
      pts.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.stroke();
      ctx.restore();

      // Start (hollow) + end-of-drawn-path (filled) markers.
      const [sx, sy] = pts[0];
      const [ex, ey] = pts[pts.length - 1];
      ctx.strokeStyle = traj.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = traj.color;
      ctx.beginPath();
      ctx.arc(ex, ey, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [surface, rgba, trajectories]);

  return (
    <div className="relative w-full">
      <canvas
        ref={canvasRef}
        width={RES}
        height={RES}
        className="aspect-square w-full rounded-lg border border-border"
      />
      {trajectories.length > 0 && (
        <div className="pointer-events-none absolute left-3 top-3 flex flex-col gap-1 rounded-md border border-border bg-bg/70 px-2.5 py-2 backdrop-blur-sm">
          {trajectories.map((t) => (
            <div key={t.label} className="flex items-center gap-2 text-xs">
              <span
                className="h-2 w-4 rounded-full"
                style={{ background: t.color, boxShadow: `0 0 6px ${t.color}` }}
              />
              <span className="text-fg">{t.label}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
