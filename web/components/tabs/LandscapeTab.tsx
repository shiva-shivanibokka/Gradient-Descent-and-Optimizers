"use client";

import { useMemo, useState } from "react";

import { LossSurfaceCanvas, type Trajectory } from "@/components/LossSurfaceCanvas";
import { CheckList, Field, Panel, Select, Slider } from "@/components/ui";
import { descend, OPTIMIZER_LABELS } from "@/lib/optimizers";
import { seriesColor } from "@/lib/palette";
import { SURFACE_LABELS, SURFACES } from "@/lib/surfaces";

const SURFACE_NAMES = Object.keys(SURFACE_LABELS);
const OPT_NAMES = Object.keys(OPTIMIZER_LABELS);

// Per-surface defaults tuned so the shown optimizers actually reach the marked
// minimum (verified against the Python gdo package). Stiff surfaces (Rosenbrock,
// Beale) default to adaptive optimizers, since plain GD diverges on them.
const SURFACE_DEFAULTS: Record<string, { lr: number; steps: number; opts: string[] }> = {
  quadratic: { lr: 0.03, steps: 250, opts: ["Batch GD", "SGD + Momentum", "Adam"] },
  rosenbrock: { lr: 0.04, steps: 3000, opts: ["Adam", "RMSProp", "AdamW"] },
  beale: { lr: 0.08, steps: 2500, opts: ["Adam", "RMSProp", "AdamW"] },
  himmelblau: { lr: 0.05, steps: 900, opts: ["Adam", "RMSProp", "AdamW"] },
};

export function LandscapeTab() {
  const [surfaceName, setSurfaceName] = useState("Quadratic (ill-conditioned)");
  const [selected, setSelected] = useState<string[]>(SURFACE_DEFAULTS.quadratic.opts);
  const [lr, setLr] = useState(SURFACE_DEFAULTS.quadratic.lr);
  const [steps, setSteps] = useState(SURFACE_DEFAULTS.quadratic.steps);

  const surface = SURFACES[SURFACE_LABELS[surfaceName]];

  const colorOf = (name: string) => seriesColor(OPT_NAMES.indexOf(name));

  // Switching surface loads a config that converges on it (user can still tweak).
  const changeSurface = (name: string) => {
    setSurfaceName(name);
    const d = SURFACE_DEFAULTS[SURFACE_LABELS[name]];
    if (d) {
      setLr(d.lr);
      setSteps(d.steps);
      setSelected(d.opts);
    }
  };

  const trajectories: Trajectory[] = useMemo(() => {
    return selected.map((name) => ({
      label: name,
      color: colorOf(name),
      // noiseScale 0 → clean deterministic curves on the 2D landscape (the RNG
      // jitter is a cosmetic stand-in for stochasticity and reads as messy here).
      path: descend(surface, OPTIMIZER_LABELS[name], lr, steps, undefined, { noiseScale: 0 }),
    }));
  }, [surface, selected, lr, steps]);

  const toggle = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));

  return (
    <div className="grid items-start gap-6 lg:grid-cols-[300px_1fr]">
      <Panel
        title="Controls"
        help="Pick a loss surface and one or more optimizers — switching surface loads a learning rate, step count, and optimizer set tuned to converge on it. Each optimizer runs from the same start point and its path animates down the surface. Lower loss = brighter region; crosshairs mark the minima."
      >
        <div className="flex flex-col gap-5">
          <Field label="Loss surface">
            <Select value={surfaceName} onChange={changeSurface} options={SURFACE_NAMES} />
          </Field>
          <Field label="Learning rate" hint={lr.toFixed(4)}>
            <Slider value={lr} min={0.001} max={0.5} step={0.001} onChange={setLr} />
          </Field>
          <Field label="Steps" hint={String(steps)}>
            <Slider value={steps} min={20} max={4000} step={20} onChange={setSteps} />
          </Field>
          <Field label="Optimizers">
            <CheckList options={OPT_NAMES} selected={selected} onToggle={toggle} colorOf={colorOf} />
          </Field>
        </div>
      </Panel>

      <div className="flex flex-col gap-3">
        <LossSurfaceCanvas surface={surface} trajectories={trajectories} />
        <p className="text-sm text-muted">
          Each path runs {steps} steps at lr&nbsp;=&nbsp;
          <span className="num text-fg">{lr.toFixed(3)}</span> on {surface.name}, animating down toward
          the crosshairs (the minima). Plain gradient descent overshoots or diverges on the harder
          surfaces; adaptive methods thread the valley into the basin.
        </p>
      </div>
    </div>
  );
}
