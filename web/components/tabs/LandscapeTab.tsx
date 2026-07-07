"use client";

import { useMemo, useState } from "react";

import { LossSurfaceCanvas, type Trajectory } from "@/components/LossSurfaceCanvas";
import { CheckList, Field, Panel, Select, Slider } from "@/components/ui";
import { descend, OPTIMIZER_LABELS } from "@/lib/optimizers";
import { seriesColor } from "@/lib/palette";
import { SURFACE_LABELS, SURFACES } from "@/lib/surfaces";

const SURFACE_NAMES = Object.keys(SURFACE_LABELS);
const OPT_NAMES = Object.keys(OPTIMIZER_LABELS);

export function LandscapeTab() {
  const [surfaceName, setSurfaceName] = useState("Quadratic (ill-conditioned)");
  const [selected, setSelected] = useState<string[]>(["Batch GD", "SGD + Momentum", "Adam"]);
  const [lr, setLr] = useState(0.012);
  const [steps, setSteps] = useState(300);

  const surface = SURFACES[SURFACE_LABELS[surfaceName]];

  const colorOf = (name: string) => seriesColor(OPT_NAMES.indexOf(name));

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
        help="Pick a loss surface and one or more optimizers. Each runs from the same start point for the chosen number of steps at the shared learning rate — then their paths animate down the surface. Lower loss = brighter region; the crosshair is the true minimum."
      >
        <div className="flex flex-col gap-5">
          <Field label="Loss surface">
            <Select value={surfaceName} onChange={setSurfaceName} options={SURFACE_NAMES} />
          </Field>
          <Field label="Learning rate" hint={lr.toFixed(4)}>
            <Slider value={lr} min={0.0001} max={0.2} step={0.0001} onChange={setLr} />
          </Field>
          <Field label="Steps" hint={String(steps)}>
            <Slider value={steps} min={20} max={600} step={20} onChange={setSteps} />
          </Field>
          <Field label="Optimizers">
            <CheckList options={OPT_NAMES} selected={selected} onToggle={toggle} colorOf={colorOf} />
          </Field>
        </div>
      </Panel>

      <div className="flex flex-col gap-3">
        <LossSurfaceCanvas surface={surface} trajectories={trajectories} />
        <p className="text-sm text-muted">
          Each path starts at the same point and follows the gradient for {steps} steps at
          lr&nbsp;=&nbsp;<span className="num text-fg">{lr.toFixed(4)}</span>. The crosshair marks the
          global minimum. Watch how momentum overshoots the {surface.name} valley while adaptive
          methods thread it.
        </p>
      </div>
    </div>
  );
}
