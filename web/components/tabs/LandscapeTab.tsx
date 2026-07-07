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
  const [surfaceName, setSurfaceName] = useState("Rosenbrock");
  const [selected, setSelected] = useState<string[]>(["Batch GD", "SGD + Momentum", "Adam"]);
  const [lr, setLr] = useState(0.01);
  const [steps, setSteps] = useState(200);

  const surface = SURFACES[SURFACE_LABELS[surfaceName]];

  const colorOf = (name: string) => seriesColor(OPT_NAMES.indexOf(name));

  const trajectories: Trajectory[] = useMemo(() => {
    return selected.map((name) => ({
      label: name,
      color: colorOf(name),
      path: descend(surface, OPTIMIZER_LABELS[name], lr, steps),
    }));
  }, [surface, selected, lr, steps]);

  const toggle = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));

  return (
    <div className="grid gap-6 lg:grid-cols-[300px_1fr]">
      <Panel>
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
