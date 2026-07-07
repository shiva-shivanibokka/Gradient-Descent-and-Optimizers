"use client";

import { useMemo, useState } from "react";

import { LineChart, type Series } from "@/components/LineChart";
import { Field, Panel, Select, Slider } from "@/components/ui";
import { trainCurve } from "@/lib/mlp";
import { OPTIMIZER_LABELS } from "@/lib/optimizers";
import { seriesColor } from "@/lib/palette";

const OPT_NAMES = Object.keys(OPTIMIZER_LABELS);

export function CompareTab() {
  const [opt1, setOpt1] = useState("Adam");
  const [opt2, setOpt2] = useState("SGD + Momentum");
  const [lr, setLr] = useState(0.05);
  const [steps, setSteps] = useState(300);

  const series: Series[] = useMemo(() => {
    const pick = [opt1, opt2];
    return pick.map((name, idx) => ({
      label: name,
      color: seriesColor(idx === 0 ? 0 : 2),
      values: trainCurve(OPTIMIZER_LABELS[name], lr, steps, { seed: 0 }),
    }));
  }, [opt1, opt2, lr, steps]);

  return (
    <div className="flex flex-col gap-6">
      <Panel
        title="Controls"
        help="Both optimizers train the same 2-layer MLP on an identical 2-class spiral — same seed, same initialization. Only the update rule differs. The chart below plots cross-entropy loss per step (log scale)."
      >
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Optimizer A">
            <Select value={opt1} onChange={setOpt1} options={OPT_NAMES} />
          </Field>
          <Field label="Optimizer B">
            <Select value={opt2} onChange={setOpt2} options={OPT_NAMES} />
          </Field>
          <Field label="Learning rate" hint={lr.toFixed(4)}>
            <Slider value={lr} min={0.001} max={0.3} step={0.001} onChange={setLr} />
          </Field>
          <Field label="Training steps" hint={String(steps)}>
            <Slider value={steps} min={50} max={800} step={50} onChange={setSteps} />
          </Field>
        </div>
      </Panel>

      <div className="flex flex-col gap-3">
        <LineChart series={series} xLabel="step" yLabel="cross-entropy loss" logY />
        <p className="text-sm text-muted">
          Both optimizers train the same 2-layer MLP on an identical 2-class spiral — same seed, same
          initialization. Only the update rule differs. Loss is shown on a log scale; try Lion at a
          ~10× smaller learning rate than Adam.
        </p>
      </div>
    </div>
  );
}
