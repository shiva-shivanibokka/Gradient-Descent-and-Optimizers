"use client";

import { useMemo, useState } from "react";

import { LineChart, type Series } from "@/components/LineChart";
import { CheckList, Field, Panel, Slider } from "@/components/ui";
import { seriesColor } from "@/lib/palette";
import { lrCurve, SCHEDULER_LABELS, type SchedOpts } from "@/lib/schedulers";

const SCHED_NAMES = Object.keys(SCHEDULER_LABELS);

// Per-scheduler options tuned to read well over the chosen epoch count.
function optsFor(slug: string, epochs: number, baseLr: number): SchedOpts {
  switch (slug) {
    case "step":
      return { stepSize: Math.max(1, Math.round(epochs / 5)), gamma: 0.5 };
    case "cosine":
      return { tMax: epochs };
    case "onecycle":
      return { maxLr: baseLr * 10, totalEpochs: epochs, pctStart: 0.3 };
    case "cyclical":
      return { maxLr: baseLr * 10, stepSizeUp: Math.max(1, Math.round(epochs / 6)) };
    case "warmup_cosine":
      return { warmupSteps: Math.max(1, Math.round(epochs * 0.15)), totalEpochs: epochs };
    default:
      return {};
  }
}

export function SchedulerTab() {
  const [selected, setSelected] = useState<string[]>([
    "CosineAnnealingLR",
    "OneCycleLR",
    "Warmup + Cosine",
  ]);
  const [baseLr, setBaseLr] = useState(0.01);
  const [epochs, setEpochs] = useState(50);

  const colorOf = (name: string) => seriesColor(SCHED_NAMES.indexOf(name));

  const series: Series[] = useMemo(() => {
    return selected.map((name) => {
      const slug = SCHEDULER_LABELS[name];
      return {
        label: name,
        color: colorOf(name),
        values: lrCurve(slug, baseLr, epochs, optsFor(slug, epochs, baseLr)),
      };
    });
  }, [selected, baseLr, epochs]);

  const toggle = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));

  return (
    <div className="flex flex-col gap-6">
      <div className="w-full lg:max-w-md">
        <Panel
          title="Controls"
          help="Each learning-rate schedule is plotted over the chosen number of epochs from your base LR. Toggle schedules to compare their shapes — warmup, decay, cyclical, and one-cycle policies."
        >
          <div className="flex flex-col gap-5">
          <Field label="Base learning rate" hint={baseLr.toFixed(4)}>
            <Slider value={baseLr} min={0.0005} max={0.1} step={0.0005} onChange={setBaseLr} />
          </Field>
          <Field label="Epochs" hint={String(epochs)}>
            <Slider value={epochs} min={10} max={100} step={5} onChange={setEpochs} />
          </Field>
            <Field label="Schedulers">
              <CheckList options={SCHED_NAMES} selected={selected} onToggle={toggle} colorOf={colorOf} />
            </Field>
          </div>
        </Panel>
      </div>

      <div className="flex flex-col gap-3">
        <LineChart series={series} xLabel="epoch" yLabel="learning rate" />
        <p className="text-sm text-muted">
          Learning-rate schedules over {epochs} epochs from a base of{" "}
          <span className="num text-fg">{baseLr.toFixed(4)}</span>. Warmup + Cosine is the Transformer
          standard; OneCycle is the fast.ai recipe.
        </p>
      </div>
    </div>
  );
}
