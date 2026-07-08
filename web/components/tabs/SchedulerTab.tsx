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
    // maxLr kept at 4× base (not the textbook 10×) so OneCycle/Cyclical don't
    // dwarf the at-or-below-base schedules (Cosine, Warmup) on the shared axis.
    case "onecycle":
      return { maxLr: baseLr * 4, totalEpochs: epochs, pctStart: 0.3 };
    case "cyclical":
      return { maxLr: baseLr * 4, stepSizeUp: Math.max(1, Math.round(epochs / 6)) };
    case "warmup_cosine":
      return { warmupSteps: Math.max(1, Math.round(epochs * 0.15)), totalEpochs: epochs };
    default:
      return {};
  }
}

export function SchedulerTab() {
  const [selected, setSelected] = useState<string[]>([]);
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
      <Panel
        title="Controls"
        help="Each learning-rate schedule is plotted over the chosen number of epochs from your base LR. Toggle schedules to compare their shapes — warmup, decay, cyclical, and one-cycle policies."
      >
        <div className="flex flex-col gap-5">
          <div className="grid gap-5 sm:grid-cols-2 lg:max-w-lg">
            <Field label="Base learning rate" hint={baseLr.toFixed(4)}>
              <Slider value={baseLr} min={0.0005} max={0.1} step={0.0005} onChange={setBaseLr} />
            </Field>
            <Field label="Epochs" hint={String(epochs)}>
              <Slider value={epochs} min={10} max={100} step={5} onChange={setEpochs} />
            </Field>
          </div>
          <Field label="Schedulers">
            <CheckList
              options={SCHED_NAMES}
              selected={selected}
              onToggle={toggle}
              colorOf={colorOf}
              wrap
            />
          </Field>
        </div>
      </Panel>

      <div className="flex flex-col gap-3">
        <LineChart series={series} xLabel="epoch" yLabel="learning rate" />
        <div className="rounded-lg border border-border bg-panel/60 p-4 text-sm leading-6 text-muted">
          <p className="eyebrow mb-2">About this tab</p>
          <p>
            A <span className="text-fg">learning-rate scheduler</span> changes the learning rate as
            training progresses instead of holding it fixed. Each curve is the LR an optimizer would
            use at every epoch under that policy — here over {epochs} epochs from a base of{" "}
            <span className="num text-fg">{baseLr.toFixed(4)}</span>. Toggle schedules to compare their
            shapes:
          </p>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            <li>
              <span className="text-fg">StepLR</span> drops the LR by a factor at fixed intervals;{" "}
              <span className="text-fg">CosineAnnealingLR</span> decays it smoothly to near zero.
            </li>
            <li>
              <span className="text-fg">Warmup + Cosine</span> ramps up first, then decays — the
              Transformer standard, since a cold start at full LR is unstable.
            </li>
            <li>
              <span className="text-fg">OneCycle</span> (fast.ai) and{" "}
              <span className="text-fg">Cyclical</span> push the LR above the base before annealing,
              which can speed up training and help escape sharp minima.
            </li>
          </ul>
          <p className="mt-2">
            Picking the right schedule often means faster convergence and better final accuracy than a
            constant LR.
          </p>
        </div>
      </div>
    </div>
  );
}
