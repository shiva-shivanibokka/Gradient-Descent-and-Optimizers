// Minimal themed controls built on native elements (accessible, zero deps).

"use client";

import type { ReactNode } from "react";

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="eyebrow">{label}</span>
        {hint !== undefined && <span className="num text-xs text-accent">{hint}</span>}
      </div>
      {children}
    </label>
  );
}

export function Select({
  value,
  onChange,
  options,
}: {
  value: string;
  onChange: (v: string) => void;
  options: string[];
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full rounded-md border border-border bg-panel-2 px-3 py-2 text-sm text-fg outline-none transition-colors hover:border-border-bright focus:border-accent"
    >
      {options.map((o) => (
        <option key={o} value={o}>
          {o}
        </option>
      ))}
    </select>
  );
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      className="w-full"
    />
  );
}

export function CheckList({
  options,
  selected,
  onToggle,
  colorOf,
}: {
  options: string[];
  selected: string[];
  onToggle: (v: string) => void;
  colorOf?: (v: string) => string;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      {options.map((o) => {
        const on = selected.includes(o);
        return (
          <button
            key={o}
            type="button"
            aria-pressed={on}
            onClick={() => onToggle(o)}
            className={`flex items-center gap-2.5 rounded-md border px-3 py-1.5 text-left text-sm transition-colors ${
              on
                ? "border-border-bright bg-panel-2 text-fg"
                : "border-border bg-transparent text-muted hover:border-border-bright"
            }`}
          >
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-full border"
              style={{
                background: on && colorOf ? colorOf(o) : "transparent",
                borderColor: colorOf ? colorOf(o) : "var(--border-bright)",
                boxShadow: on && colorOf ? `0 0 6px ${colorOf(o)}` : "none",
              }}
            />
            {o}
          </button>
        );
      })}
    </div>
  );
}

export function Button({ children, onClick }: { children: ReactNode; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="rounded-md border border-accent/40 bg-accent/10 px-4 py-2 text-sm font-medium text-accent transition-colors hover:bg-accent/20"
    >
      {children}
    </button>
  );
}

export function Panel({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-border bg-panel p-5 shadow-[0_1px_0_0_rgba(255,255,255,0.02)_inset]">
      {children}
    </div>
  );
}
