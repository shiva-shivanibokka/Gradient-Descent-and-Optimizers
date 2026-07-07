"use client";

import { useState } from "react";

import { CompareTab } from "@/components/tabs/CompareTab";
import { LandscapeTab } from "@/components/tabs/LandscapeTab";
import { SchedulerTab } from "@/components/tabs/SchedulerTab";

const TABS = [
  { id: "landscape", label: "Loss landscape", node: <LandscapeTab /> },
  { id: "compare", label: "Optimizer comparison", node: <CompareTab /> },
  { id: "scheduler", label: "Scheduler explorer", node: <SchedulerTab /> },
] as const;

const REPO = "https://github.com/shiva-shivanibokka/Gradient-Descent-and-Optimizers";

export default function Home() {
  const [active, setActive] = useState<(typeof TABS)[number]["id"]>("landscape");

  return (
    <div className="mx-auto flex min-h-full max-w-6xl flex-col px-5 py-8 sm:px-8">
      <header className="mb-8">
        <p className="eyebrow mb-2">Gradient descent · in-browser instrument</p>
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          <span className="text-fg">Optimizers, </span>
          <span className="bg-gradient-to-r from-accent to-[#9b8cff] bg-clip-text text-transparent">
            descending.
          </span>
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted">
          Watch every optimizer descend a loss surface, live in your browser. The math is a
          TypeScript port of the Python{" "}
          <a href={REPO} className="text-accent underline-offset-2 hover:underline">
            gdo
          </a>{" "}
          package, verified against it to six decimals by a parity test suite — no backend, no waiting.
        </p>
        <div className="mt-6 h-px w-full bg-gradient-to-r from-accent/50 via-border-bright to-transparent" />
      </header>

      <nav className="mb-6 flex flex-wrap gap-2" aria-label="Modes">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            onClick={() => setActive(t.id)}
            aria-current={active === t.id}
            className={`rounded-lg px-4 py-2 text-sm transition-all ${
              active === t.id
                ? "bg-panel text-fg shadow-[0_0_0_1px_var(--accent),0_0_18px_-6px_var(--accent)]"
                : "border border-border text-muted hover:border-border-bright hover:text-fg"
            }`}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className="flex-1">{TABS.find((t) => t.id === active)!.node}</main>

      <footer className="mt-12 border-t border-border pt-5 text-xs text-faint">
        <span className="num">gdo</span> · optimizer implementations verified in Python, ported to
        TypeScript for the web ·{" "}
        <a href={REPO} className="text-muted hover:text-accent">
          source
        </a>
      </footer>
    </div>
  );
}
