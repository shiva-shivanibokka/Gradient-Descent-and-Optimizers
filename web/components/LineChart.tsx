// Themed Recharts line chart for loss curves and LR schedules.

"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart as RLineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface Series {
  label: string;
  color: string;
  values: number[];
}

export function LineChart({
  series,
  xLabel,
  yLabel,
  logY = false,
}: {
  series: Series[];
  xLabel: string;
  yLabel: string;
  logY?: boolean;
}) {
  const n = Math.max(0, ...series.map((s) => s.values.length));
  const data = Array.from({ length: n }, (_, i) => {
    const row: Record<string, number> = { i };
    for (const s of series) if (i < s.values.length) row[s.label] = s.values[i];
    return row;
  });

  return (
    <div className="h-[380px] w-full rounded-lg border border-border bg-panel-2/40 p-3">
      <ResponsiveContainer width="100%" height="100%">
        <RLineChart data={data} margin={{ top: 8, right: 16, bottom: 20, left: 8 }}>
          <CartesianGrid stroke="var(--grid)" strokeDasharray="0" vertical={false} />
          <XAxis
            dataKey="i"
            stroke="var(--faint)"
            tick={{ fill: "var(--muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            tickLine={false}
            label={{ value: xLabel, position: "insideBottom", offset: -8, fill: "var(--muted)", fontSize: 11 }}
          />
          <YAxis
            scale={logY ? "log" : "linear"}
            domain={logY ? ["auto", "auto"] : [0, "auto"]}
            allowDataOverflow={false}
            stroke="var(--faint)"
            tick={{ fill: "var(--muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            tickLine={false}
            width={52}
            label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "var(--muted)", fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--border-bright)",
              borderRadius: 8,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
            }}
            labelStyle={{ color: "var(--muted)" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {series.map((s) => (
            <Line
              key={s.label}
              type="monotone"
              dataKey={s.label}
              stroke={s.color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </RLineChart>
      </ResponsiveContainer>
    </div>
  );
}
