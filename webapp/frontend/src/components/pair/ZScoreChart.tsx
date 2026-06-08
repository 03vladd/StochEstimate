import {
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SignalPoint } from "@/api/types";
import { formatDate } from "@/lib/utils";

interface Props {
  signals: SignalPoint[];
  longThreshold: number;
  shortThreshold: number;
}

const SIGNAL_COLORS: Record<string, string> = {
  LONG:  "#22c55e",
  SHORT: "#ef4444",
  EXIT:  "#f59e0b",
  NONE:  "transparent",
};

export default function ZScoreChart({ signals, longThreshold, shortThreshold }: Props) {
  const data = signals.map((s, i) => {
    const prev = signals[i - 1];
    const isTransition = !prev || s.signal_type !== prev.signal_type;
    const showDot = s.signal_type !== "NONE" && isTransition;
    return {
      date: s.date,
      z: parseFloat(s.z_score.toFixed(3)),
      signal: s.signal_type,
      showDot,
      color: SIGNAL_COLORS[s.signal_type] ?? "transparent",
    };
  });

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ComposedChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <XAxis
          dataKey="date"
          tickFormatter={(v: string) => {
            const d = new Date(v);
            return `${d.toLocaleString("default", { month: "short" })} '${String(d.getFullYear()).slice(2)}`;
          }}
          tick={{ fill: "#525252", fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
          tickLine={false}
          axisLine={false}
          minTickGap={60}
        />
        <YAxis
          tickFormatter={(v: number) => `${v.toFixed(1)}σ`}
          tick={{ fill: "#525252", fontSize: 10, fontFamily: "JetBrains Mono, monospace" }}
          tickLine={false}
          axisLine={false}
          width={40}
          domain={[
            (dataMin: number) => Math.min(dataMin, -longThreshold) - 0.3,
            (dataMax: number) => Math.max(dataMax,  shortThreshold) + 0.3,
          ]}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload as (typeof data)[0];
            return (
              <div className="rounded-lg border border-border bg-card px-3 py-2 text-xs">
                <p className="text-muted-foreground font-mono">{formatDate(d.date)}</p>
                <p className="mt-0.5">
                  z = <span className="font-mono text-foreground">{d.z.toFixed(3)}σ</span>
                </p>
                {d.showDot && (
                  <p style={{ color: SIGNAL_COLORS[d.signal] }} className="font-mono font-semibold mt-0.5">
                    {d.signal}
                  </p>
                )}
              </div>
            );
          }}
        />
        <ReferenceLine y={shortThreshold}  stroke="#ef4444" strokeDasharray="3 4" strokeWidth={1} strokeOpacity={0.6} />
        <ReferenceLine y={-longThreshold}  stroke="#22c55e" strokeDasharray="3 4" strokeWidth={1} strokeOpacity={0.6} />
        <ReferenceLine y={0}               stroke="#2a2a2a" strokeWidth={1} />
        <ReferenceLine y={0.5}             stroke="#2a2a2a" strokeDasharray="2 5" strokeWidth={1} strokeOpacity={0.5} />
        <ReferenceLine y={-0.5}            stroke="#2a2a2a" strokeDasharray="2 5" strokeWidth={1} strokeOpacity={0.5} />
        <Line
          type="monotone"
          dataKey="z"
          stroke="#a3a3a3"
          strokeWidth={1.5}
          activeDot={{ r: 3, fill: "#ededed", strokeWidth: 0 }}
          dot={(props: { cx?: number; cy?: number; payload?: { signal: string; showDot: boolean; color: string } }) => {
            const { cx = 0, cy = 0, payload } = props;
            if (!payload || !payload.showDot) return <g key={`${cx}-${cy}`} />;
            return (
              <circle
                key={`${cx}-${cy}`}
                cx={cx}
                cy={cy}
                r={4}
                fill={payload.color}
                stroke="#0a0a0a"
                strokeWidth={1.5}
              />
            );
          }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
