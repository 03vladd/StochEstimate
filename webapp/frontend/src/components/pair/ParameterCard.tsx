import type { EstimationData } from "@/api/types";
import { formatNumber, halfLife } from "@/lib/utils";

interface Props {
  estimation: EstimationData;
}

function Param({
  label,
  value,
  ci,
  sub,
}: {
  label: string;
  value: string;
  ci?: string;
  sub?: string;
}) {
  return (
    <div className="px-5 py-4">
      <p className="text-xs text-muted-foreground mb-2">{label}</p>
      <p className="font-mono text-2xl font-medium tabular-nums">{value}</p>
      {ci && <p className="font-mono text-xs text-muted-foreground/60 mt-1">{ci}</p>}
      {sub && <p className="text-xs text-muted-foreground mt-1">{sub}</p>}
    </div>
  );
}

export default function ParameterCard({ estimation: e }: Props) {
  return (
    <div className="grid grid-cols-3 border border-border/60 rounded-lg divide-x divide-border/60 bg-card">
      <Param
        label="θ  mean reversion"
        value={formatNumber(e.theta)}
        ci={`[${formatNumber(e.theta_ci_lower)}, ${formatNumber(e.theta_ci_upper)}]`}
        sub={`half-life ≈ ${halfLife(e.theta)}`}
      />
      <Param
        label="μ  equilibrium"
        value={formatNumber(e.mu)}
        ci={`[${formatNumber(e.mu_ci_lower)}, ${formatNumber(e.mu_ci_upper)}]`}
      />
      <Param
        label="σ  volatility"
        value={formatNumber(e.sigma)}
        ci={`[${formatNumber(e.sigma_ci_lower)}, ${formatNumber(e.sigma_ci_upper)}]`}
      />
    </div>
  );
}
