import type { EstimationData, MLEData } from "@/api/types";
import { formatNumber, halfLife } from "@/lib/utils";

interface Props {
  estimation: EstimationData;
  mle: MLEData;
}

function Row({
  label,
  lstm,
  lstmCi,
  mle,
  highlight,
}: {
  label: string;
  lstm: string;
  lstmCi: string;
  mle: string;
  highlight?: boolean;
}) {
  return (
    <tr className={highlight ? "bg-amber-500/5" : ""}>
      <td className="py-2.5 pr-4 text-xs text-muted-foreground font-mono">{label}</td>
      <td className="py-2.5 pr-4 font-mono text-sm font-medium">{lstm}</td>
      <td className="py-2.5 pr-4 font-mono text-xs text-muted-foreground">{lstmCi}</td>
      <td className="py-2.5 font-mono text-sm text-muted-foreground">{mle}</td>
    </tr>
  );
}

export default function MLEComparisonPanel({ estimation: e, mle: m }: Props) {
  const thetaDiff = Math.abs(e.theta - m.theta);
  const sigmaDiff = Math.abs(e.sigma - m.sigma);

  return (
    <div className="border border-border/60 rounded-lg bg-card">
      <div className="px-5 py-3 border-b border-border/60">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
          LSTM-Robust vs Gaussian MLE
        </p>
        <p className="text-xs text-muted-foreground/60 mt-1">
          MLE assumes clean Gaussian innovations — differences reveal jump contamination sensitivity.
        </p>
      </div>

      <div className="px-5 py-4">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/60">
              <th className="text-left pb-2 text-xs font-medium text-muted-foreground/60">Param</th>
              <th className="text-left pb-2 text-xs font-medium text-muted-foreground/60">LSTM-Robust</th>
              <th className="text-left pb-2 text-xs font-medium text-muted-foreground/60">95% CI (MC)</th>
              <th className="text-left pb-2 text-xs font-medium text-muted-foreground/60">Gaussian MLE</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/60">
            <Row
              label="θ"
              lstm={formatNumber(e.theta)}
              lstmCi={`[${formatNumber(e.theta_ci_lower)}, ${formatNumber(e.theta_ci_upper)}]`}
              mle={formatNumber(m.theta)}
              highlight={thetaDiff > 0.02}
            />
            <Row
              label="μ"
              lstm={formatNumber(e.mu)}
              lstmCi={`[${formatNumber(e.mu_ci_lower)}, ${formatNumber(e.mu_ci_upper)}]`}
              mle={formatNumber(m.mu)}
            />
            <Row
              label="σ"
              lstm={formatNumber(e.sigma)}
              lstmCi={`[${formatNumber(e.sigma_ci_lower)}, ${formatNumber(e.sigma_ci_upper)}]`}
              mle={formatNumber(m.sigma)}
              highlight={sigmaDiff > 0.1}
            />
          </tbody>
        </table>

        <div className="mt-4 grid grid-cols-2 gap-4 border-t border-border/60 pt-4 text-xs">
          <div>
            <p className="text-muted-foreground/60 mb-0.5">Half-life (LSTM-R)</p>
            <p className="font-mono text-muted-foreground">{halfLife(e.theta)}</p>
          </div>
          <div>
            <p className="text-muted-foreground/60 mb-0.5">Half-life (MLE)</p>
            <p className="font-mono text-muted-foreground">{halfLife(m.theta)}</p>
          </div>
          <div>
            <p className="text-muted-foreground/60 mb-0.5">MC samples</p>
            <p className="font-mono text-muted-foreground">{e.n_mc_samples.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-muted-foreground/60 mb-0.5">MLE log-likelihood</p>
            <p className="font-mono text-muted-foreground">
              {m.log_likelihood != null ? formatNumber(m.log_likelihood, 2) : m.success ? "—" : "failed"}
            </p>
          </div>
        </div>

        {(thetaDiff > 0.02 || sigmaDiff > 0.1) && (
          <p className="mt-4 text-xs text-amber-400 bg-amber-500/10 border border-amber-500/20 rounded-md px-3 py-2">
            Notable divergence between LSTM-Robust and MLE — consistent with jump contamination. LSTM-Robust is the recommended estimator.
          </p>
        )}
      </div>
    </div>
  );
}
