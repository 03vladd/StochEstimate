import type { ValidationData } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Check, X } from "lucide-react";
import { formatNumber } from "@/lib/utils";

interface Props {
  validation: ValidationData;
}

function TestRow({ label, passed }: { label: string; passed: boolean }) {
  return (
    <div className="flex items-center justify-between px-4 py-2.5">
      <span className="text-sm text-muted-foreground">{label}</span>
      {passed ? (
        <Check className="h-3.5 w-3.5 text-green-400 shrink-0" />
      ) : (
        <X className="h-3.5 w-3.5 text-red-400 shrink-0" />
      )}
    </div>
  );
}

export default function ValidationPanel({ validation: v }: Props) {
  const confidenceVariant = v.confidence_level as "HIGH" | "MEDIUM" | "LOW" | "NOT_OU";

  return (
    <div className="border border-border/60 rounded-lg bg-card">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/60">
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest">OU Validation</p>
        <Badge variant={confidenceVariant}>{v.confidence_level}</Badge>
      </div>

      <div className="divide-y divide-border/60">
        <TestRow label="Stationarity (ADF)" passed={v.stationarity_passed} />
        <TestRow label="No linear drift" passed={v.drift_passed} />
        <TestRow label="Constant volatility" passed={v.volatility_passed} />
        <TestRow label="Exponential ACF decay" passed={v.acf_passed} />
        <TestRow label="Normality (informational)" passed={v.normality_passed} />
      </div>

      <div className="flex items-center justify-between px-4 py-2.5 border-t border-border/60 bg-white/[0.02] rounded-b-lg">
        <span className="text-xs text-muted-foreground">{v.tests_passed_count}/4 core tests passed</span>
        {v.acf_r_squared != null && (
          <span className="text-xs text-muted-foreground font-mono">
            ACF R² = {formatNumber(v.acf_r_squared, 3)}
            {v.acf_theta != null && ` · θ = ${formatNumber(v.acf_theta, 4)}`}
          </span>
        )}
      </div>
    </div>
  );
}
