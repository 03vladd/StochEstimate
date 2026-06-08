import { useNavigate } from "react-router-dom";
import type { PairSummary } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Trash2, ChevronRight } from "lucide-react";
import { pairsApi } from "@/api/pairs";
import { cn } from "@/lib/utils";

interface Props {
  pair: PairSummary;
  onRemoved: (pairId: string) => void;
}

const SIGNAL_BORDER: Record<string, string> = {
  LONG:  "#22c55e",
  SHORT: "#ef4444",
  EXIT:  "#f59e0b",
  NONE:  "transparent",
};

function ZScoreBar({ z }: { z: number }) {
  const clamped = Math.max(-3, Math.min(3, z));
  const pct = ((clamped + 3) / 6) * 100;
  const color =
    z <= -1.5 ? "bg-green-500" : z >= 1.5 ? "bg-red-500" : "bg-muted-foreground/40";

  return (
    <div className="relative w-20 h-1 bg-muted rounded-full overflow-hidden">
      <div
        className={cn("absolute top-0 h-full rounded-full", color)}
        style={{
          left: "50%",
          width: `${Math.abs(pct - 50)}%`,
          transform: pct >= 50 ? "none" : "translateX(-100%)",
        }}
      />
      <div className="absolute top-0 left-1/2 w-px h-full bg-border" />
    </div>
  );
}

export default function PairRow({ pair, onRemoved }: Props) {
  const navigate = useNavigate();

  const remove = async (e: React.MouseEvent) => {
    e.stopPropagation();
    await pairsApi.remove(pair.pair_id);
    onRemoved(pair.pair_id);
  };

  const signal = pair.latest_signal_type ?? "NONE";
  const signalVariant = signal.toLowerCase() as "long" | "short" | "exit" | "none";
  const confidenceVariant = (pair.confidence_level ?? "none") as "HIGH" | "MEDIUM" | "LOW" | "NOT_OU" | "none";

  return (
    <div
      style={{ borderLeft: `2px solid ${SIGNAL_BORDER[signal]}` }}
      className="flex items-center gap-4 px-4 py-3.5 hover:bg-white/[0.02] cursor-pointer transition-colors group"
      onClick={() => navigate(`/pairs/${pair.pair_id}`)}
    >
      {/* Tickers */}
      <div className="flex-1 min-w-0">
        <span className="font-mono font-medium text-sm">{pair.ticker1}</span>
        <span className="text-muted-foreground/40 mx-1.5 text-sm">/</span>
        <span className="font-mono font-medium text-sm">{pair.ticker2}</span>
        {pair.sector && (
          <span className="ml-2.5 text-xs text-muted-foreground hidden sm:inline">{pair.sector}</span>
        )}
      </div>

      {/* Confidence */}
      {pair.confidence_level ? (
        <Badge variant={confidenceVariant}>{pair.confidence_level}</Badge>
      ) : (
        <span className="text-muted-foreground/40 text-xs w-8 text-center">—</span>
      )}

      {/* Z-score */}
      <div className="hidden sm:flex items-center gap-2.5">
        {pair.latest_z_score != null ? (
          <>
            <ZScoreBar z={pair.latest_z_score} />
            <span className="font-mono text-xs w-14 text-right tabular-nums text-muted-foreground">
              {pair.latest_z_score >= 0 ? "+" : ""}{pair.latest_z_score.toFixed(2)}σ
            </span>
          </>
        ) : (
          <span className="text-muted-foreground/40 text-xs">—</span>
        )}
      </div>

      {/* Signal */}
      {signal !== "NONE" ? (
        <Badge variant={signalVariant}>{signal}</Badge>
      ) : (
        <span className="text-xs text-muted-foreground/40 w-10 text-center font-mono">—</span>
      )}

      {/* Actions */}
      <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-muted-foreground/50 hover:text-red-400"
          onClick={remove}
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>
      <ChevronRight className="h-3.5 w-3.5 text-muted-foreground/30" />
    </div>
  );
}
