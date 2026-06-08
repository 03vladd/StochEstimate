import { useEffect, useRef, useState } from "react";
import { pairsApi } from "@/api/pairs";
import type { JobStatus } from "@/api/types";
import { useToast } from "@/components/ui/toast";

interface Props {
  pairId: string;
  pairName?: string;
  onComplete: () => void;
}

export default function JobPoller({ pairId, pairName, onComplete }: Props) {
  const { toast } = useToast();
  const [job, setJob] = useState<JobStatus | null>(null);
  const firedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    firedRef.current = false;

    const poll = async () => {
      try {
        const { data } = await pairsApi.jobStatus(pairId);
        if (cancelled) return;
        setJob(data);
        if (data.status === "complete") {
          if (!firedRef.current) {
            firedRef.current = true;
            onComplete();
            toast({
              title: "Analysis ready",
              description: pairName ? `${pairName} — results updated` : "Results updated",
              variant: "success",
            });
          }
        } else if (data.status === "failed") {
          toast({
            title: "Analysis failed",
            description: data.error_message ?? "Unknown error",
            variant: "error",
          });
        } else {
          setTimeout(poll, 3000);
        }
      } catch {
        if (!cancelled) setTimeout(poll, 5000);
      }
    };

    poll();
    return () => { cancelled = true; };
  }, [pairId, pairName, onComplete, toast]);

  if (!job || job.status === "complete") return null;

  if (job.status === "failed") {
    return (
      <div className="flex items-center gap-3 border border-red-500/20 bg-red-500/10 rounded-lg px-4 py-3">
        <div className="h-1.5 w-1.5 rounded-full bg-red-400 shrink-0" />
        <div>
          <p className="text-xs font-medium text-red-400">Analysis failed</p>
          <p className="text-xs text-muted-foreground mt-0.5">{job.error_message ?? "Unknown error"}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 border border-border/60 rounded-lg px-4 py-3 bg-white/[0.02]">
      <div className="h-1.5 w-1.5 rounded-full bg-amber-400 animate-pulse shrink-0" />
      <div>
        <p className="text-xs font-medium text-foreground">
          {job.status === "pending" ? "Queued" : "Running analysis"}
        </p>
        <p className="text-xs text-muted-foreground mt-0.5">
          Fetching prices → cointegration → OU validation → LSTM → MLE → signals
        </p>
      </div>
    </div>
  );
}
