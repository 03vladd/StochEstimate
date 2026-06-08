import { useCallback, useEffect, useState } from "react";
import { useAuth } from "@/store/auth";
import { pairsApi } from "@/api/pairs";
import type { PairSummary } from "@/api/types";
import { Button } from "@/components/ui/button";
import AddPairForm from "@/components/dashboard/AddPairForm";
import PairRow from "@/components/dashboard/PairRow";
import { RefreshCw } from "lucide-react";
import { useToast } from "@/components/ui/toast";

export default function DashboardPage() {
  const { user, logout } = useAuth();
  const { toast } = useToast();
  const [pairs, setPairs] = useState<PairSummary[]>([]);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const { data } = await pairsApi.list();
      setPairs(data);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleAdded = (_pairId: string) => {
    load();
    toast({ title: "Pair queued", description: "Analysis starting in the background", variant: "info" });
  };
  const handleRemoved = (pairId: string) => {
    setPairs((prev) => prev.filter((p) => p.pair_id !== pairId));
  };

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/60 sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
        <div className="max-w-3xl mx-auto px-6 h-12 flex items-center justify-between">
          <span className="text-sm font-semibold tracking-tight">StochEstimate</span>
          <div className="flex items-center gap-4">
            <button
              onClick={() => window.dispatchEvent(new CustomEvent("open-command-palette"))}
              className="hidden sm:flex items-center gap-1.5 text-xs font-mono text-muted-foreground/40 hover:text-muted-foreground/70 border border-border/40 rounded-md px-2 py-1 transition-colors"
            >
              <span>⌘K</span>
            </button>
            <span className="text-xs font-mono text-muted-foreground/50 hidden md:block">{user?.email}</span>
            <button
              onClick={logout}
              className="text-xs text-muted-foreground/60 hover:text-foreground transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-6 py-10 space-y-10">
        {/* Add pair */}
        <div>
          <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest mb-3">
            Add a pair
          </p>
          <AddPairForm onAdded={handleAdded} />
          <p className="text-xs text-muted-foreground mt-2 leading-relaxed">
            Select a pair to run the full OU pipeline — cointegration, validation, LSTM estimation, and trading signals.
          </p>
        </div>

        {/* Watchlist */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              Watchlist
            </p>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-muted-foreground hover:text-foreground"
              onClick={load}
              disabled={loading}
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
            </Button>
          </div>

          {loading ? (
            <div className="border border-border/60 rounded-lg">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center gap-4 px-4 py-3.5 border-b border-border/60 last:border-0">
                  <div className="h-3 w-24 rounded-sm bg-muted animate-pulse" />
                  <div className="h-3 w-12 rounded-sm bg-muted animate-pulse" />
                </div>
              ))}
            </div>
          ) : pairs.length === 0 ? (
            <div className="border border-border/60 border-dashed rounded-lg px-6 py-12 text-center">
              <p className="text-sm text-muted-foreground">No pairs tracked yet.</p>
              <p className="text-xs text-muted-foreground/60 mt-1">Add your first pair above to get started.</p>
            </div>
          ) : (
            <div className="border border-border/60 rounded-lg divide-y divide-border/60">
              {pairs.map((p) => (
                <PairRow key={p.pair_id} pair={p} onRemoved={handleRemoved} />
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
