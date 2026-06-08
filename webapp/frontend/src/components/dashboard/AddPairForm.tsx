import { useState } from "react";
import { pairsApi } from "@/api/pairs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  onAdded: (pairId: string) => void;
}

const QUICK_PICKS: [string, string][] = [
  ["PFE",  "MRK"],
  ["AMZN", "WMT"],
  ["JPM",  "GS"],
  ["KO",   "PEP"],
  ["XOM",  "CVX"],
  ["MSFT", "GOOGL"],
];

const WINDOW_OPTIONS = [
  { value: 6,  label: "6 mo" },
  { value: 12, label: "1 yr" },
  { value: 18, label: "18 mo" },
  { value: 24, label: "2 yr" },
];

export default function AddPairForm({ onAdded }: Props) {
  const [t1, setT1] = useState("");
  const [t2, setT2] = useState("");
  const [window, setWindow] = useState(12);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (ticker1 = t1, ticker2 = t2) => {
    const a = ticker1.trim().toUpperCase();
    const b = ticker2.trim().toUpperCase();
    if (!a || !b) return;
    if (a === b) { setError("Tickers must be different"); return; }

    setError("");
    setLoading(true);
    try {
      const { data } = await pairsApi.submit(a, b, window);
      setT1("");
      setT2("");
      onAdded(data.pair_id);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
        "Failed to add pair";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    submit();
  };

  return (
    <div className="space-y-3">
      <form onSubmit={handleSubmit} className="flex items-center gap-2 flex-wrap">
        <Input
          placeholder="AAPL"
          value={t1}
          onChange={(e) => setT1(e.target.value.toUpperCase())}
          className="font-mono w-28 text-center"
          maxLength={10}
          required
        />
        <span className="text-muted-foreground/40 text-sm select-none">/</span>
        <Input
          placeholder="MSFT"
          value={t2}
          onChange={(e) => setT2(e.target.value.toUpperCase())}
          className="font-mono w-28 text-center"
          maxLength={10}
          required
        />
        <div className="flex items-center gap-1 border border-border/60 rounded-md p-0.5">
          {WINDOW_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setWindow(opt.value)}
              className={cn(
                "text-xs font-mono px-2 py-1 rounded transition-colors",
                window === opt.value
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
        <Button type="submit" disabled={loading || !t1.trim() || !t2.trim()} className="gap-1.5 shrink-0">
          <Plus className="h-3.5 w-3.5" />
          {loading ? "Adding…" : "Track pair"}
        </Button>
      </form>

      {error && (
        <p className="text-xs text-red-400">{error}</p>
      )}

      {/* Quick picks */}
      <div className="flex flex-wrap gap-1.5">
        {QUICK_PICKS.map(([a, b]) => (
          <button
            key={`${a}/${b}`}
            type="button"
            disabled={loading}
            onClick={() => submit(a, b)}
            className={cn(
              "font-mono text-xs px-2.5 py-1 rounded-md border border-border/60 text-muted-foreground",
              "hover:border-border hover:text-foreground hover:bg-white/[0.03] transition-colors",
              "disabled:opacity-40 disabled:pointer-events-none"
            )}
          >
            {a}/{b}
          </button>
        ))}
      </div>
    </div>
  );
}
