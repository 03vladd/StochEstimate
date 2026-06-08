import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { pairsApi } from "@/api/pairs";
import type { PairSummary } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/components/ui/toast";
import { cn } from "@/lib/utils";
import { Search, Plus, ArrowRight, CornerDownLeft } from "lucide-react";

function parseTickers(q: string): [string, string] | null {
  const parts = q.trim().toUpperCase().split(/[\s/,]+/).filter(Boolean);
  if (
    parts.length === 2 &&
    parts[0].length >= 1 && parts[0].length <= 10 &&
    parts[1].length >= 1 && parts[1].length <= 10
  ) {
    return [parts[0], parts[1]];
  }
  return null;
}

type PaletteItem =
  | { kind: "pair"; pair: PairSummary }
  | { kind: "add"; t1: string; t2: string };

export default function CommandPalette() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [pairs, setPairs] = useState<PairSummary[]>([]);
  const [activeIdx, setActiveIdx] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Global Cmd+K / Ctrl+K shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // External trigger via custom event
  useEffect(() => {
    const handler = () => setOpen(true);
    window.addEventListener("open-command-palette", handler);
    return () => window.removeEventListener("open-command-palette", handler);
  }, []);

  // On open: reset state, focus input, load pairs
  useEffect(() => {
    if (!open) return;
    setQuery("");
    setActiveIdx(0);
    setTimeout(() => inputRef.current?.focus(), 10);
    pairsApi.list().then(({ data }) => setPairs(data)).catch(() => {});
  }, [open]);

  const tickers = parseTickers(query);

  const filteredPairs = query.trim()
    ? pairs.filter((p) => {
        const q = query.toUpperCase();
        return (
          p.ticker1.startsWith(q) ||
          p.ticker2.startsWith(q) ||
          `${p.ticker1}/${p.ticker2}`.includes(q.replace(/\s+/, "/"))
        );
      })
    : pairs;

  const items: PaletteItem[] = [
    ...filteredPairs.map((p): PaletteItem => ({ kind: "pair", pair: p })),
    ...(tickers ? [{ kind: "add" as const, t1: tickers[0], t2: tickers[1] }] : []),
  ];

  const safeIdx = Math.max(0, Math.min(activeIdx, items.length - 1));

  const selectItem = useCallback(
    async (item: PaletteItem) => {
      if (item.kind === "pair") {
        setOpen(false);
        navigate(`/pairs/${item.pair.pair_id}`);
      } else {
        setSubmitting(true);
        try {
          await pairsApi.submit(item.t1, item.t2);
          setOpen(false);
          toast({
            title: "Pair queued",
            description: `${item.t1}/${item.t2} — analysis starting`,
            variant: "info",
          });
          navigate("/dashboard");
        } catch (err: unknown) {
          const msg =
            (err as { response?: { data?: { detail?: string } } })?.response?.data
              ?.detail ?? "Failed to add pair";
          toast({ title: "Error", description: msg, variant: "error" });
        } finally {
          setSubmitting(false);
        }
      }
    },
    [navigate, toast]
  );

  const onKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActiveIdx((i) => Math.min(i + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActiveIdx((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      const item = items[safeIdx];
      if (item) selectItem(item);
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  };

  // Reset active index when query changes
  useEffect(() => { setActiveIdx(0); }, [query]);

  if (!open) return null;

  const hasPairs = filteredPairs.length > 0;
  const hasAction = !!tickers;
  const isEmpty = !hasPairs && !hasAction && query.trim() !== "";
  const isBlankEmpty = !hasPairs && !query.trim();

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[16vh] px-4"
      onMouseDown={() => setOpen(false)}
    >
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

      {/* Panel */}
      <div
        className="relative w-full max-w-lg rounded-xl border border-border/80 bg-card shadow-2xl shadow-black/60 overflow-hidden"
        onMouseDown={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 h-12 border-b border-border/60">
          <Search className="h-4 w-4 text-muted-foreground/40 shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search pairs or type AAPL MSFT to track…"
            className="flex-1 bg-transparent text-sm placeholder:text-muted-foreground/30 focus:outline-none font-mono"
            autoComplete="off"
            spellCheck={false}
          />
          <kbd className="text-[10px] text-muted-foreground/25 font-mono border border-border/40 rounded px-1.5 py-0.5">
            esc
          </kbd>
        </div>

        {/* Results */}
        <div ref={listRef} className="max-h-[300px] overflow-y-auto">
          {/* Watchlist */}
          {hasPairs && (
            <section>
              <p className="px-4 pt-3 pb-1.5 text-[10px] font-medium text-muted-foreground/40 uppercase tracking-widest">
                Watchlist
              </p>
              {filteredPairs.map((pair, i) => {
                const isActive = safeIdx === i;
                const signal = pair.latest_signal_type ?? "NONE";
                const sigVariant = signal.toLowerCase() as "long" | "short" | "exit" | "none";
                const confVariant = (pair.confidence_level ?? "none") as
                  | "HIGH" | "MEDIUM" | "LOW" | "NOT_OU" | "none";
                return (
                  <div
                    key={pair.pair_id}
                    className={cn(
                      "flex items-center gap-3 px-4 py-2.5 cursor-pointer transition-colors",
                      isActive ? "bg-white/[0.05]" : "hover:bg-white/[0.02]"
                    )}
                    onMouseDown={() => selectItem({ kind: "pair", pair })}
                    onMouseEnter={() => setActiveIdx(i)}
                  >
                    <span className="font-mono text-sm font-medium flex-1 truncate">
                      {pair.ticker1}
                      <span className="text-muted-foreground/30 mx-1">/</span>
                      {pair.ticker2}
                      {pair.sector && (
                        <span className="ml-2 text-xs text-muted-foreground/40 font-sans font-normal">
                          {pair.sector}
                        </span>
                      )}
                    </span>
                    {pair.confidence_level && (
                      <Badge variant={confVariant} className="text-[10px] px-1.5 py-0">
                        {pair.confidence_level}
                      </Badge>
                    )}
                    {pair.latest_z_score != null && (
                      <span className="font-mono text-xs text-muted-foreground/60 tabular-nums">
                        {pair.latest_z_score >= 0 ? "+" : ""}
                        {pair.latest_z_score.toFixed(2)}σ
                      </span>
                    )}
                    {signal !== "NONE" && (
                      <Badge variant={sigVariant} className="text-[10px] px-1.5 py-0">
                        {signal}
                      </Badge>
                    )}
                    <ArrowRight className="h-3 w-3 text-muted-foreground/20 shrink-0" />
                  </div>
                );
              })}
            </section>
          )}

          {/* Track new pair action */}
          {hasAction && (
            <section>
              <p className="px-4 pt-3 pb-1.5 text-[10px] font-medium text-muted-foreground/40 uppercase tracking-widest">
                Add pair
              </p>
              <div
                className={cn(
                  "flex items-center gap-3 px-4 py-2.5 cursor-pointer transition-colors",
                  safeIdx === filteredPairs.length
                    ? "bg-white/[0.05]"
                    : "hover:bg-white/[0.02]"
                )}
                onMouseDown={() =>
                  !submitting && selectItem({ kind: "add", t1: tickers![0], t2: tickers![1] })
                }
                onMouseEnter={() => setActiveIdx(filteredPairs.length)}
              >
                <Plus className="h-3.5 w-3.5 text-muted-foreground/40 shrink-0" />
                <span className="text-sm flex-1">
                  Track{" "}
                  <span className="font-mono font-medium">
                    {tickers![0]}/{tickers![1]}
                  </span>
                </span>
                {submitting ? (
                  <span className="text-xs text-muted-foreground/30 font-mono">adding…</span>
                ) : (
                  <CornerDownLeft className="h-3 w-3 text-muted-foreground/25" />
                )}
              </div>
            </section>
          )}

          {/* Empty states */}
          {isEmpty && (
            <p className="px-4 py-8 text-sm text-center text-muted-foreground/40">
              No pairs match.{" "}
              <span className="font-mono">
                Try "{query.trim().split(/\s+/)[0]} MSFT"
              </span>{" "}
              to track a new pair.
            </p>
          )}
          {isBlankEmpty && (
            <p className="px-4 py-8 text-sm text-center text-muted-foreground/40">
              No pairs tracked yet. Type two tickers to get started.
            </p>
          )}
        </div>

        {/* Footer hints */}
        <div className="border-t border-border/60 px-4 py-2 flex items-center gap-4">
          <span className="text-[10px] text-muted-foreground/25 font-mono">↑↓ navigate</span>
          <span className="text-[10px] text-muted-foreground/25 font-mono">↵ select</span>
          <span className="text-[10px] text-muted-foreground/25 font-mono">esc close</span>
        </div>
      </div>
    </div>
  );
}
