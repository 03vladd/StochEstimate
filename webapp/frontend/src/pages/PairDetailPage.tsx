import { useCallback, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { pairsApi } from "@/api/pairs";
import type { PairDetail } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import ZScoreChart from "@/components/pair/ZScoreChart";
import ParameterCard from "@/components/pair/ParameterCard";
import ValidationPanel from "@/components/pair/ValidationPanel";
import MLEComparisonPanel from "@/components/pair/MLEComparisonPanel";
import JobPoller from "@/components/pair/JobPoller";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { formatDate, analysisDateRange } from "@/lib/utils";

const SIGNAL_GLOW: Record<string, string> = {
  LONG:  "radial-gradient(ellipse 100% 260px at 50% -40px, rgba(34,197,94,0.08), transparent)",
  SHORT: "radial-gradient(ellipse 100% 260px at 50% -40px, rgba(239,68,68,0.08), transparent)",
  EXIT:  "radial-gradient(ellipse 100% 260px at 50% -40px, rgba(245,158,11,0.06), transparent)",
  NONE:  "none",
};

export default function PairDetailPage() {
  const { pairId } = useParams<{ pairId: string }>();
  const navigate = useNavigate();
  const [detail, setDetail] = useState<PairDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [needsPolling, setNeedsPolling] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [mode, setMode] = useState<"simple" | "expert">("simple");

  const load = useCallback(async () => {
    if (!pairId) return;
    setLoading(true);
    try {
      const { data } = await pairsApi.detail(pairId);
      setDetail(data);
      setNeedsPolling(!data.estimation);
    } finally {
      setLoading(false);
    }
  }, [pairId]);

  useEffect(() => { load(); }, [load]);

  const handlePollingComplete = useCallback(() => {
    setRefreshing(false);
    load();
  }, [load]);

  async function handleRefresh() {
    if (!pairId) return;
    setRefreshing(true);
    try {
      await pairsApi.refresh(pairId);
      setNeedsPolling(true);
    } catch {
      setRefreshing(false);
    }
  }

  const latestSignal = detail?.signals.length
    ? detail.signals[detail.signals.length - 1]
    : null;

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-sm text-muted-foreground font-mono">Loading…</p>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center gap-4">
        <p className="text-sm text-muted-foreground">Pair not found.</p>
        <Button variant="outline" size="sm" onClick={() => navigate("/dashboard")}>
          Back to dashboard
        </Button>
      </div>
    );
  }

  const confidenceVariant = (detail.validation?.confidence_level ?? "none") as
    | "HIGH" | "MEDIUM" | "LOW" | "NOT_OU" | "none";

  const signalType = latestSignal?.signal_type ?? "NONE";
  const signalGlow = SIGNAL_GLOW[signalType] ?? "none";

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/60 sticky top-0 z-10 bg-background/95 backdrop-blur-sm">
        <div className="max-w-4xl mx-auto px-6 h-12 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate("/dashboard")}
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <ArrowLeft className="h-4 w-4" />
            </button>
            <div className="flex items-center gap-2.5">
              <span className="font-mono font-semibold text-sm">
                {detail.ticker1}
                <span className="text-muted-foreground/40 mx-1">/</span>
                {detail.ticker2}
              </span>
              {detail.sector && (
                <span className="text-xs text-muted-foreground hidden sm:block">{detail.sector}</span>
              )}
              {detail.validation && (
                <Badge variant={confidenceVariant}>{detail.validation.confidence_level}</Badge>
              )}
              {detail.window_months && detail.estimation && (
                <span className="text-xs text-muted-foreground/50 font-mono hidden sm:block">
                  {analysisDateRange(detail.estimation.computed_at, detail.window_months)}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Tabs value={mode} onValueChange={(v) => setMode(v as "simple" | "expert")}>
              <TabsList className="h-7">
                <TabsTrigger value="simple" className="h-6 px-3">Simple</TabsTrigger>
                <TabsTrigger value="expert" className="h-6 px-3">Expert</TabsTrigger>
              </TabsList>
            </Tabs>
            <Button
              variant="outline"
              size="sm"
              className="text-xs h-7"
              onClick={handleRefresh}
              disabled={refreshing || needsPolling}
            >
              <RefreshCw className={`h-3 w-3 mr-1.5 ${refreshing ? "animate-spin" : ""}`} />
              {refreshing ? "Queued…" : "Refresh"}
            </Button>
          </div>
        </div>
      </header>

      <main
        className="max-w-4xl mx-auto px-6 py-6 space-y-5"
        style={{ backgroundImage: signalGlow }}
      >
        {needsPolling && pairId && (
          <JobPoller
            pairId={pairId}
            pairName={`${detail.ticker1}/${detail.ticker2}`}
            onComplete={handlePollingComplete}
          />
        )}

        {detail.estimation && (
          <>
            {/* Signal banner */}
            {latestSignal && latestSignal.signal_type !== "NONE" && (
              <div
                className={`px-4 py-2.5 text-xs flex items-center justify-between rounded-lg border ${
                  latestSignal.signal_type === "LONG"
                    ? "bg-green-500/10 border-green-500/20 text-green-400"
                    : latestSignal.signal_type === "SHORT"
                    ? "bg-red-500/10 border-red-500/20 text-red-400"
                    : "bg-amber-500/10 border-amber-500/20 text-amber-400"
                }`}
              >
                <span className="flex items-center gap-2.5">
                  <span className="font-mono font-semibold tracking-wide">{latestSignal.signal_type}</span>
                  <span className="text-current/60">
                    z = <span className="font-mono">{latestSignal.z_score.toFixed(2)}σ</span>
                  </span>
                </span>
                <span className="text-current/50 font-mono">{formatDate(latestSignal.date)}</span>
              </div>
            )}

            {/* Z-score chart */}
            <div className="border border-border/60 rounded-lg bg-card overflow-hidden">
              <div className="px-5 pt-4 pb-1 flex items-center justify-between">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
                  Z-Score History
                </p>
                {detail.window_months && detail.estimation && (
                  <p className="text-xs text-muted-foreground/50 font-mono">
                    {analysisDateRange(detail.estimation.computed_at, detail.window_months)}
                  </p>
                )}
              </div>
              <div className="px-2 pb-3">
                <ZScoreChart
                  signals={detail.signals}
                  longThreshold={detail.settings.long_threshold}
                  shortThreshold={detail.settings.short_threshold}
                />
              </div>
              <div className="flex gap-5 text-xs text-muted-foreground px-5 pb-4">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-px bg-green-500" />
                  Long (−{detail.settings.long_threshold}σ)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-px bg-red-500" />
                  Short (+{detail.settings.short_threshold}σ)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-px bg-amber-500" />
                  Exit (±0.5σ)
                </span>
              </div>
            </div>

            {/* Parameters */}
            <ParameterCard estimation={detail.estimation} />

            {/* Mode content */}
            {mode === "simple" ? (
              <SimpleView detail={detail} onReload={load} />
            ) : (
              <ExpertView detail={detail} />
            )}
          </>
        )}

        {!detail.estimation && !needsPolling && (
          <div className="border border-border/60 border-dashed rounded-lg px-6 py-12 text-center">
            <p className="text-sm text-muted-foreground">No analysis data yet.</p>
            <p className="text-xs text-muted-foreground/60 mt-1">Analysis may still be running — refresh in a moment.</p>
          </div>
        )}
      </main>
    </div>
  );
}

// ── Simple mode ────────────────────────────────────────────────────────────────

function SimpleView({ detail, onReload }: { detail: PairDetail; onReload: () => void }) {
  const e = detail.estimation!;
  const v = detail.validation;
  const latestSignal = detail.signals.length ? detail.signals[detail.signals.length - 1] : null;
  const [generating, setGenerating] = useState(false);

  async function handleGenerate() {
    if (!detail.pair_id) return;
    setGenerating(true);
    const previousNarration = detail.narration;
    try {
      await pairsApi.generateNarration(detail.pair_id);
      const poll = setInterval(async () => {
        const { data } = await pairsApi.detail(detail.pair_id);
        if (data.narration && data.narration !== previousNarration) {
          clearInterval(poll);
          setGenerating(false);
          onReload();
        }
      }, 2000);
    } catch {
      setGenerating(false);
    }
  }

  const halfLifeDays = Math.log(2) / e.theta;
  const halfLifeStr =
    halfLifeDays > 30
      ? `about ${Math.round(halfLifeDays / 7)} weeks`
      : `about ${Math.round(halfLifeDays)} trading days`;

  let confidenceExplain = "";
  if (v) {
    const cl = v.confidence_level;
    if (cl === "HIGH")
      confidenceExplain = "All four statistical tests pass — this pair shows strong mean-reverting behaviour consistent with the OU model.";
    else if (cl === "MEDIUM")
      confidenceExplain = "Three of four tests pass. The pair is likely mean-reverting, but some characteristics are marginal.";
    else if (cl === "LOW")
      confidenceExplain = "Two of four tests pass. Mean reversion is weak — use signals alongside other confirmation.";
    else
      confidenceExplain = "The spread failed the core stationarity or autocorrelation test. OU-based signals are not reliable for this pair.";
  }

  return (
    <div className="space-y-4">
      {/* AI narration */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              Analysis Summary
            </CardTitle>
            <span className="text-xs text-muted-foreground/50 font-mono">Claude Haiku</span>
          </div>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground leading-relaxed space-y-3">
          {detail.narration ? (
            <>
              {detail.narration.split(/\n\n+/).map((para, i) => (
                <p key={i}>{para}</p>
              ))}
              <button
                className="text-xs text-muted-foreground/50 hover:text-muted-foreground transition-colors flex items-center gap-1.5 mt-2"
                onClick={handleGenerate}
                disabled={generating}
              >
                <RefreshCw className={`h-3 w-3 ${generating ? "animate-spin" : ""}`} />
                {generating ? "Regenerating…" : "Regenerate"}
              </button>
            </>
          ) : (
            <div className="flex items-center gap-3">
              <Button size="sm" variant="outline" onClick={handleGenerate} disabled={generating}>
                {generating ? (
                  <><RefreshCw className="h-3 w-3 mr-1.5 animate-spin" />Generating…</>
                ) : (
                  "Generate AI summary"
                )}
              </Button>
              {generating && (
                <span className="text-xs text-muted-foreground/60">Usually 5–10 seconds</span>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Plain-language explainer */}
      {!detail.narration && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              What does this mean?
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3.5 text-sm text-muted-foreground leading-relaxed">
            <p>
              <span className="text-foreground font-medium">Mean reversion speed (θ = {e.theta.toFixed(4)}):</span>{" "}
              When the spread diverges, it historically takes {halfLifeStr} to return halfway to its
              long-run average. Faster reversion means tighter, more reliable trading windows.
            </p>
            <p>
              <span className="text-foreground font-medium">Equilibrium level (μ = {e.mu.toFixed(4)}):</span>{" "}
              The spread tends to drift back toward this value over time.
            </p>
            <p>
              <span className="text-foreground font-medium">Volatility (σ = {e.sigma.toFixed(4)}):</span>{" "}
              How much daily noise is in the spread. Higher volatility means wider confidence bands.
            </p>
            {latestSignal && (
              <p>
                <span className="text-foreground font-medium">
                  Current z-score = {latestSignal.z_score.toFixed(2)}σ:
                </span>{" "}
                {latestSignal.signal_type === "LONG"
                  ? `The spread is ${Math.abs(latestSignal.z_score).toFixed(1)}σ below its mean — historically a long-entry zone.`
                  : latestSignal.signal_type === "SHORT"
                  ? `The spread is ${latestSignal.z_score.toFixed(1)}σ above its mean — historically a short-entry zone.`
                  : latestSignal.signal_type === "EXIT"
                  ? "The spread is near its equilibrium — a typical exit zone."
                  : "The spread is within normal bounds — no active signal."}
              </p>
            )}
            {v && <p>{confidenceExplain}</p>}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// ── Expert mode ────────────────────────────────────────────────────────────────

function ExpertView({ detail }: { detail: PairDetail }) {
  return (
    <div className="space-y-5">
      {detail.validation && <ValidationPanel validation={detail.validation} />}

      {detail.cointegration && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              Engle-Granger Cointegration
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-5 text-sm">
              <div>
                <p className="text-xs text-muted-foreground mb-1">Hedge ratio</p>
                <p className="font-mono font-medium">{detail.cointegration.hedge_ratio.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">ADF statistic</p>
                <p className="font-mono font-medium">{detail.cointegration.adf_statistic.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">p-value</p>
                <p className="font-mono font-medium">{detail.cointegration.adf_pvalue.toFixed(4)}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground mb-1">Cointegrated?</p>
                <p className={`font-mono font-medium ${detail.cointegration.cointegrated ? "text-green-400" : "text-red-400"}`}>
                  {detail.cointegration.cointegrated ? "Yes" : "No"}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {detail.estimation && detail.mle && (
        <MLEComparisonPanel estimation={detail.estimation} mle={detail.mle} />
      )}

      {detail.estimation && (
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-xs font-medium text-muted-foreground uppercase tracking-widest">
              MC Dropout Uncertainty
            </CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 sm:grid-cols-4 gap-5 text-sm">
            <div>
              <p className="text-xs text-muted-foreground mb-1">θ std</p>
              <p className="font-mono font-medium">±{detail.estimation.theta_std.toFixed(5)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">σ std</p>
              <p className="font-mono font-medium">±{detail.estimation.sigma_std.toFixed(5)}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">MC samples</p>
              <p className="font-mono font-medium">{detail.estimation.n_mc_samples.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground mb-1">Model</p>
              <p className="font-mono font-medium">{detail.estimation.model_version}</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
