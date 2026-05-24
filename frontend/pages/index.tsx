import * as React from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { toast } from "sonner";

import { AnalyzeForm } from "@/components/analyze/AnalyzeForm";
import { CredibilityKpis } from "@/components/analyze/CredibilityKpis";
import { MultiAxisInterpretation } from "@/components/analyze/MultiAxisInterpretation";
import { PipelineTrace } from "@/components/analyze/PipelineTrace";
import { RegimeHeadline } from "@/components/analyze/RegimeHeadline";
import {
  RegimeHistoryStrip,
  type RegimeHistoryEntry,
} from "@/components/analyze/RegimeHistoryStrip";
import { SentenceStrikeXaiPanel } from "@/components/analyze/SentenceStrikeXaiPanel";
import { WatchlistChips } from "@/components/analyze/WatchlistChips";
import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchHistory,
  fetchHistoryRun,
  postAnalyze,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { DEFAULT_TEXT } from "@/lib/analyze/constants";
import { toStance } from "@/lib/analyze/format";
import type { AnalyzeRequest, AnalyzeResult, HistoryEntry, Horizon } from "@/lib/analyze/types";

const HORIZON_VALUES = new Set<Horizon>(["1d", "3d", "5d", "10d"]);

function defaultRequest(): AnalyzeRequest {
  return {
    text: DEFAULT_TEXT,
    date: new Date().toISOString().slice(0, 10),
    symbol: "^GSPC",
    horizon: "10d",
    include_realized: false,
    include_xai: true,
  };
}

function takeArgmaxRegime(entry: HistoryEntry, detailPayload: AnalyzeResult | null): string | null {
  const regime = detailPayload?.regime_classification;
  if (regime?.argmax_class) return regime.argmax_class;
  // Older history rows may not carry the regime field; surface stance as the
  // last-resort tag so the strip is not blank on legacy data. Normalise
  // through toStance() first so uppercase / LABEL_n rows resolve too.
  const stance = toStance(entry.stance);
  if (stance === "hawkish") return "high";
  if (stance === "dovish") return "calm";
  if (stance === "neutral") return "normal";
  return null;
}

function parseHorizonParam(value: unknown): Horizon | null {
  return typeof value === "string" && HORIZON_VALUES.has(value as Horizon)
    ? (value as Horizon)
    : null;
}

export default function WorkspacePage() {
  const router = useRouter();
  const [request, setRequest] = React.useState<AnalyzeRequest>(defaultRequest);
  const [result, setResult] = React.useState<AnalyzeResult | null>(null);
  const [baselineResult, setBaselineResult] = React.useState<AnalyzeResult | null>(null);
  const [struck, setStruck] = React.useState<Set<number>>(() => new Set());
  const [counterfactualLoading, setCounterfactualLoading] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [historyEntries, setHistoryEntries] = React.useState<RegimeHistoryEntry[]>([]);

  // Calendar / cross-page deep links land here with ?date=&symbol=&horizon=&kind=.
  React.useEffect(() => {
    if (!router.isReady) return;
    const queryDate = router.query.date;
    const querySymbol = router.query.symbol;
    const queryHorizon = router.query.horizon;
    const queryKind = router.query.kind;
    if (typeof queryDate === "string" && queryDate) {
      setRequest((prev) => ({ ...prev, date: queryDate }));
    }
    if (typeof querySymbol === "string" && querySymbol) {
      setRequest((prev) => ({ ...prev, symbol: querySymbol }));
    }
    const validHorizon = parseHorizonParam(queryHorizon);
    if (validHorizon) {
      setRequest((prev) => ({ ...prev, horizon: validHorizon }));
    }
    if (typeof queryDate !== "string" || !queryDate || typeof queryKind !== "string" || !queryKind) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const url = `${apiBaseUrl}/documents/by-date?date=${encodeURIComponent(queryDate)}&kind=${encodeURIComponent(queryKind)}`;
        const response = await fetch(url);
        if (!response.ok) {
          if (response.status === 404) {
            toast.info(`No FOMC ${queryKind} on disk for ${queryDate}; paste the text manually.`);
          }
          return;
        }
        const payload = await response.json();
        if (cancelled || typeof payload?.text !== "string" || !payload.text) return;
        setRequest((prev) => ({ ...prev, text: payload.text }));
        toast.success(`Prefilled FOMC ${payload.kind} from ${queryDate}.`);
      } catch (err) {
        toast.error((err as Error).message || "Could not load document for this date.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router.isReady, router.query.date, router.query.symbol, router.query.horizon, router.query.kind, apiBaseUrl]);

  // Small slice of past runs for the realized-vs-predicted strip. Detail
  // is fetched lazily for each surfaced row so the strip can read the
  // regime argmax off the persisted payload. An AbortController scoped to
  // the effect cancels every in-flight request on symbol change or unmount,
  // so rapid asset switches don't leave a dozen XHRs racing each other.
  React.useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    (async () => {
      try {
        const list = await fetchHistory(
          apiBaseUrl,
          { symbol: request.symbol, limit: 12 },
          signal,
        );
        if (signal.aborted) return;
        const items = list.items.slice(0, 12);
        const entries: RegimeHistoryEntry[] = await Promise.all(
          items.map(async (entry) => {
            let payload: AnalyzeResult | null = null;
            try {
              const detail = await fetchHistoryRun(apiBaseUrl, entry.id, signal);
              payload = (detail.payload || null) as AnalyzeResult | null;
            } catch {
              // Detail fetch is best-effort; the strip still renders the stance fallback.
            }
            return {
              runId: entry.id,
              documentDate: entry.document_date,
              argmax: takeArgmaxRegime(entry, payload),
              realized: null,
            };
          }),
        );
        if (!signal.aborted) {
          // Newest first → reverse so the strip reads left-to-right chronologically.
          setHistoryEntries(entries.reverse());
        }
      } catch {
        // History pull is best-effort; aborted requests land here too.
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, request.symbol]);

  const handleSubmit = async () => {
    setLoading(true);
    setStruck(new Set());
    try {
      const next = await postAnalyze(apiBaseUrl, { ...request, mask_sentence_indices: [] });
      setResult(next);
      setBaselineResult(next);
      toast.success("Analysis complete");
    } catch (err) {
      setResult(null);
      setBaselineResult(null);
      const message =
        (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail ||
        (err as Error).message ||
        "Request failed. Is the backend running?";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  // Monotonic id for counterfactual requests. The user can strike /
  // unstrike sentences faster than /analyze can respond; without this
  // guard a slower earlier response could land after a newer one and
  // paint stale state. We snapshot the seq before the await and only
  // commit when the response still matches the most recent issued id.
  const counterfactualSeqRef = React.useRef(0);

  const runCounterfactual = React.useCallback(
    async (mask: Set<number>) => {
      if (!baselineResult) return;
      counterfactualSeqRef.current += 1;
      const ticket = counterfactualSeqRef.current;
      setCounterfactualLoading(true);
      try {
        const indices = Array.from(mask).sort((a, b) => a - b);
        const next = await postAnalyze(apiBaseUrl, {
          ...request,
          mask_sentence_indices: indices,
        });
        if (ticket === counterfactualSeqRef.current) {
          setResult(next);
        }
      } catch (err) {
        if (ticket !== counterfactualSeqRef.current) return;
        const message =
          (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail ||
          (err as Error).message ||
          "Counterfactual request failed.";
        toast.error(message);
      } finally {
        if (ticket === counterfactualSeqRef.current) {
          setCounterfactualLoading(false);
        }
      }
    },
    [apiBaseUrl, baselineResult, request],
  );

  const handleStruckChange = React.useCallback(
    (next: Set<number>) => {
      setStruck(next);
      // Empty mask restores the baseline rather than firing an empty-mask round trip.
      if (next.size === 0) {
        setResult(baselineResult);
        return;
      }
      runCounterfactual(next);
    },
    [baselineResult, runCounterfactual],
  );

  const regimeHistorySpark = React.useMemo(
    () =>
      historyEntries.map((entry) => ({
        documentDate: entry.documentDate,
        argmax: entry.argmax,
        realized: entry.realized ?? null,
      })),
    [historyEntries],
  );

  return (
    <>
      <Head>
        <title>Fed Pulse — vol-regime workspace</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar
          result={result}
          loading={loading || counterfactualLoading}
          symbol={request.symbol}
          documentDate={request.date}
        />
        <main id="main-content" tabIndex={-1} className="container space-y-5 py-6 focus:outline-none">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div className="space-y-1">
              <h1 className="text-2xl font-semibold tracking-tight">Workspace</h1>
              <p className="max-w-2xl text-sm text-muted-foreground">
                Paste an FOMC excerpt and the classifier returns a calibrated 10-day vol-regime set,
                the multi-axis breakdown, sentence attribution, credibility KPIs, and the full pipeline
                trace. Everything is read off the live backend.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="numeric text-[10px]">
                horizon · 10d
              </Badge>
              <Badge variant="outline" className="numeric text-[10px]">
                target · vol regime
              </Badge>
            </div>
          </div>

          <AnalyzeForm
            value={request}
            onChange={setRequest}
            onSubmit={handleSubmit}
            loading={loading}
          />

          <WatchlistChips
            currentSymbol={request.symbol}
            onSelect={(symbol) => setRequest((prev) => ({ ...prev, symbol }))}
          />

          {loading && !result ? (
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
              <Skeleton className="h-32 w-full xl:col-span-4" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-64 w-full xl:col-span-4" />
            </div>
          ) : null}

          {result ? (
            <>
              {result.regime_classification ? (
                <RegimeHeadline
                  regime={result.regime_classification}
                  sentiment={result.sentiment}
                  symbol={request.symbol}
                  documentDate={request.date}
                  history={regimeHistorySpark}
                />
              ) : (
                <EmptyState
                  title="Calibrated regime card not in this build"
                  description={
                    <div className="space-y-2">
                      <p>
                        The deployed forecaster is regression-mode — it predicts close and volatility numerically
                        but does not bucket them into a calibrated{" "}
                        <span className="numeric">calm / normal / high</span> set. The classification head and its
                        conformal sidecar (<code>softmax_quantile</code>) ship as part of #216 / Round 1.
                      </p>
                      <p className="text-muted-foreground">
                        Every other workspace surface below — multi-axis breakdown, sentence attribution,
                        credibility KPIs, pipeline trace, history strip — is live against the current checkpoint.
                      </p>
                    </div>
                  }
                />
              )}

              <div className="grid gap-4 xl:grid-cols-2">
                {result.multi_axis ? (
                  <MultiAxisInterpretation multiAxis={result.multi_axis} />
                ) : (
                  <EmptyState
                    variant="inline"
                    title="Multi-axis checkpoint absent"
                    description="Train and deploy the multi-axis classifier to populate stance, factor, certainty, topic."
                  />
                )}
                {result.credibility ? (
                  <CredibilityKpis credibility={result.credibility} />
                ) : (
                  <EmptyState
                    variant="inline"
                    title="Credibility features unavailable"
                    description="No embedding or FRED cache attached on this host yet."
                  />
                )}
              </div>

              {result.xai ? (
                <SentenceStrikeXaiPanel
                  xai={result.xai}
                  struck={struck}
                  onMaskChange={handleStruckChange}
                  baselineResult={baselineResult}
                  currentResult={result}
                  loading={counterfactualLoading}
                />
              ) : null}

              <PipelineTrace result={result} inputText={request.text} />

              <RegimeHistoryStrip entries={historyEntries} symbol={request.symbol} />
            </>
          ) : null}
        </main>
      </div>
    </>
  );
}
