import * as React from "react";
import Head from "next/head";
import dynamic from "next/dynamic";
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
import { WatchlistChips } from "@/components/analyze/WatchlistChips";
import { Header } from "@/components/shell/header";
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
import type { AnalyzeRequest, AnalyzeResult, HistoryEntry } from "@/lib/analyze/types";

const XaiPanel = dynamic(
  () => import("@/components/analyze/XaiPanel").then((m) => m.XaiPanel),
  { ssr: false, loading: () => null },
);

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
  // Older history rows may not carry the regime field; surface stance as the last-resort tag
  // so the strip is not blank on legacy data.
  if (entry.stance === "hawkish") return "high";
  if (entry.stance === "dovish") return "calm";
  if (entry.stance === "neutral") return "normal";
  return null;
}

export default function WorkspacePage() {
  const router = useRouter();
  const [request, setRequest] = React.useState<AnalyzeRequest>(defaultRequest);
  const [result, setResult] = React.useState<AnalyzeResult | null>(null);
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
    if (typeof queryHorizon === "string" && queryHorizon) {
      setRequest((prev) => ({ ...prev, horizon: queryHorizon as AnalyzeRequest["horizon"] }));
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
  // regime argmax off the persisted payload.
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const list = await fetchHistory(apiBaseUrl, { symbol: request.symbol, limit: 12 });
        if (cancelled) return;
        const items = list.items.slice(0, 12);
        const entries: RegimeHistoryEntry[] = await Promise.all(
          items.map(async (entry) => {
            let payload: AnalyzeResult | null = null;
            try {
              const detail = await fetchHistoryRun(apiBaseUrl, entry.id);
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
        if (!cancelled) {
          // Newest first → reverse so the strip reads left-to-right chronologically.
          setHistoryEntries(entries.reverse());
        }
      } catch {
        // History pull is best-effort.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, request.symbol]);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const next = await postAnalyze(apiBaseUrl, request);
      setResult(next);
      toast.success("Analysis complete");
    } catch (err) {
      setResult(null);
      const message =
        (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail ||
        (err as Error).message ||
        "Request failed. Is the backend running?";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

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
                  title="Regime classifier disabled"
                  description={
                    <p>
                      The current backend checkpoint is regression-mode or lacks a conformal sidecar with{" "}
                      <code>softmax_quantile</code>. Multi-axis, XAI, and credibility still surface below.
                    </p>
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

              {result.xai ? <XaiPanel xai={result.xai} /> : null}

              <PipelineTrace result={result} inputText={request.text} />

              <RegimeHistoryStrip entries={historyEntries} symbol={request.symbol} />
            </>
          ) : null}
        </main>
      </div>
    </>
  );
}
