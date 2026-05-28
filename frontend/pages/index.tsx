import * as React from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { toast } from "sonner";

import { AnalyzeForm } from "@/components/analyze/AnalyzeForm";
import { CredibilityKpis } from "@/components/analyze/CredibilityKpis";
import { HistoricalAnalogPanel } from "@/components/analyze/HistoricalAnalogPanel";
import { MarketReactionPanel } from "@/components/analyze/MarketReactionPanel";
import { MultiAxisInterpretation } from "@/components/analyze/MultiAxisInterpretation";
import { PipelineTrace } from "@/components/analyze/PipelineTrace";
import { PolicyActionCard } from "@/components/analyze/PolicyActionCard";
import { RegimeHeadline } from "@/components/analyze/RegimeHeadline";
import {
  RegimeHistoryStrip,
  type RegimeHistoryEntry,
} from "@/components/analyze/RegimeHistoryStrip";
import { SentenceStrikeXaiPanel } from "@/components/analyze/SentenceStrikeXaiPanel";
import { TrajectoryPanel } from "@/components/analyze/TrajectoryPanel";
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
  postAnalyzeAnalogs,
  postAnalyzeMarket,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { DEFAULT_TEXT } from "@/lib/analyze/constants";
import { toStance } from "@/lib/analyze/format";
import { useEvaluationCoverage } from "@/lib/analyze/useEvaluationCoverage";
import type {
  AnalogsResponse,
  AnalyzeRequest,
  AnalyzeResult,
  HistoryEntry,
  Horizon,
  MarketReactionPanelResponse,
} from "@/lib/analyze/types";
import {
  DEFAULT_HORIZON,
  DEFAULT_SYMBOL,
  HORIZON_VALUES as HORIZON_VALUE_LIST,
  loadWorkspacePrefs,
} from "@/lib/workspace-prefs";
import { LegacyForecastCard } from "@/components/analyze/LegacyForecastCard";

const HORIZON_VALUES = new Set<Horizon>(HORIZON_VALUE_LIST);

// Initial request used by both SSR and the client's first paint. localStorage
// prefs are applied after mount inside the component (see usePrefHydration)
// so SSR + client agree and React does not log a hydration mismatch.
function defaultRequest(): AnalyzeRequest {
  return {
    text: DEFAULT_TEXT,
    date: new Date().toISOString().slice(0, 10),
    symbol: DEFAULT_SYMBOL,
    horizon: DEFAULT_HORIZON,
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
  const [marketPanel, setMarketPanel] = React.useState<MarketReactionPanelResponse | null>(null);
  const [analogsPanel, setAnalogsPanel] = React.useState<AnalogsResponse | null>(null);
  const [analogsLoading, setAnalogsLoading] = React.useState(false);
  const [struck, setStruck] = React.useState<Set<number>>(() => new Set());
  const [counterfactualLoading, setCounterfactualLoading] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [historyEntries, setHistoryEntries] = React.useState<RegimeHistoryEntry[]>([]);
  const coverage = useEvaluationCoverage(apiBaseUrl, { symbol: request.symbol });

  // Apply saved workspace prefs (default symbol / horizon) after mount.
  // Doing this in an effect rather than the initial state preserves the
  // SSR ↔ hydration agreement; otherwise a user with non-default prefs
  // gets a hydration mismatch warning on first paint.
  React.useEffect(() => {
    const prefs = loadWorkspacePrefs();
    setRequest((prev) => {
      if (prev.symbol === prefs.defaultSymbol && prev.horizon === prefs.defaultHorizon) {
        return prev;
      }
      return { ...prev, symbol: prefs.defaultSymbol, horizon: prefs.defaultHorizon };
    });
  }, []);

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

  // #317 finding #14: monotonic seq for /analyze/market fetches. A
  // second submit before the first market fetch resolves bumps the
  // seq; the late-arriving resolver checks the seq and skips state
  // commit if it does not match the most recent issued id. Mirrors
  // the counterfactualSeqRef pattern below. The analogs panel uses
  // the same guard so a stale slow-arriving retrieval response cannot
  // overwrite the newest submit.
  const marketPanelSeqRef = React.useRef(0);
  const analogsPanelSeqRef = React.useRef(0);

  const handleSubmit = async () => {
    setLoading(true);
    setStruck(new Set());
    marketPanelSeqRef.current += 1;
    analogsPanelSeqRef.current += 1;
    const marketTicket = marketPanelSeqRef.current;
    const analogsTicket = analogsPanelSeqRef.current;
    setAnalogsLoading(true);
    try {
      // #317 finding #13: dispatch /analyze + /analyze/market in
      // parallel rather than sequentially -- the market panel does
      // not depend on the /analyze response payload. Halves the
      // user-visible latency on a submit click. Wrapped via
      // ``Promise.allSettled`` so a failure on the optional market
      // panel does not abort the primary /analyze surface. The
      // analogs fetch joins the same fan-out.
      const sharedRequest = { ...request, mask_sentence_indices: [] };
      const [analyzeRes, marketRes, analogsRes] = await Promise.allSettled([
        postAnalyze(apiBaseUrl, sharedRequest),
        postAnalyzeMarket(apiBaseUrl, sharedRequest),
        postAnalyzeAnalogs(apiBaseUrl, {
          text: request.text,
          k: 5,
          as_of_date: request.date,
        }),
      ]);
      if (analyzeRes.status === "fulfilled") {
        setResult(analyzeRes.value);
        setBaselineResult(analyzeRes.value);
        toast.success("Analysis complete");
      } else {
        setResult(null);
        setBaselineResult(null);
        const err = analyzeRes.reason;
        const message =
          (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail ||
          (err as Error).message ||
          "Request failed. Is the backend running?";
        toast.error(message);
      }
      // Commit the market panel only when the seq still matches the
      // most recent submit. Older in-flight fetches that arrive late
      // never overwrite a newer submit's result.
      if (marketTicket === marketPanelSeqRef.current) {
        setMarketPanel(
          marketRes.status === "fulfilled" ? marketRes.value : null,
        );
      }
      if (analogsTicket === analogsPanelSeqRef.current) {
        setAnalogsPanel(
          analogsRes.status === "fulfilled" ? analogsRes.value : null,
        );
      }
    } finally {
      setLoading(false);
      if (analogsTicket === analogsPanelSeqRef.current) {
        setAnalogsLoading(false);
      }
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
      marketPanelSeqRef.current += 1;
      const ticket = counterfactualSeqRef.current;
      const marketTicket = marketPanelSeqRef.current;
      setCounterfactualLoading(true);
      try {
        const indices = Array.from(mask).sort((a, b) => a - b);
        // #317 finding #15: refetch the market panel alongside the
        // sentence-strike counterfactual so the per-head rates cards
        // reflect the masked input rather than staying stale on the
        // baseline forward pass. The historical analog panel is
        // intentionally NOT refetched here: ``AnalogsRequest`` has no
        // ``mask_sentence_indices`` field, so a refetch would resend
        // the same original text and produce the same top-k. The
        // panel is contextual ("statements that sound like the one
        // you pasted"), not attributional, so freezing it on the
        // baseline retrieval is the right semantics under a strike.
        const sharedRequest = { ...request, mask_sentence_indices: indices };
        const [analyzeRes, marketRes] = await Promise.allSettled([
          postAnalyze(apiBaseUrl, sharedRequest),
          postAnalyzeMarket(apiBaseUrl, sharedRequest),
        ]);
        if (
          ticket === counterfactualSeqRef.current
          && analyzeRes.status === "fulfilled"
        ) {
          setResult(analyzeRes.value);
        }
        if (marketTicket === marketPanelSeqRef.current) {
          setMarketPanel(
            marketRes.status === "fulfilled" ? marketRes.value : null,
          );
        }
        if (analyzeRes.status === "rejected") {
          const err = analyzeRes.reason;
          const message =
            (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data?.detail ||
            (err as Error).message ||
            "Counterfactual request failed.";
          toast.error(message);
        }
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
        <title>Fed Pulse — Volatility Regime workspace</title>
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
                Paste an FOMC excerpt and the model returns a calibrated 10-day Volatility
                Regime prediction, a sentiment breakdown, a per-sentence explanation,
                credibility checks, and a full pipeline trace. Everything comes from the live
                backend.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="numeric text-[10px]">
                horizon · 10 days
              </Badge>
              <Badge variant="outline" className="numeric text-[10px]">
                target · Volatility Regime
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
                  empiricalCoverage={coverage.data?.empirical ?? null}
                  empiricalCoverageSampleSize={coverage.data?.sample_size ?? null}
                />
              ) : result.prediction?.close != null ? (
                <LegacyForecastCard
                  prediction={result.prediction}
                  market={result.market}
                  documentDate={request.date}
                />
              ) : (
                <EmptyState
                  title="Volatility Regime card not available in this build"
                  description={
                    <div className="space-y-2">
                      <p>
                        The active forecaster is in numeric mode — it predicts close and
                        volatility as single numbers rather than a calibrated{" "}
                        <span className="numeric">calm / normal / high</span> prediction set.
                        The Volatility Regime classifier and its calibration data are not loaded.
                      </p>
                      <p className="text-muted-foreground">
                        Every other workspace panel below — sentiment breakdown, per-sentence
                        explanation, credibility checks, pipeline trace, and history strip — is
                        live against the current model.
                      </p>
                    </div>
                  }
                />
              )}

              {result.policy_action ? (
                <PolicyActionCard action={result.policy_action} />
              ) : null}

              <div className="grid gap-4 xl:grid-cols-2">
                {result.multi_axis ? (
                  <MultiAxisInterpretation multiAxis={result.multi_axis} />
                ) : (
                  <EmptyState
                    variant="inline"
                    title="Sentiment breakdown not loaded"
                    description="Load the sentiment model to populate stance, factor, certainty, and topic."
                  />
                )}
                {result.credibility ? (
                  <CredibilityKpis credibility={result.credibility} />
                ) : (
                  <EmptyState
                    variant="inline"
                    title="Credibility signals unavailable"
                    description="The required embedding model and historical rate cache are not loaded on this host yet."
                  />
                )}
              </div>

              {marketPanel && (marketPanel.rates.length > 0 || marketPanel.vol_regime) ? (
                <MarketReactionPanel panel={marketPanel} />
              ) : null}

              <HistoricalAnalogPanel analogs={analogsPanel} loading={analogsLoading} />


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

              <TrajectoryPanel
                apiBaseUrl={apiBaseUrl}
                asOfDate={request.date}
                historyLength={12}
              />

              <PipelineTrace result={result} inputText={request.text} />

              <RegimeHistoryStrip entries={historyEntries} symbol={request.symbol} />
            </>
          ) : null}
        </main>
      </div>
    </>
  );
}
