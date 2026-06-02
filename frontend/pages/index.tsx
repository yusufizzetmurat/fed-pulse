import * as React from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { toast } from "sonner";

import { AnalyzeForm } from "@/components/analyze/AnalyzeForm";
import { CredibilityKpis } from "@/components/analyze/CredibilityKpis";
import { ExpectedVolumeCard } from "@/components/analyze/ExpectedVolumeCard";
import { FuturesConsensusPanel } from "@/components/analyze/FuturesConsensusPanel";
import { HistoricalAnalogPanel } from "@/components/analyze/HistoricalAnalogPanel";
import { MarketReactionPanel } from "@/components/analyze/MarketReactionPanel";
import { MonetaryPolicySurpriseChip } from "@/components/analyze/MonetaryPolicySurpriseChip";
import { MultiAxisInterpretation } from "@/components/analyze/MultiAxisInterpretation";
import { PipelineTrace } from "@/components/analyze/PipelineTrace";
import { PolicyActionCard } from "@/components/analyze/PolicyActionCard";
import { HarAccuracyPanel } from "@/components/analyze/HarAccuracyPanel";
import { HarRegimeHeadline } from "@/components/analyze/HarRegimeHeadline";
import { RvAccuracyPanel } from "@/components/analyze/RvAccuracyPanel";
import {
  RegimeHistoryStrip,
  type RegimeHistoryEntry,
} from "@/components/analyze/RegimeHistoryStrip";
import { SecondOpinionRegime } from "@/components/analyze/SecondOpinionRegime";
import { HistoricalContextBadge } from "@/components/analyze/HistoricalContextBadge";
import { SemanticDiffPanel } from "@/components/analyze/SemanticDiffPanel";
import { SentenceStrikeXaiPanel } from "@/components/analyze/SentenceStrikeXaiPanel";
import { TldrCard } from "@/components/analyze/TldrCard";
import { VolatilityOutlookCard } from "@/components/analyze/VolatilityOutlookCard";
import { WorkspaceMetaStrip } from "@/components/analyze/WorkspaceMetaStrip";
import { TrajectoryPanel } from "@/components/analyze/TrajectoryPanel";
import { WatchlistChips } from "@/components/analyze/WatchlistChips";
import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import { EmptyState } from "@/components/ui/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import {
  fetchExpectedVolumeForecast,
  fetchFuturesConsensus,
  fetchHarTercileBacktest,
  fetchRecentStanceScores,
  fetchHistoryRealizedBatch,
  fetchLatestMpSurprise,
  fetchRealizedVolForecast,
  fetchRvBacktest,
  fetchSemanticDiff,
  postAnalyze,
  postAnalyzeAnalogs,
  postAnalyzeMarket,
} from "@/lib/analyze/api";
import { DEFAULT_TEXT, HAR_TERCILE_SUPPORTED_SYMBOLS } from "@/lib/analyze/constants";
import { errorMessage } from "@/lib/analyze/errors";
import { toStance } from "@/lib/analyze/format";
import {
  useHarBaselines,
  useSharedContext,
  useSharedCoverage,
  useSharedRecentHistory,
} from "@/lib/analyze/shared-context";
import type {
  AnalogsResponse,
  AnalyzeRequest,
  AnalyzeResult,
  ExpectedVolumeForecastResponse,
  FuturesConsensusResponse,
  HarTercileBacktestResponse,
  HistoryEntry,
  Horizon,
  MarketReactionPanelResponse,
  MonetaryPolicySurpriseResponse,
  RealizedVolForecastResponse,
  RvBacktestResponse,
  SemanticDiffResponse,
  StanceContextResponse,
} from "@/lib/analyze/types";
import {
  DEFAULT_HORIZON,
  DEFAULT_SYMBOL,
  HORIZON_VALUES as HORIZON_VALUE_LIST,
  loadWorkspacePrefs,
} from "@/lib/workspace-prefs";
import { LegacyForecastCard } from "@/components/analyze/LegacyForecastCard";
import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";

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

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="border-t border-border mt-6 mb-1 pt-3">
      <p className="text-xs uppercase tracking-wide text-muted-foreground">{label}</p>
    </div>
  );
}

function takeArgmaxRegime(entry: HistoryEntry): string | null {
  // ``argmax_regime`` is denormalised onto every persisted history row,
  // so the list endpoint already carries enough signal to render the
  // strip without fanning out a /history/{id} fetch per item.
  if (entry.argmax_regime) return entry.argmax_regime;
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
  const { apiBaseUrl, refreshRecentHistory } = useSharedContext();
  const [historyEntries, setHistoryEntries] = React.useState<RegimeHistoryEntry[]>([]);
  const [volForecast, setVolForecast] = React.useState<RealizedVolForecastResponse | null>(null);
  const [volForecastLoading, setVolForecastLoading] = React.useState(false);
  const [volForecastError, setVolForecastError] = React.useState<string | null>(null);
  const [expectedVolume, setExpectedVolume] =
    React.useState<ExpectedVolumeForecastResponse | null>(null);
  const [expectedVolumeLoading, setExpectedVolumeLoading] = React.useState(false);
  const [expectedVolumeError, setExpectedVolumeError] = React.useState<string | null>(null);
  const [latestMpSurprise, setLatestMpSurprise] =
    React.useState<MonetaryPolicySurpriseResponse | null>(null);
  const [latestMpSurpriseLoading, setLatestMpSurpriseLoading] = React.useState(false);
  const [futuresConsensus, setFuturesConsensus] =
    React.useState<FuturesConsensusResponse | null>(null);
  const [futuresConsensusLoading, setFuturesConsensusLoading] = React.useState(false);
  const [semanticDiff, setSemanticDiff] = React.useState<SemanticDiffResponse | null>(null);
  const [semanticDiffLoading, setSemanticDiffLoading] = React.useState(false);
  const coverage = useSharedCoverage(request.symbol);
  const recentHistory = useSharedRecentHistory(request.symbol, 12);
  const harBaselines = useHarBaselines(request.symbol);
  const [stanceContext, setStanceContext] = React.useState<StanceContextResponse | null>(null);
  const [harBacktest, setHarBacktest] =
    React.useState<HarTercileBacktestResponse | null>(null);
  const [harBacktestLoading, setHarBacktestLoading] = React.useState(false);
  const [harBacktestError, setHarBacktestError] = React.useState<string | null>(null);
  const [rvBacktest, setRvBacktest] = React.useState<RvBacktestResponse | null>(null);
  const [rvBacktestLoading, setRvBacktestLoading] = React.useState(false);
  const [rvBacktestError, setRvBacktestError] = React.useState<string | null>(null);

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

  // Pending request to auto-submit after a deep-link prefill so the user
  // lands on a populated workspace without an extra click. ``handleSubmit``
  // is declared further down — we keep this in a ref so the effect that
  // schedules the submit doesn't need to retrigger when handleSubmit
  // changes identity each render.
  const autoSubmitPendingRef = React.useRef<AnalyzeRequest | null>(null);

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
        setRequest((prev) => {
          const next = { ...prev, text: payload.text };
          // Mark the next state as the one that should auto-submit. The
          // submit fires from a follow-up effect that watches request.text
          // so handleSubmit sees the populated request.
          autoSubmitPendingRef.current = next;
          return next;
        });
        toast.success(`Prefilled FOMC ${payload.kind} from ${queryDate}.`);
      } catch (err) {
        toast.error(errorMessage(err, "Could not load document for this date."));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router.isReady, router.query.date, router.query.symbol, router.query.horizon, router.query.kind, apiBaseUrl]);

  // Small slice of past runs for the realized-vs-predicted strip. The
  // list endpoint already carries ``argmax_regime`` per row so the
  // strip renders without a per-row /history/{id} detail fetch.
  // Realised regimes load via the batched /history-realized endpoint
  // (one round trip, not 12). The list itself is hoisted into
  // SharedContext so the workspace and any other consumer share a
  // single fetch.
  React.useEffect(() => {
    const items = recentHistory.data?.items ?? null;
    if (!items) {
      if (historyEntries.length > 0) setHistoryEntries([]);
      return;
    }
    const slice = items.slice(0, 12);
    const base: RegimeHistoryEntry[] = slice.map((entry) => ({
      runId: entry.id,
      documentDate: entry.document_date,
      argmax: takeArgmaxRegime(entry),
      realized: null,
    }));
    // Newest first → reverse so the strip reads left-to-right chronologically.
    setHistoryEntries(base.slice().reverse());

    if (slice.length === 0) return;
    const controller = new AbortController();
    (async () => {
      try {
        const batch = await fetchHistoryRealizedBatch(
          apiBaseUrl,
          slice.map((entry) => entry.id),
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setHistoryEntries(
          base
            .map((entry) => {
              const realized = batch.items[entry.runId]?.realized_regime ?? null;
              return { ...entry, realized };
            })
            .reverse(),
        );
      } catch {
        // History pull is best-effort; the strip still renders without realised.
      }
    })();
    return () => {
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiBaseUrl, recentHistory.data]);

  // The volatility outlook card is market-only and refreshes per symbol;
  // it does not wait for /analyze. A 503 from the backend (model artifact
  // missing on a fresh checkout) is surfaced as an inline error message
  // rather than blanking the card.
  React.useEffect(() => {
    const controller = new AbortController();
    setVolForecastLoading(true);
    setVolForecastError(null);
    (async () => {
      try {
        const data = await fetchRealizedVolForecast(
          apiBaseUrl,
          request.symbol,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setVolForecast(data);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setVolForecast(null);
          setVolForecastError(errorMessage(err, "Forecast unavailable."));
        }
      } finally {
        if (!controller.signal.aborted) {
          setVolForecastLoading(false);
        }
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, request.symbol]);

  // Trailing stance-score baseline for the rolling-z dashboard tile.
  // Fired on every symbol change and after each fresh /analyze so the
  // window stays aligned with the latest persisted runs. A network
  // failure here is silent — the StanceTile falls back to its raw
  // confidence rendering when context is null.
  React.useEffect(() => {
    const controller = new AbortController();
    (async () => {
      try {
        const data = await fetchRecentStanceScores(
          apiBaseUrl,
          { symbol: request.symbol, horizon: request.horizon, n: 12 },
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setStanceContext(data);
        }
      } catch {
        if (!controller.signal.aborted) {
          setStanceContext(null);
        }
      }
    })();
    return () => {
      controller.abort();
    };
    // ``result`` is the dep on purpose: a fresh /analyze response is a
    // new object reference, so the effect refires and we pick up the
    // row that was just persisted to history.
  }, [apiBaseUrl, request.symbol, request.horizon, result]);

  // Expected Volume forecast card is HAR-volume over market history,
  // market-data only. 503 (artifact missing) renders the unavailable
  // placeholder rather than a generic error toast. The AbortController
  // cleanup is the only concurrency guard we need: a re-render or
  // StrictMode remount tears down the prior fetch before the next one
  // commits state, matching the volForecast effect convention above.
  React.useEffect(() => {
    const controller = new AbortController();
    setExpectedVolumeLoading(true);
    setExpectedVolumeError(null);
    (async () => {
      try {
        const data = await fetchExpectedVolumeForecast(
          apiBaseUrl,
          request.symbol,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setExpectedVolume(data);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setExpectedVolume(null);
          setExpectedVolumeError(errorMessage(err, "HAR-volume forecast unavailable."));
        }
      } finally {
        // setState after unmount is a no-op in React 18; clearing
        // loading unconditionally avoids a sticky spinner if the
        // controller aborts mid-fetch on real navigation.
        setExpectedVolumeLoading(false);
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, request.symbol]);

  // HAR-tercile backtest panel — the endpoint is ^GSPC-only (matches
  // the regime/baselines constraint upstream). The fetcher folds a
  // 503 (downstream artifact failure) into null; the panel renders
  // the empty state when the symbol is supported but there are no
  // resolved runs yet. For non-GSPC symbols we don't fire at all and
  // surface a tailored "unavailable" placeholder.
  React.useEffect(() => {
    const controller = new AbortController();
    if (request.symbol !== "^GSPC") {
      setHarBacktest(null);
      setHarBacktestLoading(false);
      setHarBacktestError(null);
      return;
    }
    setHarBacktestLoading(true);
    setHarBacktestError(null);
    (async () => {
      try {
        const data = await fetchHarTercileBacktest(
          apiBaseUrl,
          request.symbol,
          10,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setHarBacktest(data);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setHarBacktest(null);
          setHarBacktestError(
            errorMessage(err, "HAR-tercile backtest unavailable."),
          );
        }
      } finally {
        setHarBacktestLoading(false);
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, request.symbol]);

  // QLIKE-RV band coverage panel — same ^GSPC-only constraint as the
  // HAR-tercile backtest (the RV artifact is SPX-trained). The
  // fetcher folds a 503 (model / history unavailable) into null so
  // the panel renders its tailored "unavailable" branch.
  React.useEffect(() => {
    const controller = new AbortController();
    if (request.symbol !== "^GSPC") {
      setRvBacktest(null);
      setRvBacktestLoading(false);
      setRvBacktestError(null);
      return;
    }
    setRvBacktestLoading(true);
    setRvBacktestError(null);
    (async () => {
      try {
        const data = await fetchRvBacktest(
          apiBaseUrl,
          request.symbol,
          10,
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setRvBacktest(data);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setRvBacktest(null);
          setRvBacktestError(
            errorMessage(err, "RV backtest unavailable."),
          );
        }
      } finally {
        setRvBacktestLoading(false);
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, request.symbol]);

  // MP-surprise chip is descriptive and global (latest FOMC event).
  // 503 is normalised to null inside the fetcher; here we just render
  // the unavailable placeholder.
  React.useEffect(() => {
    const controller = new AbortController();
    setLatestMpSurpriseLoading(true);
    (async () => {
      try {
        const data = await fetchLatestMpSurprise(apiBaseUrl, controller.signal);
        if (!controller.signal.aborted) {
          setLatestMpSurprise(data);
        }
      } catch {
        if (!controller.signal.aborted) {
          setLatestMpSurprise(null);
        }
      } finally {
        setLatestMpSurpriseLoading(false);
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl]);

  // FRED futures-consensus panel pulls the short-end DGS proxy on
  // mount and on every request.date change so the consensus tracks
  // the workspace as-of date. 503 already collapses to null in the
  // fetcher.
  React.useEffect(() => {
    const controller = new AbortController();
    setFuturesConsensusLoading(true);
    (async () => {
      try {
        const data = await fetchFuturesConsensus(apiBaseUrl, {
          asOf: request.date,
          signal: controller.signal,
        });
        if (!controller.signal.aborted) {
          setFuturesConsensus(data);
        }
      } catch {
        if (!controller.signal.aborted) {
          setFuturesConsensus(null);
        }
      } finally {
        setFuturesConsensusLoading(false);
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, request.date]);

  // Semantic-diff panel is gated on /analyze submit completion. The
  // diff describes "the just-analyzed statement vs the prior" — there
  // is no use case for diffing mid-typing, and POST /fomc/semantic-diff
  // runs difflib + topic emphasis server-side so per-keystroke fan-out
  // would hammer the backend on every paste. Each submit bumps a seq
  // and snapshots the submitted (text, date) into refs; the effect
  // depends on the seq alone, so re-typing in the textarea after a
  // submit cannot retrigger the POST until the next submit.
  const [semanticDiffSeq, setSemanticDiffSeq] = React.useState(0);
  const semanticDiffInputRef = React.useRef<{ text: string; date: string } | null>(null);
  // The semantic diff is anchored on the submitted text; if the user
  // changes the symbol (or wipes the date) without resubmitting, the
  // previously-rendered diff is stale and should drop out until the
  // next submit reseeds it.
  React.useEffect(() => {
    setSemanticDiff(null);
  }, [request.symbol]);
  React.useEffect(() => {
    if (semanticDiffSeq === 0) return;
    const submitted = semanticDiffInputRef.current;
    if (!submitted || !submitted.text.trim()) {
      setSemanticDiff(null);
      setSemanticDiffLoading(false);
      return;
    }
    const controller = new AbortController();
    setSemanticDiffLoading(true);
    (async () => {
      try {
        const data = await fetchSemanticDiff(
          apiBaseUrl,
          { current_date: submitted.date, current_text: submitted.text },
          controller.signal,
        );
        if (!controller.signal.aborted) {
          setSemanticDiff(data);
        }
      } catch {
        if (!controller.signal.aborted) {
          setSemanticDiff(null);
        }
      } finally {
        setSemanticDiffLoading(false);
      }
    })();
    return () => {
      controller.abort();
    };
  }, [apiBaseUrl, semanticDiffSeq]);

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
    // Snapshot the submitted (text, date) for the semantic-diff effect
    // and bump the seq so it fires exactly once per submit, independent
    // of any post-submit typing in the textarea.
    semanticDiffInputRef.current = { text: request.text, date: request.date };
    setSemanticDiffSeq((prev) => prev + 1);
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
        // Force the history strip to re-fetch so the row just
        // persisted by /analyze shows up in the Past 12 runs card.
        refreshRecentHistory(request.symbol, 12);
        toast.success("Analysis complete");
      } else {
        setResult(null);
        setBaselineResult(null);
        toast.error(errorMessage(analyzeRes.reason));
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
          toast.error(errorMessage(analyzeRes.reason));
        }
      } finally {
        if (ticket === counterfactualSeqRef.current) {
          setCounterfactualLoading(false);
        }
      }
    },
    [apiBaseUrl, baselineResult, request],
  );

  // Auto-fire handleSubmit once the deep-link prefill flushes into the
  // request state. We compare against the ref the prefill effect parked
  // there so a manual edit between prefill and run doesn't accidentally
  // trigger this branch.
  React.useEffect(() => {
    const pending = autoSubmitPendingRef.current;
    if (pending && request.text === pending.text && request.date === pending.date) {
      autoSubmitPendingRef.current = null;
      handleSubmit();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request.text, request.date]);

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

  // Approximate market-only top-pick probability via the most recent
  // history row for the same symbol that ran with no pasted text. The
  // RegimeHeadline uses this to render a "text contribution ±X.Xpp"
  // chip so the user can see at-a-glance how much the text channel
  // shifted the prediction relative to a market-only baseline.
  const marketOnlyArgmaxProb = React.useMemo<number | null>(() => {
    const items = recentHistory.data?.items ?? [];
    for (const entry of items) {
      const excerpt = (entry.text_excerpt ?? "").trim();
      if (excerpt.length > 0) continue;
      if (entry.argmax_probability == null) continue;
      return entry.argmax_probability;
    }
    return null;
  }, [recentHistory.data]);

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
          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-end sm:justify-between">
            <div className="space-y-1">
              <h1 className="text-2xl font-semibold tracking-tight">Workspace</h1>
              <p className="max-w-2xl text-sm text-muted-foreground">
                Paste an FOMC excerpt and the model returns a calibrated Volatility
                Regime prediction, a sentiment breakdown, a per-sentence explanation,
                credibility checks, and a full pipeline trace. Everything comes from the live
                backend.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <HistoricalContextBadge result={result} documentDate={request.date} />
              <Badge variant="outline" className="numeric text-[10px]">
                forecast curve · {request.horizon}
              </Badge>
            </div>
          </div>

          <AnalyzeForm
            value={request}
            onChange={setRequest}
            onSubmit={handleSubmit}
            loading={loading}
            onSampleLoad={(next) => {
              // Wipe stale analysis state before swapping in the sample's
              // request so the cards below the form do not keep rendering
              // the previous run's regime / market / analogs / multi-axis
              // output attributed to the new sample's date and symbol.
              setResult(null);
              setBaselineResult(null);
              setMarketPanel(null);
              setAnalogsPanel(null);
              setRequest(next);
            }}
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

          {/* SPINE forecast / cross-check zone — HAR-tercile headline,
              QLIKE-RV outlook, the late-fusion second-opinion cross-check,
              and the HAR-volume forecast. Second opinion sits inside the
              forecast zone deliberately: it is a regime-classifier
              cross-check (text+market vs market-only), not a descriptive
              stance summary. The accuracy diagnostics ride at the very
              bottom of the workspace so the reader sees the headline
              forecasts first and the backtest surfaces last. */}
          <SectionDivider label="Forecasts" />
          <WorkspaceSection
            title="HAR-tercile regime headline"
            description="3-class forward-vol classifier · Low / Medium / High"
            variant="forecast"
            collapsible
            storageKey="workspace-card:har-headline"
          >
            <HarRegimeHeadline
              baselines={harBaselines.data}
              loading={harBaselines.loading}
              error={harBaselines.error}
              symbol={request.symbol}
            />
          </WorkspaceSection>
          <WorkspaceSection
            title="Volatility outlook"
            description="QLIKE-RV forward outlook"
            variant="forecast"
            collapsible
            storageKey="workspace-card:vol-outlook"
          >
            <VolatilityOutlookCard
              forecast={volForecast}
              loading={volForecastLoading}
              error={volForecastError}
            />
          </WorkspaceSection>
          {result?.regime_classification ? (
            <WorkspaceSection
              title="Second opinion · late-fusion regime"
              description="Text+market vs market-only regime cross-check"
              variant="forecast"
              collapsible
              storageKey="workspace-card:second-opinion"
            >
              <SecondOpinionRegime
                regime={result.regime_classification}
                sentiment={result.sentiment}
                symbol={request.symbol}
                documentDate={request.date}
                history={regimeHistorySpark}
                empiricalCoverage={coverage.data?.empirical ?? null}
                empiricalCoverageSampleSize={coverage.data?.sample_size ?? null}
                marketOnlyArgmaxProb={marketOnlyArgmaxProb}
                harBaselines={harBaselines.data}
              />
            </WorkspaceSection>
          ) : null}
          <ExpectedVolumeCard
            forecast={expectedVolume}
            loading={expectedVolumeLoading}
            error={expectedVolumeError}
            symbol={request.symbol}
            collapsible
            storageKey="workspace-card:expected-volume"
          />

          {/* SPINE boundary — descriptive panels follow. These are
              text-derived or realized-rate commentary and never feed the
              forecast cards above. */}
          <div className="border-t-2 border-dashed border-border/70 mt-8 pt-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Descriptive context
            </p>
            <p className="mt-1 text-xs text-muted-foreground">
              Text- and realized-rate panels. Descriptive only; these signals
              never feed the forecast cards above.
            </p>
          </div>
          <MonetaryPolicySurpriseChip
            data={latestMpSurprise}
            loading={latestMpSurpriseLoading}
            collapsible
            storageKey="workspace-card:mp-surprise"
          />
          <FuturesConsensusPanel
            data={futuresConsensus}
            loading={futuresConsensusLoading}
            collapsible
            storageKey="workspace-card:futures-consensus"
          />
          <SemanticDiffPanel
            data={semanticDiff}
            loading={semanticDiffLoading}
            collapsible
            storageKey="workspace-card:semantic-diff"
          />

          {result ? (
            <>
              <SectionDivider label="Statement analysis" />
              <TldrCard result={result} />
              <WorkspaceMetaStrip result={result} />

              {!result.regime_classification && result.prediction?.close != null ? (
                <LegacyForecastCard
                  prediction={result.prediction}
                  market={result.market}
                  documentDate={request.date}
                />
              ) : !result.regime_classification ? (
                <EmptyState
                  title="Volatility Regime card unavailable."
                  description={
                    <div className="space-y-2">
                      <p>
                        The active checkpoint runs in regression mode. Re-run with a
                        classification-capable checkpoint to populate the calibrated{" "}
                        <span className="numeric">calm / normal / high</span> prediction set.
                      </p>
                      <p className="text-muted-foreground">
                        Sentiment breakdown, per-sentence explanation, credibility checks,
                        pipeline trace, and history strip below still render against the
                        current model.
                      </p>
                    </div>
                  }
                />
              ) : null}

              {result.policy_action ? (
                <PolicyActionCard action={result.policy_action} />
              ) : null}

              {marketPanel && marketPanel.rates.length > 0 ? (
                <MarketReactionPanel panel={marketPanel} />
              ) : null}

              <SectionDivider label="Sentiment and context" />
              <div className="grid gap-4 xl:grid-cols-2">
                {result.multi_axis ? (
                  <MultiAxisInterpretation
                    multiAxis={result.multi_axis}
                    stanceContext={stanceContext}
                    history={
                      stanceContext?.history.length
                        ? {
                            // ``recent-stance-scores`` returns the trailing
                            // window newest-first; KpiTile sparklines render
                            // oldest-first, so reverse before threading.
                            stance: stanceContext.history
                              .slice()
                              .reverse()
                              .map((p) => p.stance_score),
                          }
                        : undefined
                    }
                  />
                ) : (
                  <EmptyState
                    variant="inline"
                    title="Sentiment breakdown unavailable."
                    description="The active checkpoint did not return stance or certainty for this passage."
                  />
                )}
                {result.credibility ? (
                  <CredibilityKpis credibility={result.credibility} />
                ) : (
                  <EmptyState
                    variant="inline"
                    title="Credibility signals unavailable."
                    description="The credibility checks need the embedding cache and the FRED rate history to be warm-loaded; this passage produced no signals."
                  />
                )}
              </div>

              <HistoricalAnalogPanel analogs={analogsPanel} loading={analogsLoading} />

              <SectionDivider label="Model internals" />
              <PipelineTrace result={result} inputText={request.text} />

              {(baselineResult?.xai ?? result.xai) ? (
                <SentenceStrikeXaiPanel
                  xai={(baselineResult?.xai ?? result.xai)!}
                  struck={struck}
                  onMaskChange={handleStruckChange}
                  baselineResult={baselineResult}
                  currentResult={result}
                  loading={counterfactualLoading}
                />
              ) : null}

              <RegimeHistoryStrip entries={historyEntries} symbol={request.symbol} />
            </>
          ) : null}

          {/* Trajectory + diagnostic-accuracy zone — these always render
              against history (independent of the most recent /analyze
              submit) and sit at the bottom of the workspace so the reader
              treats them as diagnostic surfaces, not headline forecasts. */}
          <WorkspaceSection
            title="Stance trajectory"
            description="Embedding projection of recent FOMC stance history"
            variant="descriptive"
            collapsible
            storageKey="workspace-card:trajectory"
          >
            <TrajectoryPanel
              apiBaseUrl={apiBaseUrl}
              asOfDate={request.date}
              historyLength={12}
            />
          </WorkspaceSection>

          <p
            className="mt-4 text-xs uppercase tracking-wide text-muted-foreground"
            data-testid="workspace-accuracy-caption"
          >
            Forecast accuracy (backtest)
          </p>
          {HAR_TERCILE_SUPPORTED_SYMBOLS.includes(request.symbol) ? (
            <HarAccuracyPanel
              data={harBacktest}
              loading={harBacktestLoading}
              error={harBacktestError}
              symbol={request.symbol}
              collapsible
              storageKey="workspace-card:har-accuracy"
            />
          ) : null}
          {/* RV / QLIKE-DLq backtest remains SPX-only because the
              QLIKE-DLq artifact pins per-fold coefficients on SPX RV
              and is not yet refit per asset. */}
          {request.symbol === "^GSPC" ? (
            <RvAccuracyPanel
              data={rvBacktest}
              loading={rvBacktestLoading}
              error={rvBacktestError}
              symbol={request.symbol}
              collapsible
              storageKey="workspace-card:rv-accuracy"
            />
          ) : null}
        </main>
      </div>
    </>
  );
}
