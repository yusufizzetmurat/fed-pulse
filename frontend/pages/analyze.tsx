import * as React from "react";
import Head from "next/head";
import dynamic from "next/dynamic";
import { useRouter } from "next/router";
import { toast } from "sonner";

import { AnalyzeForm } from "@/components/analyze/AnalyzeForm";
import { ErrorBadges } from "@/components/analyze/ErrorBadges";
import { ForecastChart } from "@/components/analyze/ForecastChart";
import { MarketContext } from "@/components/analyze/MarketContext";
import { PredictionCards } from "@/components/analyze/PredictionCards";
import { RealTrainStatus } from "@/components/analyze/RealTrainStatus";
import { SentimentCard } from "@/components/analyze/SentimentCard";
import { WatchlistChips } from "@/components/analyze/WatchlistChips";
import { Header } from "@/components/shell/header";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchTrainJob, postAnalyze, resolveApiBaseUrl } from "@/lib/analyze/api";
import {
  DEFAULT_TEXT,
  REAL_TRAIN_POLL_INTERVAL_MS,
  REAL_TRAIN_POLL_MAX,
} from "@/lib/analyze/constants";
import {
  buildCloseSeries,
  buildVolatilitySeries,
  computeErrorMetrics,
} from "@/lib/analyze/derive";
import { bandLabel } from "@/lib/analyze/format";
import type { AnalyzeRequest, AnalyzeResult, TrainJobState } from "@/lib/analyze/types";

// Lazy-load fixture-driven panels so the fixture module never ships to the
// default analyze bundle. Only loaded when the toggle is on or when the API
// returns the v2 fields.
const PreviewPanels = dynamic(() => import("@/components/analyze/PreviewPanels"), {
  ssr: false,
  loading: () => null,
});

function defaultRequest(): AnalyzeRequest {
  return {
    text: DEFAULT_TEXT,
    date: new Date().toISOString().slice(0, 10),
    symbol: "^GSPC",
    forecast_mode: "fast",
    horizon: "3d",
    include_realized: false,
    include_xai: true,
  };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function trainJobMessage(state: TrainJobState): string {
  if (state.message) return state.message;
  if (state.status === "running") {
    return "Real Train running on 252-day history…";
  }
  if (state.status === "queued") return "Real Train queued…";
  if (state.status === "succeeded") return "Real Train completed. Rendering results.";
  return "";
}

export default function AnalyzePage() {
  const router = useRouter();
  const [request, setRequest] = React.useState<AnalyzeRequest>(defaultRequest);
  const [result, setResult] = React.useState<AnalyzeResult | null>(null);
  const [trainJob, setTrainJob] = React.useState<TrainJobState | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [previewV2, setPreviewV2] = React.useState(false);
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);

  React.useEffect(() => {
    if (!router.isReady) return;
    const queryDate = router.query.date;
    const queryKind = router.query.kind;
    if (typeof queryDate !== "string" || !queryDate) return;
    setRequest((current) => ({ ...current, date: queryDate }));
    if (typeof queryKind !== "string" || !queryKind) return;
    // Calendar -> Analyze deep link. Pull the FOMC statement / minutes
    // text for this date so the textarea is prefilled when the page
    // mounts; the user clicks 'Submit' instead of pasting.
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
        setRequest((current) => ({ ...current, text: payload.text }));
        toast.success(`Prefilled FOMC ${payload.kind} from ${queryDate}.`);
      } catch (err) {
        const message =
          (err as { message?: string })?.message || "Could not load document for this date.";
        toast.error(message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router.isReady, router.query.date, router.query.kind, apiBaseUrl]);

  const handleSubmit = async () => {
    setLoading(true);
    setTrainJob(null);
    try {
      const response = await postAnalyze(apiBaseUrl, request);
      if (response.mode === "result") {
        setResult(response.result);
        toast.success("Forecast ready");
        return;
      }

      setResult(null);
      setTrainJob(response.job);
      toast.info("Real Train queued — polling for completion");

      for (let i = 0; i < REAL_TRAIN_POLL_MAX; i += 1) {
        await sleep(REAL_TRAIN_POLL_INTERVAL_MS);
        const next = await fetchTrainJob(apiBaseUrl, response.job.job_id);
        setTrainJob({ ...next, message: trainJobMessage(next) });
        if (next.status === "succeeded") {
          setResult(next.result ?? null);
          toast.success("Real Train completed");
          return;
        }
        if (next.status === "failed") {
          throw new Error(next.error || "Real Train job failed.");
        }
      }
      throw new Error("Real Train timed out while waiting for completion.");
    } catch (err) {
      setResult(null);
      const message =
        (err as { response?: { data?: { detail?: string } }; message?: string })?.response?.data
          ?.detail ||
        (err as Error).message ||
        "Request failed. Ensure the backend is running.";
      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const closeSeries = React.useMemo(() => buildCloseSeries(result), [result]);
  const volatilitySeries = React.useMemo(() => buildVolatilitySeries(result), [result]);
  const errorMetrics = React.useMemo(() => computeErrorMetrics(result), [result]);

  const splitTimestamp = result?.series?.timestamps?.[result.series.timestamps.length - 1];
  const volScale = result?.series?.volatility_scale || { suggested_ymin: 0.0, suggested_ymax: 1.0 };
  const confidenceLevel = Math.round(Number(result?.series?.forecast_confidence_level || 0.8) * 100);
  const confidenceLabel = bandLabel(confidenceLevel, result?.series?.forecast_band_source);
  const hasCloseConfidence = Boolean(result?.series?.forecast_close_lower?.length);
  const hasVolConfidence = Boolean(result?.series?.forecast_volatility_lower?.length);

  return (
    <>
      <Head>
        <title>Analyze — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main id="main-content" tabIndex={-1} className="container space-y-6 py-8 focus:outline-none">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div className="space-y-2">
              <h1 className="text-3xl font-semibold tracking-tight">Analyze</h1>
              <p className="max-w-2xl text-muted-foreground">
                Score an FOMC excerpt and project asset close + volatility with confidence bands.
              </p>
            </div>
            <Button
              variant={previewV2 ? "default" : "outline"}
              size="sm"
              onClick={() => setPreviewV2((value) => !value)}
            >
              {previewV2 ? "Hide v2 preview" : "Preview v2 panels"}
            </Button>
          </div>

          <AnalyzeForm
            value={request}
            onChange={setRequest}
            onSubmit={handleSubmit}
            loading={loading}
          />

          <WatchlistChips
            currentSymbol={request.symbol}
            onSelect={(symbol) => setRequest((value) => ({ ...value, symbol }))}
          />

          {trainJob ? <RealTrainStatus job={trainJob} /> : null}

          {loading && !result ? (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              <Skeleton className="h-28 w-full" />
              <Skeleton className="h-28 w-full" />
              <Skeleton className="h-28 w-full" />
            </div>
          ) : null}

          {result ? (
            <>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                <SentimentCard sentiment={result.sentiment} />
                <div className="md:col-span-2 xl:col-span-2">
                  <PredictionCards result={result} />
                </div>
              </div>

              {result.multi_axis || previewV2 ? (
                <PreviewPanels slot="cards" multiAxis={result.multi_axis} />
              ) : null}

              <ErrorBadges result={result} metrics={errorMetrics} />

              <div className="grid gap-4 xl:grid-cols-2">
                <ForecastChart
                  title="Close forecast"
                  description={`Forecast line and ${confidenceLabel} over the requested horizon.`}
                  data={closeSeries}
                  kind="close"
                  splitTimestamp={splitTimestamp}
                  includeRealized={request.include_realized}
                  hasConfidence={hasCloseConfidence}
                  confidenceLabel={confidenceLabel}
                />
                <ForecastChart
                  title="Volatility forecast"
                  description={`Forecast line and ${confidenceLabel} over the requested horizon.`}
                  data={volatilitySeries}
                  kind="volatility"
                  splitTimestamp={splitTimestamp}
                  includeRealized={request.include_realized}
                  hasConfidence={hasVolConfidence}
                  confidenceLabel={confidenceLabel}
                  yDomain={[
                    Number(volScale.suggested_ymin ?? 0),
                    Number(volScale.suggested_ymax ?? 1),
                  ]}
                />
              </div>

              {result.xai || previewV2 ? (
                <PreviewPanels slot="xai" xai={result.xai} />
              ) : null}
              {result.credibility || previewV2 ? (
                <PreviewPanels slot="credibility" credibility={result.credibility} />
              ) : null}

              <MarketContext result={result} />
            </>
          ) : null}
        </main>
      </div>
    </>
  );
}
