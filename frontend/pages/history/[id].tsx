import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import dynamic from "next/dynamic";
import { ArrowLeft, Download, GitCompare } from "lucide-react";
import { toast } from "sonner";

import { ErrorBadges } from "@/components/analyze/ErrorBadges";
import { ForecastChart } from "@/components/analyze/ForecastChart";
import { MarketContext } from "@/components/analyze/MarketContext";
import { PredictionCards } from "@/components/analyze/PredictionCards";
import { SentimentCard } from "@/components/analyze/SentimentCard";
import { Header } from "@/components/shell/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchHistoryRun, resolveApiBaseUrl } from "@/lib/analyze/api";
import { downloadRunCsv } from "@/lib/export/run-export";
import { downloadRunPdf } from "@/lib/export/pdf";
import {
  buildCloseSeries,
  buildVolatilitySeries,
  computeErrorMetrics,
} from "@/lib/analyze/derive";
import { bandLabel, stanceLabel, toStance } from "@/lib/analyze/format";
import type { AnalyzeResult, HistoryDetail } from "@/lib/analyze/types";

const PreviewPanels = dynamic(() => import("@/components/analyze/PreviewPanels"), {
  ssr: false,
  loading: () => null,
});

function detailToResult(detail: HistoryDetail | null): AnalyzeResult | null {
  if (!detail) return null;
  return (detail.payload || {}) as AnalyzeResult;
}

export default function HistoryDetailPage() {
  const router = useRouter();
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [detail, setDetail] = React.useState<HistoryDetail | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);

  const runId = React.useMemo(() => {
    const value = router.query.id;
    return typeof value === "string" ? value : null;
  }, [router.query.id]);

  React.useEffect(() => {
    if (!router.isReady) return;
    if (!runId) return;
    let cancelled = false;
    setLoading(true);
    setErrorMessage(null);
    fetchHistoryRun(apiBaseUrl, runId)
      .then((data) => {
        if (!cancelled) setDetail(data);
      })
      .catch((err) => {
        if (cancelled) return;
        const message = (err as Error).message || "Run not found.";
        setErrorMessage(message);
        toast.error(message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, router.isReady, runId]);

  const result = React.useMemo(() => detailToResult(detail), [detail]);
  const closeSeries = React.useMemo(() => buildCloseSeries(result), [result]);
  const volatilitySeries = React.useMemo(() => buildVolatilitySeries(result), [result]);
  const errorMetrics = React.useMemo(() => computeErrorMetrics(result), [result]);
  const splitTimestamp = result?.series?.timestamps?.[result.series.timestamps.length - 1];
  const confidenceLevel = Math.round(Number(result?.series?.forecast_confidence_level || 0.8) * 100);
  const confidenceLabel = bandLabel(confidenceLevel, result?.series?.forecast_band_source);
  const hasCloseConfidence = Boolean(result?.series?.forecast_close_lower?.length);
  const hasVolConfidence = Boolean(result?.series?.forecast_volatility_lower?.length);
  const volScale = result?.series?.volatility_scale || { suggested_ymin: 0.0, suggested_ymax: 1.0 };
  const stance = detail ? toStance(detail.stance) : "unknown";

  return (
    <>
      <Head>
        <title>{detail ? `Run ${detail.document_date} — Fed Pulse` : "History run — Fed Pulse"}</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main className="container space-y-6 py-8">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Button asChild variant="ghost" size="sm">
                <Link href="/history">
                  <ArrowLeft className="h-4 w-4" />
                  History
                </Link>
              </Button>
              {detail ? (
                <h1 className="flex flex-wrap items-center gap-2 text-2xl font-semibold tracking-tight">
                  <span>{detail.document_date}</span>
                  <Badge variant="outline">{detail.symbol}</Badge>
                  <Badge variant="outline">{detail.horizon}</Badge>
                  <Badge variant="outline" className="capitalize">{detail.forecast_mode}</Badge>
                  <Badge variant={stance === "hawkish" ? "hawkish" : stance === "dovish" ? "dovish" : stance === "neutral" ? "neutral" : "outline"}>
                    {stanceLabel(stance)}
                  </Badge>
                </h1>
              ) : (
                <h1 className="text-2xl font-semibold tracking-tight">History run</h1>
              )}
            </div>
            {detail ? (
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    try {
                      downloadRunCsv(detail);
                    } catch (err) {
                      toast.error((err as Error).message || "CSV export failed.");
                    }
                  }}
                >
                  <Download className="h-4 w-4" />
                  Export CSV
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    downloadRunPdf(detail).catch((err) => {
                      toast.error((err as Error).message || "PDF export failed.");
                    });
                  }}
                >
                  <Download className="h-4 w-4" />
                  Export PDF
                </Button>
                <Button asChild variant="outline" size="sm">
                  <Link href={`/compare?a=${detail.id}`}>
                    <GitCompare className="h-4 w-4" />
                    Compare with…
                  </Link>
                </Button>
              </div>
            ) : null}
          </div>

          {loading ? (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              <Skeleton className="h-28 w-full" />
              <Skeleton className="h-28 w-full" />
              <Skeleton className="h-28 w-full" />
            </div>
          ) : errorMessage ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                {errorMessage}
              </CardContent>
            </Card>
          ) : detail && result ? (
            <>
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                <SentimentCard sentiment={result.sentiment} />
                <div className="md:col-span-2 xl:col-span-2">
                  <PredictionCards result={result} />
                </div>
              </div>

              {result.multi_axis ? <PreviewPanels slot="cards" multiAxis={result.multi_axis} /> : null}

              <ErrorBadges result={result} metrics={errorMetrics} />

              <div className="grid gap-4 xl:grid-cols-2">
                <ForecastChart
                  title="Close forecast"
                  description={`Forecast line and ${confidenceLabel} over the requested horizon.`}
                  data={closeSeries}
                  kind="close"
                  splitTimestamp={splitTimestamp}
                  includeRealized={Boolean(result.series?.realized_timestamps?.length)}
                  hasConfidence={hasCloseConfidence}
                  confidenceLabel={confidenceLabel}
                />
                <ForecastChart
                  title="Volatility forecast"
                  description={`Forecast line and ${confidenceLabel} over the requested horizon.`}
                  data={volatilitySeries}
                  kind="volatility"
                  splitTimestamp={splitTimestamp}
                  includeRealized={Boolean(result.series?.realized_timestamps?.length)}
                  hasConfidence={hasVolConfidence}
                  confidenceLabel={confidenceLabel}
                  yDomain={[
                    Number(volScale.suggested_ymin ?? 0),
                    Number(volScale.suggested_ymax ?? 1),
                  ]}
                />
              </div>

              {result.xai ? <PreviewPanels slot="xai" xai={result.xai} /> : null}
              {result.credibility ? <PreviewPanels slot="credibility" credibility={result.credibility} /> : null}

              <MarketContext result={result} />

              {detail.text_excerpt ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Submitted text</CardTitle>
                    <CardDescription>First {detail.text_excerpt.length} characters of the analysed document.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="whitespace-pre-wrap rounded-md border border-border bg-muted/30 p-3 font-mono text-xs leading-relaxed">
                      {detail.text_excerpt}
                    </p>
                  </CardContent>
                </Card>
              ) : null}
            </>
          ) : (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                Run not found.
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
