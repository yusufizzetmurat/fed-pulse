import * as React from "react";
import Head from "next/head";
import { useRouter } from "next/router";
import { ArrowDownRight, ArrowRight, ArrowUpRight, GitCompare } from "lucide-react";
import { toast } from "sonner";

import { MultiAxisCards } from "@/components/analyze/MultiAxisCards";
import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchHistory, fetchHistoryRun, resolveApiBaseUrl } from "@/lib/analyze/api";
import {
  computeCompareDelta,
  computeMultiAxisDelta,
  describeStanceShift,
  type CompareDelta,
  type MultiAxisDelta,
} from "@/lib/analyze/compare";
import { downloadCompareCsv } from "@/lib/export/compare-export";
import { downloadComparePdf } from "@/lib/export/pdf";
import { stanceLabel, toStance } from "@/lib/analyze/format";
import type { AnalyzeResult, HistoryDetail, HistoryEntry } from "@/lib/analyze/types";

const SLOT_LABELS = { a: "Run A", b: "Run B" } as const;
type Slot = keyof typeof SLOT_LABELS;

function formatDelta(value: number | null, fractionDigits = 2): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "" : "";
  return `${sign}${value.toFixed(fractionDigits)}`;
}

function deltaColorClass(value: number | null): string {
  if (value == null || value === 0) return "text-muted-foreground";
  return value > 0 ? "text-hawkish" : "text-dovish";
}

function DeltaIcon({ value }: { value: number | null }) {
  if (value == null || value === 0) return <ArrowRight className="h-3.5 w-3.5" />;
  return value > 0 ? <ArrowUpRight className="h-3.5 w-3.5" /> : <ArrowDownRight className="h-3.5 w-3.5" />;
}

function RunSlotCard({
  slot,
  detail,
  loading,
  entries,
  selectedId,
  onSelect,
}: {
  slot: Slot;
  detail: HistoryDetail | null;
  loading: boolean;
  entries: HistoryEntry[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}) {
  const stance = detail ? toStance(detail.stance) : "unknown";
  const stanceVariant: "hawkish" | "dovish" | "neutral" | "outline" =
    stance === "hawkish" ? "hawkish" : stance === "dovish" ? "dovish" : stance === "neutral" ? "neutral" : "outline";

  return (
    <Card>
      <CardHeader className="space-y-3">
        <CardDescription>{SLOT_LABELS[slot]}</CardDescription>
        <Select
          value={selectedId ?? ""}
          onValueChange={(value) => onSelect(value || null)}
        >
          <SelectTrigger aria-label={`Select ${SLOT_LABELS[slot]}`}>
            <SelectValue placeholder="Pick a run…" />
          </SelectTrigger>
          <SelectContent>
            {entries.map((entry) => (
              <SelectItem key={entry.id} value={entry.id}>
                {entry.document_date} · {entry.symbol} · {stanceLabel(toStance(entry.stance))}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent className="space-y-4">
        {loading ? (
          <div className="space-y-2">
            <Skeleton className="h-6 w-1/2" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-4 w-2/3" />
          </div>
        ) : !detail ? (
          <p className="text-sm text-muted-foreground">
            No run selected. Pick one from the dropdown to compare.
          </p>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge variant={stanceVariant}>{stanceLabel(stance)}</Badge>
              <span className="text-muted-foreground">{detail.symbol}</span>
              <span className="text-muted-foreground">·</span>
              <span className="text-muted-foreground">{detail.horizon}</span>
              <span className="text-muted-foreground">·</span>
              <span className="font-mono text-xs text-muted-foreground">
                {detail.document_date}
              </span>
            </div>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
              <dt className="text-muted-foreground">Sentiment score</dt>
              <dd className="text-right font-mono">
                {detail.sentiment_score != null ? detail.sentiment_score.toFixed(3) : "—"}
              </dd>
            </dl>
            {detail.text_excerpt ? (
              <p className="line-clamp-3 rounded-md border border-border bg-muted/30 p-3 text-xs text-muted-foreground">
                {detail.text_excerpt}
              </p>
            ) : null}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function MultiAxisDeltaCard({ delta }: { delta: MultiAxisDelta }) {
  const hasSignal =
    delta.stanceRankDelta != null ||
    delta.factorDelta != null ||
    delta.certaintyShift !== "unknown" ||
    delta.topicChanged != null;
  if (!hasSignal) return null;
  const stanceDir =
    delta.stanceRankDelta == null
      ? "—"
      : delta.stanceRankDelta > 0
      ? `+${delta.stanceRankDelta.toFixed(1)} hawkish`
      : delta.stanceRankDelta < 0
      ? `${delta.stanceRankDelta.toFixed(1)} dovish`
      : "0";
  const certaintyMessage =
    delta.certaintyShift === "more_decisive"
      ? "A more decisive"
      : delta.certaintyShift === "more_tentative"
      ? "A more tentative"
      : delta.certaintyShift === "unchanged"
      ? "Certainty unchanged"
      : "Certainty unknown";
  return (
    <Card>
      <CardHeader>
        <CardTitle>Multi-axis Δ A − B</CardTitle>
        <CardDescription>
          Per-axis deltas from the multi-axis schema. Missing axes appear as
          "—" when at least one side does not carry that axis.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-0">
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-4 py-2 text-left">Axis</th>
              <th className="px-4 py-2 text-right">Δ A − B</th>
              <th className="px-4 py-2 text-right">Confidence Δ</th>
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-border">
              <td className="px-4 py-2 font-mono">stance</td>
              <td className="px-4 py-2 text-right font-mono">{stanceDir}</td>
              <td className="px-4 py-2 text-right font-mono">
                {delta.stanceConfidenceDelta != null
                  ? `${delta.stanceConfidenceDelta >= 0 ? "+" : ""}${delta.stanceConfidenceDelta.toFixed(2)}`
                  : "—"}
              </td>
            </tr>
            <tr className="border-b border-border">
              <td className="px-4 py-2 font-mono">factor</td>
              <td className="px-4 py-2 text-right font-mono">
                {delta.factorDelta != null
                  ? `${delta.factorDelta >= 0 ? "+" : ""}${delta.factorDelta.toFixed(2)}`
                  : "—"}
              </td>
              <td className="px-4 py-2 text-right font-mono">
                {delta.factorConfidenceDelta != null
                  ? `${delta.factorConfidenceDelta >= 0 ? "+" : ""}${delta.factorConfidenceDelta.toFixed(2)}`
                  : "—"}
              </td>
            </tr>
            <tr className="border-b border-border">
              <td className="px-4 py-2 font-mono">certainty</td>
              <td className="px-4 py-2 text-right text-xs text-muted-foreground">
                {certaintyMessage}
              </td>
              <td className="px-4 py-2 text-right font-mono">
                {delta.certaintyConfidenceDelta != null
                  ? `${delta.certaintyConfidenceDelta >= 0 ? "+" : ""}${delta.certaintyConfidenceDelta.toFixed(2)}`
                  : "—"}
              </td>
            </tr>
            <tr>
              <td className="px-4 py-2 font-mono">topic</td>
              <td className="px-4 py-2 text-right text-xs text-muted-foreground" colSpan={2}>
                {delta.topicChanged == null
                  ? "—"
                  : delta.topicChanged
                  ? "primary topic changed"
                  : "primary topic unchanged"}
              </td>
            </tr>
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function MultiAxisSideBySide({
  detailA,
  detailB,
}: {
  detailA: HistoryDetail;
  detailB: HistoryDetail;
}) {
  const ma = ((detailA.payload || {}) as AnalyzeResult).multi_axis;
  const mb = ((detailB.payload || {}) as AnalyzeResult).multi_axis;
  if (!ma && !mb) return null;
  return (
    <div className="grid gap-4 xl:grid-cols-2">
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">Run A · multi-axis</p>
        {ma ? <MultiAxisCards multiAxis={ma} /> : (
          <Card>
            <CardContent className="py-6 text-center text-sm text-muted-foreground">
              Run A has no multi-axis payload.
            </CardContent>
          </Card>
        )}
      </div>
      <div className="space-y-2">
        <p className="text-xs uppercase tracking-wide text-muted-foreground">Run B · multi-axis</p>
        {mb ? <MultiAxisCards multiAxis={mb} /> : (
          <Card>
            <CardContent className="py-6 text-center text-sm text-muted-foreground">
              Run B has no multi-axis payload.
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

function DeltaSummary({ delta }: { delta: CompareDelta }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitCompare className="h-4 w-4 text-primary" />
          Δ A − B
        </CardTitle>
        <CardDescription>{describeStanceShift(delta.stanceShift)}</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
          <div>
            <dt className="text-xs uppercase tracking-wide text-muted-foreground">Sentiment score</dt>
            <dd className={`flex items-center gap-1 font-mono ${deltaColorClass(delta.scoreDelta)}`}>
              <DeltaIcon value={delta.scoreDelta} />
              {formatDelta(delta.scoreDelta, 3)}
            </dd>
          </div>
        </dl>
        <p className="mt-3 text-xs text-muted-foreground">
          Regime + multi-axis deltas land in a follow-up — this view is interim while the compare page
          is realigned to the vol-regime classifier.
        </p>
      </CardContent>
    </Card>
  );
}

export default function ComparePage() {
  const router = useRouter();
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [entries, setEntries] = React.useState<HistoryEntry[]>([]);
  const [entriesLoading, setEntriesLoading] = React.useState(true);
  const [detailA, setDetailA] = React.useState<HistoryDetail | null>(null);
  const [detailB, setDetailB] = React.useState<HistoryDetail | null>(null);
  const [loadingA, setLoadingA] = React.useState(false);
  const [loadingB, setLoadingB] = React.useState(false);

  const aId = React.useMemo(() => {
    const value = router.query.a;
    return typeof value === "string" ? value : null;
  }, [router.query.a]);
  const bId = React.useMemo(() => {
    const value = router.query.b;
    return typeof value === "string" ? value : null;
  }, [router.query.b]);

  React.useEffect(() => {
    let cancelled = false;
    setEntriesLoading(true);
    fetchHistory(apiBaseUrl, { limit: 50, offset: 0 })
      .then((result) => {
        if (!cancelled) setEntries(result.items);
      })
      .catch((err) => {
        if (!cancelled) toast.error((err as Error).message || "Failed to load history list.");
      })
      .finally(() => {
        if (!cancelled) setEntriesLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  // Each slot loads independently so picking a single run still renders
  // that side while the other slot is empty.
  React.useEffect(() => {
    let cancelled = false;
    if (!aId) {
      setDetailA(null);
      return;
    }
    setLoadingA(true);
    fetchHistoryRun(apiBaseUrl, aId)
      .then((detail) => {
        if (!cancelled) setDetailA(detail);
      })
      .catch((err) => {
        if (!cancelled) toast.error((err as Error).message || "Failed to load run A.");
      })
      .finally(() => {
        if (!cancelled) setLoadingA(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, aId]);

  React.useEffect(() => {
    let cancelled = false;
    if (!bId) {
      setDetailB(null);
      return;
    }
    setLoadingB(true);
    fetchHistoryRun(apiBaseUrl, bId)
      .then((detail) => {
        if (!cancelled) setDetailB(detail);
      })
      .catch((err) => {
        if (!cancelled) toast.error((err as Error).message || "Failed to load run B.");
      })
      .finally(() => {
        if (!cancelled) setLoadingB(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, bId]);

  const handleSelect = React.useCallback(
    (slot: Slot, id: string | null) => {
      const next = { ...router.query };
      if (id) {
        next[slot] = id;
      } else {
        delete next[slot];
      }
      router.replace({ pathname: "/compare", query: next }, undefined, { shallow: true });
    },
    [router],
  );

  const delta = React.useMemo(() => {
    if (!detailA || !detailB) return null;
    return computeCompareDelta(detailA, detailB);
  }, [detailA, detailB]);

  const multiAxisDelta = React.useMemo(() => {
    if (!detailA || !detailB) return null;
    return computeMultiAxisDelta(detailA, detailB);
  }, [detailA, detailB]);

  const handleExportCsv = React.useCallback(() => {
    if (!detailA || !detailB) return;
    try {
      downloadCompareCsv(detailA, detailB);
    } catch (err) {
      toast.error((err as Error).message || "CSV export failed.");
    }
  }, [detailA, detailB]);

  const handleExportPdf = React.useCallback(() => {
    if (!detailA || !detailB) return;
    downloadComparePdf(detailA, detailB).catch((err) => {
      toast.error((err as Error).message || "PDF export failed.");
    });
  }, [detailA, detailB]);

  return (
    <>
      <Head>
        <title>Compare runs — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main id="main-content" tabIndex={-1} className="container space-y-6 py-8 focus:outline-none">
          <div className="space-y-2">
            <h1 className="text-3xl font-semibold tracking-tight">Compare runs</h1>
            <p className="max-w-2xl text-muted-foreground">
              Pick two past analyses and see the stance, prediction, and confidence deltas side by
              side. Selections are sticky in the URL — share the link to send a paired view.
            </p>
          </div>

          {entriesLoading ? (
            <div className="grid gap-4 md:grid-cols-2">
              <Skeleton className="h-48 w-full" />
              <Skeleton className="h-48 w-full" />
            </div>
          ) : entries.length === 0 ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                No runs yet — submit at least two analyses before using this page.
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-4 md:grid-cols-2">
              <RunSlotCard
                slot="a"
                detail={detailA}
                loading={loadingA}
                entries={entries}
                selectedId={aId}
                onSelect={(id) => handleSelect("a", id)}
              />
              <RunSlotCard
                slot="b"
                detail={detailB}
                loading={loadingB}
                entries={entries}
                selectedId={bId}
                onSelect={(id) => handleSelect("b", id)}
              />
            </div>
          )}

          {delta ? <DeltaSummary delta={delta} /> : null}
          {multiAxisDelta ? <MultiAxisDeltaCard delta={multiAxisDelta} /> : null}
          {detailA && detailB ? (
            <>
              <MultiAxisSideBySide detailA={detailA} detailB={detailB} />
              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={handleExportCsv}>
                  Export CSV
                </Button>
                <Button variant="outline" size="sm" onClick={handleExportPdf}>
                  Export PDF
                </Button>
              </div>
            </>
          ) : null}
        </main>
      </div>
    </>
  );
}
