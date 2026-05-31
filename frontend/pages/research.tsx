import * as React from "react";
import Head from "next/head";
import { FlaskConical } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "sonner";

import { DecisionsLink } from "@/components/research/DecisionsLink";
import { JobsLink } from "@/components/research/JobsLink";
import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { fetchResearchArtifacts, resolveApiBaseUrl } from "@/lib/analyze/api";
import { errorMessage } from "@/lib/analyze/errors";
import type {
  ArtifactFile,
  CrossBankTransferSection,
  EncoderBakeoffSection,
  ResearchArtifactsResponse,
  TransferMatrixCell,
} from "@/lib/analyze/types";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatNumberOrDash(value: number | null | undefined, digits: number = 3): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

function formatCi(low: number | null, high: number | null): string {
  if (low == null || high == null) return "—";
  return `[${low.toFixed(3)}, ${high.toFixed(3)}]`;
}

function heatmapColor(value: number, min: number, max: number): string {
  // Two-stop ramp through neutral muted to emerald 500 — readable in both themes.
  if (max <= min) return "rgba(16, 185, 129, 0.15)";
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const alpha = 0.12 + 0.55 * t;
  return `rgba(16, 185, 129, ${alpha.toFixed(3)})`;
}

const BAKEOFF_TOOLTIP_STYLE: React.CSSProperties = {
  background: "hsl(var(--popover))",
  color: "hsl(var(--popover-foreground))",
  border: "1px solid hsl(var(--border))",
  borderRadius: 6,
  padding: "6px 8px",
  fontSize: 12,
};

function bakeoffBarColor(value: number, min: number, max: number): string {
  if (max <= min) return "hsl(var(--primary))";
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  // Sequential primary-tinted ramp; lighter at low scores.
  const alpha = 0.35 + 0.6 * t;
  return `hsla(var(--primary) / ${alpha.toFixed(3)})`;
}

interface BakeoffBarDatum {
  name: string;
  macroF1: number;
  ciLow: number | null;
  ciHigh: number | null;
  // recharts error bars accept a [low, high] tuple-difference.
  errorOffsets: [number, number];
}

function buildBakeoffBarData(section: EncoderBakeoffSection): BakeoffBarDatum[] {
  return [...section.rows]
    .sort((a, b) => b.macro_f1_mean - a.macro_f1_mean)
    .map((row) => {
      const low = row.macro_f1_ci_low;
      const high = row.macro_f1_ci_high;
      const offsetLow = low != null ? Math.max(0, row.macro_f1_mean - low) : 0;
      const offsetHigh = high != null ? Math.max(0, high - row.macro_f1_mean) : 0;
      return {
        name: row.encoder_key,
        macroF1: row.macro_f1_mean,
        ciLow: low,
        ciHigh: high,
        errorOffsets: [offsetLow, offsetHigh] as [number, number],
      };
    });
}

function bakeoffCallout(section: EncoderBakeoffSection): string | null {
  if (section.rows.length < 2) return null;
  const sorted = [...section.rows].sort((a, b) => b.macro_f1_mean - a.macro_f1_mean);
  const leader = sorted[0];
  const runner = sorted[1];
  const gapPoints = (leader.macro_f1_mean - runner.macro_f1_mean) * 100;
  if (!Number.isFinite(gapPoints) || gapPoints <= 0) return null;
  return `${leader.encoder_key} leads the overall F1 score by ${gapPoints.toFixed(1)} percentage points over ${runner.encoder_key}.`;
}

function crossBankCallout(
  section: CrossBankTransferSection,
  cellMap: Map<string, TransferMatrixCell>,
  sources: string[],
  targets: string[],
): string | null {
  if (sources.length === 0 || targets.length === 0) return null;
  // Largest in-domain to off-diagonal drop across all rows.
  let worstDrop = 0;
  let worstSource = "";
  let worstTarget = "";
  for (const src of sources) {
    const inDomain = cellMap.get(`${src}|${src}`);
    if (!inDomain) continue;
    for (const tgt of targets) {
      if (tgt === src) continue;
      const cell = cellMap.get(`${src}|${tgt}`);
      if (!cell) continue;
      const drop = inDomain.metric - cell.metric;
      if (drop > worstDrop) {
        worstDrop = drop;
        worstSource = src;
        worstTarget = tgt;
      }
    }
  }
  if (worstDrop <= 0 || !worstSource || !worstTarget) return null;
  const metricLabel = section.metric_name.replace("_", "-");
  return `Transferring from ${worstSource} to ${worstTarget} drops ${metricLabel} by ${(worstDrop * 100).toFixed(1)} percentage points compared with training and evaluating on the same bank.`;
}

function EncoderBakeoffPane({ section }: { section: EncoderBakeoffSection }) {
  if (!section.available || section.rows.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Encoder bake-off</CardTitle>
          <CardDescription>
            Per-encoder macro-F1 with seed-set confidence intervals.
          </CardDescription>
        </CardHeader>
        <CardContent className="py-10 text-center text-sm text-muted-foreground">
          No bake-off artefacts under <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">data/artifacts/phase3/</code>.
          Run <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">make train-batch TRAINING_PACKAGE_ID=&lt;id&gt;</code>
          and then the bake-off aggregator to populate this view.
        </CardContent>
      </Card>
    );
  }
  const barData = buildBakeoffBarData(section);
  const macroValues = barData.map((d) => d.macroF1);
  const minF1 = macroValues.length ? Math.min(...macroValues) : 0;
  const maxF1 = macroValues.length ? Math.max(...macroValues) : 1;
  const callout = bakeoffCallout(section);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Encoder bake-off</CardTitle>
        <CardDescription>
          {section.rows.length} encoders, coverage {section.coverage ? `${(section.coverage * 100).toFixed(0)}%` : "—"} block-bootstrap CI.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 p-0">
        {barData.length > 0 ? (
          <div className="px-4 pt-4">
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} margin={{ top: 12, right: 16, bottom: 24, left: 0 }}>
                  <CartesianGrid stroke="hsl(var(--border))" strokeDasharray="2 3" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                    interval={0}
                    angle={-30}
                    textAnchor="end"
                  />
                  <YAxis
                    tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                    domain={[0, 1]}
                    tickFormatter={(v) => Number(v).toFixed(2)}
                  />
                  <Tooltip
                    cursor={{ fill: "hsl(var(--muted) / 0.4)" }}
                    contentStyle={BAKEOFF_TOOLTIP_STYLE}
                    formatter={(value, _name, ctx) => {
                      const d = ctx?.payload as BakeoffBarDatum | undefined;
                      if (!d) return [String(value), "macro-F1"];
                      const ci =
                        d.ciLow != null && d.ciHigh != null
                          ? ` (95% CI ${d.ciLow.toFixed(3)}–${d.ciHigh.toFixed(3)})`
                          : "";
                      return [`${d.macroF1.toFixed(3)}${ci}`, "macro-F1"];
                    }}
                  />
                  <Bar dataKey="macroF1" isAnimationActive={false}>
                    {barData.map((d) => (
                      <Cell key={d.name} fill={bakeoffBarColor(d.macroF1, minF1, maxF1)} />
                    ))}
                    <ErrorBar dataKey="errorOffsets" stroke="hsl(var(--muted-foreground))" width={4} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            {callout ? (
              <p className="mt-2 text-xs text-muted-foreground">{callout}</p>
            ) : null}
          </div>
        ) : null}
        <table className="w-full text-sm">
          <thead className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="px-4 py-2 text-left">Encoder</th>
              <th className="px-4 py-2 text-left">Checkpoint</th>
              <th className="px-4 py-2 text-right">Seeds</th>
              <th className="px-4 py-2 text-right">Macro-F1</th>
              <th className="px-4 py-2 text-right">95% CI</th>
              <th className="px-4 py-2 text-right">Weighted-F1</th>
              <th className="px-4 py-2 text-right">Accuracy</th>
              <th className="px-4 py-2 text-right">κ</th>
            </tr>
          </thead>
          <tbody>
            {section.rows.map((row) => (
              <tr key={row.encoder_key} className="border-b border-border last:border-0">
                <td className="px-4 py-2 font-medium">{row.encoder_key}</td>
                <td className="px-4 py-2 font-mono text-xs text-muted-foreground">{row.checkpoint || "—"}</td>
                <td className="px-4 py-2 text-right font-mono text-muted-foreground">{row.seeds.length}</td>
                <td className="px-4 py-2 text-right font-mono">{formatNumberOrDash(row.macro_f1_mean)}</td>
                <td className="px-4 py-2 text-right font-mono text-muted-foreground">
                  {formatCi(row.macro_f1_ci_low, row.macro_f1_ci_high)}
                </td>
                <td className="px-4 py-2 text-right font-mono">{formatNumberOrDash(row.weighted_f1_mean)}</td>
                <td className="px-4 py-2 text-right font-mono">{formatNumberOrDash(row.accuracy_mean)}</td>
                <td className="px-4 py-2 text-right font-mono">{formatNumberOrDash(row.cohen_kappa)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}

function CrossBankTransferPane({ section }: { section: CrossBankTransferSection }) {
  if (!section.available || section.cells.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Cross-CB transfer matrix</CardTitle>
          <CardDescription>
            Source bank → target bank, macro-F1 of a model trained on source evaluated on target.
          </CardDescription>
        </CardHeader>
        <CardContent className="py-10 text-center text-sm text-muted-foreground">
          No transfer-matrix artefacts under <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">data/artifacts/cross_bank/</code>.
        </CardContent>
      </Card>
    );
  }
  // Build the cell lookup so we can render a dense matrix even when the
  // backend sends only the sparse cells array.
  const cellMap = new Map<string, TransferMatrixCell>();
  section.cells.forEach((cell) => {
    cellMap.set(`${cell.source}|${cell.target}`, cell);
  });
  const sources = section.sources.length
    ? section.sources
    : Array.from(new Set(section.cells.map((c) => c.source)));
  const targets = section.targets.length
    ? section.targets
    : Array.from(new Set(section.cells.map((c) => c.target)));
  const metrics = section.cells.map((c) => c.metric);
  const min = Math.min(...metrics);
  const max = Math.max(...metrics);
  const callout = crossBankCallout(section, cellMap, sources, targets);
  const renderCellTooltip = (src: string, tgt: string, cell: TransferMatrixCell | undefined) => {
    if (!cell) return undefined;
    const inDomain = cellMap.get(`${src}|${src}`);
    if (!inDomain || src === tgt) {
      return `Trained on ${src}, evaluated on ${tgt}: ${cell.metric.toFixed(3)} ${section.metric_name.replace("_", "-")}.`;
    }
    const delta = cell.metric - inDomain.metric;
    const deltaLabel =
      delta >= 0
        ? `+${(delta * 100).toFixed(1)}pp vs in-domain`
        : `${(delta * 100).toFixed(1)}pp vs in-domain`;
    return `Trained on ${src}, evaluated on ${tgt}: ${cell.metric.toFixed(3)} ${section.metric_name.replace("_", "-")}. ${deltaLabel}.`;
  };
  return (
    <Card>
      <CardHeader>
        <CardTitle>Cross-CB transfer matrix</CardTitle>
        <CardDescription>
          {section.metric_name.replace("_", "-")}. Rows are training banks, columns are evaluation banks. Heatmap on top, numeric values below.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4 overflow-auto">
        <div>
          <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">Heatmap</p>
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="border border-border bg-muted/30 px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  source ↓ / target →
                </th>
                {targets.map((tgt) => (
                  <th key={tgt} className="border border-border bg-muted/30 px-3 py-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    {tgt}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sources.map((src) => (
                <tr key={`heat-${src}`}>
                  <th className="border border-border bg-muted/20 px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                    {src}
                  </th>
                  {targets.map((tgt) => {
                    const cell = cellMap.get(`${src}|${tgt}`);
                    return (
                      <td
                        key={`heat-${src}-${tgt}`}
                        className="border border-border px-3 py-3 text-center font-mono text-[10px]"
                        style={cell ? { backgroundColor: heatmapColor(cell.metric, min, max) } : undefined}
                        title={renderCellTooltip(src, tgt, cell)}
                      >
                        {cell ? "" : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {callout ? (
            <p className="mt-2 text-xs text-muted-foreground">{callout}</p>
          ) : null}
        </div>
        <div>
          <p className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">Values</p>
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="border border-border bg-muted/30 px-3 py-2 text-left text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  source ↓ / target →
                </th>
                {targets.map((tgt) => (
                  <th key={tgt} className="border border-border bg-muted/30 px-3 py-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    {tgt}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sources.map((src) => (
                <tr key={`val-${src}`}>
                  <th className="border border-border bg-muted/20 px-3 py-2 text-left text-xs font-medium text-muted-foreground">
                    {src}
                  </th>
                  {targets.map((tgt) => {
                    const cell = cellMap.get(`${src}|${tgt}`);
                    return (
                      <td
                        key={`val-${src}-${tgt}`}
                        className="border border-border px-3 py-2 text-right font-mono text-xs"
                        title={renderCellTooltip(src, tgt, cell)}
                      >
                        {cell ? cell.metric.toFixed(3) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function describeArtifactFile(relativePath: string): string {
  const last = relativePath.split("/").pop() ?? "";
  if (last.endsWith(".parquet")) return "Tabular cache for downstream loaders.";
  if (last.endsWith(".pt") || last.endsWith(".bin")) return "Model checkpoint weights.";
  if (last.includes("aggregate")) return "Aggregated metrics across seeds.";
  if (last.includes("conformal")) return "Conformal calibration sidecar.";
  if (last.includes("breakdown")) return "Per-class breakdown for the evaluation harness.";
  if (last.includes("transfer")) return "Cross-bank transfer evaluation row.";
  if (last.endsWith(".json")) return "Structured evaluation artefact.";
  if (last.endsWith(".csv")) return "CSV table of evaluation rows.";
  if (last.endsWith(".md")) return "Markdown notes for this run.";
  return "Research artefact.";
}

function ArtifactsExplorer({
  sections,
}: {
  sections: Record<string, ArtifactFile[]>;
}) {
  const sectionEntries = Object.entries(sections);
  const totalFiles = sectionEntries.reduce((acc, [, files]) => acc + files.length, 0);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Downloads</CardTitle>
        <CardDescription>
          {totalFiles} files across {sectionEntries.length} sections. Each file lists its size, last update, and a short note on what it is.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {sectionEntries.map(([section, files]) => (
          <div key={section} className="space-y-1">
            <div className="flex items-baseline gap-2">
              <h3 className="font-mono text-sm font-medium">{section}/</h3>
              <span className="text-xs text-muted-foreground">{files.length} files</span>
            </div>
            {files.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No files in this section. Run the relevant pipeline to populate it.
              </p>
            ) : (
              <ul className="space-y-1.5">
                {files.slice(0, 20).map((file) => (
                  <li
                    key={file.relative_path}
                    className="space-y-0.5 border-b border-border/40 pb-1.5 last:border-0"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-xs text-muted-foreground">
                      <span>{file.relative_path}</span>
                      <span>
                        {formatBytes(file.size_bytes)} · updated {file.modified_at.slice(0, 19)}
                      </span>
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      {describeArtifactFile(file.relative_path)}
                    </p>
                  </li>
                ))}
                {files.length > 20 ? (
                  <li className="text-xs text-muted-foreground">
                    +{files.length - 20} more files
                  </li>
                ) : null}
              </ul>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

export default function ResearchPage() {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [data, setData] = React.useState<ResearchArtifactsResponse | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let cancelled = false;
    fetchResearchArtifacts(apiBaseUrl)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) toast.error(errorMessage(err, "Failed to load research artefacts."));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl]);

  return (
    <>
      <Head>
        <title>Research — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main id="main-content" tabIndex={-1} className="container space-y-5 py-6 focus:outline-none">
          <div className="space-y-1">
            <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight">
              <FlaskConical className="h-6 w-6 text-primary" />
              Research console
            </h1>
            <p className="max-w-2xl text-sm text-muted-foreground">
              Research artefacts the model is built on. The Bake-off compares text encoders.
              The Transfer matrix shows how a model trained on one central bank&apos;s statements
              performs on another&apos;s. Files lists the raw artefact JSONs you can download.
            </p>
          </div>

          {loading ? (
            <div className="space-y-3">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-48 w-full" />
              <Skeleton className="h-48 w-full" />
            </div>
          ) : data ? (
            <Tabs defaultValue="bakeoff" className="w-full">
              <TabsList className="flex w-full flex-wrap justify-start">
                <TabsTrigger value="bakeoff">Bake-off</TabsTrigger>
                <TabsTrigger value="transfer">Transfer</TabsTrigger>
                <TabsTrigger value="decisions">Decisions</TabsTrigger>
                <TabsTrigger value="jobs">Jobs</TabsTrigger>
                <TabsTrigger value="files">Downloads</TabsTrigger>
              </TabsList>
              <TabsContent value="bakeoff" className="space-y-3">
                <EncoderBakeoffPane section={data.encoder_bakeoff} />
                {data.encoder_bakeoff.source_files.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {data.encoder_bakeoff.source_files.map((f) => (
                      <Badge key={f} variant="outline" className="font-mono text-[10px]">
                        {f}
                      </Badge>
                    ))}
                  </div>
                ) : null}
              </TabsContent>
              <TabsContent value="transfer" className="space-y-3">
                <CrossBankTransferPane section={data.cross_bank_transfer} />
                {data.cross_bank_transfer.source_files.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {data.cross_bank_transfer.source_files.map((f) => (
                      <Badge key={f} variant="outline" className="font-mono text-[10px]">
                        {f}
                      </Badge>
                    ))}
                  </div>
                ) : null}
              </TabsContent>
              <TabsContent value="decisions">
                <DecisionsLink />
              </TabsContent>
              <TabsContent value="jobs">
                <JobsLink />
              </TabsContent>
              <TabsContent value="files">
                <ArtifactsExplorer sections={data.sections} />
              </TabsContent>
            </Tabs>
          ) : (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground">
                Could not load artefacts. Make sure the backend is running.
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
