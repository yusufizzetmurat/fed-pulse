import * as React from "react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  fetchResearchRegistry,
  postAnalyze,
  postAnalyzeAnalogs,
  postResearchBacktest,
  resolveApiBaseUrl,
} from "@/lib/analyze/api";
import { errorMessage } from "@/lib/analyze/errors";
import { SAMPLE_STATEMENTS } from "@/lib/analyze/sample-statements";
import type {
  AnalogsResponse,
  AnalyzeResult,
  BacktestPositionEntry,
  BacktestResponse,
  ResearchRegistryResponse,
} from "@/lib/analyze/types";
import { cn } from "@/lib/utils";

const SAMPLE_TEXT = SAMPLE_STATEMENTS[0].text;
const MAX_TEXT_CHARS = 12000;

type Surface = "dual" | "cls";

function formatDelta(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(4)}`;
}

function formatPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

function deltaTone(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "text-muted-foreground";
  return value >= 0 ? "text-emerald-700" : "text-red-700";
}

export function TerminalTab(): JSX.Element {
  const baseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [text, setText] = React.useState<string>(SAMPLE_TEXT);
  const [analysisDate, setAnalysisDate] = React.useState<string>(
    () => new Date().toISOString().slice(0, 10),
  );
  const [surface, setSurface] = React.useState<Surface>("dual");
  const [includeRejected, setIncludeRejected] = React.useState<boolean>(false);
  const [registry, setRegistry] = React.useState<ResearchRegistryResponse | null>(null);
  const [activeEncoder, setActiveEncoder] = React.useState<string>("");
  const [analyze, setAnalyze] = React.useState<AnalyzeResult | null>(null);
  const [analogs, setAnalogs] = React.useState<AnalogsResponse | null>(null);
  const [running, setRunning] = React.useState<boolean>(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    fetchResearchRegistry(baseUrl, { surface, includeRejected })
      .then((response) => {
        if (cancelled) return;
        setRegistry(response);
        if (response.rows.length > 0) {
          setActiveEncoder((prev) =>
            response.rows.some((row: { encoder_alias: string }) => row.encoder_alias === prev)
              ? prev
              : response.rows[0].encoder_alias,
          );
        }
      })
      .catch((err) => {
        if (cancelled) return;
        setError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, surface, includeRejected]);

  const runAnalysis = React.useCallback(async () => {
    if (!text.trim()) return;
    setRunning(true);
    setError(null);
    try {
      const [analyzeResult, analogsResult] = await Promise.all([
        postAnalyze(baseUrl, {
          text,
          date: analysisDate,
          symbol: "^GSPC",
          horizon: "10d",
          include_realized: false,
        }),
        postAnalyzeAnalogs(baseUrl, { text, k: 3 }),
      ]);
      setAnalyze(analyzeResult);
      setAnalogs(analogsResult);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setRunning(false);
    }
  }, [baseUrl, text, analysisDate]);

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        Backtest runs automatically after analysis completes.
      </p>
      <ActiveCheckpointBar
        registry={registry}
        activeEncoder={activeEncoder}
        onActiveEncoderChange={setActiveEncoder}
        surface={surface}
        onSurfaceChange={setSurface}
        includeRejected={includeRejected}
        onIncludeRejectedChange={setIncludeRejected}
      />

      <Card>
        <CardHeader>
          <CardTitle>Statement</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value.slice(0, MAX_TEXT_CHARS))}
            maxLength={MAX_TEXT_CHARS}
            className="w-full h-32 rounded border border-border bg-background p-3 text-sm font-mono"
            placeholder="Paste FOMC statement text…"
            aria-label="FOMC statement text"
          />
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <label htmlFor="console-sample">Load sample</label>
            <span>
              {text.length.toLocaleString()} / {MAX_TEXT_CHARS.toLocaleString()} chars
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <select
              id="console-sample"
              value=""
              onChange={(event) => {
                const sample = SAMPLE_STATEMENTS.find((entry) => entry.id === event.target.value);
                if (!sample) return;
                setText(sample.text.slice(0, MAX_TEXT_CHARS));
                setAnalysisDate(sample.date);
              }}
              className="rounded border border-border bg-background px-2 py-1 text-sm"
            >
              <option value="">Load sample statement…</option>
              {SAMPLE_STATEMENTS.map((sample) => (
                <option key={sample.id} value={sample.id}>
                  {sample.label}
                </option>
              ))}
            </select>
            <label className="text-xs text-muted-foreground" htmlFor="console-date">
              Analysis date
            </label>
            <input
              id="console-date"
              type="date"
              value={analysisDate}
              onChange={(event) => setAnalysisDate(event.target.value)}
              className="rounded border border-border bg-background px-2 py-1 text-sm"
            />
            <Button onClick={runAnalysis} disabled={running || !text.trim()}>
              {running ? "Running…" : "Run analysis"}
            </Button>
            {error && (
              <span className="text-xs text-red-700" role="alert">
                {error}
              </span>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <StancePanel analyze={analyze} />
        <VolRegimePanel analyze={analyze} />
        <AnalogsPanel analogs={analogs} />
        <BacktestPanel
          baseUrl={baseUrl}
          triggerRun={!running && analyze !== null}
          triggerKey={analyze}
        />
      </div>

      <RegistryStrip
        registry={registry}
        surface={surface}
        includeRejected={includeRejected}
      />
    </div>
  );
}

interface ActiveCheckpointBarProps {
  registry: ResearchRegistryResponse | null;
  activeEncoder: string;
  onActiveEncoderChange: (next: string) => void;
  surface: Surface;
  onSurfaceChange: (next: Surface) => void;
  includeRejected: boolean;
  onIncludeRejectedChange: (next: boolean) => void;
}

function ActiveCheckpointBar({
  registry,
  activeEncoder,
  onActiveEncoderChange,
  surface,
  onSurfaceChange,
  includeRejected,
  onIncludeRejectedChange,
}: ActiveCheckpointBarProps): JSX.Element {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between gap-3">
          <span>Active checkpoint</span>
          <div className="flex items-center gap-3 text-xs font-normal text-muted-foreground">
            <label htmlFor="surface-picker" className="flex items-center gap-1">
              surface
              <select
                id="surface-picker"
                value={surface}
                onChange={(event) => onSurfaceChange(event.target.value as Surface)}
                className="rounded border border-border bg-background px-2 py-1"
              >
                <option value="dual">dual</option>
                <option value="cls">cls</option>
              </select>
            </label>
            <label className="flex items-center gap-1">
              <input
                type="checkbox"
                checked={includeRejected}
                onChange={(event) => onIncludeRejectedChange(event.target.checked)}
              />
              show rejected
            </label>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <select
          value={activeEncoder}
          onChange={(event) => onActiveEncoderChange(event.target.value)}
          className="w-full rounded border border-border bg-background px-3 py-2 text-sm font-mono"
        >
          {(registry?.rows ?? []).map((row) => {
            const delta = surface === "dual" ? row.delta_dual : row.delta_cls;
            return (
              <option key={row.encoder_alias} value={row.encoder_alias}>
                {row.encoder_display} — Δ{surface} {formatDelta(delta)}
              </option>
            );
          })}
        </select>
        {registry && (
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>
              {registry.rows.length} shown · {registry.rejected_count} rejected
              {" · "}TP={registry.training_package_id}
            </span>
            <span>head={registry.head} · seeds={registry.seeds.join(",")}</span>
          </div>
        )}
        <p className="text-xs text-muted-foreground">
          Display only: checkpoint selection drives the Registry strip ranking;
          the analysis panels above run on the backend&apos;s active checkpoint.
        </p>
      </CardContent>
    </Card>
  );
}

function StancePanel({ analyze }: { analyze: AnalyzeResult | null }): JSX.Element {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Stance</CardTitle>
      </CardHeader>
      <CardContent className="text-sm">
        {analyze ? (
          <div className="space-y-2">
            <Row
              label="Label"
              value={(analyze.sentiment?.label ?? "—").toUpperCase()}
            />
            <Row
              label="Score"
              value={analyze.sentiment?.score?.toFixed(3) ?? "—"}
            />
            <Row
              label="OOD"
              value={
                analyze.sentiment?.is_in_distribution === false
                  ? `OUT (energy ${analyze.sentiment?.ood_energy?.toFixed(2) ?? "—"})`
                  : `in-distribution (${analyze.sentiment?.ood_energy?.toFixed(2) ?? "—"})`
              }
              tone={analyze.sentiment?.is_in_distribution === false ? "warn" : undefined}
            />
            {analyze.multi_axis?.stance && (
              <Row
                label="Multi-axis stance"
                value={`${analyze.multi_axis.stance.label} (${analyze.multi_axis.stance.confidence.toFixed(2)})`}
              />
            )}
          </div>
        ) : (
          <Hint />
        )}
      </CardContent>
    </Card>
  );
}

function VolRegimePanel({ analyze }: { analyze: AnalyzeResult | null }): JSX.Element {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Vol regime (10d fwd)</CardTitle>
        <CardDescription className="text-xs">
          Argmax class needs a fine-tuned regime checkpoint loaded on the backend; otherwise the row reads &mdash;.
        </CardDescription>
      </CardHeader>
      <CardContent className="text-sm">
        {analyze ? (
          <div className="space-y-2">
            <Row
              label="Argmax class"
              value={analyze.regime_classification?.argmax_class ?? "—"}
            />
            <Row
              label="Predicted vol"
              value={analyze.prediction?.volatility?.toFixed(4) ?? "—"}
            />
            <Row
              label="Predicted close"
              value={analyze.prediction?.close?.toFixed(2) ?? "—"}
            />
            <Row
              label="Latest close"
              value={analyze.market?.close?.toFixed(2) ?? "—"}
            />
            {analyze.regime_classification?.coverage != null && (
              <Row
                label="Coverage"
                value={`${(analyze.regime_classification.coverage * 100).toFixed(1)}%`}
              />
            )}
          </div>
        ) : (
          <Hint />
        )}
      </CardContent>
    </Card>
  );
}

function AnalogsPanel({ analogs }: { analogs: AnalogsResponse | null }): JSX.Element {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Historical analogs</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {!analogs ? (
          <Hint />
        ) : analogs.analogs.length === 0 ? (
          <div className="text-muted-foreground">No analogs found.</div>
        ) : (
          analogs.analogs.map((analog) => (
            <div
              key={analog.event_date}
              className="space-y-1 border-l-2 border-border pl-3"
            >
              <div className="flex items-center justify-between">
                <span className="font-mono">{analog.event_date}</span>
                <span className="text-xs text-muted-foreground">
                  cos {analog.similarity.toFixed(3)}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 font-mono text-xs">
                <span>
                  5d S&amp;P:{" "}
                  <span className={deltaTone(analog.subsequent_close_pct_5d)}>
                    {formatPct(analog.subsequent_close_pct_5d)}
                  </span>
                </span>
                <span>
                  20d S&amp;P:{" "}
                  <span className={deltaTone(analog.subsequent_close_pct_20d)}>
                    {formatPct(analog.subsequent_close_pct_20d)}
                  </span>
                </span>
              </div>
              <div className="line-clamp-2 text-xs text-muted-foreground">
                {analog.excerpt}
              </div>
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}

// Historical FOMC dates with hard-coded stance proxies (hawkish=-1
// short S&P, dovish=+1 long, neutral=0). Conservative reads of the
// published Fed direction at each meeting so the backtest panel can
// render real Sharpe / HitRate / MaxDD numbers from real S&P forward
// returns without depending on a live predicted-stance fan-out.
const DEMO_BACKTEST_POSITIONS: BacktestPositionEntry[] = [
  { date: "2022-03-16", position: -1 }, // 25bp hike, start of cycle
  { date: "2022-06-15", position: -1 }, // 75bp hike, surprise
  { date: "2022-11-02", position: -1 }, // 75bp hike, sustained tightening
  { date: "2023-02-01", position: -1 }, // 25bp hike, decelerating
  { date: "2023-07-26", position: -1 }, // 25bp hike, last in cycle
  { date: "2023-12-13", position: 0 }, // hold, neutral
  { date: "2024-03-20", position: 0 }, // hold, neutral
  { date: "2024-09-18", position: 1 }, // 50bp cut, dovish
];

interface BacktestPanelProps {
  baseUrl: string;
  triggerRun?: boolean;
  // Identity-tracked sentinel so the auto-run effect fires once per
  // analysis result rather than on every parent re-render.
  triggerKey?: unknown;
}

function BacktestPanel({ baseUrl, triggerRun, triggerKey }: BacktestPanelProps): JSX.Element {
  const [result, setResult] = React.useState<BacktestResponse | null>(null);
  const [running, setRunning] = React.useState<boolean>(false);
  const [error, setError] = React.useState<string | null>(null);
  const lastTriggerKeyRef = React.useRef<unknown>(null);

  const run = React.useCallback(async () => {
    setRunning(true);
    setError(null);
    try {
      const response = await postResearchBacktest(baseUrl, {
        positions: DEMO_BACKTEST_POSITIONS,
        symbol: "^GSPC",
        horizon_days: 5,
      });
      setResult(response);
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setRunning(false);
    }
  }, [baseUrl]);

  React.useEffect(() => {
    if (!triggerRun) return;
    if (triggerKey == null) return;
    if (lastTriggerKeyRef.current === triggerKey) return;
    lastTriggerKeyRef.current = triggerKey;
    void run();
  }, [triggerRun, triggerKey, run]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Backtest (stance-directional, 5d)</span>
          <Button size="sm" onClick={run} disabled={running}>
            {running ? "Running…" : result ? "Re-run" : "Run backtest"}
          </Button>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm">
        {error && (
          <span className="text-xs text-red-700" role="alert">
            {error}
          </span>
        )}
        {!result && !running && (
          <p className="text-xs text-muted-foreground">
            Eight historical FOMC dates with proxied stances
            (hawkish=-1 short S&amp;P, dovish=+1 long). Computes Sharpe /
            HitRate / MaxDD vs buy-and-hold on real ^GSPC 5d forward
            returns. <strong>Note:</strong> the 2022&ndash;2024 window
            covers the Fed hiking cycle&apos;s sharp equity sell-off, so
            short-SPX bets dominate the sample, so Sharpe and hit-rate
            reflect a directionally favourable period, not out-of-sample
            model skill.
          </p>
        )}
        {result && (
          <>
            <div className="grid grid-cols-3 gap-3 text-sm">
              <Row label="Sharpe" value={result.sharpe?.toFixed(2) ?? "—"} />
              <Row
                label="HitRate"
                value={
                  result.hit_rate != null
                    ? `${(result.hit_rate * 100).toFixed(1)}%`
                    : "—"
                }
              />
              <Row
                label="MaxDD"
                value={
                  result.max_dd_pct != null
                    ? `${result.max_dd_pct.toFixed(2)}%`
                    : "—"
                }
                tone={result.max_dd_pct != null && result.max_dd_pct < -5 ? "warn" : undefined}
              />
            </div>
            <div className="grid grid-cols-3 gap-3 text-xs text-muted-foreground">
              <Row label="cum return" value={formatPct(result.cum_return_pct)} />
              <Row label="buy-and-hold" value={formatPct(result.benchmark_cum_pct)} />
              <Row
                label="alpha"
                value={formatPct(result.alpha_cum_pct)}
              />
            </div>
            <p className="text-xs text-muted-foreground">
              {result.n_trades} / {result.trades.length} dates realized ·
              horizon {result.horizon_days}d · {result.symbol}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

function RegistryStrip({
  registry,
  surface,
  includeRejected,
}: {
  registry: ResearchRegistryResponse | null;
  surface: Surface;
  includeRejected: boolean;
}): JSX.Element | null {
  if (!registry) return null;
  const title = includeRejected
    ? "Registry (all encoders, losers dimmed)"
    : `Registry (winners on ${surface})`;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>{title}</span>
          <span className="text-xs font-normal text-muted-foreground">
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-xs">
            <thead className="text-muted-foreground">
              <tr>
                <th className="py-1 text-left">Encoder</th>
                <th className="px-2 text-right">dual F1</th>
                <th className="px-2 text-right">cls F1</th>
                <th className="px-2 text-right">Δ dual</th>
                <th className="px-2 text-right">Δ cls</th>
                <th className="pl-3 text-left">Notes</th>
              </tr>
            </thead>
            <tbody>
              {registry.rows.map((row) => (
                <tr
                  key={row.encoder_alias}
                  className={cn(
                    "border-t border-border",
                    row.is_winner ? undefined : "opacity-50",
                  )}
                >
                  <td className="py-1 font-mono">{row.encoder_display}</td>
                  <td className="px-2 text-right font-mono">
                    {row.dual_f1?.toFixed(4) ?? "—"}
                  </td>
                  <td className="px-2 text-right font-mono">
                    {row.cls_f1?.toFixed(4) ?? "—"}
                  </td>
                  <td
                    className={cn(
                      "px-2 text-right font-mono",
                      deltaTone(row.delta_dual),
                    )}
                  >
                    {formatDelta(row.delta_dual)}
                  </td>
                  <td
                    className={cn(
                      "px-2 text-right font-mono",
                      deltaTone(row.delta_cls),
                    )}
                  >
                    {formatDelta(row.delta_cls)}
                  </td>
                  <td className="pl-3 text-muted-foreground">{row.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function Row({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "warn";
}): JSX.Element {
  return (
    <div className="flex items-center justify-between gap-3">
      <span className="text-muted-foreground">{label}</span>
      <span
        className={cn(
          "font-mono",
          tone === "warn" ? "text-amber-500" : undefined,
        )}
      >
        {value}
      </span>
    </div>
  );
}

function Hint(): JSX.Element {
  return (
    <span className="text-muted-foreground">
      Run an analysis to populate this panel.
    </span>
  );
}
