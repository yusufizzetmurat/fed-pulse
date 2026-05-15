import { Activity, GitBranch, Scale, Timer } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { CredibilityResponse } from "@/lib/analyze/types";

interface CredibilityPanelProps {
  credibility: CredibilityResponse;
  previewMode?: boolean;
}

function driftTone(value: number): "hawkish" | "dovish" | "neutral" {
  if (value > 0.6) return "dovish";
  if (value < 0.3) return "hawkish";
  return "neutral";
}

function driftLabel(value: number): string {
  if (value > 0.6) return "High drift";
  if (value < 0.3) return "Steady";
  return "Drifting";
}

function MiniTrend({ trend }: { trend: number[] }) {
  if (!trend.length) return null;
  const max = Math.max(...trend, 1e-6);
  return (
    <div className="flex h-6 items-end gap-0.5">
      {trend.map((value, idx) => {
        const pct = (value / max) * 100;
        return (
          <div
            key={idx}
            className="w-2 rounded-sm bg-primary/60"
            style={{ height: `${Math.max(pct, 8)}%` }}
            aria-hidden="true"
          />
        );
      })}
    </div>
  );
}

function gapTone(value?: number): "hawkish" | "dovish" | "neutral" {
  if (value == null) return "neutral";
  if (value > 0.05) return "hawkish";
  if (value < -0.05) return "dovish";
  return "neutral";
}

function formatGap(value?: number): string {
  if (value == null) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
}

export function CredibilityPanel({ credibility, previewMode }: CredibilityPanelProps) {
  const driftScore = Number.isFinite(credibility.drift_score) ? credibility.drift_score : 0;
  const tone = driftTone(driftScore);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Central-bank credibility
          {previewMode ? (
            <Badge variant="outline" className="ml-2 text-[10px] uppercase tracking-wide">
              Preview · fixture
            </Badge>
          ) : null}
        </CardTitle>
        <CardDescription>
          Four-axis credibility check on the issuing institution at this statement date.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <GitBranch className="h-3.5 w-3.5" /> Drift vs. prior 4 statements
            </span>
            <Badge variant={tone}>{driftLabel(driftScore)} · {driftScore.toFixed(2)}</Badge>
          </div>
          <Progress value={driftScore} />
          {credibility.drift_trend?.length ? (
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Trend</span>
              <MiniTrend trend={credibility.drift_trend} />
            </div>
          ) : null}
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
            <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-muted-foreground">
              <span>Realized vs. stated</span>
              <Scale className="h-3.5 w-3.5" />
            </div>
            <div className="mt-1 flex items-center justify-between">
              <strong>{formatGap(credibility.realized_vs_stated_gap)}</strong>
              <Badge variant={gapTone(credibility.realized_vs_stated_gap)} className="text-[10px]">
                {credibility.realized_vs_stated_gap == null ? "—" : "90d corr"}
              </Badge>
            </div>
          </div>
          <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
            <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-muted-foreground">
              <span>Market-implied gap</span>
              <Scale className="h-3.5 w-3.5" />
            </div>
            <div className="mt-1 flex items-center justify-between">
              <strong>{formatGap(credibility.market_implied_gap)}</strong>
              <Badge variant={gapTone(credibility.market_implied_gap)} className="text-[10px]">
                {credibility.market_implied_gap == null ? "—" : "vs OIS"}
              </Badge>
            </div>
          </div>
          <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
            <div className="flex items-center justify-between text-[11px] uppercase tracking-wide text-muted-foreground">
              <span>Since reversal</span>
              <Timer className="h-3.5 w-3.5" />
            </div>
            <div className="mt-1 flex items-center justify-between">
              <strong>
                {credibility.months_since_reversal == null
                  ? "—"
                  : `${credibility.months_since_reversal} mo`}
              </strong>
              <Badge variant="outline" className="text-[10px]">stance</Badge>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
