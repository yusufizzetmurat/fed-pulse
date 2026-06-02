import * as React from "react";
import { Compass, Gauge, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { KpiTile } from "@/components/ui/kpi-tile";
import { stanceLabel } from "@/lib/analyze/format";
import type {
  MultiAxisResponse,
  MultiAxisStance,
  StanceContextResponse,
} from "@/lib/analyze/types";

interface MultiAxisInterpretationProps {
  multiAxis: MultiAxisResponse;
  // Optional sparkline series of prior runs for each axis (newest last).
  history?: {
    stance?: Array<number | null>;
    factor?: Array<number | null>;
    certainty?: Array<number | null>;
  };
  // Trailing stance-score baseline. When provided with at least two
  // usable history rows, the StanceTile renders the current run as a
  // z-score against this baseline instead of the raw confidence number.
  // The validity study showed relative ordering carries signal but the
  // absolute level is mis-centred (dovish bias) — the z-score makes
  // the dashboard claim what the instrument is actually validated for.
  stanceContext?: StanceContextResponse | null;
}

function currentStanceScore(stance: MultiAxisStance): number | null {
  // s = P(hawkish) - P(dovish); matches the validity study's anchor.
  const dist = stance.distribution;
  if (!dist) return null;
  const hawk = dist.hawkish;
  const dove = dist.dovish;
  if (typeof hawk !== "number" && typeof dove !== "number") return null;
  return (hawk ?? 0) - (dove ?? 0);
}

interface StanceTileProps {
  stance: MultiAxisStance;
  history?: Array<number | null>;
  context?: StanceContextResponse | null;
}

function StanceTile({ stance, history, context }: StanceTileProps) {
  const variant: "hawkish" | "dovish" | "neutral" =
    stance.label === "hawkish"
      ? "hawkish"
      : stance.label === "dovish"
      ? "dovish"
      : "neutral";

  const score = currentStanceScore(stance);
  // ``std == null`` is the backend's signal for "degenerate trailing
  // window" (constant series or float-precision residue). Combining
  // it with ``> 0`` guards both the null and the exact-zero edges
  // under a single check; ``Number.isFinite`` also rejects ±Infinity
  // that would otherwise render as ``+Infσ``.
  const hasUsableContext =
    context != null &&
    context.mean != null &&
    context.std != null &&
    Number.isFinite(context.std) &&
    context.std > 0 &&
    context.n >= 2 &&
    score != null;
  const z = hasUsableContext
    ? (score! - context!.mean!) / context!.std!
    : null;

  const zTone = z == null ? "neutral" : z > 0.5 ? "hawkish" : z < -0.5 ? "dovish" : "neutral";
  const zLabel = z == null ? null : `${z >= 0 ? "+" : ""}${z.toFixed(2)}σ`;
  const zTitle = z == null
    ? undefined
    : `Rolling z-score against the last ${context!.n} runs` +
      ` (mean ${context!.mean!.toFixed(2)}, std ${context!.std!.toFixed(2)}).` +
      ` Relative ordering — the instrument is validated for this; the raw absolute level is dovish-skewed.`;

  return (
    <KpiTile
      label="Stance"
      icon={<Compass className="h-3.5 w-3.5" />}
      value={
        <span className="flex items-center gap-2">
          <span className="capitalize">{stanceLabel(stance.label)}</span>
          {zLabel != null ? (
            <Badge
              variant={zTone}
              className="text-[10px]"
              title={zTitle}
              data-testid="stance-zscore"
            >
              {zLabel}
            </Badge>
          ) : (
            <Badge variant={variant} className="text-[10px]">
              {stance.confidence.toFixed(2)}
            </Badge>
          )}
        </span>
      }
      sparkline={history}
      caption={
        zLabel != null
          ? "Rolling z-score vs recent meetings. Hawkish (+) favours tighter policy."
          : "Hawkish (+) favours tighter policy; Dovish (−) favours easier policy"
      }
    />
  );
}

function FactorTile({
  factor,
  history,
}: {
  factor: NonNullable<MultiAxisResponse["factor"]>;
  history?: Array<number | null>;
}) {
  const value = factor.value;
  const tone = value > 0.05 ? "up" : value < -0.05 ? "down" : "neutral";
  // §6.13 / #328: the factor axis carries near-zero label coverage on
  // the canonical training pool, and the backend gates the card off
  // when coverage falls below 0.01. If a card still arrives here the
  // checkpoint stamped a non-zero coverage — render but flag the axis
  // as low-confidence so the surface does not oversell it.
  return (
    <KpiTile
      label="Factor"
      icon={<Gauge className="h-3.5 w-3.5" />}
      value={
        <span className="numeric">
          {value >= 0 ? "+" : ""}
          {value.toFixed(2)}
        </span>
      }
      delta={factor.confidence}
      deltaFormatter={(v) => `±${v.toFixed(2)}`}
      tone={tone}
      sparkline={history}
      sparklineTone={tone}
      caption="Forward-guidance vs near-term rate shock · limited training data"
    />
  );
}

function CertaintyTile({
  certainty,
  history,
}: {
  certainty: NonNullable<MultiAxisResponse["certainty"]>;
  history?: Array<number | null>;
}) {
  return (
    <KpiTile
      label="Certainty"
      icon={<Target className="h-3.5 w-3.5" />}
      value={<span className="capitalize">{certainty.label}</span>}
      delta={certainty.confidence}
      deltaFormatter={(v) => v.toFixed(2)}
      sparkline={history}
      caption="How firmly the wording commits to a stance"
    />
  );
}

export function MultiAxisInterpretation({
  multiAxis,
  history,
  stanceContext,
}: MultiAxisInterpretationProps) {
  const stanceHistory = history?.stance;
  const factorHistory = history?.factor;
  const certaintyHistory = history?.certainty;
  const allAxesNull =
    !multiAxis.stance && !multiAxis.factor && !multiAxis.certainty;

  if (allAxesNull) {
    return (
      <div className="rounded-md border border-dashed border-border bg-muted/20 p-4 text-xs text-muted-foreground sm:col-span-2">
        <p className="mb-1 text-[10px] uppercase tracking-wide text-foreground">
          Sentiment breakdown returned no labels
        </p>
        <p>
          The sentiment model ran but produced no labels for any axis. Paste a
          longer FOMC excerpt and re-run.
        </p>
      </div>
    );
  }

  // The forward-guidance vs near-term-shock factor was retired in the
  // 2026-05-12 multi-axis pivot — the training pool never carried enough
  // labels to estimate it. The card stays in the type so legacy
  // checkpoints can still surface it, but we no longer reserve a placeholder
  // tile or "hidden" badge when it is absent.
  return (
    <div className="space-y-2">
      <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
        Sentiment breakdown
      </Badge>
      <div className="grid gap-3 sm:grid-cols-2">
        {multiAxis.stance ? (
          <StanceTile
            stance={multiAxis.stance}
            history={stanceHistory}
            context={stanceContext}
          />
        ) : null}
        {multiAxis.factor ? <FactorTile factor={multiAxis.factor} history={factorHistory} /> : null}
        {multiAxis.certainty ? (
          <CertaintyTile certainty={multiAxis.certainty} history={certaintyHistory} />
        ) : null}
      </div>
    </div>
  );
}
