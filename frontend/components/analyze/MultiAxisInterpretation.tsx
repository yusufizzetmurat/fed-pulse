import * as React from "react";
import { Compass, Gauge, Layers, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { KpiTile } from "@/components/ui/kpi-tile";
import { stanceLabel } from "@/lib/analyze/format";
import type { MultiAxisResponse, MultiAxisStance } from "@/lib/analyze/types";

interface MultiAxisInterpretationProps {
  multiAxis: MultiAxisResponse;
  // Optional sparkline series of prior runs for each axis (newest last).
  history?: {
    stance?: Array<number | null>;
    factor?: Array<number | null>;
    certainty?: Array<number | null>;
  };
}

function StanceTile({ stance, history }: { stance: MultiAxisStance; history?: Array<number | null> }) {
  const variant: "hawkish" | "dovish" | "neutral" =
    stance.label === "hawkish"
      ? "hawkish"
      : stance.label === "dovish"
      ? "dovish"
      : "neutral";
  return (
    <KpiTile
      label="Stance"
      icon={<Compass className="h-3.5 w-3.5" />}
      value={
        <span className="flex items-center gap-2">
          <span className="capitalize">{stanceLabel(stance.label)}</span>
          <Badge variant={variant} className="text-[10px]">
            {stance.confidence.toFixed(2)}
          </Badge>
        </span>
      }
      sparkline={history}
      caption="Hawkish (+) / Dovish (−)"
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
      caption="GSS forward-guidance vs target-shock axis"
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
      caption="How firmly the language commits"
    />
  );
}

function TopicTile({ topic }: { topic: NonNullable<MultiAxisResponse["topic"]> }) {
  const display = (topic.label ?? topic.primary ?? "other").toString();
  return (
    <KpiTile
      label="Topic"
      icon={<Layers className="h-3.5 w-3.5" />}
      value={
        <span className="capitalize">{display.replace(/_/g, " ")}</span>
      }
      delta={topic.confidence}
      deltaFormatter={(v) => v.toFixed(2)}
      caption={
        topic.secondary?.length
          ? `also: ${topic.secondary.map((t) => t.replace(/_/g, " ")).join(", ")}`
          : "primary topic"
      }
    />
  );
}

export function MultiAxisInterpretation({
  multiAxis,
  history,
}: MultiAxisInterpretationProps) {
  const stanceHistory = history?.stance;
  const factorHistory = history?.factor;
  const certaintyHistory = history?.certainty;
  const allAxesNull =
    !multiAxis.stance && !multiAxis.factor && !multiAxis.certainty && !multiAxis.topic;

  if (allAxesNull) {
    return (
      <div className="rounded-md border border-dashed border-border bg-muted/20 p-4 text-xs text-muted-foreground sm:col-span-2">
        <p className="mb-1 text-[10px] uppercase tracking-wide text-foreground">
          Multi-axis classifier returned no axis labels
        </p>
        <p>
          The classifier ran but every head returned null. The most common cause is a missing
          checkpoint at <code className="rounded bg-muted px-1">backend/models/multi_axis_classifier.pt</code>
          {" "}or a passage shorter than the encoder context window. Train and deploy the
          checkpoint, or paste a longer FOMC statement to populate stance, factor, certainty,
          and topic.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {multiAxis.stance ? <StanceTile stance={multiAxis.stance} history={stanceHistory} /> : null}
      {multiAxis.factor ? <FactorTile factor={multiAxis.factor} history={factorHistory} /> : null}
      {multiAxis.certainty ? (
        <CertaintyTile certainty={multiAxis.certainty} history={certaintyHistory} />
      ) : null}
      {multiAxis.topic ? <TopicTile topic={multiAxis.topic} /> : null}
    </div>
  );
}
