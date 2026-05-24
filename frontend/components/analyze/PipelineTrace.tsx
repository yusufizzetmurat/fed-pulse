import * as React from "react";
import { ChevronDown, Cpu, FileInput, Layers, ShieldCheck, Workflow } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type {
  AnalyzeResult,
  MultiAxisResponse,
  RegimeClassificationResponse,
  SentimentResponse,
} from "@/lib/analyze/types";

interface PipelineTraceProps {
  result: AnalyzeResult;
  inputText: string;
}

interface Step {
  key: string;
  title: string;
  icon: React.ReactNode;
  summary: string;
  body: React.ReactNode;
}

function countWords(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function buildSteps({
  result,
  inputText,
}: PipelineTraceProps): Step[] {
  const sentiment = result.sentiment;
  const multiAxis = result.multi_axis;
  const regime = result.regime_classification;
  const encoderKey = (result.model as { encoder_key?: string } | undefined)?.encoder_key;
  const xaiSentences = result.xai?.sentences?.length ?? 0;

  const ingestStep: Step = {
    key: "ingest",
    title: "Ingest",
    icon: <FileInput className="h-3.5 w-3.5" />,
    summary: `${countWords(inputText)} tokens · ${inputText.length} chars · ${xaiSentences} sentences`,
    body: (
      <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
        <dt className="text-muted-foreground">Character count</dt>
        <dd className="numeric text-right">{inputText.length}</dd>
        <dt className="text-muted-foreground">Word count</dt>
        <dd className="numeric text-right">{countWords(inputText)}</dd>
        <dt className="text-muted-foreground">Sentences scored</dt>
        <dd className="numeric text-right">{xaiSentences}</dd>
        <dt className="text-muted-foreground">Excerpt</dt>
        <dd className="text-right text-muted-foreground">
          {inputText.slice(0, 80) + (inputText.length > 80 ? "…" : "")}
        </dd>
      </dl>
    ),
  };

  const encodeStep: Step = {
    key: "encode",
    title: "Encode",
    icon: <Layers className="h-3.5 w-3.5" />,
    summary: encoderKey
      ? `Encoder: ${encoderKey}`
      : "Encoder: classifier-side embedding (default)",
    body: (
      <dl className="grid gap-x-4 gap-y-1 text-xs sm:grid-cols-2">
        <dt className="text-muted-foreground">Encoder alias</dt>
        <dd className="text-right">{encoderKey ?? "default"}</dd>
        <dt className="text-muted-foreground">OOD energy</dt>
        <dd className="numeric text-right">
          {sentiment?.ood_energy != null ? sentiment.ood_energy.toFixed(3) : "—"}
        </dd>
        <dt className="text-muted-foreground">OOD threshold</dt>
        <dd className="numeric text-right">
          {sentiment?.ood_threshold != null ? sentiment.ood_threshold.toFixed(3) : "—"}
        </dd>
        <dt className="text-muted-foreground">In-distribution</dt>
        <dd className="text-right">
          {sentiment?.is_in_distribution == null ? (
            "—"
          ) : sentiment.is_in_distribution ? (
            <Badge variant="dovish" className="text-[10px]">yes</Badge>
          ) : (
            <Badge variant="hawkish" className="text-[10px]">no</Badge>
          )}
        </dd>
      </dl>
    ),
  };

  const multiAxisStep: Step = {
    key: "multi-axis",
    title: "Multi-axis head",
    icon: <Workflow className="h-3.5 w-3.5" />,
    summary: multiAxis ? multiAxisSummary(multiAxis) : "Multi-axis classifier checkpoint absent",
    body: multiAxis ? (
      <ul className="grid gap-1 text-xs sm:grid-cols-2">
        <li className="flex items-center justify-between">
          <span className="text-muted-foreground">stance</span>
          <span className="numeric capitalize">
            {multiAxis.stance?.label ?? "—"} · {multiAxis.stance?.confidence.toFixed(2) ?? "—"}
          </span>
        </li>
        <li className="flex items-center justify-between">
          <span className="text-muted-foreground">factor</span>
          <span className="numeric">
            {multiAxis.factor
              ? `${multiAxis.factor.value >= 0 ? "+" : ""}${multiAxis.factor.value.toFixed(2)} · ±${multiAxis.factor.confidence.toFixed(2)}`
              : "—"}
          </span>
        </li>
        <li className="flex items-center justify-between">
          <span className="text-muted-foreground">certainty</span>
          <span className="numeric capitalize">
            {multiAxis.certainty?.label ?? "—"}{" "}
            {multiAxis.certainty ? `· ${multiAxis.certainty.confidence.toFixed(2)}` : ""}
          </span>
        </li>
        <li className="flex items-center justify-between">
          <span className="text-muted-foreground">topic</span>
          <span className="numeric capitalize">
            {(multiAxis.topic?.label ?? multiAxis.topic?.primary ?? "—")
              .toString()
              .replace(/_/g, " ")}{" "}
            {multiAxis.topic ? `· ${multiAxis.topic.confidence.toFixed(2)}` : ""}
          </span>
        </li>
      </ul>
    ) : (
      <p className="text-xs text-muted-foreground">
        No multi-axis classifier checkpoint at <code>backend/models/text_multi_axis_best.pt</code>.
        The stance card is populated from the legacy sentiment classifier fallback.
      </p>
    ),
  };

  const regimeStep: Step = {
    key: "regime",
    title: "Regime head",
    icon: <Cpu className="h-3.5 w-3.5" />,
    summary: regime
      ? `softmax argmax: ${regime.argmax_class} · set size ${regime.set_size}`
      : "Regime classifier disabled (regression-only checkpoint or no conformal sidecar)",
    body: regime ? (
      <div className="space-y-2 text-xs">
        <dl className="grid gap-x-4 gap-y-1 sm:grid-cols-2">
          <dt className="text-muted-foreground">Argmax class</dt>
          <dd className="text-right capitalize">{regime.argmax_class}</dd>
          <dt className="text-muted-foreground">Set composition</dt>
          <dd className="text-right capitalize">{regime.predicted_set.join(", ") || "—"}</dd>
          <dt className="text-muted-foreground">Set size</dt>
          <dd className="numeric text-right">{regime.set_size}</dd>
          <dt className="text-muted-foreground">Coverage (nominal)</dt>
          <dd className="numeric text-right">{Math.round(regime.coverage * 100)}%</dd>
        </dl>
        <p className="text-[11px] text-muted-foreground">
          Calibrated APS: classes whose softmax mass meets the calibration quantile are included in the set.
        </p>
        <div className="space-y-1">
          {Object.entries(regime.distribution).map(([label, prob]) => (
            <div key={label} className="flex items-center justify-between">
              <span className="capitalize">{label}</span>
              <span className="numeric">{(prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>
    ) : (
      <p className="text-xs text-muted-foreground">
        The active checkpoint is regression-mode or no <code>.conformal.json</code> sidecar with{" "}
        <code>softmax_quantile</code> was found. Train a classification-mode checkpoint to populate this step.
      </p>
    ),
  };

  const calibrationStep: Step = {
    key: "calibration",
    title: "Calibration",
    icon: <ShieldCheck className="h-3.5 w-3.5" />,
    summary: regime
      ? `Calibrated split-conformal · ${Math.round(regime.coverage * 100)}% nominal`
      : "Calibration manifest absent",
    body: regime ? (
      <p className="text-xs text-muted-foreground">
        Split-conformal prediction set at {Math.round(regime.coverage * 100)}% nominal coverage.
        Empirical coverage tracked separately on the Performance page once enough resolved runs exist.
      </p>
    ) : (
      <p className="text-xs text-muted-foreground">
        Calibration is only available when the regime head is active. Performance metrics on the
        Performance page surface empirical coverage independently.
      </p>
    ),
  };

  return [ingestStep, encodeStep, multiAxisStep, regimeStep, calibrationStep];
}

function multiAxisSummary(multiAxis: MultiAxisResponse): string {
  const parts: string[] = [];
  if (multiAxis.stance) parts.push(`stance ${multiAxis.stance.label}`);
  if (multiAxis.factor) parts.push(`factor ${multiAxis.factor.value.toFixed(2)}`);
  if (multiAxis.certainty) parts.push(`certainty ${multiAxis.certainty.label}`);
  if (multiAxis.topic) parts.push(`topic ${(multiAxis.topic.label ?? multiAxis.topic.primary ?? "—").toString()}`);
  return parts.join(" · ") || "axes absent";
}

export function PipelineTrace(props: PipelineTraceProps) {
  const [openKey, setOpenKey] = React.useState<string>("regime");
  const steps = React.useMemo(() => buildSteps(props), [props]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Workflow className="h-4 w-4 text-primary" />
          Pipeline trace
        </CardTitle>
        <CardDescription>
          What the backend ran end-to-end. Click any step to expand its diagnostics.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-1.5">
        {steps.map((step, idx) => {
          const open = openKey === step.key;
          return (
            <div
              key={step.key}
              className={cn(
                "rounded-md border border-border bg-muted/20",
                open && "bg-muted/40",
              )}
            >
              <button
                type="button"
                onClick={() => setOpenKey(open ? "" : step.key)}
                className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-expanded={open}
              >
                <span className="flex items-center gap-2">
                  <span className="numeric text-[10px] text-muted-foreground">
                    {String(idx + 1).padStart(2, "0")}
                  </span>
                  <span className="text-muted-foreground">{step.icon}</span>
                  <span className="text-sm font-medium">{step.title}</span>
                </span>
                <span className="flex items-center gap-2">
                  <span className="hidden text-xs text-muted-foreground sm:inline">{step.summary}</span>
                  <ChevronDown
                    className={cn(
                      "h-3.5 w-3.5 text-muted-foreground transition-transform",
                      open && "rotate-180",
                    )}
                    aria-hidden="true"
                  />
                </span>
              </button>
              {open ? (
                <div className="border-t border-border px-3 py-3 sm:hidden">
                  <p className="mb-2 text-xs text-muted-foreground">{step.summary}</p>
                  {step.body}
                </div>
              ) : null}
              {open ? (
                <div className="hidden border-t border-border px-3 py-3 sm:block">{step.body}</div>
              ) : null}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
