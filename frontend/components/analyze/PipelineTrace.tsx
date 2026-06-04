import * as React from "react";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  ChevronDown,
  Cpu,
  FileInput,
  Highlighter,
  Layers,
  ShieldCheck,
  Workflow,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { friendlyEncoderName } from "@/lib/analyze/encoders";
import { cn } from "@/lib/utils";
import type {
  AnalyzeResult,
  MultiAxisResponse,
} from "@/lib/analyze/types";

interface PipelineTraceProps {
  result: AnalyzeResult;
  inputText: string;
}

interface Step {
  key: string;
  title: string;
  blurb: string;
  icon: React.ReactNode;
  summary: string;
  state: "ok" | "warn" | "absent";
  body: React.ReactNode;
}

function countWords(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function regimeBarClass(label: string): string {
  if (label === "calm") return "bg-dovish";
  if (label === "high") return "bg-hawkish";
  return "bg-neutral";
}

function MiniBar({
  label,
  value,
  max = 1,
  tone = "neutral",
}: {
  label: string;
  value: number | null | undefined;
  max?: number;
  tone?: "neutral" | "hawkish" | "dovish" | "up" | "down";
}) {
  const pct =
    value == null || !Number.isFinite(value) || max <= 0
      ? 0
      : Math.max(0, Math.min(1, value / max)) * 100;
  const fillClass =
    tone === "hawkish"
      ? "bg-hawkish"
      : tone === "dovish"
      ? "bg-dovish"
      : tone === "up"
      ? "bg-up"
      : tone === "down"
      ? "bg-down"
      : "bg-primary/70";
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="numeric text-foreground">
          {value == null || !Number.isFinite(value) ? "—" : value.toFixed(2)}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full", fillClass)} style={{ width: `${pct}%` }} aria-hidden="true" />
      </div>
    </div>
  );
}

function CoverageGauge({ coverage }: { coverage: number }) {
  const pct = Math.round(coverage * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">Confidence level</span>
        <span className="numeric text-foreground">{pct}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full bg-up" style={{ width: `${pct}%` }} aria-hidden="true" />
      </div>
    </div>
  );
}

function multiAxisSummary(multiAxis: MultiAxisResponse): string {
  const parts: string[] = [];
  if (multiAxis.stance) parts.push(`stance ${multiAxis.stance.label}`);
  if (multiAxis.factor) parts.push(`factor ${multiAxis.factor.value.toFixed(2)}`);
  if (multiAxis.certainty) parts.push(`certainty ${multiAxis.certainty.label}`);
  return parts.join(" · ") || "no sentiment labels";
}

function buildSteps({ result, inputText }: PipelineTraceProps): Step[] {
  const multiAxis = result.multi_axis;
  const regime = result.regime_classification;
  const encoderKey = (result.model as { encoder_key?: string } | undefined)?.encoder_key;
  const xaiSentences = result.xai?.sentences?.length ?? 0;
  const oodEnergy = result.sentiment?.ood_energy ?? null;
  const oodThreshold = result.sentiment?.ood_threshold ?? null;
  const inDistribution = result.sentiment?.is_in_distribution ?? null;

  const ingestStep: Step = {
    key: "ingest",
    title: "Ingest",
    blurb:
      "Splits the document into sentences and records the source. Every downstream step operates on these sentence units.",
    icon: <FileInput className="h-3.5 w-3.5" />,
    state: "ok",
    summary: `${countWords(inputText)} words · ${inputText.length} chars · ${xaiSentences} sentences`,
    body: (
      <div className="space-y-3">
        <dl className="grid gap-x-4 gap-y-1.5 text-sm sm:grid-cols-2">
          <dt className="text-muted-foreground">Character count</dt>
          <dd className="numeric text-right">{inputText.length}</dd>
          <dt className="text-muted-foreground">Word count</dt>
          <dd className="numeric text-right">{countWords(inputText)}</dd>
          <dt className="text-muted-foreground">Sentences scored</dt>
          <dd className="numeric text-right">{xaiSentences}</dd>
        </dl>
        <div className="rounded-md border border-border bg-muted/30 p-2 font-mono text-xs leading-relaxed text-muted-foreground">
          {inputText.slice(0, 220)}
          {inputText.length > 220 ? "…" : ""}
        </div>
      </div>
    ),
  };

  const encodeStep: Step = {
    key: "encode",
    title: "Encode",
    blurb:
      "Runs the text through the language model and emits the encoder embedding the downstream heads consume.",
    icon: <Layers className="h-3.5 w-3.5" />,
    state: "ok",
    summary: encoderKey
      ? `Model variant: ${friendlyEncoderName(encoderKey)}`
      : "Model variant: default",
    body: (
      <div className="space-y-1.5 text-sm text-muted-foreground">
        <p>
          Model variant: <span className="numeric text-foreground">{encoderKey ?? "default"}</span>.
        </p>
        {inDistribution === null ? (
          <p>OOD signal not available on this checkpoint.</p>
        ) : (
          <p>
            OOD check:{" "}
            <span
              className={
                inDistribution
                  ? "text-foreground numeric"
                  : "text-hawkish numeric font-semibold"
              }
            >
              {inDistribution ? "in-distribution" : "out-of-distribution"}
            </span>
            {oodEnergy !== null && oodThreshold !== null ? (
              <>
                {" "}
                · Mahalanobis distance{" "}
                <span className="numeric text-foreground">{oodEnergy.toFixed(1)}</span>{" "}
                vs threshold{" "}
                <span className="numeric text-foreground">{oodThreshold.toFixed(1)}</span>
              </>
            ) : null}
            .
          </p>
        )}
      </div>
    ),
  };

  const multiAxisStep: Step = {
    key: "multi-axis",
    title: "Sentiment breakdown",
    blurb:
      "Three predictions from the same shared model: stance, hawkish / dovish score, and certainty. Calibrated against the labelled FOMC corpus.",
    icon: <Workflow className="h-3.5 w-3.5" />,
    state: multiAxis ? "ok" : "absent",
    summary: multiAxis ? multiAxisSummary(multiAxis) : "Sentiment model not loaded",
    body: multiAxis ? (
      <div className="space-y-3">
        <ul className="grid gap-1.5 text-sm sm:grid-cols-2">
          <li className="flex items-center justify-between">
            <span className="text-muted-foreground">stance</span>
            <span className="numeric capitalize">
              {multiAxis.stance?.label ?? "—"}{" "}
              {multiAxis.stance ? `· ${multiAxis.stance.confidence.toFixed(2)}` : ""}
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
        </ul>
        <div className="grid gap-2 sm:grid-cols-2">
          {multiAxis.stance ? (
            <MiniBar
              label="stance confidence"
              value={multiAxis.stance.confidence}
              tone={
                multiAxis.stance.label === "hawkish"
                  ? "hawkish"
                  : multiAxis.stance.label === "dovish"
                  ? "dovish"
                  : "neutral"
              }
            />
          ) : null}
          {multiAxis.factor ? (
            <MiniBar
              label="factor magnitude"
              value={Math.abs(multiAxis.factor.value)}
              tone={multiAxis.factor.value >= 0 ? "hawkish" : "dovish"}
            />
          ) : null}
          {multiAxis.certainty ? (
            <MiniBar label="certainty confidence" value={multiAxis.certainty.confidence} />
          ) : null}
        </div>
      </div>
    ) : (
      <p className="text-sm text-muted-foreground">
        Sentiment breakdown model isn't loaded. The stance card falls back to the
        legacy sentiment classifier.
      </p>
    ),
  };

  const regimeStep: Step = {
    key: "regime",
    title: "Volatility Regime prediction",
    blurb:
      "Predicts the 10-day forward volatility regime (calm / normal / high) together with a calibrated prediction set so the UI can hedge across more than one class when the model is unsure.",
    icon: <Cpu className="h-3.5 w-3.5" />,
    state: regime ? "ok" : "absent",
    summary: regime
      ? `Top pick: ${regime.argmax_class} · ${regime.set_size} label${regime.set_size === 1 ? "" : "s"} in set`
      : "Numeric-mode model or calibration data not loaded",
    body: regime ? (
      <div className="space-y-3 text-sm">
        <dl className="grid gap-x-4 gap-y-1.5 sm:grid-cols-2">
          <dt className="text-muted-foreground">Top pick</dt>
          <dd className="text-right capitalize">{regime.argmax_class}</dd>
          <dt className="text-muted-foreground">Labels in prediction set</dt>
          <dd className="text-right capitalize">{regime.predicted_set.join(", ") || "—"}</dd>
          <dt className="text-muted-foreground">Set size</dt>
          <dd className="numeric text-right">{regime.set_size}</dd>
        </dl>
        <div className="space-y-2">
          {Object.entries(regime.distribution).map(([label, prob]) => {
            const inSet = regime.predicted_set.includes(label);
            return (
              <div key={label} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span
                    className={cn(
                      "flex items-center gap-1.5 capitalize",
                      inSet ? "text-foreground" : "text-muted-foreground",
                    )}
                  >
                    <span
                      className={cn("h-1.5 w-1.5 rounded-full", regimeBarClass(label))}
                      aria-hidden="true"
                    />
                    {label}
                    {inSet ? (
                      <span className="text-xs text-muted-foreground">in set</span>
                    ) : null}
                  </span>
                  <span
                    className={cn(
                      "numeric",
                      inSet ? "font-medium text-foreground" : "text-muted-foreground",
                    )}
                  >
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={prob} indicatorClassName={regimeBarClass(label)} />
              </div>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground">
          The prediction set includes every regime whose probability clears the calibration
          threshold. The threshold and confidence level come from a calibration step run on
          held-out data.
        </p>
      </div>
    ) : (
      <p className="text-sm text-muted-foreground">
        The active model is in numeric mode, or no calibration data was found. The numeric
        forecaster still produces close and volatility point predictions in the response; the
        workspace hides them because the Volatility Regime prediction is the headline.
      </p>
    ),
  };

  const xaiStep: Step = {
    key: "xai",
    title: "Per-sentence explanation",
    blurb:
      "Scores each sentence by how much it pushed the model toward its stance decision. Powers the strike-out tool that lets you remove a sentence and re-score.",
    icon: <Highlighter className="h-3.5 w-3.5" />,
    state: xaiSentences > 0 ? "ok" : "absent",
    summary:
      xaiSentences > 0
        ? `${xaiSentences} sentences scored · method ${result.xai?.method ?? "keyword"}`
        : "No sentences detected",
    body:
      xaiSentences > 0 ? (
        <div className="space-y-2">
          <p className="text-sm text-muted-foreground">
            Three highest-impact sentences. The full panel above lets you strike any of them.
          </p>
          <ul className="space-y-1.5">
            {[...(result.xai?.sentences ?? [])]
              .sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
              .slice(0, 3)
              .map((sentence, idx) => (
                <li
                  key={`${idx}-${sentence.text.slice(0, 12)}`}
                  className="flex items-start gap-2 rounded-md border border-border bg-muted/20 p-2"
                >
                  <Badge
                    variant={sentence.score >= 0 ? "hawkish" : "dovish"}
                    className="numeric text-[10px]"
                  >
                    {sentence.score >= 0 ? "+" : ""}
                    {sentence.score.toFixed(2)}
                  </Badge>
                  <span className="text-xs leading-snug text-foreground">
                    {sentence.text.length > 160
                      ? `${sentence.text.slice(0, 160)}…`
                      : sentence.text}
                  </span>
                </li>
              ))}
          </ul>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          No high-impact sentences found. This is common for very short inputs, or for text
          outside the FOMC vocabulary the explanation dictionary covers.
        </p>
      ),
  };

  const calibrationStep: Step = {
    key: "calibration",
    title: "Calibration",
    blurb:
      "How tight the prediction set is, and the confidence level it was calibrated for. Actual coverage on resolved runs is tracked on the Performance page.",
    icon: <ShieldCheck className="h-3.5 w-3.5" />,
    state: regime ? "ok" : "absent",
    summary: regime
      ? `Calibrated prediction set · ${Math.round(regime.coverage * 100)}% confidence level`
      : "Calibration data not loaded",
    body: regime ? (
      <div className="space-y-3">
        <CoverageGauge coverage={regime.coverage} />
        <dl className="grid gap-x-4 gap-y-1.5 text-sm sm:grid-cols-2">
          <dt className="text-muted-foreground">Set size</dt>
          <dd className="numeric text-right">{regime.set_size}</dd>
          <dt className="text-muted-foreground">Set label</dt>
          <dd className="text-right">{regime.set_label}</dd>
        </dl>
        <p className="text-xs text-muted-foreground">
          A tight set (size 1) means the model is confident enough to single out one regime; a
          looser set hedges across more classes. Actual coverage on resolved runs is tracked on
          the Performance page.
        </p>
      </div>
    ) : (
      <p className="text-sm text-muted-foreground">
        Calibration is only available when the Volatility Regime prediction is active. Actual
        coverage is still computed on the Performance page when realised regimes resolve.
      </p>
    ),
  };

  return [ingestStep, encodeStep, multiAxisStep, regimeStep, xaiStep, calibrationStep];
}

function StepRail({
  steps,
  activeKey,
  onSelect,
}: {
  steps: Step[];
  activeKey: string;
  onSelect: (key: string) => void;
}) {
  return (
    <ol className="flex flex-wrap items-center gap-1 text-xs">
      {steps.map((step, idx) => {
        const active = step.key === activeKey;
        const stateClass =
          step.state === "warn"
            ? "border-hawkish/60 text-hawkish"
            : step.state === "absent"
            ? "border-border text-muted-foreground"
            : "border-up/40 text-foreground";
        const StateIcon =
          step.state === "warn" ? AlertTriangle : step.state === "absent" ? null : Check;
        return (
          <li key={step.key} className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => onSelect(step.key)}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
                stateClass,
                active
                  ? "bg-background shadow-sm ring-2 ring-ring/40"
                  : "bg-muted/30 hover:bg-muted/50",
              )}
              aria-current={active ? "step" : undefined}
            >
              <span className="numeric text-[10px] text-muted-foreground">
                {String(idx + 1).padStart(2, "0")}
              </span>
              <span>{step.title}</span>
              {StateIcon ? <StateIcon className="h-3 w-3" aria-hidden="true" /> : null}
            </button>
            {idx < steps.length - 1 ? (
              <ArrowRight className="h-3 w-3 text-muted-foreground" aria-hidden="true" />
            ) : null}
          </li>
        );
      })}
    </ol>
  );
}

export function PipelineTrace({ result, inputText }: PipelineTraceProps) {
  const steps = React.useMemo(() => buildSteps({ result, inputText }), [result, inputText]);
  const [openKey, setOpenKey] = React.useState<string>("regime");

  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="space-y-1">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Workflow className="h-5 w-5 text-primary" />
            Pipeline trace
          </CardTitle>
          <CardDescription>
            End-to-end view of what the backend actually ran on this input. Click any step in the
            rail or row below for details.
          </CardDescription>
        </div>
        <StepRail steps={steps} activeKey={openKey} onSelect={setOpenKey} />
      </CardHeader>
      <CardContent className="space-y-2">
        {steps.map((step, idx) => {
          const open = openKey === step.key;
          return (
            <div
              key={step.key}
              className={cn(
                "rounded-md border border-border bg-muted/20",
                open && "bg-muted/40",
                step.state === "warn" && !open && "border-hawkish/30",
              )}
            >
              <button
                type="button"
                onClick={() => setOpenKey(open ? "" : step.key)}
                className="flex w-full items-start justify-between gap-3 px-3 py-2.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                aria-expanded={open}
              >
                <span className="flex items-start gap-2">
                  <span className="numeric mt-0.5 text-xs text-muted-foreground">
                    {String(idx + 1).padStart(2, "0")}
                  </span>
                  <span className="mt-0.5 text-muted-foreground">{step.icon}</span>
                  <span className="space-y-0.5">
                    <span className="block text-sm font-medium">{step.title}</span>
                    <span className="block text-xs text-muted-foreground">{step.blurb}</span>
                  </span>
                </span>
                <span className="flex items-center gap-2">
                  <span className="hidden text-xs text-muted-foreground sm:inline">
                    {step.summary}
                  </span>
                  <ChevronDown
                    className={cn(
                      "h-4 w-4 text-muted-foreground transition-transform",
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
