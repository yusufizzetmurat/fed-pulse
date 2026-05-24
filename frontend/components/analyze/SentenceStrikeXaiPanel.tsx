import * as React from "react";
import { Highlighter, Loader2, RotateCcw } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type {
  AnalyzeResult,
  XaiResponse,
  XaiSentence,
  XaiTokenAttribution,
} from "@/lib/analyze/types";

interface SentenceStrikeXaiPanelProps {
  xai: XaiResponse;
  // When `onMaskChange` is provided the panel renders sentences as
  // clickable; clicking toggles strike-through and emits the next mask
  // upward so the parent can re-fire /analyze.
  struck?: Set<number>;
  onMaskChange?: (next: Set<number>) => void;
  baselineResult?: AnalyzeResult | null;
  currentResult?: AnalyzeResult | null;
  loading?: boolean;
}

function sentenceBackground(score: number): string {
  const clamped = Math.max(-1, Math.min(1, score));
  const magnitude = Math.abs(clamped).toFixed(2);
  const cssVar = clamped >= 0 ? "--hawkish" : "--dovish";
  return `hsl(var(${cssVar}) / ${magnitude})`;
}

function sentenceLabel(score: number): string {
  if (score > 0.5) return "Strong hawkish pull";
  if (score > 0.15) return "Hawkish pull";
  if (score > -0.15) return "Neutral";
  if (score > -0.5) return "Dovish pull";
  return "Strong dovish pull";
}

function TokenTable({ tokens }: { tokens: XaiTokenAttribution[] }) {
  return (
    <table className="numeric w-full text-left text-[11px]">
      <thead>
        <tr className="text-muted-foreground">
          <th className="pb-0.5 pr-3 font-normal">token</th>
          <th className="pb-0.5 text-right font-normal">weight</th>
        </tr>
      </thead>
      <tbody>
        {tokens.map((token) => (
          <tr key={token.token}>
            <td className="pr-3">{token.token}</td>
            <td
              className={cn(
                "text-right font-medium",
                token.weight >= 0 ? "text-hawkish" : "text-dovish",
              )}
            >
              {token.weight >= 0 ? "+" : ""}
              {token.weight.toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function formatProbabilityDelta(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  const pct = value * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)}pp`;
}

function describeArgmaxDiff(
  baseline: AnalyzeResult | null | undefined,
  current: AnalyzeResult | null | undefined,
): React.ReactNode {
  const baselineLabel = baseline?.regime_classification?.argmax_class ?? null;
  const currentLabel = current?.regime_classification?.argmax_class ?? null;
  if (!baselineLabel || !currentLabel) return null;
  if (baselineLabel === currentLabel) {
    const probDelta =
      (current?.regime_classification?.distribution?.[currentLabel] ?? 0) -
      (baseline?.regime_classification?.distribution?.[baselineLabel] ?? 0);
    return (
      <span className="flex items-center gap-1">
        <span className="capitalize">{currentLabel}</span>
        <span className="text-muted-foreground">·</span>
        <span className="numeric">{formatProbabilityDelta(probDelta)}</span>
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1">
      <span className="capitalize text-muted-foreground">{baselineLabel}</span>
      <span>→</span>
      <span className="capitalize">{currentLabel}</span>
    </span>
  );
}

function stanceConfidenceDelta(
  baseline: AnalyzeResult | null | undefined,
  current: AnalyzeResult | null | undefined,
): number | null {
  const a = baseline?.multi_axis?.stance?.confidence;
  const b = current?.multi_axis?.stance?.confidence;
  if (a == null || b == null) return null;
  return b - a;
}

function SentenceChip({
  sentence,
  index,
  interactive,
  struck,
  onToggle,
}: {
  sentence: XaiSentence;
  index: number;
  interactive: boolean;
  struck: boolean;
  onToggle?: () => void;
}) {
  const inner = (
    <span
      className={cn(
        "rounded px-1 py-0.5 leading-relaxed transition-colors",
        interactive && "cursor-pointer hover:ring-2 hover:ring-ring",
        struck && "line-through opacity-60",
      )}
      style={struck ? undefined : { backgroundColor: sentenceBackground(sentence.score) }}
      aria-label={`Sentence ${index + 1} · ${sentenceLabel(sentence.score)} · score ${sentence.score.toFixed(2)}${struck ? " · struck" : ""}`}
    >
      {sentence.text}
    </span>
  );
  return (
    <Tooltip delayDuration={120}>
      <TooltipTrigger asChild>
        {interactive ? (
          <button
            type="button"
            onClick={onToggle}
            className="inline-block focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-pressed={struck}
          >
            {inner}
          </button>
        ) : (
          inner
        )}
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs space-y-1.5">
        <p className="numeric text-[11px] uppercase tracking-wide text-muted-foreground">
          {sentenceLabel(sentence.score)} · {sentence.score.toFixed(2)}
        </p>
        {sentence.topTokens?.length ? <TokenTable tokens={sentence.topTokens.slice(0, 5)} /> : null}
        {interactive ? (
          <p className="text-[10px] text-muted-foreground">
            Click to {struck ? "restore" : "strike out"} — the panel re-runs /analyze without this sentence.
          </p>
        ) : null}
      </TooltipContent>
    </Tooltip>
  );
}

export function SentenceStrikeXaiPanel({
  xai,
  struck,
  onMaskChange,
  baselineResult,
  currentResult,
  loading = false,
}: SentenceStrikeXaiPanelProps) {
  const interactive = Boolean(onMaskChange);
  const struckSet = struck ?? new Set<number>();
  const argmaxDiff = describeArgmaxDiff(baselineResult, currentResult);
  const stanceDelta = stanceConfidenceDelta(baselineResult, currentResult);

  const handleToggle = React.useCallback(
    (index: number) => {
      if (!onMaskChange) return;
      const next = new Set(struckSet);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      onMaskChange(next);
    },
    [onMaskChange, struckSet],
  );

  const handleReset = React.useCallback(() => {
    onMaskChange?.(new Set<number>());
  }, [onMaskChange]);

  const isEmpty = !xai.sentences.length;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <Highlighter className="h-4 w-4 text-primary" />
            Sentence attribution
            {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" /> : null}
          </CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            {struckSet.size > 0 ? (
              <Badge variant="outline" className="text-[10px]">
                {struckSet.size} struck
              </Badge>
            ) : null}
            {argmaxDiff ? (
              <Badge variant="outline" className="text-[10px]">
                <span className="text-muted-foreground">Δ regime</span>
                <span className="ml-1">{argmaxDiff}</span>
              </Badge>
            ) : null}
            {stanceDelta != null ? (
              <Badge variant="outline" className="text-[10px]">
                <span className="text-muted-foreground">Δ stance</span>
                <span
                  className={cn(
                    "numeric ml-1",
                    stanceDelta > 0 ? "text-up" : stanceDelta < 0 ? "text-down" : "",
                  )}
                >
                  {stanceDelta >= 0 ? "+" : ""}
                  {stanceDelta.toFixed(2)}
                </span>
              </Badge>
            ) : null}
            {struckSet.size > 0 ? (
              <Button variant="ghost" size="sm" onClick={handleReset}>
                <RotateCcw className="h-3.5 w-3.5" />
                Reset
              </Button>
            ) : null}
          </div>
        </div>
        <CardDescription>
          {interactive
            ? "Click a sentence to strike it out — the dashboard re-runs the classifier without it and shows the Δ above. "
            : "Hover any sentence to see the five tokens with the largest attribution. "}
          {xai.method ? `Method: ${xai.method}.` : null}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isEmpty ? (
          <p className="rounded-md border border-dashed border-border bg-muted/30 px-3 py-4 text-sm text-muted-foreground">
            No salient sentences detected. The attribution method found no tokens above the salience floor — common for
            very short inputs or text that lies outside the FOMC vocabulary the model was trained on.
          </p>
        ) : (
          <p className="space-x-1.5 text-sm leading-7">
            {xai.sentences.map((sentence, idx) => (
              <SentenceChip
                key={`${idx}-${sentence.text.slice(0, 16)}`}
                sentence={sentence}
                index={idx}
                interactive={interactive}
                struck={struckSet.has(idx)}
                onToggle={() => handleToggle(idx)}
              />
            ))}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
