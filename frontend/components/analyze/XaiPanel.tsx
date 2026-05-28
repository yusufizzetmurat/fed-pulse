import { Highlighter } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type {
  XaiPanelAttribution,
  XaiResponse,
  XaiSentence,
  XaiTokenAttribution,
} from "@/lib/analyze/types";

interface XaiPanelProps {
  xai: XaiResponse;
  previewMode?: boolean;
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
    <table className="w-full text-left font-mono text-[11px]">
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
              className={`text-right font-medium ${
                token.weight >= 0 ? "text-hawkish" : "text-dovish"
              }`}
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

function SentenceChip({ sentence }: { sentence: XaiSentence }) {
  return (
    <Tooltip delayDuration={120}>
      <TooltipTrigger asChild>
        <span
          className="cursor-help rounded px-1 py-0.5 leading-relaxed"
          style={{ backgroundColor: sentenceBackground(sentence.score) }}
          aria-label={`${sentenceLabel(sentence.score)} (${sentence.score.toFixed(2)})`}
        >
          {sentence.text}
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" className="max-w-xs space-y-1.5">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          {sentenceLabel(sentence.score)} · {sentence.score.toFixed(2)}
        </p>
        <TokenTable tokens={sentence.topTokens.slice(0, 5)} />
      </TooltipContent>
    </Tooltip>
  );
}

function familyLabel(family: string): string {
  // Cosmetic relabel — backend slice names are snake_case; render the
  // human-readable form alongside the magnitude bars.
  const map: Record<string, string> = {
    market: "Market",
    credibility: "Credibility",
    linguistic: "Linguistic",
    mp_surprise: "Policy surprise",
    multi_axis: "Sentiment breakdown",
    realized_vol: "Realised volatility",
    cross_asset: "Cross-asset",
    llm: "LLM features",
    trajectory_input: "Trajectory input",
  };
  return map[family] ?? family;
}

function panelLabel(panel: string): string {
  if (panel === "regime") return "Volatility Regime";
  if (panel.startsWith("rates_")) {
    const head = panel.slice("rates_".length);
    return `Rates · ${head}`;
  }
  if (panel === "trajectory") return "Trajectory";
  return panel;
}

function unavailableCopy(reason: string | null): string {
  switch (reason) {
    case "not_classification_mode":
      return "Panel inactive on the active model.";
    case "head_not_mounted":
      return "This prediction is not enabled on the active model.";
    case "no_multi_task_forward":
      return "Model does not expose the joint prediction needed for this view.";
    case "inference_kwarg_missing":
      return "Model inputs do not match what the explainer expects.";
    case "ig_runtime_error":
    case "unexpected_exception":
      return "Explanation engine error.";
    case "missing_stance_logits":
    case "missing_logits":
      return "Target prediction missing from the model output.";
    case "bundle_not_loaded":
      return "Model not loaded.";
    default:
      return "Explanation unavailable for this panel.";
  }
}

export function PanelAttributionRow({ panel }: { panel: XaiPanelAttribution }) {
  if (panel.unavailable) {
    return (
      <div
        className="flex items-center justify-between rounded-md border border-dashed border-border bg-muted/30 px-3 py-2"
        data-testid={`panel-attribution-${panel.panel}`}
      >
        <span className="text-xs font-medium">{panelLabel(panel.panel)}</span>
        <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
          Explanation not available
        </Badge>
        <span className="ml-2 text-[11px] text-muted-foreground">
          {unavailableCopy(panel.reason)}
        </span>
      </div>
    );
  }
  // Scale magnitudes to (0, 1] inside this panel so the longest bar
  // anchors at 100%. Sum is always >= 0 by construction.
  const maxMagnitude = panel.families.reduce(
    (acc, item) => Math.max(acc, Math.abs(item.magnitude)),
    0,
  );
  const scaleFactor = maxMagnitude > 0 ? 1 / maxMagnitude : 0;
  return (
    <div className="space-y-2" data-testid={`panel-attribution-${panel.panel}`}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold">{panelLabel(panel.panel)}</span>
        <span className="text-[10px] text-muted-foreground">
          target: {panel.target} · {panel.n_steps} attribution steps
        </span>
      </div>
      <div className="space-y-1">
        {panel.families.map((family) => {
          const width = Math.max(0, Math.min(1, Math.abs(family.magnitude) * scaleFactor)) * 100;
          const direction = family.signed >= 0 ? "--hawkish" : "--dovish";
          return (
            <div key={family.family} className="grid grid-cols-[6rem_1fr_3rem] items-center gap-2">
              <span className="text-[11px] text-muted-foreground">{familyLabel(family.family)}</span>
              <div className="h-2 rounded bg-muted/40">
                <div
                  className="h-2 rounded"
                  style={{
                    width: `${width}%`,
                    backgroundColor: `hsl(var(${direction}) / 0.7)`,
                  }}
                />
              </div>
              <span className="text-right font-mono text-[10px] text-muted-foreground">
                {family.magnitude.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function XaiPanel({ xai, previewMode }: XaiPanelProps) {
  const isEmpty = !xai.sentences.length;
  const panels = xai.panels ?? [];
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Highlighter className="h-4 w-4 text-primary" />
          Per-sentence explanation
          {previewMode ? (
            <Badge variant="outline" className="ml-2 text-[10px] uppercase tracking-wide">
              Preview · sample data
            </Badge>
          ) : null}
        </CardTitle>
        <CardDescription>
          Hover any sentence to see the five words that mattered most.
          {xai.method ? ` Method: ${xai.method}.` : null}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isEmpty ? (
          <p className="rounded-md border border-dashed border-border bg-muted/30 px-3 py-4 text-sm text-muted-foreground">
            No high-impact sentences found. The explanation method found no words above the
            sensitivity threshold. This is common for very short inputs, or text that lies
            outside the FOMC vocabulary the model was trained on.
          </p>
        ) : (
          <p className="space-x-1.5 text-sm leading-7">
            {xai.sentences.map((sentence, idx) => (
              <SentenceChip key={`${idx}-${sentence.text.slice(0, 16)}`} sentence={sentence} />
            ))}
          </p>
        )}
        {panels.length > 0 ? (
          <div className="space-y-3 border-t border-border pt-3" data-testid="panel-attributions">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
              What drove each panel · grouped by feature type
            </p>
            {panels.map((panel) => (
              <PanelAttributionRow key={panel.panel} panel={panel} />
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
