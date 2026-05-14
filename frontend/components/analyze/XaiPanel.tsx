import { Highlighter } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { XaiResponse, XaiSentence, XaiTokenAttribution } from "@/lib/analyze/types";

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

export function XaiPanel({ xai, previewMode }: XaiPanelProps) {
  const isEmpty = !xai.sentences.length;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Highlighter className="h-4 w-4 text-primary" />
          Sentence attribution
          {previewMode ? (
            <Badge variant="outline" className="ml-2 text-[10px] uppercase tracking-wide">
              Preview · fixture
            </Badge>
          ) : null}
        </CardTitle>
        <CardDescription>
          Hover any sentence to see the five tokens with the largest attribution.
          {xai.method ? ` Method: ${xai.method}.` : null}
        </CardDescription>
      </CardHeader>
      <CardContent>
        {isEmpty ? (
          <p className="rounded-md border border-dashed border-border bg-muted/30 px-3 py-4 text-sm text-muted-foreground">
            No salient sentences detected. The attribution method found no tokens above the salience floor — common for very short
            inputs or text that lies outside the FOMC vocabulary the model was trained on.
          </p>
        ) : (
          <p className="space-x-1.5 text-sm leading-7">
            {xai.sentences.map((sentence, idx) => (
              <SentenceChip key={`${idx}-${sentence.text.slice(0, 16)}`} sentence={sentence} />
            ))}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
