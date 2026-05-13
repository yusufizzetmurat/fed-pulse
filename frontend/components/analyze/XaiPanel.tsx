import { Highlighter } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { XaiResponse, XaiSentence, XaiTokenAttribution } from "@/lib/analyze/types";

interface XaiPanelProps {
  xai: XaiResponse;
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

export function XaiPanel({ xai }: XaiPanelProps) {
  if (!xai.sentences.length) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Highlighter className="h-4 w-4 text-primary" />
          Sentence attribution
        </CardTitle>
        <CardDescription>
          Hover any sentence to see the five tokens with the largest attribution.
          {xai.method ? ` Method: ${xai.method}.` : null}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <TooltipProvider>
          <p className="space-x-1.5 text-sm leading-7">
            {xai.sentences.map((sentence, idx) => (
              <SentenceChip key={`${idx}-${sentence.text.slice(0, 16)}`} sentence={sentence} />
            ))}
          </p>
        </TooltipProvider>
      </CardContent>
    </Card>
  );
}
