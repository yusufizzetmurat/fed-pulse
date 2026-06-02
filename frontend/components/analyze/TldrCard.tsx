import * as React from "react";

import { Card, CardContent } from "@/components/ui/card";
import type {
  AnalyzeResult,
  MultiAxisStance,
  RegimeClassificationResponse,
} from "@/lib/analyze/types";

function regimeLine(regime: RegimeClassificationResponse | null | undefined): string | null {
  if (!regime?.argmax_class) return null;
  if (regime.argmax_class === "high") {
    return "Predicts elevated realised volatility over the next 10 days.";
  }
  if (regime.argmax_class === "calm") {
    return "Predicts low realised volatility over the next 10 days.";
  }
  if (regime.argmax_class === "normal") {
    return "Predicts realised volatility in line with the recent baseline.";
  }
  return null;
}

function stanceWord(stance: MultiAxisStance | null | undefined): string | null {
  if (!stance) return null;
  if (stance.label === "hawkish") return "hawkish";
  if (stance.label === "dovish") return "dovish";
  if (stance.label === "neutral") return "neutral";
  return null;
}

/**
 * One-line plain-English summary of the analysis. Reads the regime
 * argmax + multi-axis stance from the existing analyze response.
 * Renders nothing when the model has neither field.
 */
export function TldrCard({ result }: { result: AnalyzeResult }) {
  const stance = stanceWord(result.multi_axis?.stance ?? null);
  const regime = result.regime_classification ?? null;
  const lowConviction =
    regime?.argmax_class === "normal" &&
    (regime?.distribution?.normal ?? 0) < 0.5;
  const headlinePrefix =
    stance === "hawkish"
      ? "This statement is hawkish."
      : stance === "dovish"
      ? "This statement is dovish."
      : stance === "neutral"
      ? "This statement is neutral."
      : null;
  const tail = lowConviction
    ? "The model has low conviction."
    : regimeLine(regime);
  if (!headlinePrefix && !tail) return null;
  const text = [headlinePrefix, tail].filter(Boolean).join(" ");
  return (
    <Card>
      <CardContent className="px-4 py-3 text-sm text-muted-foreground">
        <span className="font-medium text-foreground">TL;DR</span>{" "}
        <span>{text}</span>
      </CardContent>
    </Card>
  );
}
