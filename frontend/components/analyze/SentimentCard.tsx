import { AlertTriangle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { stanceLabel, toStance } from "@/lib/analyze/format";
import type { SentimentResponse } from "@/lib/analyze/types";

interface SentimentCardProps {
  sentiment?: SentimentResponse;
}

function formatEnergy(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(2);
}

export function SentimentCard({ sentiment }: SentimentCardProps) {
  const rawLabel = sentiment?.label;
  const stance = toStance(rawLabel);
  const score = Number(sentiment?.score ?? 0);
  const isUnknown = stance === "unknown";
  const badgeVariant: "hawkish" | "dovish" | "neutral" | "outline" =
    stance === "hawkish" ? "hawkish" : stance === "dovish" ? "dovish" : stance === "neutral" ? "neutral" : "outline";
  const energy = sentiment?.ood_energy;
  const threshold = sentiment?.ood_threshold;
  const inDist = sentiment?.is_in_distribution;
  const isOod = inDist === false;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Stance</CardDescription>
        <CardTitle className="text-2xl">
          {isUnknown ? "Sentiment unavailable" : stanceLabel(stance)}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <Badge variant={badgeVariant}>
            {isUnknown ? `raw: ${String(rawLabel ?? "n/a")}` : `${stanceLabel(stance)} · ${score.toFixed(3)}`}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {isUnknown ? "Backend returned a non-stance label; check sentiment-model load." : "model confidence"}
          </span>
        </div>
        {isOod ? (
          <div
            className="flex items-start gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs"
            role="status"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" aria-hidden="true" />
            <div className="space-y-0.5">
              <strong className="text-amber-700 dark:text-amber-300">Out of distribution</strong>
              <p className="text-muted-foreground">
                Text doesn&apos;t look like the FOMC corpus the classifier was trained on; treat the stance as low confidence.
              </p>
              <p className="font-mono text-[11px] text-muted-foreground">
                energy {formatEnergy(energy)} &gt; threshold {formatEnergy(threshold)}
              </p>
            </div>
          </div>
        ) : inDist === true && energy != null ? (
          <p className="font-mono text-[11px] text-muted-foreground">
            in-distribution · energy {formatEnergy(energy)} ≤ {formatEnergy(threshold)}
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
