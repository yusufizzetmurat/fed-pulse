import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { stanceLabel, toStance } from "@/lib/analyze/format";
import type { SentimentResponse } from "@/lib/analyze/types";

interface SentimentCardProps {
  sentiment?: SentimentResponse;
}

export function SentimentCard({ sentiment }: SentimentCardProps) {
  const rawLabel = sentiment?.label;
  const stance = toStance(rawLabel);
  const score = Number(sentiment?.score ?? 0);
  const isUnknown = stance === "unknown";
  const badgeVariant: "hawkish" | "dovish" | "neutral" | "outline" =
    stance === "hawkish" ? "hawkish" : stance === "dovish" ? "dovish" : stance === "neutral" ? "neutral" : "outline";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Stance</CardDescription>
        <CardTitle className="text-2xl">
          {isUnknown ? "Sentiment unavailable" : stanceLabel(stance)}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex items-center justify-between">
        <Badge variant={badgeVariant}>
          {isUnknown ? `raw: ${String(rawLabel ?? "n/a")}` : `${stanceLabel(stance)} · ${score.toFixed(3)}`}
        </Badge>
        <span className="text-xs text-muted-foreground">
          {isUnknown ? "Backend returned a non-stance label; check sentiment-model load." : "model confidence"}
        </span>
      </CardContent>
    </Card>
  );
}
