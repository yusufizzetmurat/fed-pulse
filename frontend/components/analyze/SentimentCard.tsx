import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { stanceLabel, toStance } from "@/lib/analyze/format";
import type { SentimentResponse } from "@/lib/analyze/types";

interface SentimentCardProps {
  sentiment?: SentimentResponse;
}

export function SentimentCard({ sentiment }: SentimentCardProps) {
  const stance = toStance(sentiment?.label);
  const score = Number(sentiment?.score ?? 0);
  const badgeVariant =
    stance === "hawkish" ? "hawkish" : stance === "dovish" ? "dovish" : stance === "neutral" ? "neutral" : "outline";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Stance</CardDescription>
        <CardTitle className="text-2xl">{stanceLabel(stance)}</CardTitle>
      </CardHeader>
      <CardContent className="flex items-center justify-between">
        <Badge variant={badgeVariant}>
          {stanceLabel(stance)} · {score.toFixed(3)}
        </Badge>
        <span className="text-xs text-muted-foreground">model confidence</span>
      </CardContent>
    </Card>
  );
}
