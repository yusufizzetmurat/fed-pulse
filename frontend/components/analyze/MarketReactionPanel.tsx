import { ArrowDownRight, ArrowUpRight, Minus, TrendingUp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type {
  MarketReactionPanelResponse,
  RatesDirectionalBucket,
  RatesReactionCard as RatesReactionCardData,
  VolRegimeReactionCard as VolRegimeReactionCardData,
} from "@/lib/analyze/types";

const HEAD_LABELS: Record<RatesReactionCardData["head"], string> = {
  "2y": "2y yield",
  "5y": "5y yield",
  terminal: "Terminal rate",
};

const BUCKET_ORDER: RatesDirectionalBucket[] = ["easing", "neutral", "tightening"];

function bucketTone(bucket: RatesDirectionalBucket): "hawkish" | "dovish" | "neutral" {
  if (bucket === "tightening") return "hawkish";
  if (bucket === "easing") return "dovish";
  return "neutral";
}

function BucketIcon({ bucket }: { bucket: RatesDirectionalBucket }) {
  const tone = bucketTone(bucket);
  if (tone === "hawkish") return <ArrowUpRight className="h-4 w-4 text-hawkish" />;
  if (tone === "dovish") return <ArrowDownRight className="h-4 w-4 text-dovish" />;
  return <Minus className="h-4 w-4 text-neutral" />;
}

function formatBps(value: number): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)} bps`;
}

interface RatesReactionCardProps {
  card: RatesReactionCardData;
}

export function RatesReactionCard({ card }: RatesReactionCardProps) {
  // #317 finding #10: when the checkpoint exposes no aux classifier
  // for this head the backend returns ``directional_bucket: null`` and
  // ``bucket_probabilities: null``; render an explicit "aux classifier
  // unavailable" badge rather than fabricating an argmax on
  // non-existent probabilities.
  const hasBucket = card.directional_bucket != null && card.bucket_probabilities != null;
  const tone = hasBucket ? bucketTone(card.directional_bucket!) : "neutral";
  const bandText =
    card.lower_bps != null && card.upper_bps != null
      ? `${formatBps(card.lower_bps)} … ${formatBps(card.upper_bps)}`
      : "Band unavailable";
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <TrendingUp className="h-3.5 w-3.5" />
          {HEAD_LABELS[card.head]}
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl">
          <span className="numeric">{formatBps(card.point_bps)}</span>
          {hasBucket ? (
            <Badge variant={tone} className="flex items-center gap-1 capitalize">
              <BucketIcon bucket={card.directional_bucket!} />
              {card.directional_bucket}
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
              Aux classifier unavailable
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1 text-xs text-muted-foreground">
          <p>
            5-day post-event change predicted by the rates head.
          </p>
          <p>
            <span className="font-mono">{bandText}</span>
            {card.coverage != null ? (
              <span className="ml-2 text-[10px] uppercase tracking-wide text-muted-foreground">
                {(card.coverage * 100).toFixed(0)}% conformal
              </span>
            ) : null}
          </p>
          {card.predicted_set != null && card.predicted_set.length > 0 ? (
            <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
              calibrated set: {`{${card.predicted_set.join(", ")}}`}
            </p>
          ) : null}
        </div>
        {hasBucket ? (
          <div className="space-y-2">
            {BUCKET_ORDER.map((bucket) => {
              const value = card.bucket_probabilities![bucket] ?? 0;
              return (
                <div key={bucket} className="space-y-1">
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span className="capitalize">{bucket}</span>
                    <span className="font-medium text-foreground">
                      {(value * 100).toFixed(0)}%
                    </span>
                  </div>
                  <Progress
                    value={value * 100}
                    indicatorClassName={
                      bucket === "tightening"
                        ? "bg-hawkish"
                        : bucket === "easing"
                        ? "bg-dovish"
                        : "bg-neutral"
                    }
                  />
                </div>
              );
            })}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

interface VolRegimeReactionCardProps {
  card: VolRegimeReactionCardData;
}

function VolRegimeCard({ card }: VolRegimeReactionCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <TrendingUp className="h-3.5 w-3.5" />
          Vol regime
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl capitalize">
          <span>{card.regime_label}</span>
          <Badge variant="outline">
            {card.predicted_set.length > 0 ? `{${card.predicted_set.join(", ")}}` : "—"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {Object.entries(card.regime_probabilities).map(([label, value]) => (
          <div key={label} className="space-y-1">
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span className="capitalize">{label}</span>
              <span className="font-medium text-foreground">{(value * 100).toFixed(0)}%</span>
            </div>
            <Progress value={value * 100} />
          </div>
        ))}
        {card.log_rv_point != null ? (
          <p className="text-xs text-muted-foreground pt-1">
            Dual-head log(RV) prediction: <span className="font-mono">{card.log_rv_point.toFixed(3)}</span>
          </p>
        ) : null}
        {card.coverage != null ? (
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
            {(card.coverage * 100).toFixed(0)}% conformal
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

interface MarketReactionPanelProps {
  panel: MarketReactionPanelResponse;
}

export function MarketReactionPanel({ panel }: MarketReactionPanelProps) {
  const hasContent = panel.rates.length > 0 || panel.vol_regime != null;
  if (!hasContent) {
    return null;
  }
  return (
    <div className="space-y-2">
      <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
        Market reaction panel
      </Badge>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {panel.rates.map((card) => (
          <RatesReactionCard key={card.head} card={card} />
        ))}
        {panel.vol_regime ? <VolRegimeCard card={panel.vol_regime} /> : null}
      </div>
    </div>
  );
}
