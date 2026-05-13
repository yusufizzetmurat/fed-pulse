import { ArrowDownRight, ArrowUpRight, Compass, Gauge, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { stanceLabel } from "@/lib/analyze/format";
import type {
  MultiAxisResponse,
  MultiAxisStance,
  StanceAxis,
} from "@/lib/analyze/types";

interface MultiAxisCardsProps {
  multiAxis: MultiAxisResponse;
  previewMode?: boolean;
}

function stanceVariant(stance: StanceAxis): "hawkish" | "dovish" | "neutral" {
  return stance;
}

function StanceCard({ stance }: { stance: MultiAxisStance }) {
  const distribution = stance.distribution || {};
  const order: StanceAxis[] = ["hawkish", "neutral", "dovish"];
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <Compass className="h-3.5 w-3.5" />
          Stance
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl">
          <span>{stanceLabel(stance.label)}</span>
          <Badge variant={stanceVariant(stance.label)}>{stance.confidence.toFixed(2)}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {order.map((key) => {
          const value = distribution[key] ?? 0;
          return (
            <div key={key} className="space-y-1">
              <div className="flex items-center justify-between text-xs text-muted-foreground">
                <span className="capitalize">{key}</span>
                <span className="font-medium text-foreground">{value.toFixed(2)}</span>
              </div>
              <Progress
                value={value}
                indicatorClassName={
                  key === "hawkish"
                    ? "bg-hawkish"
                    : key === "dovish"
                    ? "bg-dovish"
                    : "bg-neutral"
                }
              />
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function FactorCard({ factor }: { factor: NonNullable<MultiAxisResponse["factor"]> }) {
  const value = factor.value;
  const tone: "hawkish" | "dovish" | "neutral" =
    value > 0.05 ? "hawkish" : value < -0.05 ? "dovish" : "neutral";
  const Arrow = value > 0 ? ArrowUpRight : value < 0 ? ArrowDownRight : Gauge;
  const arrowClass =
    tone === "hawkish" ? "text-hawkish" : tone === "dovish" ? "text-dovish" : "text-neutral";
  const range = factor.range;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <Gauge className="h-3.5 w-3.5" />
          Factor
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl">
          <span className="flex items-center gap-1">
            <Arrow className={`h-5 w-5 ${arrowClass}`} />
            {value >= 0 ? "+" : ""}
            {value.toFixed(2)}
          </span>
          <Badge variant={tone}>±{(factor.confidence ?? 0).toFixed(2)}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-1.5 text-xs text-muted-foreground">
        <p>Signed magnitude on the hawkish (+) ↔ dovish (−) axis.</p>
        {range ? (
          <p>
            80% range: <span className="font-mono">{range[0].toFixed(2)} … {range[1].toFixed(2)}</span>
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}

function CertaintyCard({ certainty }: { certainty: NonNullable<MultiAxisResponse["certainty"]> }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <Target className="h-3.5 w-3.5" />
          Certainty
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl">
          <span className="capitalize">{certainty.label}</span>
          <Badge variant="outline">{certainty.confidence.toFixed(2)}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <Progress value={certainty.confidence} />
        <p className="text-xs text-muted-foreground">
          How firmly the language commits to the stance.
        </p>
      </CardContent>
    </Card>
  );
}

function TopicCard({ topic }: { topic: NonNullable<MultiAxisResponse["topic"]> }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Topic</CardDescription>
        <CardTitle className="text-xl capitalize">
          {topic.primary.replace(/_/g, " ")}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Primary confidence</span>
          <span className="font-medium text-foreground">{topic.confidence.toFixed(2)}</span>
        </div>
        <Progress value={topic.confidence} />
        {topic.secondary?.length ? (
          <div className="flex flex-wrap gap-1.5 pt-1">
            {topic.secondary.map((t) => (
              <Badge key={t} variant="outline" className="capitalize">
                {t.replace(/_/g, " ")}
              </Badge>
            ))}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

export function MultiAxisCards({ multiAxis, previewMode }: MultiAxisCardsProps) {
  if (!multiAxis.stance && !multiAxis.factor && !multiAxis.certainty && !multiAxis.topic) {
    return null;
  }
  return (
    <div className="space-y-2">
      {previewMode ? (
        <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
          Multi-axis preview · fixture data
        </Badge>
      ) : null}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {multiAxis.stance ? <StanceCard stance={multiAxis.stance} /> : null}
        {multiAxis.factor ? <FactorCard factor={multiAxis.factor} /> : null}
        {multiAxis.certainty ? <CertaintyCard certainty={multiAxis.certainty} /> : null}
        {multiAxis.topic ? <TopicCard topic={multiAxis.topic} /> : null}
      </div>
    </div>
  );
}
