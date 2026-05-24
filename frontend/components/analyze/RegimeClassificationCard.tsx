import { ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import type { RegimeClassificationResponse } from "@/lib/analyze/types";

interface RegimeClassificationCardProps {
  regime: RegimeClassificationResponse;
}

const REGIME_ORDER = ["calm", "normal", "high"];

function regimeIndicatorClass(label: string): string {
  switch (label) {
    case "calm":
      return "bg-dovish";
    case "high":
      return "bg-hawkish";
    case "normal":
    default:
      return "bg-neutral";
  }
}

export function RegimeClassificationCard({ regime }: RegimeClassificationCardProps) {
  const distribution = regime.distribution ?? {};
  const coveragePct = Math.round(regime.coverage * 100);
  const known = new Set(REGIME_ORDER);
  const extraLabels = Object.keys(distribution).filter((k) => !known.has(k));
  const renderOrder = [...REGIME_ORDER, ...extraLabels];
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <ShieldCheck className="h-3.5 w-3.5" />
          Regime prediction set
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl">
          <span className="font-mono">{regime.set_label}</span>
          <Badge variant="outline">{coveragePct}% coverage</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-1.5">
          {regime.predicted_set.map((label) => (
            <Badge key={label} variant="default" className="capitalize">
              {label}
            </Badge>
          ))}
        </div>
        <div className="space-y-2">
          {renderOrder.map((key) => {
            const value = distribution[key];
            if (value === undefined) return null;
            const inSet = regime.predicted_set.includes(key);
            return (
              <div key={key} className="space-y-1">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span className="capitalize">
                    {key} {inSet ? "✓" : ""}
                  </span>
                  <span className={inSet ? "font-medium text-foreground" : "text-muted-foreground"}>
                    {value.toFixed(2)}
                  </span>
                </div>
                <Progress value={value} indicatorClassName={regimeIndicatorClass(key)} />
              </div>
            );
          })}
        </div>
        <p className="text-xs text-muted-foreground">
          Split-conformal prediction set at nominal {coveragePct}% coverage.
          Argmax: <span className="font-medium text-foreground capitalize">{regime.argmax_class}</span> · set size {regime.set_size}.
        </p>
      </CardContent>
    </Card>
  );
}
