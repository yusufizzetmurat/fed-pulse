import * as React from "react";
import { ArrowDownRight, ArrowRight, ArrowUpRight } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import { Sparkline, type SparklineTone } from "@/components/ui/sparkline";
import { cn } from "@/lib/utils";

export type KpiTone = "up" | "down" | "neutral" | "warn";

interface KpiTileProps {
  label: string;
  value: React.ReactNode;
  delta?: number | null;
  deltaFormatter?: (value: number) => string;
  caption?: React.ReactNode;
  tone?: KpiTone;
  sparkline?: Array<number | null | undefined>;
  sparklineTone?: SparklineTone;
  icon?: React.ReactNode;
  className?: string;
  emphasis?: "default" | "large";
}

const TONE_TEXT: Record<KpiTone, string> = {
  up: "text-up",
  down: "text-down",
  neutral: "text-muted-foreground",
  warn: "text-hawkish",
};

function defaultDeltaFormatter(value: number): string {
  // toFixed already emits the leading "-" for negatives; only positives
  // need an explicit "+" so the chip shows direction at a glance.
  const text = value.toFixed(2);
  return value > 0 ? `+${text}` : text;
}

export function KpiTile({
  label,
  value,
  delta,
  deltaFormatter = defaultDeltaFormatter,
  caption,
  tone,
  sparkline,
  sparklineTone,
  icon,
  className,
  emphasis = "default",
}: KpiTileProps) {
  const inferredTone: KpiTone =
    tone ?? (delta == null || delta === 0 ? "neutral" : delta > 0 ? "up" : "down");
  const DeltaIcon =
    delta == null || delta === 0 ? ArrowRight : delta > 0 ? ArrowUpRight : ArrowDownRight;
  return (
    <Card className={cn("h-full", className)}>
      <CardContent className="space-y-2 p-4">
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {label}
          </p>
          {icon ? <span className="text-muted-foreground">{icon}</span> : null}
        </div>
        <div className="flex items-baseline justify-between gap-2">
          <div
            className={cn(
              "numeric font-semibold",
              emphasis === "large" ? "text-4xl" : "text-2xl",
            )}
          >
            {value}
          </div>
          {delta != null && Number.isFinite(delta) ? (
            <div
              className={cn(
                "numeric flex items-center gap-0.5 text-sm",
                TONE_TEXT[inferredTone],
              )}
            >
              <DeltaIcon className="h-3.5 w-3.5" aria-hidden="true" />
              {deltaFormatter(delta)}
            </div>
          ) : null}
        </div>
        {sparkline?.length ? (
          <Sparkline
            values={sparkline}
            tone={sparklineTone ?? (inferredTone === "warn" ? "neutral" : inferredTone)}
            height={32}
          />
        ) : null}
        {caption ? <p className="text-xs text-muted-foreground">{caption}</p> : null}
      </CardContent>
    </Card>
  );
}
