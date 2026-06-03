import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type {
  RealisedOutcomeBlock,
  ReplayModeBlock,
} from "@/lib/analyze/types";

interface ReplayBannerProps {
  replay: ReplayModeBlock | null | undefined;
}

/**
 * Sticky "Replay mode — as of YYYY-MM-DD" banner. Renders nothing in
 * live mode (replay is null). Surfaces the fold_id + train_end so the
 * user knows which checkpoint served the prediction, plus the
 * classifier-rewind caveat as a chip.
 */
export function ReplayBanner({ replay }: ReplayBannerProps) {
  if (!replay) return null;
  return (
    <div
      role="status"
      className="rounded-md border border-amber-500/40 bg-amber-50 px-4 py-3 text-sm text-amber-900 shadow-sm dark:border-amber-500/30 dark:bg-amber-950/30 dark:text-amber-100"
    >
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="outline" className="border-amber-500/60 text-amber-800 dark:text-amber-100">
          Replay mode
        </Badge>
        <span className="font-semibold">as of {replay.as_of_date}</span>
        {replay.fold_id ? (
          <span className="text-xs text-amber-800/80 dark:text-amber-200/80">
            · fold <span className="numeric">{replay.fold_id}</span>
            {replay.train_end ? (
              <>
                {" "}· train_end <span className="numeric">{replay.train_end}</span>
              </>
            ) : null}
          </span>
        ) : null}
        {!replay.classifier_rewind ? (
          <Badge
            variant="outline"
            className="border-amber-500/40 text-[10px] uppercase tracking-wide text-amber-700 dark:text-amber-200"
            title={replay.notes.join("\n")}
          >
            classifier weights post-X
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

interface RealisedOutcomeCardProps {
  outcome: RealisedOutcomeBlock | null | undefined;
}

function _formatNumber(value: number | null | undefined, digits = 4): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(digits);
}

/**
 * "What actually happened" reveal. Collapsed by default so the user
 * makes the explicit decision to unmask realised outcomes -- the
 * replay flow is meant to surface the model's read first, then peel
 * the truth. Renders nothing when there is no outcome block (live
 * mode, or replay against a date with no forward data yet).
 */
export function RealisedOutcomeCard({ outcome }: RealisedOutcomeCardProps) {
  const [revealed, setRevealed] = React.useState(false);
  if (!outcome) return null;
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle>What actually happened</CardTitle>
            <CardDescription>
              Realised market path for the {outcome.symbol} symbol on the 1, 5
              and 10 trading days after {outcome.as_of_date}. Click reveal to
              compare against the model's prediction.
            </CardDescription>
          </div>
          <Button
            type="button"
            variant={revealed ? "outline" : "default"}
            size="sm"
            onClick={() => setRevealed((value) => !value)}
            aria-expanded={revealed}
          >
            {revealed ? "Hide" : "Reveal"}
          </Button>
        </div>
      </CardHeader>
      {revealed ? (
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-3" data-testid="realised-outcome-grid">
            {outcome.horizons.map((row) => (
              <div
                key={row.horizon}
                className="rounded-md border border-border/70 bg-muted/30 px-3 py-2"
                data-testid={`realised-outcome-h${row.horizon}`}
              >
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                  +{row.horizon}d
                </p>
                <p className="numeric mt-1 text-sm">
                  log-return: <span className="font-semibold">{_formatNumber(row.log_return)}</span>
                </p>
                <p className="numeric text-sm">
                  vol_5d: <span className="font-semibold">{_formatNumber(row.realised_volatility_5d)}</span>
                </p>
                <p className="text-[10px] text-muted-foreground">
                  bar {row.date ?? "—"}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      ) : null}
    </Card>
  );
}
