import { ArrowDownRight, ArrowUpRight, Gavel, Minus } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type {
  BalanceSheetState,
  PolicyActionResponse,
  PolicyChangeDirection,
} from "@/lib/analyze/types";

// #446: card surfaces the mechanical policy decision extracted from
// the statement text (target range, change indicator, balance-sheet
// posture). Pure render — no inference, no calibration. Mirrors the
// visual language of `RegimeClassificationCard` / `MarketReactionPanel`
// so the workspace page reads as one panel surface.

interface PolicyActionCardProps {
  action: PolicyActionResponse;
}

const BALANCE_SHEET_COPY: Record<BalanceSheetState, string> = {
  expansion: "Balance sheet: expansion (asset purchases)",
  tapering: "Balance sheet: tapering (slowing runoff pace)",
  runoff: "Balance sheet: runoff (continuing to reduce holdings)",
};

function directionTone(
  direction: PolicyChangeDirection,
): "hawkish" | "dovish" | "neutral" {
  if (direction === "hike") return "hawkish";
  if (direction === "cut") return "dovish";
  return "neutral";
}

function DirectionIcon({ direction }: { direction: PolicyChangeDirection }) {
  if (direction === "hike") return <ArrowUpRight className="h-4 w-4 text-hawkish" />;
  if (direction === "cut") return <ArrowDownRight className="h-4 w-4 text-dovish" />;
  return <Minus className="h-4 w-4 text-neutral" />;
}

function formatTargetRange(lowBp: number, highBp: number): string {
  // bps -> percent with two decimals, dropping trailing zeros on whole-
  // quarter boundaries so "5-1/4 to 5-1/2 percent" reads as "5.25% –
  // 5.50%" rather than "5.2500% – 5.5000%".
  const lowPct = (lowBp / 100).toFixed(2);
  const highPct = (highBp / 100).toFixed(2);
  return `${lowPct}% – ${highPct}%`;
}

function formatChangeMagnitude(direction: PolicyChangeDirection, magnitudeBp: number): string {
  if (direction === "hold") return "hold";
  const sign = magnitudeBp > 0 ? "+" : magnitudeBp < 0 ? "" : "";
  return `${sign}${magnitudeBp} bp`;
}

export function PolicyActionCard({ action }: PolicyActionCardProps) {
  const hasRange =
    action.target_range_low_bp != null && action.target_range_high_bp != null;
  const hasDirection = action.change_direction != null;
  // The card stays mute when extraction surfaced nothing — keeps the
  // workspace from rendering an empty "Policy action" tile on non-
  // policy text.
  if (!hasRange && !hasDirection && action.balance_sheet_state == null) {
    return null;
  }
  const direction = action.change_direction;
  const tone = direction ? directionTone(direction) : "neutral";
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <Gavel className="h-3.5 w-3.5" />
          Policy action
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-2xl">
          <span className="numeric">
            {hasRange
              ? formatTargetRange(
                  action.target_range_low_bp!,
                  action.target_range_high_bp!,
                )
              : "—"}
          </span>
          {direction != null && action.change_magnitude_bp != null ? (
            <Badge variant={tone} className="flex items-center gap-1">
              <DirectionIcon direction={direction} />
              {formatChangeMagnitude(direction, action.change_magnitude_bp)}
            </Badge>
          ) : direction != null ? (
            <Badge variant={tone} className="flex items-center gap-1 capitalize">
              <DirectionIcon direction={direction} />
              {direction}
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
              No action verb detected
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="text-xs text-muted-foreground">
          Target range for the federal funds rate, extracted from the statement text.
        </p>
        {action.balance_sheet_state != null ? (
          <p className="text-xs text-muted-foreground">
            {BALANCE_SHEET_COPY[action.balance_sheet_state]}
          </p>
        ) : (
          <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
            Balance sheet stance not detected in this statement.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
