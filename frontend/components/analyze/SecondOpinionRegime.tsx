import * as React from "react";
import { Info } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { RegimeHeadline } from "@/components/analyze/RegimeHeadline";
import type {
  HarTercileBaselineResponse,
  RegimeClassificationResponse,
  SentimentResponse,
} from "@/lib/analyze/types";

interface SecondOpinionRegimeProps {
  regime: RegimeClassificationResponse;
  sentiment?: SentimentResponse;
  history?: Array<{ documentDate: string; argmax: string | null; realized?: string | null }>;
  symbol?: string;
  documentDate?: string;
  empiricalCoverage?: number | null;
  empiricalCoverageSampleSize?: number | null;
  marketOnlyArgmaxProb?: number | null;
  // HAR baseline for the matching cross-check chip. The 1-day HAR
  // top pick is the comparison anchor — the late-fusion target is a
  // 10-day forward window, so the "disagrees with HAR" chip is only
  // approximate and surfaces a tooltip noting the horizon mismatch.
  harBaselines?: HarTercileBaselineResponse | null;
}

// Map HAR tercile labels to the late-fusion regime labels so the
// argmax comparison reads against the same 3-class vocabulary.
function harLabelToRegime(label: string | undefined): string | null {
  if (label === "low") return "calm";
  if (label === "medium") return "normal";
  if (label === "high") return "high";
  return null;
}

export function SecondOpinionRegime({
  regime,
  sentiment,
  history,
  symbol,
  documentDate,
  empiricalCoverage,
  empiricalCoverageSampleSize,
  marketOnlyArgmaxProb,
  harBaselines,
}: SecondOpinionRegimeProps) {
  const distribution = regime.distribution ?? {};
  const argmaxProb = distribution[regime.argmax_class] ?? 0;

  // Anchor the HAR cross-check on the 1-day horizon — the wiki §20
  // table where the late-fusion underperformance is largest.
  const harOneDay = React.useMemo(() => {
    if (!harBaselines) return null;
    return harBaselines.horizons.find((h) => h.h === 1) ?? null;
  }, [harBaselines]);

  const harRegimeEquivalent = harLabelToRegime(harOneDay?.top_pick);
  const disagreesWithHar =
    harRegimeEquivalent !== null && harRegimeEquivalent !== regime.argmax_class;

  // Low-confidence collapse chip: the production checkpoint's
  // training-mode majority class is "calm". When the late-fusion
  // argmax pins to that class with a weak probability, surface a
  // chip telling the user the model is at its fallback.
  const isCollapsed = regime.argmax_class === "calm" && argmaxProb < 0.65;

  return (
    <section
      aria-label="Second opinion · late-fusion text+market classifier"
      className="space-y-3"
    >
      <div className="flex flex-col gap-2 rounded-md border border-dashed border-border bg-muted/30 p-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
            Second opinion
          </Badge>
          <span className="text-xs text-muted-foreground">
            Late-fusion text+market classifier
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {disagreesWithHar ? (
            <Badge
              variant="hawkish"
              className="text-[10px] uppercase tracking-wide"
              title="HAR 1-day vs late-fusion 10-day are different horizons, so the comparison is approximate."
            >
              Disagrees with HAR
            </Badge>
          ) : null}
          {isCollapsed ? (
            <Badge
              variant="outline"
              className="text-[10px] uppercase tracking-wide"
              title="Model is sitting at its training-mode majority class. Treat as a no-signal fallback."
            >
              Low-confidence collapse
            </Badge>
          ) : null}
          <Badge
            variant="outline"
            className="numeric text-[10px]"
            title="Late-fusion macro-F1 from wiki §20 (1-day forward-RV regime task)."
          >
            macro-F1 0.592 (wiki §20, 1-day)
          </Badge>
        </div>
      </div>
      <div className="flex items-start gap-2 rounded-md border border-border bg-muted/20 p-3 text-xs text-muted-foreground">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden="true" />
        <p>
          Weaker than the HAR baseline above. Per the wiki §20 eval, the text+market
          fusion underperforms HAR-tercile by ~0.10 macro-F1 at 1-day; the text channel
          itself robustly hurts the model (95% CI [-0.022, -0.009] at 1-day). Surfaced
          for transparency.
        </p>
      </div>
      <RegimeHeadline
        regime={regime}
        sentiment={sentiment}
        history={history}
        symbol={symbol}
        documentDate={documentDate}
        empiricalCoverage={empiricalCoverage}
        empiricalCoverageSampleSize={empiricalCoverageSampleSize}
        marketOnlyArgmaxProb={marketOnlyArgmaxProb}
      />
    </section>
  );
}
