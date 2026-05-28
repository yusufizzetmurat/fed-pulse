import * as React from "react";
import { ChevronDown, ChevronUp, History, Sparkles } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { EmptyState } from "@/components/ui/empty-state";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type {
  AnalogCard,
  AnalogVolRegime,
  AnalogsResponse,
} from "@/lib/analyze/types";
import { EvidenceLink } from "@/components/analyze/EvidenceLink";

const VOL_REGIME_ORDER: AnalogVolRegime[] = ["calm", "normal", "high"];

const VOL_REGIME_LABEL: Record<AnalogVolRegime, string> = {
  calm: "Calm",
  normal: "Normal",
  high: "High",
};

// Segment colour for the "what happened next" mini-chart. The backend
// only exposes a coarse calm/normal/high bucket — surfacing the raw
// forward_realized_vol_10d would leak the supervised target, so the
// indicator visualises the bucket itself rather than a price series.
const VOL_REGIME_BAR: Record<AnalogVolRegime, string> = {
  calm: "bg-dovish",
  normal: "bg-neutral",
  high: "bg-hawkish",
};

const STANCE_BADGE: Record<string, "hawkish" | "dovish" | "neutral"> = {
  hawkish: "hawkish",
  dovish: "dovish",
  neutral: "neutral",
};

function stanceTone(value: string | null | undefined): "hawkish" | "dovish" | "neutral" | null {
  if (!value) return null;
  const key = value.toLowerCase();
  return STANCE_BADGE[key] ?? null;
}

function formatSimilarity(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

interface WhatHappenedNextProps {
  bucket: AnalogVolRegime | null;
}

function WhatHappenedNext({ bucket }: WhatHappenedNextProps) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
        <span>What happened next · 10-day volatility</span>
        <span className="numeric capitalize text-foreground">
          {bucket ? VOL_REGIME_LABEL[bucket] : "—"}
        </span>
      </div>
      <div
        className="flex items-stretch gap-1"
        role="img"
        aria-label={
          bucket
            ? `10-day realised volatility after the event: ${VOL_REGIME_LABEL[bucket]}`
            : "Post-event volatility unavailable"
        }
      >
        {VOL_REGIME_ORDER.map((segment) => {
          const active = segment === bucket;
          return (
            <div
              key={segment}
              className={cn(
                "h-2 flex-1 rounded-sm",
                active ? VOL_REGIME_BAR[segment] : "bg-muted",
              )}
            />
          );
        })}
      </div>
    </div>
  );
}

interface AnalogCardItemProps {
  card: AnalogCard;
}

function AnalogCardItem({ card }: AnalogCardItemProps) {
  const [expanded, setExpanded] = React.useState(false);
  const stance = stanceTone(card.axis_stance);
  // The /analyze/analogs endpoint truncates to ~280 chars on the
  // server; treat that ceiling as the "looks truncated" trigger so the
  // expand affordance never appears on a short statement that's
  // already fully rendered.
  const looksTruncated = card.excerpt.length >= 280;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardDescription className="flex items-center gap-1.5">
          <History className="h-3.5 w-3.5" />
          {card.event_date}
        </CardDescription>
        <CardTitle className="flex items-center justify-between text-base">
          <span className="numeric">{formatSimilarity(card.similarity)} similar</span>
          {stance ? (
            <Badge variant={stance} className="capitalize">
              {card.axis_stance}
            </Badge>
          ) : (
            <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
              stance unknown
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <WhatHappenedNext bucket={card.subsequent_vol_regime} />
        <div className="space-y-1">
          <p
            className={cn(
              "text-xs leading-relaxed text-foreground/80",
              expanded ? "" : "line-clamp-4",
            )}
          >
            {card.excerpt}
          </p>
          {looksTruncated ? (
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="inline-flex items-center gap-1 text-[11px] font-medium text-muted-foreground hover:text-foreground focus:outline-none focus:ring-1 focus:ring-ring rounded-sm"
              aria-expanded={expanded}
            >
              {expanded ? (
                <>
                  <ChevronUp className="h-3 w-3" /> Collapse
                </>
              ) : (
                <>
                  <ChevronDown className="h-3 w-3" /> Show full excerpt
                </>
              )}
            </button>
          ) : null}
        </div>
      </CardContent>
    </Card>
  );
}

export interface HistoricalAnalogPanelProps {
  analogs: AnalogsResponse | null;
  loading?: boolean;
  // Cosine similarity floor for what counts as a useful analog. Cards
  // below the floor are filtered client-side and trigger the
  // "no analogs above threshold" empty state when the entire top-k
  // falls under the bar. The default mirrors the threshold used by
  // the retrieval evaluation harness for "loosely related" pairs.
  similarityThreshold?: number;
  // How many analog cards to render. The endpoint defaults to k=5;
  // the panel surfaces the top 3 per the §16 spec.
  topK?: number;
}

const DEFAULT_THRESHOLD = 0.4;
const DEFAULT_TOP_K = 3;

export function HistoricalAnalogPanel({
  analogs,
  loading = false,
  similarityThreshold = DEFAULT_THRESHOLD,
  topK = DEFAULT_TOP_K,
}: HistoricalAnalogPanelProps) {
  const headerBadge = (
    <div className="flex flex-wrap items-center gap-2">
      <Badge variant="outline" className="text-[10px] uppercase tracking-wide">
        Historical analogs
      </Badge>
      <EvidenceLink section="6.16" label="Method notes · retrieval quality" />
    </div>
  );

  if (loading) {
    return (
      <div className="space-y-2">
        {headerBadge}
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: topK }).map((_, idx) => (
            <Card key={idx}>
              <CardHeader className="pb-2">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="mt-2 h-5 w-32" />
              </CardHeader>
              <CardContent className="space-y-3">
                <Skeleton className="h-2 w-full" />
                <Skeleton className="h-12 w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (!analogs) {
    return null;
  }

  if (analogs.index_size === 0) {
    return (
      <div className="space-y-2">
        {headerBadge}
        <EmptyState
          variant="card"
          icon={<History className="h-5 w-5" />}
          title="Analog index not loaded"
          description={
            <span>
              The historical analog index is not available. Train and load the retrieval
              model against the training corpus, then refresh.
            </span>
          }
        />
      </div>
    );
  }

  const filtered = analogs.analogs
    .filter((card) => card.similarity >= similarityThreshold)
    .slice(0, topK);

  if (filtered.length === 0) {
    return (
      <div className="space-y-2">
        {headerBadge}
        <EmptyState
          variant="card"
          icon={<Sparkles className="h-5 w-5" />}
          title="No close analogs found"
          description={
            <span>
              The retrieval model scored {analogs.analogs.length} candidate
              {analogs.analogs.length === 1 ? "" : "s"} but none crossed the{" "}
              <span className="numeric">{formatSimilarity(similarityThreshold)}</span>{" "}
              similarity threshold for this statement.
            </span>
          }
        />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {headerBadge}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {filtered.map((card, idx) => (
          // event_date alone is not guaranteed unique — the retrieval
          // index dedupes by text_hash, so the same date can carry an
          // intermeeting statement and a correction. Composite key on
          // (event_date, similarity, idx) keeps React reconciliation
          // honest so the per-card expand state cannot bleed.
          <AnalogCardItem
            key={`${card.event_date}-${card.similarity.toFixed(6)}-${idx}`}
            card={card}
          />
        ))}
      </div>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        Past FOMC statements most similar to the current text, ranked by a
        retrieval model. The post-event volatility marker shows a coarse
        calm / normal / high band only — raw values are hidden so the
        analog does not give away the answer. Index size:{" "}
        <span className="numeric">{analogs.index_size.toLocaleString()}</span>
        {" "}past statements · model variant{" "}
        <code className="rounded bg-muted px-1">{analogs.encoder_alias}</code>.
      </p>
    </div>
  );
}
