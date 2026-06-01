import * as React from "react";

import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";
import { cn } from "@/lib/utils";
import type {
  SemanticDiffResponse,
  SemanticDiffSpan,
  SemanticDiffTopic,
} from "@/lib/analyze/types";

// Descriptive workspace panel that surfaces the realized text change
// between the pasted statement and its strict-prior FOMC statement.
// Two sub-views ride inside one ``WorkspaceSection variant="descriptive"``:
//
//   - Wording changes: token-level redline with added (emerald),
//     removed (red strikethrough), substituted (paired chips), and
//     unchanged runs collapsed past ``UNCHANGED_RUN_KEEP_TOKENS_FRONT``.
//   - Emphasis shifts: six-topic emphasis bars rendering current vs
//     prior with a signed delta chip per row.
//
// The panel is intentionally never wired into the forecast cards —
// the SPINE contract for ``descriptive`` variants is that they
// explain the realized text/realized-rate signal and do not feed any
// quantitative forecast head.

// Maximum number of words to keep in an unchanged run before
// collapsing to a "…" placeholder. The backend already truncates
// very long runs; this is the frontend-side belt-and-braces that
// kicks in for the typical paragraph-length equal stretches.
export const UNCHANGED_RUN_KEEP_TOKENS_FRONT = 25;

function formatEventDate(iso: string): string {
  const parsed = new Date(`${iso}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    timeZone: "UTC",
  });
}

function formatPercentShare(value: number): string {
  // Topic emphasis shares come off the backend in [0, 1]. Render as
  // an integer percent so the bars line up with the visual width.
  return `${Math.round(value * 100)}%`;
}

function formatDelta(value: number): string {
  const pct = value * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(1)} pp`;
}

function truncateUnchanged(text: string): string {
  const tokens = text.split(/\s+/).filter(Boolean);
  if (tokens.length <= UNCHANGED_RUN_KEEP_TOKENS_FRONT) {
    return tokens.join(" ");
  }
  const halfHead = Math.floor(UNCHANGED_RUN_KEEP_TOKENS_FRONT / 2);
  const halfTail = UNCHANGED_RUN_KEEP_TOKENS_FRONT - halfHead;
  return [
    tokens.slice(0, halfHead).join(" "),
    "…",
    tokens.slice(-halfTail).join(" "),
  ].join(" ");
}

function RedlineSpan({ span }: { span: SemanticDiffSpan }) {
  if (span.kind === "unchanged") {
    return (
      <span
        className="text-muted-foreground"
        data-testid="semantic-diff-span-unchanged"
      >
        {truncateUnchanged(span.text)}{" "}
      </span>
    );
  }
  if (span.kind === "added") {
    return (
      <span
        className="rounded bg-emerald-500/15 px-1 py-0.5 text-emerald-700 dark:text-emerald-300"
        data-testid="semantic-diff-span-added"
      >
        {span.text}{" "}
      </span>
    );
  }
  if (span.kind === "removed") {
    return (
      <span
        className="rounded bg-red-500/15 px-1 py-0.5 text-red-700 line-through dark:text-red-300"
        data-testid="semantic-diff-span-removed"
      >
        {span.text}{" "}
      </span>
    );
  }
  // substituted — paired chip: prior strikethrough, new emphasised.
  return (
    <span
      className="inline-flex items-center gap-1 rounded border border-dashed border-border bg-muted/40 px-1.5 py-0.5"
      data-testid="semantic-diff-span-substituted"
    >
      {span.paired_text ? (
        <span className="text-red-700 line-through dark:text-red-300">
          {span.paired_text}
        </span>
      ) : null}
      <span className="text-emerald-700 dark:text-emerald-300">
        {span.text}
      </span>
    </span>
  );
}

function TopicRow({ topic }: { topic: SemanticDiffTopic }) {
  const currentPct = Math.round(topic.current_emphasis * 100);
  const priorPct = Math.round(topic.prior_emphasis * 100);
  const deltaPositive = topic.delta > 0;
  const deltaNegative = topic.delta < 0;
  return (
    <li
      className="space-y-1 rounded-md border border-border/60 bg-background/60 p-2"
      data-testid="semantic-diff-topic-row"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-medium">{topic.topic}</span>
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
            deltaPositive && "bg-emerald-500/15 text-emerald-700 dark:text-emerald-300",
            deltaNegative && "bg-red-500/15 text-red-700 dark:text-red-300",
            !deltaPositive && !deltaNegative && "bg-muted text-muted-foreground",
          )}
          data-testid="semantic-diff-topic-delta"
        >
          {formatDelta(topic.delta)}
        </span>
      </div>
      <div
        className="grid grid-cols-[auto_1fr_auto] items-center gap-2 text-[11px] text-muted-foreground"
        aria-label={`${topic.topic} current emphasis`}
      >
        <span className="w-12">Current</span>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-primary"
            style={{ width: `${currentPct}%` }}
          />
        </div>
        <span className="numeric tabular-nums">
          {formatPercentShare(topic.current_emphasis)}
        </span>
      </div>
      <div
        className="grid grid-cols-[auto_1fr_auto] items-center gap-2 text-[11px] text-muted-foreground"
        aria-label={`${topic.topic} prior emphasis`}
      >
        <span className="w-12">Prior</span>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
          <div
            className="h-full bg-muted-foreground/60"
            style={{ width: `${priorPct}%` }}
          />
        </div>
        <span className="numeric tabular-nums">
          {formatPercentShare(topic.prior_emphasis)}
        </span>
      </div>
      {topic.sample_phrases.length > 0 ? (
        <div className="flex flex-wrap gap-1 pt-1">
          {topic.sample_phrases.map((phrase) => (
            <span
              key={phrase}
              className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
            >
              {phrase}
            </span>
          ))}
        </div>
      ) : null}
    </li>
  );
}

function ColdStartBanner({ summary }: { summary: string }) {
  return (
    <div
      className="rounded-md border border-dashed border-border bg-muted/30 p-3 text-xs text-muted-foreground"
      data-testid="semantic-diff-cold-start"
    >
      <p className="font-medium text-foreground">
        Earliest statement in dataset
      </p>
      <p className="mt-1">{summary}</p>
    </div>
  );
}

// Informational banner for the silent-null edge cases the backend
// surfaces via ``SemanticDiffResponse.status``. We never blank the
// panel on an edge case — the user gets a parseable reason instead.
function StatusBanner({
  title,
  summary,
  testId,
}: {
  title: string;
  summary: string;
  testId: string;
}) {
  return (
    <div
      className="rounded-md border border-dashed border-border bg-muted/30 p-3 text-xs text-muted-foreground"
      data-testid={testId}
    >
      <p className="font-medium text-foreground">{title}</p>
      <p className="mt-1">{summary}</p>
    </div>
  );
}

export interface SemanticDiffPanelProps {
  data: SemanticDiffResponse | null;
  loading?: boolean;
}

export function SemanticDiffPanel({
  data,
  loading = false,
}: SemanticDiffPanelProps) {
  if (loading) {
    return (
      <WorkspaceSection
        title="Semantic diff vs prior statement"
        description="Token redline + topic emphasis shifts (descriptive)"
        variant="descriptive"
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="semantic-diff-loading"
        >
          Loading prior-statement comparison…
        </p>
      </WorkspaceSection>
    );
  }

  if (!data) {
    return (
      <WorkspaceSection
        title="Semantic diff vs prior statement"
        description="Token redline + topic emphasis shifts (descriptive)"
        variant="descriptive"
      >
        <p
          className="text-xs text-muted-foreground"
          data-testid="semantic-diff-unavailable"
        >
          Semantic-diff feed unavailable. The comparison will appear here once
          the strict-prior statement is reachable.
        </p>
      </WorkspaceSection>
    );
  }

  // Surface the silent-null edge cases the backend signalled via
  // ``status``. ``no_prior`` keeps the existing cold-start banner so
  // older clients (status undefined + empty lists) still match.
  if (data.status === "no_input") {
    return (
      <WorkspaceSection
        title="Semantic diff vs prior statement"
        description="Token redline + topic emphasis shifts (descriptive)"
        variant="descriptive"
      >
        <StatusBanner
          title="Input too short to diff"
          summary={data.summary}
          testId="semantic-diff-no-input"
        />
      </WorkspaceSection>
    );
  }
  if (data.status === "non_english") {
    return (
      <WorkspaceSection
        title="Semantic diff vs prior statement"
        description="Token redline + topic emphasis shifts (descriptive)"
        variant="descriptive"
      >
        <StatusBanner
          title="Non-Latin text — diff not run"
          summary={data.summary}
          testId="semantic-diff-non-english"
        />
      </WorkspaceSection>
    );
  }

  const isColdStart =
    data.status === "no_prior" ||
    !data.prior_date ||
    (data.token_spans.length === 0 && data.topic_deltas.length === 0);

  const description = isColdStart
    ? "Token redline + topic emphasis shifts (descriptive)"
    : `Versus the ${formatEventDate(data.prior_date)} statement (descriptive)`;

  return (
    <WorkspaceSection
      title="Semantic diff vs prior statement"
      description={description}
      variant="descriptive"
    >
      {isColdStart ? (
        <ColdStartBanner summary={data.summary} />
      ) : (
        <div className="space-y-4">
          <section data-testid="semantic-diff-wording-section">
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Wording changes
            </h4>
            <p className="text-sm leading-relaxed">
              {data.token_spans.map((span, index) => (
                <RedlineSpan key={`${span.kind}-${index}`} span={span} />
              ))}
            </p>
          </section>
          <section data-testid="semantic-diff-emphasis-section">
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Emphasis shifts
            </h4>
            <ul className="space-y-2">
              {data.topic_deltas.map((topic) => (
                <TopicRow key={topic.topic} topic={topic} />
              ))}
            </ul>
            <p
              className="mt-3 text-[11px] text-muted-foreground"
              data-testid="semantic-diff-summary"
            >
              {data.summary}
            </p>
          </section>
        </div>
      )}
    </WorkspaceSection>
  );
}

export default SemanticDiffPanel;
