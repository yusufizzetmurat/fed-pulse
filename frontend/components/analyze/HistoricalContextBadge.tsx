import * as React from "react";

import { Badge } from "@/components/ui/badge";
import { fetchHistory, fetchHistoryRun, resolveApiBaseUrl } from "@/lib/analyze/api";
import type {
  AnalyzeResult,
  HistoryDetail,
  PolicyChangeDirection,
} from "@/lib/analyze/types";

interface HistoricalContextBadgeProps {
  result: AnalyzeResult | null;
  documentDate: string;
}

function ordinal(n: number): string {
  const v = n % 100;
  if (v >= 11 && v <= 13) return `${n}th`;
  const last = n % 10;
  if (last === 1) return `${n}st`;
  if (last === 2) return `${n}nd`;
  if (last === 3) return `${n}rd`;
  return `${n}th`;
}

function directionWord(direction: PolicyChangeDirection): string {
  if (direction === "hike") return "hike";
  if (direction === "cut") return "cut";
  return "hold";
}

/**
 * Subtle "Nth consecutive hold" / "N meetings since last cut" chip
 * that sits near the workspace title. Computes the streak from the
 * current result's policy_action and the policy_action persisted on
 * recent history runs. Renders nothing when the data isn't sufficient.
 */
export function HistoricalContextBadge({ result, documentDate }: HistoricalContextBadgeProps) {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [history, setHistory] = React.useState<HistoryDetail[] | null>(null);

  const currentDirection = result?.policy_action?.change_direction ?? null;

  React.useEffect(() => {
    if (!currentDirection) return;
    const controller = new AbortController();
    const { signal } = controller;
    (async () => {
      try {
        const list = await fetchHistory(apiBaseUrl, { limit: 24 }, signal);
        if (signal.aborted) return;
        const details = await Promise.all(
          list.items
            .filter((entry) => entry.document_date < documentDate)
            .slice(0, 16)
            .map((entry) => fetchHistoryRun(apiBaseUrl, entry.id, signal).catch(() => null)),
        );
        if (signal.aborted) return;
        setHistory(details.filter((d): d is HistoryDetail => d != null));
      } catch {
        // Best-effort badge; silent failure.
      }
    })();
    return () => controller.abort();
  }, [apiBaseUrl, currentDirection, documentDate]);

  if (!currentDirection) return null;

  // Group history runs by document_date and take the latest run per date
  // so a re-analysed meeting doesn't double-count.
  const byDate = new Map<string, PolicyChangeDirection | null>();
  if (history) {
    const sorted = [...history].sort((a, b) =>
      a.document_date < b.document_date ? 1 : -1,
    );
    for (const run of sorted) {
      if (byDate.has(run.document_date)) continue;
      const payload = (run.payload || {}) as AnalyzeResult;
      const dir = payload?.policy_action?.change_direction ?? null;
      byDate.set(run.document_date, dir);
    }
  }
  const orderedPriorDirections = [...byDate.entries()]
    .sort((a, b) => (a[0] < b[0] ? 1 : -1))
    .map(([, dir]) => dir);

  // "Nth consecutive X" — count how many prior meetings (immediately
  // preceding) shared the current decision.
  let consecutive = 1;
  for (const dir of orderedPriorDirections) {
    if (dir === currentDirection) {
      consecutive += 1;
    } else {
      break;
    }
  }

  if (consecutive >= 2) {
    return (
      <Badge variant="outline" className="text-[10px]">
        {ordinal(consecutive)} consecutive {directionWord(currentDirection)}
      </Badge>
    );
  }

  // Alternative framing: "N meetings since last cut/hike" when the
  // current decision flipped direction.
  if (currentDirection !== "hold") {
    const target: PolicyChangeDirection = currentDirection === "cut" ? "cut" : "hike";
    let gap = 0;
    for (const dir of orderedPriorDirections) {
      if (dir === target) break;
      gap += 1;
    }
    if (gap > 0 && gap < orderedPriorDirections.length) {
      return (
        <Badge variant="outline" className="text-[10px]">
          {gap} meeting{gap === 1 ? "" : "s"} since last {directionWord(target)}
        </Badge>
      );
    }
  }

  return null;
}
