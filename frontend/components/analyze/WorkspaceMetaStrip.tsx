import * as React from "react";

import { Badge } from "@/components/ui/badge";
import type { AnalyzeResult } from "@/lib/analyze/types";

interface WorkspaceMetaStripProps {
  result: AnalyzeResult;
}

/**
 * Subtle row of small badges that surface meeting-level metadata
 * discovered on the analyze response. Each badge hides silently when
 * its backing field is null/missing so the strip degrades cleanly on
 * older checkpoints.
 */
export function WorkspaceMetaStrip({ result }: WorkspaceMetaStripProps) {
  const votesFor = result.votes_for ?? null;
  const votesAgainst = result.votes_against ?? null;
  const dissent = result.dissent_direction ?? null;
  const hasPressConf = result.has_press_conf === 1;
  const showVote = votesFor != null && votesAgainst != null;
  const showPress = hasPressConf;
  if (!showVote && !showPress) return null;

  const dissentSuffix = (() => {
    if (!showVote) return "";
    if ((votesAgainst as number) <= 0) return "(unanimous)";
    const count = votesAgainst as number;
    const direction = dissent ? `${dissent} dissent` : "dissent";
    return `(${count} ${direction}${count > 1 ? "s" : ""})`;
  })();

  return (
    <div className="flex flex-wrap items-center gap-2">
      {showVote ? (
        <Badge variant="outline" className="text-[10px]">
          Vote: {votesFor}&ndash;{votesAgainst} {dissentSuffix}
        </Badge>
      ) : null}
      {showPress ? (
        <Badge variant="outline" className="text-[10px]">
          Press conf Q&amp;A included
        </Badge>
      ) : null}
    </div>
  );
}
