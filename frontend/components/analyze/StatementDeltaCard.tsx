import * as React from "react";
import { ChevronDown, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { AnalyzeResult } from "@/lib/analyze/types";

interface StatementDeltaCardProps {
  result: AnalyzeResult;
}

/**
 * Inline redline of the current statement vs the previous one. Hidden
 * when the strict-prior diff isn't available on the response. The
 * card is collapsed by default so it doesn't dominate the workspace
 * on every analysis.
 */
export function StatementDeltaCard({ result }: StatementDeltaCardProps) {
  const inserted = result.statement_delta_inserted ?? [];
  const deleted = result.statement_delta_deleted ?? [];
  const [open, setOpen] = React.useState(false);
  if (inserted.length === 0 && deleted.length === 0) return null;
  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle className="text-base">What changed since last statement</CardTitle>
            <CardDescription>
              Inline diff against the previous FOMC statement. Insertions in green; deletions
              struck through.
            </CardDescription>
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-7 px-2"
            onClick={() => setOpen((value) => !value)}
            aria-expanded={open}
            aria-label={open ? "Collapse redline" : "Expand redline"}
          >
            {open ? (
              <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5" aria-hidden="true" />
            )}
            {open ? "Hide" : "Show"}
          </Button>
        </div>
      </CardHeader>
      {open ? (
        <CardContent className="space-y-3 text-sm leading-relaxed">
          {deleted.length > 0 ? (
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Removed</p>
              <p>
                {deleted.map((span, idx) => (
                  <span
                    key={`del-${idx}`}
                    className="mr-1 rounded bg-hawkish/10 px-1 text-hawkish line-through decoration-hawkish"
                  >
                    {span.text}
                  </span>
                ))}
              </p>
            </div>
          ) : null}
          {inserted.length > 0 ? (
            <div className="space-y-1">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">Added</p>
              <p>
                {inserted.map((span, idx) => (
                  <span
                    key={`ins-${idx}`}
                    className="mr-1 rounded bg-dovish/10 px-1 text-dovish"
                  >
                    {span.text}
                  </span>
                ))}
              </p>
            </div>
          ) : null}
        </CardContent>
      ) : null}
    </Card>
  );
}
