import * as React from "react";
import Link from "next/link";
import { CircleDashed, History } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface RegimeHistoryEntry {
  runId: string;
  documentDate: string;
  argmax: string | null;
  realized?: string | null;
}

interface RegimeHistoryStripProps {
  entries: RegimeHistoryEntry[];
  symbol?: string;
}

function regimeColor(label: string | null | undefined): string {
  if (!label) return "bg-muted text-muted-foreground";
  if (label === "calm") return "bg-dovish/20 text-dovish";
  if (label === "high") return "bg-hawkish/20 text-hawkish";
  if (label === "normal") return "bg-neutral/20 text-foreground";
  return "bg-muted text-foreground";
}

function hitBorder(predicted: string | null, realized: string | null | undefined): string {
  if (!predicted || !realized) return "border-border";
  return predicted === realized ? "border-up" : "border-down";
}

export function RegimeHistoryStrip({ entries, symbol }: RegimeHistoryStripProps) {
  if (entries.length === 0) return null;
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <History className="h-4 w-4 text-primary" />
          Past runs vs realized
        </CardTitle>
        <CardDescription>
          Predicted regime (top chip) and realized regime (bottom chip) for the last {entries.length} runs
          {symbol ? ` on ${symbol}` : ""}. Green border = hit, red border = miss, no border = unresolved.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2 overflow-x-auto pb-1">
          {entries.map((entry) => {
            const borderClass = hitBorder(entry.argmax, entry.realized ?? null);
            return (
              <Link
                key={entry.runId}
                href={`/history/${entry.runId}`}
                className={cn(
                  "flex min-w-[88px] flex-col rounded-md border-2 bg-background/40 p-1.5 text-center transition hover:bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  borderClass,
                )}
              >
                <span className="numeric text-[10px] text-muted-foreground">
                  {entry.documentDate}
                </span>
                <span
                  className={cn(
                    "mt-0.5 inline-flex items-center justify-center rounded-sm px-1.5 py-0.5 text-[11px] font-medium capitalize",
                    regimeColor(entry.argmax),
                  )}
                >
                  {entry.argmax ?? "—"}
                </span>
                <span className="my-0.5 inline-block h-px w-full bg-border" aria-hidden="true" />
                {entry.realized ? (
                  <span
                    className={cn(
                      "inline-flex items-center justify-center rounded-sm px-1.5 py-0.5 text-[11px] capitalize",
                      regimeColor(entry.realized),
                    )}
                  >
                    {entry.realized}
                  </span>
                ) : (
                  <span className="inline-flex items-center justify-center gap-1 text-[10px] text-muted-foreground">
                    <CircleDashed className="h-3 w-3" aria-hidden="true" /> pending
                  </span>
                )}
              </Link>
            );
          })}
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <Badge variant="dovish" className="text-[9px]">calm</Badge>
            low realized vol
          </span>
          <span className="flex items-center gap-1">
            <Badge variant="neutral" className="text-[9px]">normal</Badge>
            mid realized vol
          </span>
          <span className="flex items-center gap-1">
            <Badge variant="hawkish" className="text-[9px]">high</Badge>
            high realized vol
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
