import * as React from "react";

import { cn } from "@/lib/utils";
import type { ConfusionRow } from "@/lib/analyze/performance";

interface ConfusionMatrixProps {
  rows: ConfusionRow[];
  classes: readonly string[];
  className?: string;
}

function intensityClass(share: number): string {
  if (share >= 0.7) return "bg-primary/80 text-primary-foreground";
  if (share >= 0.4) return "bg-primary/50 text-primary-foreground";
  if (share >= 0.2) return "bg-primary/30 text-foreground";
  if (share > 0) return "bg-primary/15 text-foreground";
  return "bg-muted/30 text-muted-foreground";
}

export function ConfusionMatrix({ rows, classes, className }: ConfusionMatrixProps) {
  const totalResolved = rows.reduce((sum, row) => sum + row.total, 0);
  if (totalResolved === 0) {
    return (
      <p className="text-xs text-muted-foreground">
        No resolved runs yet — submit analyses past their 10d window to populate the matrix.
      </p>
    );
  }
  return (
    <div className={cn("overflow-x-auto", className)}>
      <table className="w-full border-separate" style={{ borderSpacing: 2 }}>
        <caption className="sr-only">
          Confusion matrix: rows are the realised regime, columns are the predicted argmax.
        </caption>
        <thead>
          <tr>
            <th scope="col" className="px-2 py-1 text-left text-[10px] uppercase tracking-wide text-muted-foreground">
              truth / pred
            </th>
            {classes.map((klass) => (
              <th
                key={klass}
                scope="col"
                className="px-2 py-1 text-center text-[10px] uppercase tracking-wide capitalize text-muted-foreground"
              >
                {klass}
              </th>
            ))}
            <th
              scope="col"
              className="px-2 py-1 text-right text-[10px] uppercase tracking-wide text-muted-foreground"
            >
              support
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.truth}>
              <th
                scope="row"
                className="px-2 py-1 text-left text-xs font-medium capitalize text-muted-foreground"
              >
                {row.truth}
              </th>
              {classes.map((klass) => {
                const count = row.counts[klass] ?? 0;
                const share = row.total > 0 ? count / row.total : 0;
                return (
                  <td
                    key={klass}
                    className={cn(
                      "numeric h-row-sm rounded-sm px-2 text-center text-xs",
                      intensityClass(share),
                    )}
                    title={`truth ${row.truth} → predicted ${klass}: ${count} / ${row.total}`}
                  >
                    {count}
                  </td>
                );
              })}
              <td className="numeric px-2 text-right text-xs text-muted-foreground">{row.total}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
