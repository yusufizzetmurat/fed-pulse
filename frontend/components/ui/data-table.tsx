import * as React from "react";
import { ChevronDown, ChevronUp, ChevronsUpDown } from "lucide-react";

import { cn } from "@/lib/utils";

export interface DataTableColumn<T> {
  key: string;
  header: React.ReactNode;
  align?: "left" | "right" | "center";
  numeric?: boolean;
  sortable?: boolean;
  width?: string;
  render: (row: T, rowIndex: number) => React.ReactNode;
  sortValue?: (row: T) => number | string | null | undefined;
}

interface DataTableProps<T> {
  rows: T[];
  columns: DataTableColumn<T>[];
  rowKey: (row: T, rowIndex: number) => string;
  caption?: React.ReactNode;
  rowHref?: (row: T) => string | null | undefined;
  rowClassName?: string;
  emptyState?: React.ReactNode;
  density?: "sm" | "md" | "lg";
  onRowClick?: (row: T) => void;
  className?: string;
}

const DENSITY_HEIGHT = { sm: "h-row-sm", md: "h-row-md", lg: "h-row-lg" } as const;

export function DataTable<T>({
  rows,
  columns,
  rowKey,
  caption,
  rowHref,
  rowClassName,
  emptyState,
  density = "sm",
  onRowClick,
  className,
}: DataTableProps<T>) {
  const [sortKey, setSortKey] = React.useState<string | null>(null);
  const [sortDir, setSortDir] = React.useState<"asc" | "desc">("desc");

  const sortedRows = React.useMemo(() => {
    if (!sortKey) return rows;
    const column = columns.find((c) => c.key === sortKey);
    if (!column?.sortable) return rows;
    const extractor = column.sortValue ?? ((row: T) => column.render(row, 0) as any);
    const direction = sortDir === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const av = extractor(a);
      const bv = extractor(b);
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return (av - bv) * direction;
      return String(av).localeCompare(String(bv)) * direction;
    });
  }, [rows, columns, sortKey, sortDir]);

  const toggleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  if (rows.length === 0 && emptyState) {
    return <div className="px-4 py-8 text-center">{emptyState}</div>;
  }

  return (
    <div className={cn("w-full overflow-x-auto", className)}>
      <table className="w-full border-collapse text-sm">
        {caption ? (
          <caption className="px-4 py-2 text-left text-xs text-muted-foreground">{caption}</caption>
        ) : null}
        <thead className="sticky top-0 border-b border-border bg-muted/40 text-[10px] uppercase tracking-wide text-muted-foreground">
          <tr>
            {columns.map((col) => {
              const align =
                col.align === "right"
                  ? "text-right"
                  : col.align === "center"
                  ? "text-center"
                  : "text-left";
              const SortIcon =
                sortKey !== col.key
                  ? ChevronsUpDown
                  : sortDir === "asc"
                  ? ChevronUp
                  : ChevronDown;
              return (
                <th
                  key={col.key}
                  scope="col"
                  className={cn("px-3 py-2 font-medium", align)}
                  style={col.width ? { width: col.width } : undefined}
                >
                  {col.sortable ? (
                    <button
                      type="button"
                      onClick={() => toggleSort(col.key)}
                      className="inline-flex items-center gap-1 hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    >
                      <span>{col.header}</span>
                      <SortIcon className="h-3 w-3" aria-hidden="true" />
                    </button>
                  ) : (
                    col.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row, rowIndex) => {
            const key = rowKey(row, rowIndex);
            const href = rowHref?.(row);
            const interactive = Boolean(href) || Boolean(onRowClick);
            return (
              <tr
                key={key}
                className={cn(
                  DENSITY_HEIGHT[density],
                  "border-b border-border last:border-0",
                  interactive && "cursor-pointer hover:bg-muted/40",
                  rowClassName,
                )}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((col) => {
                  const align =
                    col.align === "right"
                      ? "text-right"
                      : col.align === "center"
                      ? "text-center"
                      : "text-left";
                  return (
                    <td
                      key={col.key}
                      className={cn(
                        "px-3 py-1.5",
                        align,
                        col.numeric && "numeric whitespace-nowrap",
                      )}
                    >
                      {col.render(row, rowIndex)}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
