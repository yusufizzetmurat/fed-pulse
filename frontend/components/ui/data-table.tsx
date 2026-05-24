import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/router";
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
  const router = useRouter();
  const [sortKey, setSortKey] = React.useState<string | null>(null);
  const [sortDir, setSortDir] = React.useState<"asc" | "desc">("desc");

  const sortedRows = React.useMemo(() => {
    if (!sortKey) return rows;
    const column = columns.find((c) => c.key === sortKey);
    // Sorting requires an explicit sortValue extractor — `render` returns
    // React nodes that would coerce to "[object Object]" and silently
    // no-op the sort. Columns without sortValue stay unsorted even when
    // marked sortable.
    if (!column?.sortable || !column.sortValue) return rows;
    const extractor = column.sortValue;
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
              const ariaSort: "ascending" | "descending" | "none" | undefined =
                col.sortable
                  ? sortKey === col.key
                    ? sortDir === "asc"
                      ? "ascending"
                      : "descending"
                    : "none"
                  : undefined;
              return (
                <th
                  key={col.key}
                  scope="col"
                  aria-sort={ariaSort}
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
            const href = rowHref?.(row) ?? null;
            const interactive = Boolean(href) || Boolean(onRowClick);
            const handleRowClick = (event: React.MouseEvent<HTMLTableRowElement>) => {
              if (onRowClick) {
                onRowClick(row);
                return;
              }
              if (!href) return;
              // Modifier-aware: cmd/ctrl/middle-click should still open in a
              // new tab, matching the existing inline <Link> behaviour on
              // legacy table cells.
              if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.button === 1) {
                return;
              }
              router.push(href).catch(() => {
                /* router.push failures are rare; fallback to a full nav. */
                window.location.assign(href);
              });
            };
            return (
              // Whole-row click is a mouse convenience; the keyboard /
              // screen-reader path is the Link injected into the first
              // cell below when rowHref is set. role="link" on <tr> is
              // not valid ARIA, so the row keeps its implicit row
              // semantics and we lean on the inner anchor for nav.
              <tr
                key={key}
                className={cn(
                  DENSITY_HEIGHT[density],
                  "border-b border-border last:border-0",
                  interactive && "cursor-pointer hover:bg-muted/40 focus-within:bg-muted/40",
                  rowClassName,
                )}
                onClick={interactive ? handleRowClick : undefined}
              >
                {columns.map((col, colIndex) => {
                  const align =
                    col.align === "right"
                      ? "text-right"
                      : col.align === "center"
                      ? "text-center"
                      : "text-left";
                  const inner = col.render(row, rowIndex);
                  // The first column is the conventional row title in
                  // every table on this branch; wrap it in a Next Link
                  // when rowHref is set so the row has a real anchor
                  // for keyboard tab and screen-reader semantics.
                  const cellContent =
                    href && colIndex === 0 ? (
                      <Link
                        href={href}
                        className="rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                        onClick={(event) => event.stopPropagation()}
                      >
                        {inner}
                      </Link>
                    ) : (
                      inner
                    );
                  return (
                    <td
                      key={col.key}
                      className={cn(
                        "px-3 py-1.5",
                        align,
                        col.numeric && "numeric whitespace-nowrap",
                      )}
                    >
                      {cellContent}
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
