import * as React from "react";
import { useRouter } from "next/router";
import { Activity, Calendar, FlaskConical, GitCompare, History as HistoryIcon, LineChart, Search } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { fetchFomcCalendar, resolveApiBaseUrl } from "@/lib/analyze/api";
import { useSymbols } from "@/lib/analyze/useSymbols";
import { cn } from "@/lib/utils";

interface PaletteEntry {
  id: string;
  label: string;
  hint?: string;
  group: "Pages" | "FOMC dates" | "Symbols";
  icon?: React.ComponentType<{ className?: string }>;
  perform: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (next: boolean) => void;
}

const PAGE_ENTRIES = [
  { id: "page-workspace", label: "Workspace", hint: "/", icon: Activity, href: "/" },
  { id: "page-history", label: "History", hint: "/history", icon: HistoryIcon, href: "/history" },
  { id: "page-compare", label: "Compare", hint: "/compare", icon: GitCompare, href: "/compare" },
  { id: "page-calendar", label: "Calendar", hint: "/calendar", icon: Calendar, href: "/calendar" },
  { id: "page-performance", label: "Performance", hint: "/performance", icon: LineChart, href: "/performance" },
  { id: "page-research", label: "Research", hint: "/research", icon: FlaskConical, href: "/research" },
];

export function CommandPalette({ open, onOpenChange }: CommandPaletteProps) {
  const router = useRouter();
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const { symbols } = useSymbols();
  const [query, setQuery] = React.useState("");
  const [activeIndex, setActiveIndex] = React.useState(0);
  const [meetings, setMeetings] = React.useState<Array<{ date: string; daysUntil: number | null }>>([]);

  React.useEffect(() => {
    if (!open) return;
    setQuery("");
    setActiveIndex(0);
  }, [open]);

  React.useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    fetchFomcCalendar(apiBaseUrl, { upcoming_limit: 6, past_limit: 6 })
      .then((response) => {
        if (controller.signal.aborted) return;
        const today = Date.now();
        const all = [...response.upcoming, ...response.past].map((meeting) => {
          const target = new Date(`${meeting.meeting_date}T00:00:00Z`).getTime();
          const days = Number.isFinite(target)
            ? Math.ceil((target - today) / 86_400_000)
            : null;
          return { date: meeting.meeting_date, daysUntil: days };
        });
        setMeetings(all);
      })
      .catch(() => {
        // Best-effort; an offline calendar just hides the date group.
      });
    return () => controller.abort();
  }, [open, apiBaseUrl]);

  const navigate = React.useCallback(
    (href: string) => {
      onOpenChange(false);
      router.push(href).catch(() => window.location.assign(href));
    },
    [onOpenChange, router],
  );

  const entries = React.useMemo<PaletteEntry[]>(() => {
    const all: PaletteEntry[] = [];
    for (const page of PAGE_ENTRIES) {
      all.push({
        id: page.id,
        label: page.label,
        hint: page.hint,
        group: "Pages",
        icon: page.icon,
        perform: () => navigate(page.href),
      });
    }
    for (const meeting of meetings.slice(0, 8)) {
      const tag =
        meeting.daysUntil == null
          ? ""
          : meeting.daysUntil > 0
          ? `in ${meeting.daysUntil}d`
          : meeting.daysUntil === 0
          ? "today"
          : `${Math.abs(meeting.daysUntil)}d ago`;
      all.push({
        id: `fomc-${meeting.date}`,
        label: meeting.date,
        hint: tag,
        group: "FOMC dates",
        icon: Calendar,
        perform: () =>
          navigate(`/?date=${encodeURIComponent(meeting.date)}&kind=statement`),
      });
    }
    for (const symbol of symbols.slice(0, 24)) {
      all.push({
        id: `symbol-${symbol.symbol}`,
        label: symbol.symbol,
        hint: symbol.name,
        group: "Symbols",
        icon: undefined,
        perform: () =>
          navigate(`/?symbol=${encodeURIComponent(symbol.symbol)}`),
      });
    }
    if (!query.trim()) return all;
    const needle = query.trim().toLowerCase();
    return all.filter(
      (entry) =>
        entry.label.toLowerCase().includes(needle) ||
        (entry.hint?.toLowerCase().includes(needle) ?? false),
    );
  }, [meetings, navigate, query, symbols]);

  React.useEffect(() => {
    if (activeIndex >= entries.length) {
      setActiveIndex(0);
    }
  }, [activeIndex, entries.length]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((prev) => (entries.length === 0 ? 0 : (prev + 1) % entries.length));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex((prev) =>
        entries.length === 0 ? 0 : (prev - 1 + entries.length) % entries.length,
      );
    } else if (event.key === "Enter") {
      event.preventDefault();
      entries[activeIndex]?.perform();
    }
  };

  // Group entries for the visual list, preserving the order they were
  // accumulated above so Pages always show first.
  const grouped = React.useMemo(() => {
    const buckets = new Map<string, PaletteEntry[]>();
    entries.forEach((entry) => {
      const list = buckets.get(entry.group) ?? [];
      list.push(entry);
      buckets.set(entry.group, list);
    });
    return [...buckets.entries()];
  }, [entries]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl gap-0 p-0">
        <DialogHeader className="space-y-1 px-4 py-3">
          <DialogTitle className="text-sm">Command palette</DialogTitle>
          <DialogDescription className="text-[11px] text-muted-foreground">
            Jump to a page, FOMC date, or asset. Arrow keys navigate; Enter selects.
          </DialogDescription>
        </DialogHeader>
        <div className="border-t border-border px-3 py-2">
          <div className="relative">
            <Search
              className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              autoFocus
              placeholder="Search pages, FOMC dates, symbols…"
              className="pl-7 text-sm"
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setActiveIndex(0);
              }}
              onKeyDown={handleKeyDown}
              aria-label="Command palette search"
            />
          </div>
        </div>
        <div className="max-h-72 overflow-y-auto border-t border-border">
          {entries.length === 0 ? (
            <p className="px-4 py-6 text-center text-xs text-muted-foreground">
              No matches.
            </p>
          ) : (
            grouped.map(([group, list]) => (
              <div key={group} className="py-1">
                <p className="px-3 pt-1 text-[10px] uppercase tracking-wide text-muted-foreground">
                  {group}
                </p>
                <ul role="listbox" aria-label={group}>
                  {list.map((entry) => {
                    const overall = entries.indexOf(entry);
                    const active = overall === activeIndex;
                    const Icon = entry.icon;
                    return (
                      <li key={entry.id}>
                        <button
                          type="button"
                          role="option"
                          aria-selected={active}
                          onMouseEnter={() => setActiveIndex(overall)}
                          onClick={() => entry.perform()}
                          className={cn(
                            "flex w-full items-center justify-between gap-3 px-3 py-1.5 text-left text-sm",
                            active ? "bg-muted text-foreground" : "text-muted-foreground hover:bg-muted/60",
                          )}
                        >
                          <span className="flex items-center gap-2">
                            {Icon ? <Icon className="h-3.5 w-3.5" aria-hidden="true" /> : null}
                            <span className="numeric">{entry.label}</span>
                          </span>
                          {entry.hint ? (
                            <span className="text-[11px] text-muted-foreground">{entry.hint}</span>
                          ) : null}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
