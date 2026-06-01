import * as React from "react";
import { ChevronDown } from "lucide-react";

import { cn } from "@/lib/utils";

// Shared Workspace primitive that enforces the SPINE separation between
// forecast cards and descriptive panels at a glance.
//
// - variant="forecast"    market-data-only forecast cards (HAR-tercile,
//                         QLIKE-RV, Expected Volume). Gets a solid top
//                         accent border and a "Forecast" corner badge so
//                         a reader immediately knows the numbers below
//                         are model predictions over market history.
// - variant="descriptive" text- or realized-derived panels (MP-surprise,
//                         FRED futures consensus, semantic diff). Gets a
//                         dashed border and a muted background so it
//                         reads as commentary, never as a forecast input.
//
// The `tone` prop is a light style hook for the descriptive variant
// when the panel sits inside a denser layout and wants to drop the
// background tint.
//
// When `collapsible` is true the header gains a chevron toggle that
// hides/reveals the body. The open/closed state is mirrored into
// localStorage under `storageKey` so the workspace lays out the same
// way across reloads. The header stays visible while closed so the
// user can re-open the section.
export type WorkspaceSectionVariant = "forecast" | "descriptive";
export type WorkspaceSectionTone = "default" | "muted";

export interface WorkspaceSectionProps {
  title: string;
  description?: string;
  variant: WorkspaceSectionVariant;
  tone?: WorkspaceSectionTone;
  className?: string;
  collapsible?: boolean;
  storageKey?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

const VARIANT_LABEL: Record<WorkspaceSectionVariant, string> = {
  forecast: "Forecast",
  descriptive: "Descriptive",
};

function readPersistedOpen(storageKey: string | undefined, fallback: boolean): boolean {
  if (!storageKey) return fallback;
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (raw === null) return fallback;
    return raw === "1";
  } catch {
    return fallback;
  }
}

function writePersistedOpen(storageKey: string | undefined, open: boolean): void {
  if (!storageKey) return;
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(storageKey, open ? "1" : "0");
  } catch {
    // localStorage may throw in private mode / quota; the in-memory
    // state still drives the UI for the rest of the session.
  }
}

export function WorkspaceSection({
  title,
  description,
  variant,
  tone = "default",
  className,
  collapsible = false,
  storageKey,
  defaultOpen = true,
  children,
}: WorkspaceSectionProps) {
  const isForecast = variant === "forecast";
  const isDescriptive = variant === "descriptive";

  // Initial state is the SSR-safe `defaultOpen`; the persisted value is
  // rehydrated in an effect so the server and the client agree on the
  // first paint and React does not log a hydration mismatch.
  const [open, setOpen] = React.useState<boolean>(defaultOpen);

  React.useEffect(() => {
    if (!collapsible) return;
    setOpen(readPersistedOpen(storageKey, defaultOpen));
  }, [collapsible, storageKey, defaultOpen]);

  const handleToggle = React.useCallback(() => {
    setOpen((prev) => {
      const next = !prev;
      writePersistedOpen(storageKey, next);
      return next;
    });
  }, [storageKey]);

  // Stable body id so aria-controls can point at it. Derived from the
  // storageKey when supplied (deterministic) or from a React useId hash
  // so two unstored collapsibles on the same page still have unique ids.
  const reactId = React.useId();
  const bodyId = collapsible
    ? `workspace-section-body-${storageKey ?? reactId}`
    : undefined;

  return (
    <section
      data-variant={variant}
      data-tone={tone}
      data-collapsible={collapsible ? "true" : undefined}
      data-open={collapsible ? (open ? "true" : "false") : undefined}
      aria-label={title}
      className={cn(
        "relative rounded-xl border bg-card text-card-foreground shadow-sm",
        isForecast &&
          "border-border border-t-4 border-t-primary",
        isDescriptive &&
          "border-dashed border-border/70",
        isDescriptive && tone === "default" && "bg-muted/30",
        className,
      )}
    >
      <header className="flex items-start justify-between gap-3 p-4 pb-2">
        <div className="space-y-1">
          <h3 className="text-base font-semibold leading-tight tracking-tight">
            {title}
          </h3>
          {description ? (
            <p className="text-sm text-muted-foreground">{description}</p>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <span
            data-testid="workspace-section-badge"
            className={cn(
              "inline-flex items-center rounded-full px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide",
              isForecast && "bg-primary/10 text-primary",
              isDescriptive && "border border-dashed border-border text-muted-foreground",
            )}
          >
            {VARIANT_LABEL[variant]}
          </span>
          {collapsible ? (
            <button
              type="button"
              data-testid="workspace-section-toggle"
              onClick={handleToggle}
              aria-expanded={open}
              aria-controls={bodyId}
              aria-label={open ? `Collapse ${title}` : `Expand ${title}`}
              className={cn(
                "inline-flex h-7 w-7 items-center justify-center rounded-md border border-border bg-background text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              )}
            >
              <ChevronDown
                className={cn(
                  "h-4 w-4 transition-transform duration-200",
                  !open && "-rotate-90",
                )}
                aria-hidden="true"
              />
            </button>
          ) : null}
        </div>
      </header>
      {collapsible ? (
        <div
          id={bodyId}
          data-testid="workspace-section-body"
          hidden={!open}
          className={cn(
            "grid overflow-hidden transition-[grid-template-rows] duration-200 ease-out",
            open ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
          )}
        >
          <div className="min-h-0">
            <div className="p-4 pt-2">{children}</div>
          </div>
        </div>
      ) : (
        <div className="p-4 pt-2">{children}</div>
      )}
    </section>
  );
}

export default WorkspaceSection;
