import * as React from "react";

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
export type WorkspaceSectionVariant = "forecast" | "descriptive";
export type WorkspaceSectionTone = "default" | "muted";

export interface WorkspaceSectionProps {
  title: string;
  description?: string;
  variant: WorkspaceSectionVariant;
  tone?: WorkspaceSectionTone;
  className?: string;
  children: React.ReactNode;
}

const VARIANT_LABEL: Record<WorkspaceSectionVariant, string> = {
  forecast: "Forecast",
  descriptive: "Descriptive",
};

export function WorkspaceSection({
  title,
  description,
  variant,
  tone = "default",
  className,
  children,
}: WorkspaceSectionProps) {
  const isForecast = variant === "forecast";
  const isDescriptive = variant === "descriptive";
  return (
    <section
      data-variant={variant}
      data-tone={tone}
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
        <span
          data-testid="workspace-section-badge"
          className={cn(
            "inline-flex shrink-0 items-center rounded-full px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide",
            isForecast && "bg-primary/10 text-primary",
            isDescriptive && "border border-dashed border-border text-muted-foreground",
          )}
        >
          {VARIANT_LABEL[variant]}
        </span>
      </header>
      <div className="p-4 pt-2">{children}</div>
    </section>
  );
}

export default WorkspaceSection;
