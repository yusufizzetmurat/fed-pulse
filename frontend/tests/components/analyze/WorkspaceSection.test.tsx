import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { WorkspaceSection } from "@/components/analyze/WorkspaceSection";

describe("WorkspaceSection", () => {
  it("renders title and children for the forecast variant with a Forecast badge", () => {
    render(
      <WorkspaceSection
        title="Expected Volume"
        description="HAR-based volume forecast"
        variant="forecast"
      >
        <p>forecast body</p>
      </WorkspaceSection>,
    );
    expect(screen.getByText("Expected Volume")).toBeInTheDocument();
    expect(screen.getByText("HAR-based volume forecast")).toBeInTheDocument();
    expect(screen.getByText("forecast body")).toBeInTheDocument();
    const badge = screen.getByTestId("workspace-section-badge");
    expect(badge).toHaveTextContent(/forecast/i);
  });

  it("renders the descriptive variant with a dashed-border and Descriptive badge", () => {
    render(
      <WorkspaceSection title="Monetary policy surprise" variant="descriptive">
        <p>panel body</p>
      </WorkspaceSection>,
    );
    const badge = screen.getByTestId("workspace-section-badge");
    expect(badge).toHaveTextContent(/descriptive/i);
    const section = screen.getByLabelText("Monetary policy surprise");
    expect(section).toHaveAttribute("data-variant", "descriptive");
    // The dashed border is what visually separates a descriptive panel
    // from a forecast card; the class is the load-bearing signal.
    expect(section.className).toMatch(/border-dashed/);
  });

  it("forecast variant applies the top accent border, not the dashed border", () => {
    render(
      <WorkspaceSection title="Realized vol" variant="forecast">
        <p>x</p>
      </WorkspaceSection>,
    );
    const section = screen.getByLabelText("Realized vol");
    expect(section).toHaveAttribute("data-variant", "forecast");
    expect(section.className).toMatch(/border-t-4/);
    expect(section.className).not.toMatch(/border-dashed/);
  });

  it("muted tone on the descriptive variant drops the tinted background", () => {
    render(
      <WorkspaceSection title="Futures consensus" variant="descriptive" tone="muted">
        <p>x</p>
      </WorkspaceSection>,
    );
    const section = screen.getByLabelText("Futures consensus");
    expect(section).toHaveAttribute("data-tone", "muted");
    expect(section.className).not.toMatch(/bg-muted\/30/);
  });

  it("omits the description paragraph when none is supplied", () => {
    render(
      <WorkspaceSection title="Semantic diff" variant="descriptive">
        <p>x</p>
      </WorkspaceSection>,
    );
    expect(screen.queryByText(/HAR-based/)).not.toBeInTheDocument();
  });
});
