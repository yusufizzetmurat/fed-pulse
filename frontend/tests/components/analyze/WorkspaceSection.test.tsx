import { beforeEach, describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";

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

  describe("collapsible mode", () => {
    beforeEach(() => {
      window.localStorage.clear();
    });

    it("toggles the body and updates aria-expanded when the chevron is clicked", () => {
      render(
        <WorkspaceSection
          title="HAR headline"
          variant="forecast"
          collapsible
          storageKey="workspace-card:test-har-headline"
        >
          <p>headline body</p>
        </WorkspaceSection>,
      );
      const toggle = screen.getByTestId("workspace-section-toggle");
      const body = screen.getByTestId("workspace-section-body");
      // Default open: aria-expanded=true, body visible, persisted "1".
      expect(toggle).toHaveAttribute("aria-expanded", "true");
      expect(body).not.toHaveAttribute("hidden");
      expect(screen.getByText("headline body")).toBeInTheDocument();

      fireEvent.click(toggle);
      expect(toggle).toHaveAttribute("aria-expanded", "false");
      expect(body).toHaveAttribute("hidden");
      // Header itself stays visible so the user can re-open the card.
      expect(screen.getByText("HAR headline")).toBeInTheDocument();
      expect(
        window.localStorage.getItem("workspace-card:test-har-headline"),
      ).toBe("0");
    });

    it("rehydrates the persisted closed state on mount", () => {
      window.localStorage.setItem("workspace-card:rehydrate-test", "0");
      render(
        <WorkspaceSection
          title="Persisted card"
          variant="descriptive"
          collapsible
          storageKey="workspace-card:rehydrate-test"
        >
          <p>persisted body</p>
        </WorkspaceSection>,
      );
      const toggle = screen.getByTestId("workspace-section-toggle");
      // The persisted "0" should re-open as closed after the rehydrate
      // effect commits; the header / chevron stay in the DOM regardless.
      expect(toggle).toHaveAttribute("aria-expanded", "false");
      const body = screen.getByTestId("workspace-section-body");
      expect(body).toHaveAttribute("hidden");
    });

    it("wires aria-controls to the body element id when collapsible", () => {
      render(
        <WorkspaceSection
          title="Wired controls"
          variant="forecast"
          collapsible
          storageKey="workspace-card:aria-wired"
        >
          <p>body</p>
        </WorkspaceSection>,
      );
      const toggle = screen.getByTestId("workspace-section-toggle");
      const body = screen.getByTestId("workspace-section-body");
      const bodyId = body.getAttribute("id");
      expect(bodyId).toBeTruthy();
      expect(toggle).toHaveAttribute("aria-controls", bodyId!);
      // aria-label flips with state so screen readers don't read a stale
      // "Collapse X" target once the section is already collapsed.
      expect(toggle).toHaveAttribute("aria-label", "Collapse Wired controls");
      fireEvent.click(toggle);
      expect(toggle).toHaveAttribute("aria-label", "Expand Wired controls");
    });
  });
});
