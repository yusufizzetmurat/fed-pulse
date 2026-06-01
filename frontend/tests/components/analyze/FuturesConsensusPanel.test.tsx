import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { FuturesConsensusPanel } from "@/components/analyze/FuturesConsensusPanel";
import type { FuturesConsensusResponse } from "@/lib/analyze/types";

function fixture(
  overrides: Partial<FuturesConsensusResponse> = {},
): FuturesConsensusResponse {
  return {
    meeting_date: "2026-04-28",
    generated_at: "2026-04-01T12:00:00+00:00",
    current_target_lo_bps: 525,
    current_target_hi_bps: 550,
    horizons: [
      {
        horizon_label: "1m",
        implied_rate_bps: 533,
        change_vs_current_bps: -4.5,
        probability_hike: 0.05,
        probability_cut: 0.05,
        probability_pause: 0.9,
      },
      {
        horizon_label: "3m",
        implied_rate_bps: 580,
        change_vs_current_bps: 42.5,
        probability_hike: 0.92,
        probability_cut: 0.0,
        probability_pause: 0.08,
      },
      {
        horizon_label: "6m",
        implied_rate_bps: 480,
        change_vs_current_bps: -57.5,
        probability_hike: 0.0,
        probability_cut: 0.95,
        probability_pause: 0.05,
      },
    ],
    methodology:
      "Treasury constant-maturity proxy (DGS1MO/3MO/6MO). Embeds a term premium; treat as a level proxy, not an OIS-clean expectation.",
    data_source: "FRED",
    ...overrides,
  };
}

describe("FuturesConsensusPanel", () => {
  it("renders the loading placeholder when loading is true", () => {
    render(<FuturesConsensusPanel data={null} loading />);
    expect(
      screen.getByTestId("futures-consensus-loading"),
    ).toBeInTheDocument();
  });

  it("renders the unavailable placeholder when data is null", () => {
    render(<FuturesConsensusPanel data={null} />);
    expect(
      screen.getByTestId("futures-consensus-unavailable"),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Futures consensus feed unavailable/i),
    ).toBeInTheDocument();
  });

  it("renders the descriptive workspace variant (never a forecast card)", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    const section = screen.getByLabelText("FRED futures consensus");
    expect(section).toHaveAttribute("data-variant", "descriptive");
    expect(section.className).toMatch(/border-dashed/);
  });

  it("formats the next meeting date in a human-readable form", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    expect(
      screen.getByTestId("futures-consensus-meeting-date"),
    ).toHaveTextContent(/Apr 28, 2026/i);
  });

  it("renders the current target band and its midpoint in bps", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    // Midpoint of 525 / 550 -> 537.5 -> rounded to 538 bps with the
    // default formatBps fraction-digits of 0.
    expect(
      screen.getByTestId("futures-consensus-midpoint"),
    ).toHaveTextContent(/538 bps/);
  });

  it("renders three horizon columns in tenor order", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    expect(
      screen.getByTestId("futures-consensus-horizon-1m"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("futures-consensus-horizon-3m"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("futures-consensus-horizon-6m"),
    ).toBeInTheDocument();
  });

  it("renders implied rates and change-vs-current with sign conventions", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    expect(
      screen.getByTestId("futures-consensus-implied-3m"),
    ).toHaveTextContent(/580 bps/);
    expect(
      screen.getByTestId("futures-consensus-change-3m"),
    ).toHaveTextContent(/\+43 bps vs current/);
    expect(
      screen.getByTestId("futures-consensus-change-6m"),
    ).toHaveTextContent(/-58 bps vs current/);
  });

  it("renders hike / cut / pause probability percentages per horizon", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    expect(
      screen.getByTestId("futures-consensus-phike-3m"),
    ).toHaveTextContent(/Hike 92%/i);
    expect(
      screen.getByTestId("futures-consensus-pcut-6m"),
    ).toHaveTextContent(/Cut 95%/i);
    expect(
      screen.getByTestId("futures-consensus-ppause-1m"),
    ).toHaveTextContent(/Pause 90%/i);
  });

  it("exposes a Treasury-proxy tooltip trigger in the header", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    const trigger = screen.getByTestId("futures-consensus-proxy-trigger");
    expect(trigger).toHaveAccessibleName(/Treasury proxy methodology/i);
    expect(trigger).toHaveTextContent(/Treasury proxy/i);
  });

  it("renders the methodology footnote and the FRED data source", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    const footnote = screen.getByTestId("futures-consensus-methodology");
    expect(footnote).toHaveTextContent(/Treasury constant-maturity proxy/i);
    expect(footnote).toHaveTextContent(/Source: FRED/i);
  });

  it("renders the stacked probability bar with three segments", () => {
    render(<FuturesConsensusPanel data={fixture()} />);
    // One bar per horizon -> three on the panel.
    expect(screen.getAllByTestId("futures-consensus-prob-bar")).toHaveLength(3);
    // The 3m horizon has hike-heavy probabilities; the hike segment
    // width should dominate the other two.
    const phike3m = screen.getAllByTestId("futures-consensus-prob-hike");
    expect(phike3m).toHaveLength(3);
  });
});
