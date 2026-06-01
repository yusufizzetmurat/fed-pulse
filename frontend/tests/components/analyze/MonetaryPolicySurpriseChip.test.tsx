import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { MonetaryPolicySurpriseChip } from "@/components/analyze/MonetaryPolicySurpriseChip";
import type { MonetaryPolicySurpriseResponse } from "@/lib/analyze/types";

function fixture(
  overrides: Partial<MonetaryPolicySurpriseResponse> = {},
): MonetaryPolicySurpriseResponse {
  return {
    event_date: "2026-04-29",
    mp_surprise_level_bps: 12.0,
    direction: "hawkish",
    magnitude_bps: 12.0,
    is_intermeeting: false,
    ff_target_prior_bps: 362.5,
    ...overrides,
  };
}

describe("MonetaryPolicySurpriseChip", () => {
  it("renders the unavailable placeholder when data is null", () => {
    render(<MonetaryPolicySurpriseChip data={null} />);
    expect(screen.getByTestId("mp-surprise-unavailable")).toBeInTheDocument();
    expect(
      screen.getByText(/MP-surprise feed unavailable/i),
    ).toBeInTheDocument();
  });

  it("renders the loading placeholder when loading is true", () => {
    render(<MonetaryPolicySurpriseChip data={null} loading />);
    expect(screen.getByTestId("mp-surprise-loading")).toBeInTheDocument();
  });

  it("renders the descriptive workspace variant (never a forecast card)", () => {
    render(<MonetaryPolicySurpriseChip data={fixture()} />);
    const section = screen.getByLabelText("Monetary policy surprise");
    // Descriptive panels are dashed-border + descriptive variant.
    expect(section).toHaveAttribute("data-variant", "descriptive");
    expect(section.className).toMatch(/border-dashed/);
  });

  it("renders the hawkish badge + magnitude for a positive surprise", () => {
    render(<MonetaryPolicySurpriseChip data={fixture()} />);
    const direction = screen.getByTestId("mp-surprise-direction");
    expect(direction).toHaveTextContent(/Hawkish/i);
    const magnitude = screen.getByTestId("mp-surprise-magnitude");
    // Magnitude is unsigned in the headline (sign carried by the badge).
    expect(magnitude).toHaveTextContent("12 bps");
  });

  it("renders the dovish badge for a negative surprise", () => {
    render(
      <MonetaryPolicySurpriseChip
        data={fixture({
          mp_surprise_level_bps: -14.5,
          direction: "dovish",
          magnitude_bps: 14.5,
        })}
      />,
    );
    expect(screen.getByTestId("mp-surprise-direction")).toHaveTextContent(
      /Dovish/i,
    );
    // ``formatBps`` rounds to whole bps by default — 14.5 rounds up.
    expect(screen.getByTestId("mp-surprise-magnitude")).toHaveTextContent(
      "15 bps",
    );
  });

  it("renders the no-surprise badge inside the band", () => {
    render(
      <MonetaryPolicySurpriseChip
        data={fixture({
          mp_surprise_level_bps: 1.2,
          direction: "no_surprise",
          magnitude_bps: 1.2,
        })}
      />,
    );
    expect(screen.getByTestId("mp-surprise-direction")).toHaveTextContent(
      /No surprise/i,
    );
  });

  it("flags intermeeting actions when the flag is set", () => {
    render(
      <MonetaryPolicySurpriseChip
        data={fixture({ is_intermeeting: true })}
      />,
    );
    expect(screen.getByText(/Intermeeting/i)).toBeInTheDocument();
  });

  it("formats the event date as a human-readable string", () => {
    render(<MonetaryPolicySurpriseChip data={fixture()} />);
    expect(screen.getByTestId("mp-surprise-event-date")).toHaveTextContent(
      /Apr 29, 2026/i,
    );
  });

  it("renders the prior target line when ff_target_prior_bps is present", () => {
    render(<MonetaryPolicySurpriseChip data={fixture()} />);
    expect(screen.getByText(/prior target/i)).toBeInTheDocument();
    expect(screen.getByText(/362\.5 bps|363 bps/)).toBeInTheDocument();
  });

  it("omits the prior target line when ff_target_prior_bps is null", () => {
    render(
      <MonetaryPolicySurpriseChip
        data={fixture({ ff_target_prior_bps: null })}
      />,
    );
    expect(screen.queryByText(/prior target/i)).toBeNull();
  });

  it("exposes a methodology tooltip trigger for the descriptive caveat", () => {
    render(<MonetaryPolicySurpriseChip data={fixture()} />);
    const trigger = screen.getByTestId("mp-surprise-methodology-trigger");
    expect(trigger).toHaveAccessibleName(/methodology/i);
  });
});
