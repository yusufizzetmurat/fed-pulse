import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { PolicyActionCard } from "@/components/analyze/PolicyActionCard";
import type { PolicyActionResponse } from "@/lib/analyze/types";

function hikeAction(): PolicyActionResponse {
  return {
    target_range_low_bp: 525,
    target_range_high_bp: 550,
    change_direction: "hike",
    change_magnitude_bp: 25,
    balance_sheet_state: "runoff",
  };
}

function holdAction(): PolicyActionResponse {
  return {
    target_range_low_bp: 525,
    target_range_high_bp: 550,
    change_direction: "hold",
    change_magnitude_bp: 0,
    balance_sheet_state: "runoff",
  };
}

function cutAction(): PolicyActionResponse {
  return {
    target_range_low_bp: 475,
    target_range_high_bp: 500,
    change_direction: "cut",
    change_magnitude_bp: -50,
    balance_sheet_state: "runoff",
  };
}

describe("PolicyActionCard", () => {
  it("renders the target range as a percent badge on a hike", () => {
    render(<PolicyActionCard action={hikeAction()} />);
    expect(screen.getByText(/Policy action/i)).toBeInTheDocument();
    expect(screen.getByText("5.25% – 5.50%")).toBeInTheDocument();
    expect(screen.getByText(/\+25 bp/)).toBeInTheDocument();
  });

  it("renders 'hold' on a no-change action", () => {
    render(<PolicyActionCard action={holdAction()} />);
    // The change-indicator badge body reads exactly "hold" on a
    // zero-magnitude action; assert against that node specifically so
    // the assertion does not collide with the balance-sheet text or
    // copy that also contains the word "hold".
    expect(
      screen.getByText((_, node) => node?.textContent === "hold"),
    ).toBeInTheDocument();
  });

  it("renders a negative change indicator on a cut", () => {
    render(<PolicyActionCard action={cutAction()} />);
    expect(screen.getByText(/-50 bp/)).toBeInTheDocument();
    expect(screen.getByText("4.75% – 5.00%")).toBeInTheDocument();
  });

  it("renders the balance-sheet posture row when populated", () => {
    render(<PolicyActionCard action={hikeAction()} />);
    expect(screen.getByText(/runoff/i)).toBeInTheDocument();
  });

  it("renders the unavailable badge when balance-sheet posture is null", () => {
    const action: PolicyActionResponse = {
      ...hikeAction(),
      balance_sheet_state: null,
    };
    render(<PolicyActionCard action={action} />);
    expect(screen.getByText(/Balance sheet stance unavailable/i)).toBeInTheDocument();
  });

  it("returns null on an all-null payload (non-policy text)", () => {
    const empty: PolicyActionResponse = {
      target_range_low_bp: null,
      target_range_high_bp: null,
      change_direction: null,
      change_magnitude_bp: null,
      balance_sheet_state: null,
    };
    const { container } = render(<PolicyActionCard action={empty} />);
    expect(container).toBeEmptyDOMElement();
  });
});
