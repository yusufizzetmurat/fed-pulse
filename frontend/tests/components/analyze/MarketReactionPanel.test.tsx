import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { MarketReactionPanel, RatesReactionCard } from "@/components/analyze/MarketReactionPanel";
import type { MarketReactionPanelResponse } from "@/lib/analyze/types";

function fixture(): MarketReactionPanelResponse {
  return {
    rates: [
      {
        head: "2y",
        point_bps: 4.5,
        lower_bps: 1.0,
        upper_bps: 8.0,
        coverage: 0.8,
        directional_bucket: "tightening",
        bucket_probabilities: { easing: 0.1, neutral: 0.3, tightening: 0.6 },
        predicted_set: ["tightening"],
      },
      {
        head: "5y",
        point_bps: -2.0,
        lower_bps: -6.0,
        upper_bps: 2.0,
        coverage: 0.8,
        directional_bucket: "neutral",
        bucket_probabilities: { easing: 0.2, neutral: 0.55, tightening: 0.25 },
        predicted_set: ["neutral"],
      },
      {
        head: "terminal",
        point_bps: 10.0,
        lower_bps: 5.0,
        upper_bps: 15.0,
        coverage: 0.8,
        directional_bucket: "tightening",
        bucket_probabilities: { easing: 0.05, neutral: 0.1, tightening: 0.85 },
        predicted_set: null,
      },
    ],
    vol_regime: {
      log_rv_point: -3.0,
      log_rv_lower: null,
      log_rv_upper: null,
      regime_label: "high",
      regime_probabilities: { calm: 0.1, normal: 0.3, high: 0.6 },
      predicted_set: ["high"],
      coverage: 0.8,
    },
    encoder_alias: "finbert_fomc",
    checkpoint_path: "/tmp/forecaster_best.pt",
  };
}

describe("MarketReactionPanel", () => {
  it("renders one card per rates head and never the duplicate Volatility Regime tile", () => {
    // The Volatility Regime tile used to render here alongside the
    // rates cards; it was a duplicate of the SecondOpinionRegime card
    // at workspace slot 3. The rates surfaces are still the panel's
    // load-bearing content.
    render(<MarketReactionPanel panel={fixture()} />);
    expect(screen.getByText(/2y yield/i)).toBeInTheDocument();
    expect(screen.getByText(/5y yield/i)).toBeInTheDocument();
    expect(screen.getByText(/Terminal rate/i)).toBeInTheDocument();
    expect(screen.queryByText(/Volatility Regime/i)).not.toBeInTheDocument();
  });

  it("shows the directional bucket badge", () => {
    render(<MarketReactionPanel panel={fixture()} />);
    const tightenings = screen.getAllByText(/tightening/i);
    expect(tightenings.length).toBeGreaterThan(0);
  });

  it("returns null when the panel has no rates", () => {
    const empty: MarketReactionPanelResponse = {
      rates: [],
      vol_regime: null,
      encoder_alias: null,
      checkpoint_path: null,
    };
    const { container } = render(<MarketReactionPanel panel={empty} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("returns null when only vol_regime is set — the duplicate tile is gone", () => {
    const vol_only: MarketReactionPanelResponse = {
      rates: [],
      vol_regime: fixture().vol_regime,
      encoder_alias: null,
      checkpoint_path: null,
    };
    const { container } = render(<MarketReactionPanel panel={vol_only} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("formats the point prediction with a signed bps suffix", () => {
    render(
      <RatesReactionCard
        card={{
          head: "2y",
          point_bps: 4.5,
          lower_bps: 1.0,
          upper_bps: 8.0,
          coverage: 0.8,
          directional_bucket: "tightening",
          bucket_probabilities: { easing: 0.1, neutral: 0.3, tightening: 0.6 },
          predicted_set: null,
        }}
      />,
    );
    expect(screen.getByText(/\+4.5 bps/)).toBeInTheDocument();
  });

  it("falls back to 'Band unavailable' when no conformal sidecar is present", () => {
    render(
      <RatesReactionCard
        card={{
          head: "5y",
          point_bps: -1.0,
          lower_bps: null,
          upper_bps: null,
          coverage: null,
          directional_bucket: "neutral",
          bucket_probabilities: { easing: 0.3, neutral: 0.4, tightening: 0.3 },
          predicted_set: null,
        }}
      />,
    );
    expect(screen.getByText(/Band unavailable/i)).toBeInTheDocument();
  });

  it("renders 'Aux classifier unavailable' badge when directional_bucket is null", () => {
    // #317 finding #10: a regression-only checkpoint without aux
    // classifier surfaces null fields; the card must show the
    // explicit "not available" badge rather than a fake bucket label.
    render(
      <RatesReactionCard
        card={{
          head: "2y",
          point_bps: 3.5,
          lower_bps: null,
          upper_bps: null,
          coverage: null,
          directional_bucket: null,
          bucket_probabilities: null,
          predicted_set: null,
        }}
      />,
    );
    expect(screen.getByText(/Direction model unavailable/i)).toBeInTheDocument();
  });
});
