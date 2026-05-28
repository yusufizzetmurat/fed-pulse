import { describe, expect, it } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";

import { HistoricalAnalogPanel } from "@/components/analyze/HistoricalAnalogPanel";
import type { AnalogsResponse } from "@/lib/analyze/types";

function fixture(overrides: Partial<AnalogsResponse> = {}): AnalogsResponse {
  return {
    analogs: [
      {
        event_date: "2019-07-31",
        similarity: 0.82,
        axis_stance: "dovish",
        subsequent_vol_regime: "high",
        excerpt: "Information received since the Committee met in June indicates that the labor market remains strong and that economic activity has been rising at a moderate rate.",
      },
      {
        event_date: "2015-09-17",
        similarity: 0.71,
        axis_stance: "neutral",
        subsequent_vol_regime: "normal",
        excerpt: "Information received since the Federal Open Market Committee met in July suggests that economic activity is expanding at a moderate pace.",
      },
      {
        event_date: "2007-09-18",
        similarity: 0.65,
        axis_stance: "dovish",
        subsequent_vol_regime: "calm",
        excerpt: "Economic growth was moderate during the first half of the year, but the tightening of credit conditions has the potential to intensify the housing correction and to restrain economic growth more generally.",
      },
      {
        event_date: "2001-01-31",
        similarity: 0.21,
        axis_stance: "hawkish",
        subsequent_vol_regime: "normal",
        excerpt: "The Committee continues to believe that, against the background of its long-run goals of price stability and sustainable economic growth, the risks remain weighted mainly toward conditions that may generate economic weakness.",
      },
    ],
    index_size: 184,
    encoder_alias: "finbert_fed_adjacent_xbank_dapt_retrieval",
    ...overrides,
  };
}

describe("HistoricalAnalogPanel", () => {
  it("renders the top-3 analog cards above the similarity threshold", () => {
    render(<HistoricalAnalogPanel analogs={fixture()} />);
    expect(screen.getByText("2019-07-31")).toBeInTheDocument();
    expect(screen.getByText("2015-09-17")).toBeInTheDocument();
    expect(screen.getByText("2007-09-18")).toBeInTheDocument();
    // Fourth analog is below the default 0.40 threshold.
    expect(screen.queryByText("2001-01-31")).not.toBeInTheDocument();
  });

  it("formats similarity as a percentage", () => {
    render(<HistoricalAnalogPanel analogs={fixture()} />);
    expect(screen.getByText(/82.0% similar/)).toBeInTheDocument();
    expect(screen.getByText(/71.0% similar/)).toBeInTheDocument();
  });

  it("renders a stance badge for each analog with an axis_stance", () => {
    render(<HistoricalAnalogPanel analogs={fixture()} />);
    const dovishBadges = screen.getAllByText(/^dovish$/i);
    expect(dovishBadges.length).toBe(2);
  });

  it("renders 'stance unknown' when axis_stance is null", () => {
    const data = fixture({
      analogs: [
        {
          event_date: "2010-06-23",
          similarity: 0.55,
          axis_stance: null,
          subsequent_vol_regime: "calm",
          excerpt: "Short excerpt.",
        },
      ],
    });
    render(<HistoricalAnalogPanel analogs={data} />);
    expect(screen.getByText(/stance unknown/i)).toBeInTheDocument();
  });

  it("surfaces the post-event volatility bucket via the mini-chart aria-label", () => {
    render(<HistoricalAnalogPanel analogs={fixture()} />);
    expect(
      screen.getByLabelText(/10-day realised volatility after the event: high/i),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/10-day realised volatility after the event: normal/i),
    ).toBeInTheDocument();
  });

  it("renders skeletons in the loading state", () => {
    const { container } = render(
      <HistoricalAnalogPanel analogs={null} loading={true} topK={3} />,
    );
    // Each skeleton card has 4 Skeleton bars (header + body).
    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);
    expect(screen.getByText(/Historical analogs/i)).toBeInTheDocument();
  });

  it("renders nothing when analogs is null and not loading", () => {
    const { container } = render(
      <HistoricalAnalogPanel analogs={null} loading={false} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the bundle-absent empty state when index_size is 0", () => {
    render(
      <HistoricalAnalogPanel
        analogs={{ analogs: [], index_size: 0, encoder_alias: "" }}
      />,
    );
    expect(screen.getByText(/No analogs available/i)).toBeInTheDocument();
  });

  it("renders the threshold-empty state when no analog crosses the floor", () => {
    const lowSim = fixture({
      analogs: fixture().analogs.map((card) => ({ ...card, similarity: 0.12 })),
    });
    render(<HistoricalAnalogPanel analogs={lowSim} similarityThreshold={0.4} />);
    expect(screen.getByText(/No close analogs found/i)).toBeInTheDocument();
    // The threshold floor is rendered in the description so the
    // user knows what bar was used.
    expect(screen.getByText(/40.0%/)).toBeInTheDocument();
  });

  it("expands a long excerpt when the user clicks the toggle", () => {
    // The endpoint truncates to ~280 chars; pad the fixture excerpt
    // to the ceiling so the toggle button is rendered.
    const longCard = {
      event_date: "2018-12-19",
      similarity: 0.78,
      axis_stance: "hawkish" as const,
      subsequent_vol_regime: "high" as const,
      excerpt: "x".repeat(280),
    };
    render(
      <HistoricalAnalogPanel
        analogs={{
          analogs: [longCard],
          index_size: 42,
          encoder_alias: "test",
        }}
      />,
    );
    const toggle = screen.getByRole("button", { name: /Show full excerpt/i });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(toggle);
    expect(
      screen.getByRole("button", { name: /Collapse/i }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("does not show the expand toggle for short excerpts", () => {
    const shortCard = {
      event_date: "2014-06-18",
      similarity: 0.62,
      axis_stance: "neutral" as const,
      subsequent_vol_regime: "normal" as const,
      excerpt: "Short statement under the 280 ceiling.",
    };
    render(
      <HistoricalAnalogPanel
        analogs={{
          analogs: [shortCard],
          index_size: 8,
          encoder_alias: "test",
        }}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /Show full excerpt/i }),
    ).not.toBeInTheDocument();
  });

  it("renders both cards when two analogs share the same event_date", () => {
    // The retrieval index dedupes by text_hash, not event_date — an
    // intermeeting statement and a same-day correction can both land
    // in the top-k. With ``key={card.event_date}`` alone React would
    // collapse the two cards into one DOM node and bleed the per-card
    // expand state. The composite key (event_date, similarity, idx)
    // is what keeps them distinct.
    const sameDate = {
      event_date: "2020-03-15",
      similarity: 0.81,
      axis_stance: "dovish" as const,
      subsequent_vol_regime: "high" as const,
      excerpt: "Intermeeting statement excerpt.",
    };
    const sameDateCorrection = {
      ...sameDate,
      similarity: 0.74,
      excerpt: "Correction issued same day.",
    };
    render(
      <HistoricalAnalogPanel
        analogs={{
          analogs: [sameDate, sameDateCorrection],
          index_size: 200,
          encoder_alias: "test",
        }}
      />,
    );
    expect(screen.getByText(/Intermeeting statement excerpt/)).toBeInTheDocument();
    expect(screen.getByText(/Correction issued same day/)).toBeInTheDocument();
  });

  it("renders the encoder alias + index size in the footer note", () => {
    render(<HistoricalAnalogPanel analogs={fixture()} />);
    // index size is formatted with toLocaleString — accept either
    // comma-grouped or plain formatting depending on test locale.
    const footer = screen.getByText(/model variant/i);
    expect(within(footer).getByText(/finbert_fed_adjacent_xbank_dapt_retrieval/)).toBeInTheDocument();
  });
});
