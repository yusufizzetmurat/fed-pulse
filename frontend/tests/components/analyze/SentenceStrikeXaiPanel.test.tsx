import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { SentenceStrikeXaiPanel } from "@/components/analyze/SentenceStrikeXaiPanel";
import type { AnalyzeResult, XaiResponse } from "@/lib/analyze/types";

const XAI: XaiResponse = {
  sentences: [
    { text: "First sentence about inflation.", score: 0.6, topTokens: [] },
    { text: "Second sentence about employment.", score: -0.2, topTokens: [] },
    { text: "Third sentence about forward guidance.", score: 0.1, topTokens: [] },
  ],
  method: "fixture",
};

function makeResult(argmax: string, probability: number): AnalyzeResult {
  const distribution: Record<string, number> = {
    calm: argmax === "calm" ? probability : (1 - probability) / 2,
    normal: argmax === "normal" ? probability : (1 - probability) / 2,
    high: argmax === "high" ? probability : (1 - probability) / 2,
  };
  return {
    regime_classification: {
      predicted_set: [argmax],
      set_label: argmax,
      set_size: 1,
      coverage: 0.8,
      distribution,
      argmax_class: argmax,
    },
  };
}

describe("SentenceStrikeXaiPanel cumulative drift chart", () => {
  it("hides the chart when only the baseline point is on record", () => {
    const baseline = makeResult("normal", 0.6);
    render(
      <SentenceStrikeXaiPanel
        xai={XAI}
        struck={new Set()}
        onMaskChange={() => {}}
        baselineResult={baseline}
        currentResult={baseline}
      />,
    );
    expect(screen.queryByText(/baseline/i)).not.toBeInTheDocument();
  });

  it("renders the chart after a single strike and updates on a second", () => {
    const baseline = makeResult("normal", 0.6);
    const afterOne = makeResult("normal", 0.52);
    const afterTwo = makeResult("normal", 0.41);
    const onMaskChange = vi.fn();

    const { rerender } = render(
      <SentenceStrikeXaiPanel
        xai={XAI}
        struck={new Set()}
        onMaskChange={onMaskChange}
        baselineResult={baseline}
        currentResult={baseline}
      />,
    );

    rerender(
      <SentenceStrikeXaiPanel
        xai={XAI}
        struck={new Set([0])}
        onMaskChange={onMaskChange}
        baselineResult={baseline}
        currentResult={afterOne}
      />,
    );
    expect(screen.getByText(/across 1 strike/i)).toBeInTheDocument();
    expect(screen.getByText(/60\.0% → 52\.0%/i)).toBeInTheDocument();

    rerender(
      <SentenceStrikeXaiPanel
        xai={XAI}
        struck={new Set([0, 1])}
        onMaskChange={onMaskChange}
        baselineResult={baseline}
        currentResult={afterTwo}
      />,
    );
    expect(screen.getByText(/across 2 strikes/i)).toBeInTheDocument();
    expect(screen.getByText(/60\.0% → 41\.0%/i)).toBeInTheDocument();
  });

  it("emits the next mask on click and shows the reset button", () => {
    const baseline = makeResult("normal", 0.6);
    const onMaskChange = vi.fn();
    render(
      <SentenceStrikeXaiPanel
        xai={XAI}
        struck={new Set([0])}
        onMaskChange={onMaskChange}
        baselineResult={baseline}
        currentResult={baseline}
      />,
    );
    const resetBtn = screen.getByRole("button", { name: /reset/i });
    fireEvent.click(resetBtn);
    expect(onMaskChange).toHaveBeenCalledWith(new Set());
  });
});
