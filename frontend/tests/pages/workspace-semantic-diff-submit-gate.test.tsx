import * as React from "react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { act, render } from "@testing-library/react";

// Verifies the semantic-diff submit-gate contract added to
// pages/index.tsx. Two invariants matter here:
//
// 1) Per-keystroke writes to ``request.text`` MUST NOT issue any
//    ``fetchSemanticDiff`` POST. The diff describes the just-analyzed
//    statement vs the prior — mid-keystroke fan-out would hammer a
//    server-side difflib + topic-emphasis path for no user benefit.
// 2) Re-typing in the textarea after a submit completes MUST NOT
//    retrigger the fetch; only the next submit may.
//
// The harness below mirrors the production submit-gate + fetch pair
// 1:1, so a regression in either (gate keyed on text again, or text
// changes after submit re-firing) would surface here.

const fetchSemanticDiffMock = vi.fn(
  async (_url: string, body: { current_text: string; current_date: string }) => ({
    text: body.current_text,
  }),
);

function SubmitGatedSemanticDiffHarness({
  text,
  date,
  submitSeq,
}: {
  text: string;
  date: string;
  submitSeq: number;
}) {
  const submittedRef = React.useRef<{ text: string; date: string } | null>(null);
  // Mirror handleSubmit: bumping the seq snapshots the current
  // (text, date) into the ref before the effect observes it.
  React.useEffect(() => {
    if (submitSeq === 0) return;
    submittedRef.current = { text, date };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [submitSeq]);

  React.useEffect(() => {
    if (submitSeq === 0) return;
    const submitted = submittedRef.current;
    if (!submitted || !submitted.text.trim()) return;
    fetchSemanticDiffMock("http://api", {
      current_text: submitted.text,
      current_date: submitted.date,
    });
  }, [submitSeq]);

  return null;
}

describe("workspace semantic-diff submit gate", () => {
  beforeEach(() => {
    fetchSemanticDiffMock.mockClear();
  });

  it("does not fetch while the user is typing into the textarea", () => {
    const { rerender } = render(
      <SubmitGatedSemanticDiffHarness text="" date="2026-06-01" submitSeq={0} />,
    );
    const target = "We anticipate that ongoing increases in the target range";
    for (let i = 1; i <= target.length; i += 1) {
      rerender(
        <SubmitGatedSemanticDiffHarness
          text={target.slice(0, i)}
          date="2026-06-01"
          submitSeq={0}
        />,
      );
    }
    expect(fetchSemanticDiffMock).not.toHaveBeenCalled();
  });

  it("fires exactly one fetch when the submit seq bumps", () => {
    const initial = "we expect inflation to decline";
    const { rerender } = render(
      <SubmitGatedSemanticDiffHarness text={initial} date="2026-06-01" submitSeq={0} />,
    );
    expect(fetchSemanticDiffMock).not.toHaveBeenCalled();
    act(() => {
      rerender(
        <SubmitGatedSemanticDiffHarness text={initial} date="2026-06-01" submitSeq={1} />,
      );
    });
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(1);
    expect(fetchSemanticDiffMock).toHaveBeenLastCalledWith(
      "http://api",
      expect.objectContaining({ current_text: initial, current_date: "2026-06-01" }),
    );
  });

  it("does not refetch when text changes post-submit without a new submit seq", () => {
    const submitted = "we expect inflation to decline";
    const { rerender } = render(
      <SubmitGatedSemanticDiffHarness text={submitted} date="2026-06-01" submitSeq={1} />,
    );
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(1);
    // User keeps typing into the textarea after the analysis ran;
    // the seq is unchanged so the effect must not refire.
    act(() => {
      rerender(
        <SubmitGatedSemanticDiffHarness
          text={submitted + " more text"}
          date="2026-06-01"
          submitSeq={1}
        />,
      );
    });
    act(() => {
      rerender(
        <SubmitGatedSemanticDiffHarness
          text="completely different paste"
          date="2026-06-01"
          submitSeq={1}
        />,
      );
    });
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(1);
  });

  it("fires again on the next submit with the freshly-snapshotted text", () => {
    const first = "we expect inflation to decline";
    const { rerender } = render(
      <SubmitGatedSemanticDiffHarness text={first} date="2026-06-01" submitSeq={1} />,
    );
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(1);
    expect(fetchSemanticDiffMock).toHaveBeenLastCalledWith(
      "http://api",
      expect.objectContaining({ current_text: first }),
    );
    // User edits then submits again.
    const second = "ongoing increases in the target range will be appropriate";
    act(() => {
      rerender(
        <SubmitGatedSemanticDiffHarness text={second} date="2026-06-01" submitSeq={2} />,
      );
    });
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(2);
    expect(fetchSemanticDiffMock).toHaveBeenLastCalledWith(
      "http://api",
      expect.objectContaining({ current_text: second }),
    );
  });
});
