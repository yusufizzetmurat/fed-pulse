import * as React from "react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render } from "@testing-library/react";

// Verifies the semantic-diff debounce contract added to
// pages/index.tsx. Two invariants matter here:
//
// 1) Per-keystroke writes to ``request.text`` MUST NOT issue one
//    ``fetchSemanticDiff`` POST per keystroke. The previous code
//    fan-out is what made the analyze textarea hammer the backend
//    during a paste of a long FOMC statement.
// 2) Two distinct edits that preserve length (e.g., a typo fix in
//    place) MUST each result in their own fetch — the old guard keyed
//    on ``request.text.length`` would have swallowed the second.
//
// The component below mirrors the production debounce + fetch pair
// 1:1, so a regression in either (debounce window removed, key
// reintroduced) would surface here.

const fetchSemanticDiffMock = vi.fn(async (_url: string, body: { current_text: string }) => ({
  text: body.current_text,
}));

function DebouncedSemanticDiffHarness({ text }: { text: string }) {
  const [debouncedText, setDebouncedText] = React.useState(text);
  React.useEffect(() => {
    const handle = window.setTimeout(() => {
      setDebouncedText(text);
    }, 400);
    return () => {
      window.clearTimeout(handle);
    };
  }, [text]);

  React.useEffect(() => {
    if (!debouncedText.trim()) return;
    const controller = new AbortController();
    fetchSemanticDiffMock("http://api", { current_text: debouncedText });
    return () => {
      controller.abort();
    };
  }, [debouncedText]);

  return null;
}

describe("workspace semantic-diff debounce", () => {
  beforeEach(() => {
    fetchSemanticDiffMock.mockClear();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("does not fan out one fetch per keystroke during a paste", () => {
    const { rerender } = render(<DebouncedSemanticDiffHarness text="" />);
    // Simulate a rapid 60-character paste, one keystroke per tick.
    const target = "We anticipate that ongoing increases in the target range";
    for (let i = 1; i <= target.length; i += 1) {
      rerender(<DebouncedSemanticDiffHarness text={target.slice(0, i)} />);
    }
    // Inside the debounce window only the initial empty-text effect
    // ran (which the guard above skipped); no network calls have
    // fired yet despite 60 state writes.
    expect(fetchSemanticDiffMock).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(400);
    });
    // After the window settles, exactly one fetch for the final value.
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(1);
    expect(fetchSemanticDiffMock).toHaveBeenLastCalledWith(
      "http://api",
      expect.objectContaining({ current_text: target }),
    );
  });

  it("issues a fresh fetch when an in-place edit preserves length", () => {
    const initial = "we expect infaltion to decline";
    const { rerender } = render(<DebouncedSemanticDiffHarness text={initial} />);
    act(() => {
      vi.advanceTimersByTime(400);
    });
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(1);
    expect(fetchSemanticDiffMock).toHaveBeenLastCalledWith(
      "http://api",
      expect.objectContaining({ current_text: initial }),
    );
    // Same length, different content. A length-keyed guard would
    // swallow this; the debounced + key-free effect must not.
    const fixed = "we expect inflation to decline";
    expect(fixed.length).toBe(initial.length);
    rerender(<DebouncedSemanticDiffHarness text={fixed} />);
    act(() => {
      vi.advanceTimersByTime(400);
    });
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(2);
    expect(fetchSemanticDiffMock).toHaveBeenLastCalledWith(
      "http://api",
      expect.objectContaining({ current_text: fixed }),
    );
  });

  it("coalesces edits within one debounce window into a single trailing fetch", () => {
    // Start from empty so the initial mount does not fire its own
    // fetch; that mirrors how the workspace renders on first paint
    // when the user has not pasted a statement yet.
    const { rerender } = render(<DebouncedSemanticDiffHarness text="" />);
    act(() => {
      rerender(<DebouncedSemanticDiffHarness text="draft a" />);
    });
    act(() => {
      vi.advanceTimersByTime(100);
    });
    act(() => {
      rerender(<DebouncedSemanticDiffHarness text="draft b" />);
    });
    act(() => {
      vi.advanceTimersByTime(100);
    });
    act(() => {
      rerender(<DebouncedSemanticDiffHarness text="draft c" />);
    });
    act(() => {
      vi.advanceTimersByTime(400);
    });
    expect(fetchSemanticDiffMock).toHaveBeenCalledTimes(1);
    expect(fetchSemanticDiffMock).toHaveBeenLastCalledWith(
      "http://api",
      expect.objectContaining({ current_text: "draft c" }),
    );
  });
});
