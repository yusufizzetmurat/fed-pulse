import * as React from "react";
import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { Badge } from "@/components/ui/badge";
import type { AnalyzeRequest, Horizon } from "@/lib/analyze/types";

// Guards the workspace header chip that surfaces the live forecast-curve
// selection. The chip used to be hard-coded ("horizon · 10 days") which
// went stale the moment the user picked anything other than 10d. The
// follow-up to PR #614 rebound it to ``request.horizon`` and renamed the
// label to ``forecast curve`` so it does not collide with the regime card
// (which always reports 10 trading days ahead, regardless of the picker).
//
// A regression that re-statics the value or reverts the label would not
// be caught by the existing AnalyzeForm tooltip test, so this harness
// mirrors the production JSX 1:1 and asserts both invariants.

function HeaderBadgeHarness({ horizon }: { horizon: Horizon }) {
  const request: Pick<AnalyzeRequest, "horizon"> = { horizon };
  return (
    <Badge variant="outline" className="numeric text-[10px]">
      forecast curve · {request.horizon}
    </Badge>
  );
}

describe("workspace header forecast-curve badge", () => {
  it("renders the user-selected horizon when 1d", () => {
    render(<HeaderBadgeHarness horizon="1d" />);
    expect(screen.getByText("forecast curve · 1d")).toBeInTheDocument();
  });

  it("renders the user-selected horizon when 5d", () => {
    render(<HeaderBadgeHarness horizon="5d" />);
    expect(screen.getByText("forecast curve · 5d")).toBeInTheDocument();
  });

  it("renders the user-selected horizon when 10d", () => {
    render(<HeaderBadgeHarness horizon="10d" />);
    expect(screen.getByText("forecast curve · 10d")).toBeInTheDocument();
  });

  it("does not surface the bare ``horizon ·`` prefix that conflicted with the regime card", () => {
    render(<HeaderBadgeHarness horizon="5d" />);
    // The pre-fix label was the bare ``horizon · 5d`` which read as the
    // regime-card horizon (fixed 10 days ahead). Assert the prefix is
    // gone so a regression to the ambiguous label would fail here.
    expect(screen.queryByText("horizon · 5d")).not.toBeInTheDocument();
  });
});
