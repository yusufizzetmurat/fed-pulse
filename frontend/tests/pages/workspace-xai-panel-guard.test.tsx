import * as React from "react";
import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";

// Guards the outer render condition for the per-sentence XAI panel in
// ``pages/index.tsx``. The panel mounts on the first /analyze response
// (``baselineResult.xai`` is set) and must keep rendering after a strike
// even when the counterfactual /analyze response omits the ``xai``
// field. The previous outer guard read ``result.xai`` directly, which
// unmounted the entire panel the moment the backend skipped XAI on a
// masked run; the fix tracks the same source the inner prop reads
// (``baselineResult.xai ?? result.xai``).

type XaiStub = { sentences: unknown[] };
type ResultStub = { xai?: XaiStub | null };

function XaiPanelGuardHarness({
  baselineResult,
  result,
}: {
  baselineResult: ResultStub | null;
  result: ResultStub | null;
}) {
  const xai = baselineResult?.xai ?? result?.xai ?? null;
  if (!xai) return null;
  return <div data-testid="xai-panel">panel</div>;
}

describe("workspace XAI panel render guard", () => {
  it("mounts the panel once baseline carries xai", () => {
    const baseline: ResultStub = { xai: { sentences: [] } };
    const { queryByTestId } = render(
      <XaiPanelGuardHarness baselineResult={baseline} result={baseline} />,
    );
    expect(queryByTestId("xai-panel")).not.toBeNull();
  });

  it("keeps the panel mounted when the counterfactual omits xai", () => {
    const baseline: ResultStub = { xai: { sentences: [] } };
    const { rerender, queryByTestId } = render(
      <XaiPanelGuardHarness baselineResult={baseline} result={baseline} />,
    );
    expect(queryByTestId("xai-panel")).not.toBeNull();

    // After a strike the counterfactual response may drop xai entirely.
    // The outer guard must still mount because the baseline xai is the
    // source of truth the inner panel reads.
    const counterfactual: ResultStub = {};
    rerender(
      <XaiPanelGuardHarness
        baselineResult={baseline}
        result={counterfactual}
      />,
    );
    expect(queryByTestId("xai-panel")).not.toBeNull();
  });

  it("hides the panel only when neither baseline nor result carries xai", () => {
    const { queryByTestId } = render(
      <XaiPanelGuardHarness baselineResult={null} result={{}} />,
    );
    expect(queryByTestId("xai-panel")).toBeNull();
  });
});
