import { describe, expect, it, vi } from "vitest";
import { render as rtlRender, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AnalyzeForm } from "@/components/analyze/AnalyzeForm";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { AnalyzeRequest } from "@/lib/analyze/types";

// AnalyzeForm uses Tooltip primitives for the picker-scope info icons
// (#474/#475). Radix tooltips require a TooltipProvider ancestor; the
// production app mounts one at the page root (see pages/_app.js).
function render(ui: React.ReactElement) {
  return rtlRender(<TooltipProvider>{ui}</TooltipProvider>);
}

function baseRequest(): AnalyzeRequest {
  return {
    text: "Recent indicators…",
    date: "2024-09-18",
    symbol: "^GSPC",
    horizon: "3d",
    include_realized: false,
  };
}

describe("AnalyzeForm", () => {
  it("renders core fields with current values", () => {
    render(
      <AnalyzeForm value={baseRequest()} onChange={vi.fn()} onSubmit={vi.fn()} loading={false} />
    );
    expect(screen.getByLabelText("FOMC text")).toHaveValue("Recent indicators…");
    expect(screen.getByLabelText("Document date")).toHaveValue("2024-09-18");
    expect(screen.getByRole("button", { name: /analyze/i })).toBeEnabled();
  });

  it("calls onChange when the text is edited", () => {
    const onChange = vi.fn();
    render(
      <AnalyzeForm value={baseRequest()} onChange={onChange} onSubmit={vi.fn()} loading={false} />
    );
    fireEvent.change(screen.getByLabelText("FOMC text"), { target: { value: "Updated text" } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ text: "Updated text" }));
  });

  it("calls onSubmit on form submission", () => {
    const onSubmit = vi.fn();
    render(
      <AnalyzeForm value={baseRequest()} onChange={vi.fn()} onSubmit={onSubmit} loading={false} />
    );
    fireEvent.submit(screen.getByLabelText("FOMC text").closest("form")!);
    expect(onSubmit).toHaveBeenCalled();
  });

  it("disables the submit button while loading", () => {
    render(
      <AnalyzeForm
        value={baseRequest()}
        onChange={vi.fn()}
        onSubmit={vi.fn()}
        loading
      />
    );
    expect(screen.getByRole("button", { name: /running analysis/i })).toBeDisabled();
  });

  it("exposes picker-scope info icons next to the Asset and Horizon labels (#474/#475)", () => {
    render(
      <AnalyzeForm value={baseRequest()} onChange={vi.fn()} onSubmit={vi.fn()} loading={false} />
    );
    expect(
      screen.getByLabelText("What does the asset picker affect?")
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("What does the horizon picker affect?")
    ).toBeInTheDocument();
  });

  it("asset tooltip reveals the scope-clarification text on hover (#475)", async () => {
    const user = userEvent.setup();
    render(
      <AnalyzeForm value={baseRequest()} onChange={vi.fn()} onSubmit={vi.fn()} loading={false} />
    );
    await user.hover(
      screen.getByLabelText("What does the asset picker affect?")
    );
    // Radix renders the tooltip content in a portal AND in an
    // aria-describedby region for screen readers, so findByText matches
    // multiple nodes by design. findAllByText asserts at least one of
    // them surfaces the scope-clarification copy — that's the contract
    // the user actually experiences (the substring is unique to this
    // tooltip, so >=1 match is the right invariant).
    const matches = await screen.findAllByText(/asset-independent/i);
    expect(matches.length).toBeGreaterThan(0);
  });

  it("horizon tooltip reveals the scope-clarification text on hover (#474)", async () => {
    const user = userEvent.setup();
    render(
      <AnalyzeForm value={baseRequest()} onChange={vi.fn()} onSubmit={vi.fn()} loading={false} />
    );
    await user.hover(
      screen.getByLabelText("What does the horizon picker affect?")
    );
    const matches = await screen.findAllByText(/autoregressive prediction block/i);
    expect(matches.length).toBeGreaterThan(0);
  });
});
