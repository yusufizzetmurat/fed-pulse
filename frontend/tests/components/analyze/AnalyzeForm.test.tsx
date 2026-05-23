import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { AnalyzeForm } from "@/components/analyze/AnalyzeForm";
import type { AnalyzeRequest } from "@/lib/analyze/types";

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
});
