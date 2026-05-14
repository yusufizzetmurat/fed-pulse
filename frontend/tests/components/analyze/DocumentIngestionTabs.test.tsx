import { describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

import { DocumentIngestionTabs } from "@/components/analyze/DocumentIngestionTabs";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn() },
}));

vi.mock("axios", () => ({
  default: { post: vi.fn() },
}));

describe("DocumentIngestionTabs", () => {
  it("renders the three ingestion tabs and the paste textarea", () => {
    render(<DocumentIngestionTabs text="" onChange={vi.fn()} />);
    expect(screen.getByRole("tab", { name: /paste/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /pdf \/ docx/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /url/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/fomc text/i)).toBeInTheDocument();
  });

  it("propagates textarea edits through onChange", () => {
    const onChange = vi.fn();
    render(<DocumentIngestionTabs text="" onChange={onChange} />);
    fireEvent.change(screen.getByLabelText(/fomc text/i), {
      target: { value: "Recent indicators…" },
    });
    expect(onChange).toHaveBeenCalledWith("Recent indicators…");
  });
});
