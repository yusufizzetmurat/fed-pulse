import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { MultiAxisCards } from "@/components/analyze/MultiAxisCards";
import { SAMPLE_MULTI_AXIS } from "@/lib/analyze/fixtures";

describe("MultiAxisCards", () => {
  it("renders four cards from the canonical fixture", () => {
    render(<MultiAxisCards multiAxis={SAMPLE_MULTI_AXIS} />);
    expect(screen.getAllByText(/^stance$/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/^factor$/i)).toBeInTheDocument();
    expect(screen.getByText(/^certainty$/i)).toBeInTheDocument();
    expect(screen.getByText(/^topic$/i)).toBeInTheDocument();
    expect(screen.getAllByText(/hawkish/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/inflation persistence/i)).toBeInTheDocument();
  });

  it("returns null when every axis is undefined", () => {
    const { container } = render(<MultiAxisCards multiAxis={{}} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("hides factor card when factor is absent", () => {
    render(<MultiAxisCards multiAxis={{ stance: SAMPLE_MULTI_AXIS.stance }} />);
    expect(screen.queryByText(/^factor$/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/^stance$/i).length).toBeGreaterThan(0);
  });
});
