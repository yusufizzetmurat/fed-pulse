import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { MultiAxisCards } from "@/components/analyze/MultiAxisCards";
import { SAMPLE_MULTI_AXIS } from "@/lib/analyze/fixtures";

const EMPTY_AXIS = { stance: null, factor: null, certainty: null, topic: null };

describe("MultiAxisCards", () => {
  it("renders four cards from the canonical fixture", () => {
    render(<MultiAxisCards multiAxis={SAMPLE_MULTI_AXIS} />);
    expect(screen.getAllByText(/^stance$/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/^factor$/i)).toBeInTheDocument();
    expect(screen.getByText(/^certainty$/i)).toBeInTheDocument();
    expect(screen.getByText(/^topic$/i)).toBeInTheDocument();
    expect(screen.getAllByText(/hawkish/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/^macro$/i)).toBeInTheDocument();
  });

  it("returns null when every axis is null", () => {
    const { container } = render(<MultiAxisCards multiAxis={EMPTY_AXIS} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("hides factor / certainty / topic when only stance is available", () => {
    render(
      <MultiAxisCards
        multiAxis={{
          stance: SAMPLE_MULTI_AXIS.stance,
          factor: null,
          certainty: null,
          topic: null,
        }}
      />,
    );
    expect(screen.queryByText(/^factor$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^certainty$/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^topic$/i)).not.toBeInTheDocument();
    expect(screen.getAllByText(/^stance$/i).length).toBeGreaterThan(0);
  });
});
