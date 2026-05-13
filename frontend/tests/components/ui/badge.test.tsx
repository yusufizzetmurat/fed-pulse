import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { Badge } from "@/components/ui/badge";

describe("Badge", () => {
  it("renders the label", () => {
    render(<Badge>Hawkish</Badge>);
    expect(screen.getByText("Hawkish")).toBeInTheDocument();
  });

  it("applies the hawkish finance variant", () => {
    render(<Badge variant="hawkish">Hawkish</Badge>);
    expect(screen.getByText("Hawkish")).toHaveClass("text-hawkish");
  });

  it("applies the dovish finance variant", () => {
    render(<Badge variant="dovish">Dovish</Badge>);
    expect(screen.getByText("Dovish")).toHaveClass("text-dovish");
  });

  it("applies the neutral finance variant", () => {
    render(<Badge variant="neutral">Neutral</Badge>);
    expect(screen.getByText("Neutral")).toHaveClass("text-neutral");
  });

  it("defaults to the primary variant", () => {
    render(<Badge>Default</Badge>);
    expect(screen.getByText("Default")).toHaveClass("bg-primary");
  });
});
