import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { EvidenceLink } from "@/components/analyze/EvidenceLink";

describe("EvidenceLink", () => {
  it("renders a chip pointing at the deep-learning wiki page", () => {
    render(<EvidenceLink section="6.15" label="Three-way comparison" />);
    const link = screen.getByRole("link", { name: /evidence · §6\.15/i });
    expect(link).toHaveAttribute(
      "href",
      expect.stringContaining("github.com/yusufizzetmurat/fed-pulse/wiki/06-Deep-Learning-Roadmap"),
    );
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });

  it("appends a sub-heading anchor when caller passes one", () => {
    render(
      <EvidenceLink section="6.7" label="Honest headline reporting" anchor="#honest-headline-reporting" />,
    );
    const link = screen.getByRole("link", { name: /evidence · §6\.7/i });
    expect(link.getAttribute("href")).toMatch(/#honest-headline-reporting$/);
  });
});
