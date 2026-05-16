import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { SkipLink } from "@/components/shell/skip-link";

describe("SkipLink", () => {
  it("renders a link pointing at #main-content", () => {
    render(<SkipLink />);
    const link = screen.getByRole("link", { name: /skip to main content/i });
    expect(link).toHaveAttribute("href", "#main-content");
  });

  it("sits first in the tab order when rendered before other interactive nodes", () => {
    render(
      <>
        <SkipLink />
        <button type="button">After</button>
      </>
    );
    const link = screen.getByRole("link", { name: /skip to main content/i });
    const button = screen.getByRole("button", { name: /after/i });
    expect(link.compareDocumentPosition(button) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});
