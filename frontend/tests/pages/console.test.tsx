import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const replaceSpy = vi.fn();

vi.mock("next/router", () => ({
  useRouter: () => ({
    isReady: true,
    asPath: "/console",
    query: {},
    replace: replaceSpy,
    push: vi.fn(),
  }),
}));

describe("/console redirect shim", () => {
  it("forwards to the Research Terminal tab on mount", async () => {
    replaceSpy.mockClear();
    const { default: ConsoleRedirectPage } = await import("@/pages/console");
    render(<ConsoleRedirectPage />);
    expect(replaceSpy).toHaveBeenCalledTimes(1);
    const arg = replaceSpy.mock.calls[0][0];
    if (typeof arg === "string") {
      expect(arg).toBe("/research?tab=terminal");
    } else {
      expect(arg).toEqual({
        pathname: "/research",
        query: { tab: "terminal" },
      });
    }
  });

  it("renders a non-crashing fallback while the redirect resolves", async () => {
    replaceSpy.mockClear();
    const { default: ConsoleRedirectPage } = await import("@/pages/console");
    render(<ConsoleRedirectPage />);
    expect(screen.getByText(/Redirecting to the Research console/i)).toBeInTheDocument();
  });
});
