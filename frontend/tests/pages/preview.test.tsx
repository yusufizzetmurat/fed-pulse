import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn(), resolvedTheme: "dark" }),
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe("Preview page", () => {
  it("renders the design-system gallery with header and primitive groups", async () => {
    const { default: PreviewPage } = await import("@/pages/preview");
    render(<PreviewPage />);
    expect(screen.getByText("Fed Pulse")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /design system/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /primitives/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /form/i })).toBeInTheDocument();
  });

  it("includes the hawkish, dovish, and neutral stance badges by default", async () => {
    const { default: PreviewPage } = await import("@/pages/preview");
    render(<PreviewPage />);
    expect(screen.getByText(/hawkish · 0\.62/i)).toBeInTheDocument();
    expect(screen.getByText(/dovish · 0\.18/i)).toBeInTheDocument();
    expect(screen.getByText(/neutral · 0\.20/i)).toBeInTheDocument();
  });
});
