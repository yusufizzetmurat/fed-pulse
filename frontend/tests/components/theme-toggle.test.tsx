import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const setTheme = vi.fn();
let mockTheme = "dark";

vi.mock("next-themes", () => ({
  useTheme: () => ({
    theme: mockTheme,
    setTheme,
    resolvedTheme: mockTheme,
  }),
}));

afterEach(() => {
  setTheme.mockReset();
  mockTheme = "dark";
});

describe("ThemeToggle", () => {
  it("renders an enabled toggle button after mount", async () => {
    const { ThemeToggle } = await import("@/components/theme-toggle");
    render(<ThemeToggle />);
    const button = screen.getByRole("button");
    expect(button).toBeInTheDocument();
    expect(button).not.toBeDisabled();
  });

  it("switches to light when current theme is dark", async () => {
    mockTheme = "dark";
    const { ThemeToggle } = await import("@/components/theme-toggle");
    render(<ThemeToggle />);
    await userEvent.click(screen.getByRole("button"));
    expect(setTheme).toHaveBeenCalledWith("light");
  });

  it("switches to dark when current theme is light", async () => {
    mockTheme = "light";
    const { ThemeToggle } = await import("@/components/theme-toggle");
    render(<ThemeToggle />);
    await userEvent.click(screen.getByRole("button"));
    expect(setTheme).toHaveBeenCalledWith("dark");
  });
});
