import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("KeyboardShortcuts", () => {
  it("opens the dialog when `?` is pressed and lists the bindings", async () => {
    const { KeyboardShortcuts } = await import("@/components/shell/keyboard-shortcuts");
    render(<KeyboardShortcuts />);
    expect(screen.queryByRole("dialog")).toBeNull();

    fireEvent.keyDown(window, { key: "?" });

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(screen.getByText(/Keyboard shortcuts/i)).toBeInTheDocument();
    expect(screen.getByText(/Toggle light \/ dark theme/i)).toBeInTheDocument();
    expect(screen.getByText(/Go to Analyze/i)).toBeInTheDocument();
  });

  it("closes the dialog on Escape", async () => {
    const { KeyboardShortcuts } = await import("@/components/shell/keyboard-shortcuts");
    render(<KeyboardShortcuts />);
    fireEvent.keyDown(window, { key: "?" });
    await screen.findByRole("dialog");

    fireEvent.keyDown(window, { key: "Escape" });
    // Radix unmounts on close after a tick — assert the dialog content has gone away.
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(screen.queryByText(/Keyboard shortcuts/i)).toBeNull();
  });

  it("ignores `?` when the focus is inside a text input", async () => {
    const { KeyboardShortcuts } = await import("@/components/shell/keyboard-shortcuts");
    render(
      <>
        <input data-testid="input" />
        <KeyboardShortcuts />
      </>
    );
    const input = screen.getByTestId("input");
    input.focus();
    fireEvent.keyDown(input, { key: "?" });
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("dispatches the nav target when `g` is followed by `a`", async () => {
    const { KeyboardShortcuts } = await import("@/components/shell/keyboard-shortcuts");
    const assign = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, assign },
    });
    render(<KeyboardShortcuts />);

    fireEvent.keyDown(window, { key: "g" });
    fireEvent.keyDown(window, { key: "a" });

    expect(assign).toHaveBeenCalledTimes(1);
    expect(assign).toHaveBeenCalledWith("/analyze");
  });

  it("does not dispatch nav when the `g` window has expired", async () => {
    vi.useFakeTimers();
    const { KeyboardShortcuts } = await import("@/components/shell/keyboard-shortcuts");
    const assign = vi.fn();
    Object.defineProperty(window, "location", {
      configurable: true,
      value: { ...window.location, assign },
    });
    render(<KeyboardShortcuts />);

    fireEvent.keyDown(window, { key: "g" });
    // Advance past the 1200 ms g-sequence window before the nav key.
    vi.advanceTimersByTime(1500);
    fireEvent.keyDown(window, { key: "a" });

    expect(assign).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("clicks the theme toggle when `t` is pressed", async () => {
    const { KeyboardShortcuts } = await import("@/components/shell/keyboard-shortcuts");
    const click = vi.fn();
    const toggle = document.createElement("button");
    toggle.setAttribute("aria-label", "Toggle theme");
    toggle.addEventListener("click", click);
    document.body.appendChild(toggle);

    render(<KeyboardShortcuts />);
    fireEvent.keyDown(window, { key: "t" });

    expect(click).toHaveBeenCalledTimes(1);
    document.body.removeChild(toggle);
  });
});
