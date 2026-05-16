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
});
