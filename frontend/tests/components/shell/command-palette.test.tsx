import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

// The global setup stubs CommandPalette to null for the rest of the
// suite (it mounts under KeyboardShortcuts and would explode in pages
// that don't supply a Next router). This file imports the real
// component, so undo the stub locally.
vi.unmock("@/components/shell/command-palette");

vi.mock("next/router", () => ({
  useRouter: () => ({
    pathname: "/",
    push: vi.fn().mockResolvedValue(true),
    replace: vi.fn().mockResolvedValue(true),
  }),
}));

const fetchSymbolsMock = vi.fn();
const fetchFomcCalendarMock = vi.fn();
const fetchHistoryMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchSymbols: (...args: unknown[]) => fetchSymbolsMock(...args),
  fetchFomcCalendar: (...args: unknown[]) => fetchFomcCalendarMock(...args),
  fetchHistory: (...args: unknown[]) => fetchHistoryMock(...args),
}));

describe("CommandPalette", () => {
  it("renders symbol entries once /symbols resolves", async () => {
    fetchSymbolsMock.mockResolvedValue({
      symbols: [
        { symbol: "^GSPC", name: "S&P 500", category: "Equity", default_horizon: "10d" },
        { symbol: "^VIX", name: "VIX", category: "Volatility", default_horizon: "5d" },
      ],
    });
    fetchFomcCalendarMock.mockResolvedValue({ upcoming: [], past: [] });
    fetchHistoryMock.mockResolvedValue({ items: [] });

    const { CommandPalette } = await import("@/components/shell/command-palette");
    render(<CommandPalette open onOpenChange={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText("^GSPC")).toBeInTheDocument();
      expect(screen.getByText("^VIX")).toBeInTheDocument();
    });
  });

  it("survives without a SymbolCalendarProvider via the standalone fallback", async () => {
    // The component pulls symbols through useSharedSymbols, which now
    // detects an absent provider and falls back to a local fetchSymbols
    // call. Renders here without wrapping in SymbolCalendarProvider.
    fetchSymbolsMock.mockResolvedValue({ symbols: [] });
    fetchFomcCalendarMock.mockResolvedValue({ upcoming: [], past: [] });
    fetchHistoryMock.mockResolvedValue({ items: [] });

    const { CommandPalette } = await import("@/components/shell/command-palette");
    render(<CommandPalette open onOpenChange={vi.fn()} />);

    expect(await screen.findByText("Command palette")).toBeInTheDocument();
    expect(screen.getByText("New analysis")).toBeInTheDocument();
  });
});
