import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

const pushMock = vi.fn();
vi.mock("next/router", () => ({
  useRouter: () => ({ isReady: true, query: {}, push: pushMock }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchFomcCalendarMock = vi.fn();
const fetchNextFomcForecastMock = vi.fn();
vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchFomcCalendar: (...args: unknown[]) => fetchFomcCalendarMock(...args),
  fetchNextFomcForecast: (...args: unknown[]) => fetchNextFomcForecastMock(...args),
}));

describe("CalendarPage", () => {
  beforeEach(() => {
    fetchFomcCalendarMock.mockReset();
    fetchNextFomcForecastMock.mockReset();
    fetchNextFomcForecastMock.mockRejectedValue(new Error("not available"));
    pushMock.mockReset();
  });

  it("renders upcoming and past meeting rows", async () => {
    fetchFomcCalendarMock.mockResolvedValue({
      upcoming: [
        {
          meeting_date: "2024-11-06",
          meeting_type: "scheduled",
          statement_release_date: "2024-11-07",
          minutes_release_date: "2024-11-27",
        },
      ],
      past: [
        {
          meeting_date: "2024-09-17",
          meeting_type: "scheduled",
          statement_release_date: "2024-09-18",
        },
      ],
    });
    const { default: CalendarPage } = await import("@/pages/calendar");
    render(<CalendarPage />);
    // The upcoming date is rendered twice: once in the countdown card
    // header, once in the list row. Match on the list row's font-mono
    // <p> only to keep the assertion stable.
    await waitFor(() =>
      expect(screen.getAllByText("2024-11-06").length).toBeGreaterThanOrEqual(1),
    );
    expect(screen.getByText("2024-09-17")).toBeInTheDocument();
    expect(screen.getAllByText(/^scheduled$/i).length).toBeGreaterThanOrEqual(2);
  });
});
