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

  it("renders Statement / Minutes / Presser availability badges with both states", async () => {
    fetchFomcCalendarMock.mockResolvedValue({
      upcoming: [
        {
          meeting_date: "2024-11-06",
          meeting_type: "scheduled",
          statement_release_date: "2024-11-07",
          minutes_release_date: "2024-11-27",
          statement_available: false,
          minutes_available: false,
          press_conference_available: false,
        },
      ],
      past: [
        {
          meeting_date: "2024-09-17",
          meeting_type: "scheduled",
          statement_release_date: "2024-09-18",
          minutes_release_date: "2024-10-09",
          statement_available: true,
          minutes_available: true,
          press_conference_available: false,
        },
      ],
    });
    const { default: CalendarPage } = await import("@/pages/calendar");
    render(<CalendarPage />);

    await waitFor(() =>
      expect(screen.getAllByTestId("availability-statement").length).toBe(2),
    );

    // Past row (2024-09-17): statement + minutes on file, presser missing.
    const statementBadges = screen.getAllByTestId("availability-statement");
    const minutesBadges = screen.getAllByTestId("availability-minutes");
    const presserBadges = screen.getAllByTestId("availability-presser");
    expect(statementBadges).toHaveLength(2);
    expect(minutesBadges).toHaveLength(2);
    expect(presserBadges).toHaveLength(2);

    // Identify which badge belongs to which row by walking up to the
    // <li> that holds the meeting_date text.
    function badgeForMeeting(
      badges: HTMLElement[],
      meetingDate: string,
    ): HTMLElement {
      const match = badges.find((badge) => {
        const row = badge.closest("li");
        return row?.textContent?.includes(meetingDate) ?? false;
      });
      if (!match) throw new Error(`no badge near ${meetingDate}`);
      return match;
    }

    const pastStatement = badgeForMeeting(statementBadges, "2024-09-17");
    const pastMinutes = badgeForMeeting(minutesBadges, "2024-09-17");
    const pastPresser = badgeForMeeting(presserBadges, "2024-09-17");
    expect(pastStatement).toHaveAttribute("data-available", "true");
    expect(pastStatement).toHaveAttribute("title", "Statement on file");
    expect(pastMinutes).toHaveAttribute("data-available", "true");
    expect(pastMinutes).toHaveAttribute("title", "Minutes on file");
    expect(pastPresser).toHaveAttribute("data-available", "false");
    expect(pastPresser).toHaveAttribute("title", "Presser not collected");

    // Upcoming row (2024-11-06): all three muted/not-collected.
    const upStatement = badgeForMeeting(statementBadges, "2024-11-06");
    const upMinutes = badgeForMeeting(minutesBadges, "2024-11-06");
    const upPresser = badgeForMeeting(presserBadges, "2024-11-06");
    expect(upStatement).toHaveAttribute("data-available", "false");
    expect(upMinutes).toHaveAttribute("data-available", "false");
    expect(upPresser).toHaveAttribute("data-available", "false");
    expect(upStatement).toHaveAttribute("title", "Statement not collected");
  });

  it("renders available badges as click-through links and unavailable as plain spans", async () => {
    fetchFomcCalendarMock.mockResolvedValue({
      upcoming: [
        {
          meeting_date: "2024-11-06",
          meeting_type: "scheduled",
          statement_release_date: "2024-11-07",
          minutes_release_date: "2024-11-27",
          statement_available: false,
          minutes_available: false,
          press_conference_available: false,
        },
      ],
      past: [
        {
          meeting_date: "2024-09-17",
          meeting_type: "scheduled",
          statement_release_date: "2024-09-18",
          minutes_release_date: "2024-10-09",
          statement_available: true,
          minutes_available: true,
          press_conference_available: false,
        },
      ],
    });
    const { default: CalendarPage } = await import("@/pages/calendar");
    render(<CalendarPage />);

    await waitFor(() =>
      expect(screen.getAllByTestId("availability-statement").length).toBe(2),
    );

    function badgeForMeeting(
      badges: HTMLElement[],
      meetingDate: string,
    ): HTMLElement {
      const match = badges.find((badge) => {
        const row = badge.closest("li");
        return row?.textContent?.includes(meetingDate) ?? false;
      });
      if (!match) throw new Error(`no badge near ${meetingDate}`);
      return match;
    }

    const statementBadges = screen.getAllByTestId("availability-statement");
    const minutesBadges = screen.getAllByTestId("availability-minutes");
    const presserBadges = screen.getAllByTestId("availability-presser");

    // Past row has on-file statement + minutes — both should be anchor
    // tags pointing at the path-based viewer keyed off the per-kind
    // release date. statement uses statement_release_date (day-2 of the
    // two-day meeting); minutes uses minutes_release_date (~21 days
    // later); presser falls back to statement_release_date since the
    // press conference happens on the meeting's concluding day. A
    // blanket fallback to a single date would 404 the minutes click
    // every time the two release dates differ.
    const pastStatement = badgeForMeeting(statementBadges, "2024-09-17");
    expect(pastStatement.tagName).toBe("A");
    expect(pastStatement).toHaveAttribute(
      "href",
      "/documents/statement/2024-09-18",
    );

    const pastMinutes = badgeForMeeting(minutesBadges, "2024-09-17");
    expect(pastMinutes.tagName).toBe("A");
    expect(pastMinutes).toHaveAttribute(
      "href",
      "/documents/minutes/2024-10-09",
    );

    // Past row presser is not on file — must stay a span.
    const pastPresser = badgeForMeeting(presserBadges, "2024-09-17");
    expect(pastPresser.tagName).toBe("SPAN");
    expect(pastPresser).not.toHaveAttribute("href");

    // Upcoming row has nothing on file — all three stay spans.
    const upStatement = badgeForMeeting(statementBadges, "2024-11-06");
    const upMinutes = badgeForMeeting(minutesBadges, "2024-11-06");
    const upPresser = badgeForMeeting(presserBadges, "2024-11-06");
    expect(upStatement.tagName).toBe("SPAN");
    expect(upMinutes.tagName).toBe("SPAN");
    expect(upPresser.tagName).toBe("SPAN");
  });

  it("treats undefined availability flags on a future meeting as not-collected", async () => {
    // Real backend payloads for far-future meetings omit the
    // availability flags entirely; the badge logic must read undefined
    // as false rather than rendering an aria-broken or red "missing"
    // state.
    fetchFomcCalendarMock.mockResolvedValue({
      upcoming: [
        {
          meeting_date: "2026-12-15",
          meeting_type: "scheduled",
          statement_release_date: "2026-12-15",
          minutes_release_date: "2027-01-06",
          // availability flags intentionally omitted
        },
      ],
      past: [],
    });
    const { default: CalendarPage } = await import("@/pages/calendar");
    render(<CalendarPage />);

    await waitFor(() =>
      expect(screen.getAllByTestId("availability-statement").length).toBe(1),
    );
    const statement = screen.getByTestId("availability-statement");
    const minutes = screen.getByTestId("availability-minutes");
    const presser = screen.getByTestId("availability-presser");
    expect(statement).toHaveAttribute("data-available", "false");
    expect(minutes).toHaveAttribute("data-available", "false");
    expect(presser).toHaveAttribute("data-available", "false");
    expect(statement).toHaveAttribute("title", "Statement not collected");
  });

  it("routes each badge kind to its own release date", async () => {
    // Regression: every badge kind must use the date the backend's
    // JSON cache is keyed on. statement_release_date and
    // minutes_release_date differ by ~3 weeks, so a blanket fallback
    // would 404 the minutes click. Presser shares the meeting's
    // concluding day with the statement, so it falls back to
    // statement_release_date.
    fetchFomcCalendarMock.mockResolvedValue({
      upcoming: [],
      past: [
        {
          meeting_date: "2024-09-17",
          meeting_type: "scheduled",
          statement_release_date: "2024-09-18",
          minutes_release_date: "2024-10-09",
          statement_available: true,
          minutes_available: true,
          press_conference_available: true,
        },
      ],
    });
    const { default: CalendarPage } = await import("@/pages/calendar");
    render(<CalendarPage />);

    await waitFor(() =>
      expect(screen.getAllByTestId("availability-statement").length).toBe(1),
    );
    const statement = screen.getByTestId("availability-statement");
    const minutes = screen.getByTestId("availability-minutes");
    const presser = screen.getByTestId("availability-presser");
    expect(statement.tagName).toBe("A");
    expect(statement).toHaveAttribute(
      "href",
      "/documents/statement/2024-09-18",
    );
    expect(minutes.tagName).toBe("A");
    expect(minutes).toHaveAttribute(
      "href",
      "/documents/minutes/2024-10-09",
    );
    expect(presser.tagName).toBe("A");
    expect(presser).toHaveAttribute(
      "href",
      "/documents/press_conference/2024-09-18",
    );
  });

  it("falls back to meeting_date when a per-kind release date is missing", async () => {
    // Far-future rows can ship without statement_release_date /
    // minutes_release_date. Each badge must still produce a usable href
    // rather than rendering "/documents/minutes/undefined".
    fetchFomcCalendarMock.mockResolvedValue({
      upcoming: [],
      past: [
        {
          meeting_date: "2024-09-17",
          meeting_type: "scheduled",
          // statement / minutes release dates intentionally absent
          statement_available: true,
          minutes_available: true,
          press_conference_available: true,
        },
      ],
    });
    const { default: CalendarPage } = await import("@/pages/calendar");
    render(<CalendarPage />);

    await waitFor(() =>
      expect(screen.getAllByTestId("availability-statement").length).toBe(1),
    );
    const statement = screen.getByTestId("availability-statement");
    const minutes = screen.getByTestId("availability-minutes");
    const presser = screen.getByTestId("availability-presser");
    expect(statement).toHaveAttribute(
      "href",
      "/documents/statement/2024-09-17",
    );
    expect(minutes).toHaveAttribute(
      "href",
      "/documents/minutes/2024-09-17",
    );
    expect(presser).toHaveAttribute(
      "href",
      "/documents/press_conference/2024-09-17",
    );
  });
});
