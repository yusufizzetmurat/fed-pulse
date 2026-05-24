import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Page-level tests mock `@/lib/analyze/api` with explicit exports that
// don't include `fetchFomcCalendar`; the StatusBar that now lives under
// every page header fetches the calendar on mount, which raises a vitest
// "no such export on the mock" error. Stubbing the StatusBar to a no-op
// keeps the existing mocks intact and removes a non-essential network
// call from the test environment.
vi.mock("@/components/shell/status-bar", () => ({
  StatusBar: () => null,
}));

afterEach(() => {
  cleanup();
});
