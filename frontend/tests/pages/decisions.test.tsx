import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

vi.mock("next/router", () => ({
  useRouter: () => ({ isReady: true, query: {}, push: vi.fn() }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchNextFomcForecastMock = vi.fn();
vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchNextFomcForecast: (...args: unknown[]) => fetchNextFomcForecastMock(...args),
}));

const ORDINAL_CLASSES = ["cut_50", "cut_25", "hold", "hike_25", "hike_50", "hike_75"];

const EMPTY_RESPONSE = {
  available: false,
  artifacts_dir: "/data/artifacts/next_fomc",
  ordinal_classes: ORDINAL_CLASSES,
  model_names: [],
  upcoming_meeting: {
    meeting_date: "2026-06-16",
    meeting_type: "scheduled",
    statement_release_date: "2026-06-17",
    days_until: 31,
  },
  headline: null,
  history: [],
  metrics_full_window: {},
  metrics_ex_pandemic: {},
  feature_attribution: [],
  summary: {},
};

const POPULATED_RESPONSE = {
  available: true,
  artifacts_dir: "/data/artifacts/next_fomc",
  ordinal_classes: ORDINAL_CLASSES,
  model_names: ["ordinal_logit", "ois_baseline"],
  upcoming_meeting: {
    meeting_date: "2026-06-16",
    meeting_type: "scheduled",
    statement_release_date: "2026-06-17",
    days_until: 31,
  },
  headline: {
    target_event_date: "2026-06-16",
    target_as_of_ts: "2026-06-16T19:00:00+00:00",
    target_class: null,
    n_train_rows: 25,
    probabilities: {
      ordinal_logit: {
        cut_50: 0.02,
        cut_25: 0.18,
        hold: 0.55,
        hike_25: 0.2,
        hike_50: 0.04,
        hike_75: 0.01,
      },
      ois_baseline: {
        cut_50: 0.03,
        cut_25: 0.22,
        hold: 0.5,
        hike_25: 0.2,
        hike_50: 0.04,
        hike_75: 0.01,
      },
    },
    predicted_class: { ordinal_logit: "hold", ois_baseline: "hold" },
  },
  history: [
    {
      target_event_date: "2024-09-17",
      target_as_of_ts: "2024-09-17T19:00:00+00:00",
      target_class: "cut_25",
      n_train_rows: 12,
      probabilities: {
        ordinal_logit: {
          cut_50: 0.05,
          cut_25: 0.55,
          hold: 0.3,
          hike_25: 0.07,
          hike_50: 0.02,
          hike_75: 0.01,
        },
      },
      predicted_class: { ordinal_logit: "cut_25" },
    },
    {
      target_event_date: "2024-11-06",
      target_as_of_ts: "2024-11-06T19:00:00+00:00",
      target_class: "hold",
      n_train_rows: 13,
      probabilities: {
        ordinal_logit: {
          cut_50: 0.01,
          cut_25: 0.1,
          hold: 0.7,
          hike_25: 0.15,
          hike_50: 0.03,
          hike_75: 0.01,
        },
      },
      predicted_class: { ordinal_logit: "hold" },
    },
  ],
  metrics_full_window: {
    ordinal_logit: {
      n: 18,
      brier: 0.32,
      log_loss: 0.71,
      top1_accuracy: 0.61,
      macro_f1: 0.43,
      confusion_matrix: {},
    },
    ois_baseline: {
      n: 18,
      brier: 0.45,
      log_loss: 0.95,
      top1_accuracy: 0.5,
      macro_f1: 0.32,
      confusion_matrix: {},
    },
  },
  metrics_ex_pandemic: {},
  feature_attribution: [
    {
      subset: "ois_only",
      families: ["ois"],
      n_features: 10,
      n: 18,
      brier: 0.42,
      log_loss: 0.92,
      top1_accuracy: 0.5,
      macro_f1: 0.31,
    },
    {
      subset: "full",
      families: ["ois", "text", "linguistic", "credibility", "macro"],
      n_features: 39,
      n: 18,
      brier: 0.32,
      log_loss: 0.71,
      top1_accuracy: 0.61,
      macro_f1: 0.43,
    },
  ],
  summary: { rows_emitted: 18 },
};

describe("DecisionsPage", () => {
  beforeEach(() => {
    fetchNextFomcForecastMock.mockReset();
  });

  it("renders the empty state with the make instruction", async () => {
    fetchNextFomcForecastMock.mockResolvedValue(EMPTY_RESPONSE);
    const { default: DecisionsPage } = await import("@/pages/decisions");
    render(<DecisionsPage />);
    await waitFor(() => expect(screen.getByText(/No forecast available/i)).toBeInTheDocument());
    expect(screen.getByText(/make next-fomc/)).toBeInTheDocument();
    expect(screen.getByText(/2026-06-16/)).toBeInTheDocument();
  });

  it("renders the headline prediction with primary and baseline probabilities", async () => {
    fetchNextFomcForecastMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: DecisionsPage } = await import("@/pages/decisions");
    render(<DecisionsPage />);
    await waitFor(() => expect(screen.getByText(/Next meeting/i)).toBeInTheDocument());
    expect(screen.getAllByText(/ordinal_logit/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/OIS-implied baseline/i).length).toBeGreaterThan(0);
    // Hold prediction surfaces in the headline description (55%).
    expect(screen.getAllByText(/Hold/).length).toBeGreaterThan(0);
  });

  it("renders the feature-attribution table", async () => {
    fetchNextFomcForecastMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: DecisionsPage } = await import("@/pages/decisions");
    render(<DecisionsPage />);
    await waitFor(() => expect(screen.getByText(/Next meeting/i)).toBeInTheDocument());
    // Switch to the attribution tab.
    const user = userEvent.setup();
    const trigger = screen.getByRole("tab", { name: /Attribution/i });
    await user.click(trigger);
    await waitFor(() => expect(screen.getByText(/ois_only/)).toBeInTheDocument());
    expect(screen.getByText(/^full$/)).toBeInTheDocument();
    expect(screen.getByText(/ois, text, linguistic, credibility, macro/)).toBeInTheDocument();
  });

  it("renders the OIS-implied baseline alongside the primary headline", async () => {
    fetchNextFomcForecastMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: DecisionsPage } = await import("@/pages/decisions");
    render(<DecisionsPage />);
    await waitFor(() =>
      expect(screen.getByText(/OIS-implied baseline/i)).toBeInTheDocument(),
    );
    // The headline description mentions the active primary model.
    expect(screen.getAllByText(/ordinal_logit/i).length).toBeGreaterThan(0);
  });

  it("renders the past-12-meetings hit-rate from history rows", async () => {
    fetchNextFomcForecastMock.mockResolvedValue(POPULATED_RESPONSE);
    const { default: DecisionsPage } = await import("@/pages/decisions");
    render(<DecisionsPage />);
    // History tab is the default. Hit-rate fixture: 2 resolved meetings, both
    // ordinal_logit predictions match the realised class -> 100%.
    await waitFor(() =>
      expect(screen.getByText(/Past 12 meetings/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/2\/2/)).toBeInTheDocument();
  });
});
