import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

// #342: settings-page coverage for the inference-contract badge surface.
// Renders the settings page against three checkpoint shapes:
//
//   - sidecar present + all required kwargs supplied -> neutral outline
//     badges per kwarg, no red.
//   - sidecar present + an unknown kwarg declared -> red ``hawkish``
//     badge on the unknown kwarg via the testid the row exposes.
//   - sidecar absent -> single "legacy" neutral badge.

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

vi.mock("next/router", () => ({
  useRouter: () => ({
    isReady: true,
    query: {},
    pathname: "/settings",
    asPath: "/settings",
    push: vi.fn(),
    replace: vi.fn(),
  }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "dark", setTheme: vi.fn(), resolvedTheme: "dark" }),
}));

const fetchSettingsCheckpointsMock = vi.fn();
vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchSettingsCheckpoints: (...args: unknown[]) =>
    fetchSettingsCheckpointsMock(...args),
}));

// Stub the workspace-prefs surface so the settings page can render
// without the AssetPicker firing a real /symbols probe.
vi.mock("@/lib/workspace-prefs", () => ({
  DEFAULT_HORIZON: "3d",
  DEFAULT_SYMBOL: "^GSPC",
  HORIZON_VALUES: ["1d", "3d", "5d", "10d"],
  loadWorkspacePrefs: () => ({ defaultSymbol: "^GSPC", defaultHorizon: "3d" }),
  saveWorkspacePrefs: vi.fn(),
}));

vi.mock("@/components/analyze/AssetPicker", () => ({
  AssetPicker: () => <div data-testid="asset-picker" />,
}));

describe("SettingsPage — inference contract badges", () => {
  beforeEach(() => {
    fetchSettingsCheckpointsMock.mockReset();
  });

  it("renders a red badge when the sidecar declares an unknown kwarg", async () => {
    fetchSettingsCheckpointsMock.mockResolvedValue({
      models_dir: "/tmp/models",
      checkpoints: [
        {
          filename: "forecaster_best.pt",
          relative_path: "forecaster_best.pt",
          role: "forecaster",
          size_bytes: 1024,
          modified_at: "2026-05-15T10:00:00Z",
          is_active: true,
          output_mode: "regression",
          encoder_alias: "bert_base",
          conformal_sidecar_present: true,
          inference_contract_status: "present",
          required_kwargs: ["text_embedding", "unknown_kwarg"],
          supplied_at_inference: {
            text_embedding: true,
            unknown_kwarg: false,
          },
        },
      ],
    });

    const { default: SettingsPage } = await import("@/pages/settings");
    render(<SettingsPage />);
    await waitFor(() =>
      expect(screen.getByText("forecaster_best.pt")).toBeInTheDocument(),
    );

    // The supplied kwarg renders as a neutral / outline badge, the
    // missing one renders red. We assert via the testids the row
    // exposes so we are not coupled to the variant class names.
    expect(
      screen.getByTestId("contract-kwarg-ok-text_embedding"),
    ).toBeInTheDocument();
    const missing = screen.getByTestId("contract-kwarg-missing-unknown_kwarg");
    expect(missing).toBeInTheDocument();
    // ``hawkish`` is the red Fed-context variant — assert the class to
    // pin the red-badge surface concretely.
    expect(missing.className).toMatch(/hawkish/);
  });

  it("renders a neutral legacy badge for a sidecar_absent checkpoint", async () => {
    fetchSettingsCheckpointsMock.mockResolvedValue({
      models_dir: "/tmp/models",
      checkpoints: [
        {
          filename: "forecaster_legacy.pt",
          relative_path: "forecaster_legacy.pt",
          role: "forecaster",
          size_bytes: 2048,
          modified_at: "2025-01-01T10:00:00Z",
          is_active: false,
          output_mode: null,
          encoder_alias: null,
          conformal_sidecar_present: false,
          inference_contract_status: "sidecar_absent",
          required_kwargs: [],
          supplied_at_inference: {},
        },
      ],
    });

    const { default: SettingsPage } = await import("@/pages/settings");
    render(<SettingsPage />);
    await waitFor(() =>
      expect(screen.getByText("forecaster_legacy.pt")).toBeInTheDocument(),
    );

    const legacy = screen.getByTestId("contract-legacy-badge");
    expect(legacy).toBeInTheDocument();
    expect(legacy.textContent).toMatch(/legacy/i);
    expect(legacy.className).not.toMatch(/hawkish/);
  });

  it("does not render any kwarg badge when no kwargs are required", async () => {
    fetchSettingsCheckpointsMock.mockResolvedValue({
      models_dir: "/tmp/models",
      checkpoints: [
        {
          filename: "forecaster_minimal.pt",
          relative_path: "forecaster_minimal.pt",
          role: "forecaster",
          size_bytes: 512,
          modified_at: "2026-04-01T10:00:00Z",
          is_active: false,
          output_mode: "regression",
          encoder_alias: null,
          conformal_sidecar_present: true,
          inference_contract_status: "present",
          required_kwargs: [],
          supplied_at_inference: {},
        },
      ],
    });

    const { default: SettingsPage } = await import("@/pages/settings");
    render(<SettingsPage />);
    await waitFor(() =>
      expect(screen.getByText("forecaster_minimal.pt")).toBeInTheDocument(),
    );

    expect(screen.queryByTestId(/contract-kwarg-/)).not.toBeInTheDocument();
    expect(screen.queryByTestId("contract-legacy-badge")).not.toBeInTheDocument();
    expect(screen.getByText(/no required inputs/i)).toBeInTheDocument();
  });
});
