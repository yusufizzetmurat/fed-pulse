import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("sonner", () => ({
  toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() },
  Toaster: () => null,
}));

let mockQuery: Record<string, string> = {};

vi.mock("next/router", () => ({
  useRouter: () => ({
    isReady: true,
    query: mockQuery,
    push: vi.fn(),
  }),
}));

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const fetchDocumentDetailMock = vi.fn();

vi.mock("@/lib/analyze/api", () => ({
  resolveApiBaseUrl: () => "http://localhost:8000",
  fetchDocumentDetail: (...args: unknown[]) => fetchDocumentDetailMock(...args),
}));

describe("DocumentDetailPage", () => {
  beforeEach(() => {
    mockQuery = { type: "statement", date: "2024-09-18" };
    fetchDocumentDetailMock.mockReset();
  });

  it("renders the loading skeleton before the fetch resolves", async () => {
    let resolveFetch!: (value: unknown) => void;
    fetchDocumentDetailMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveFetch = resolve;
        }),
    );
    const { default: DocumentDetailPage } = await import(
      "@/pages/documents/[type]/[date]"
    );
    const { container } = render(<DocumentDetailPage />);
    // Skeletons render as div.animate-pulse — their presence is the
    // observable proof the loading branch is rendering.
    expect(container.querySelectorAll(".animate-pulse").length).toBeGreaterThan(0);
    expect(screen.queryByTestId("document-body")).toBeNull();
    resolveFetch?.({
      type: "statement",
      date: "2024-09-18",
      title: "Federal Reserve issues FOMC statement",
      cleaned_text: "Body.",
      source_url: null,
      scraped_at: null,
    });
    await waitFor(() =>
      expect(screen.getByTestId("document-body")).toBeInTheDocument(),
    );
  });

  it("renders the populated document body with header chips and view-source link", async () => {
    fetchDocumentDetailMock.mockResolvedValue({
      type: "statement",
      date: "2024-09-18",
      title: "Federal Reserve issues FOMC statement",
      cleaned_text:
        "The Committee decided to lower the target range for the federal funds rate.",
      source_url:
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
      scraped_at: "2026-05-30T00:00:00+00:00",
    });
    const { default: DocumentDetailPage } = await import(
      "@/pages/documents/[type]/[date]"
    );
    render(<DocumentDetailPage />);

    await waitFor(() =>
      expect(screen.getByTestId("document-body")).toBeInTheDocument(),
    );
    expect(
      screen.getByText(/target range for the federal funds rate/i),
    ).toBeInTheDocument();
    // Header chips: type label + ISO date.
    expect(screen.getByText("Statement")).toBeInTheDocument();
    expect(screen.getByText("2024-09-18")).toBeInTheDocument();
    // View-source anchor.
    const viewSource = screen.getByRole("link", { name: /view source/i });
    expect(viewSource).toHaveAttribute(
      "href",
      "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
    );
    expect(viewSource).toHaveAttribute("target", "_blank");
  });

  it("renders the 404 not-found banner when the API returns null", async () => {
    fetchDocumentDetailMock.mockResolvedValue(null);
    const { default: DocumentDetailPage } = await import(
      "@/pages/documents/[type]/[date]"
    );
    render(<DocumentDetailPage />);
    await waitFor(() =>
      expect(screen.getByTestId("document-not-found")).toBeInTheDocument(),
    );
    expect(screen.getByText(/Document not on file/i)).toBeInTheDocument();
    expect(screen.queryByTestId("document-body")).toBeNull();
  });

  it("renders the error banner on a non-404 backend failure", async () => {
    fetchDocumentDetailMock.mockRejectedValue({
      response: { status: 500, data: { detail: "boom" } },
    });
    const { default: DocumentDetailPage } = await import(
      "@/pages/documents/[type]/[date]"
    );
    render(<DocumentDetailPage />);
    await waitFor(() =>
      expect(screen.getByTestId("document-error")).toBeInTheDocument(),
    );
    expect(screen.getByText(/Document unavailable/i)).toBeInTheDocument();
  });
});
