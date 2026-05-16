import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

vi.mock("next/head", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("next/link", () => ({
  default: ({ href, children, ...rest }: any) => (
    <a href={typeof href === "string" ? href : "#"} {...rest}>
      {children}
    </a>
  ),
}));

vi.mock("next-themes", () => ({
  useTheme: () => ({ theme: "dark", setTheme: () => {}, resolvedTheme: "dark" }),
}));

describe("Error pages", () => {
  it("404 page renders the branded copy and a back-to-home link", async () => {
    const { default: NotFoundPage } = await import("@/pages/404");
    render(<NotFoundPage />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/off the map/i);
    const home = screen.getByRole("link", { name: /back to home/i });
    expect(home).toHaveAttribute("href", "/analyze");
    expect(screen.getByText(/Status 404/i)).toBeInTheDocument();
  });

  it("500 page renders the server-error copy and a back-to-home link", async () => {
    const { default: ServerErrorPage } = await import("@/pages/500");
    render(<ServerErrorPage />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/broke on our side/i);
    const home = screen.getByRole("link", { name: /back to home/i });
    expect(home).toHaveAttribute("href", "/analyze");
    expect(screen.getByText(/Status 500/i)).toBeInTheDocument();
  });

  it("_error delegates to the 404 branding for 404 statusCode", async () => {
    const { default: FallbackError } = await import("@/pages/_error");
    render(<FallbackError statusCode={404} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/off the map/i);
  });

  it("_error delegates to the 500 branding for 5xx statusCode", async () => {
    const { default: FallbackError } = await import("@/pages/_error");
    render(<FallbackError statusCode={503} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/broke on our side/i);
    expect(screen.getByText(/Status 503/i)).toBeInTheDocument();
  });

  it("_error falls through to a generic page for other status codes", async () => {
    const { default: FallbackError } = await import("@/pages/_error");
    render(<FallbackError statusCode={418} />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/unexpected error/i);
  });
});
