import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { Button } from "@/components/ui/button";

describe("Button", () => {
  it("renders the label", () => {
    render(<Button>Run analyze</Button>);
    expect(screen.getByRole("button", { name: /run analyze/i })).toBeInTheDocument();
  });

  it("invokes onClick when activated", async () => {
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Click</Button>);
    await userEvent.click(screen.getByRole("button"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("applies the destructive variant class", () => {
    render(<Button variant="destructive">Delete</Button>);
    expect(screen.getByRole("button")).toHaveClass("bg-destructive");
  });

  it("applies the outline variant class", () => {
    render(<Button variant="outline">Outline</Button>);
    expect(screen.getByRole("button")).toHaveClass("border-input");
  });

  it("respects asChild by rendering the child element", () => {
    render(
      <Button asChild>
        <a href="/somewhere">Go</a>
      </Button>
    );
    const link = screen.getByRole("link", { name: /go/i });
    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/somewhere");
  });

  it("disables and skips onClick when disabled", async () => {
    const onClick = vi.fn();
    render(
      <Button onClick={onClick} disabled>
        Disabled
      </Button>
    );
    await userEvent.click(screen.getByRole("button"));
    expect(onClick).not.toHaveBeenCalled();
  });
});
