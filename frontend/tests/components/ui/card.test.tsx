import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

describe("Card", () => {
  it("renders header / title / description / content composition", () => {
    render(
      <Card>
        <CardHeader>
          <CardTitle>Sentiment</CardTitle>
          <CardDescription>Multi-axis output</CardDescription>
        </CardHeader>
        <CardContent>
          <p>Hawkish · 0.62</p>
        </CardContent>
      </Card>
    );
    expect(screen.getByText("Sentiment")).toBeInTheDocument();
    expect(screen.getByText("Multi-axis output")).toBeInTheDocument();
    expect(screen.getByText("Hawkish · 0.62")).toBeInTheDocument();
  });

  it("forwards arbitrary props to the underlying div", () => {
    render(<Card data-testid="run-card">Body</Card>);
    expect(screen.getByTestId("run-card")).toBeInTheDocument();
  });
});
