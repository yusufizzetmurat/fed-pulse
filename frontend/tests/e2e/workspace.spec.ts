import { expect, test } from "@playwright/test";

test.describe("workspace", () => {
  test("loads with the default text and runs an analysis", async ({ page }) => {
    await page.goto("/");

    await expect(page.getByRole("heading", { name: /workspace/i, level: 1 })).toBeVisible();

    // Document ingestion tabs default to the paste view; the textarea
    // should be prefilled with the default boilerplate so the user can
    // submit immediately.
    const textarea = page.getByLabel(/fomc text/i);
    await expect(textarea).toBeVisible();
    await expect(textarea).not.toBeEmpty();

    await page.getByRole("button", { name: /analyze/i }).click();

    // Either the regime headline or the multi-axis tile grid will land
    // first depending on backend response order; we wait on the regime
    // card title because it is the demo headline.
    await expect(
      page.getByText(/vol-regime prediction set/i),
    ).toBeVisible({ timeout: 30_000 });
  });

  test("status bar shows the analysis state", async ({ page }) => {
    await page.goto("/");
    const statusbar = page.getByRole("status");
    await expect(statusbar).toBeVisible();
    await expect(statusbar).toContainText(/symbol/i);
  });
});
