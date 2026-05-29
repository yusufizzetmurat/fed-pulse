import { expect, test } from "@playwright/test";

test.describe("FOMC calendar", () => {
  test("upcoming-meeting click prefills the workspace date input", async ({ page }) => {
    await page.goto("/calendar");
    await expect(page.getByRole("heading", { name: /fomc calendar/i, level: 1 })).toBeVisible();

    // The Analyze action is a button (router.push to /?date=…),
    // not an anchor — selector must match a button, and we capture the
    // resulting URL after navigation rather than reading an href.
    const analyzeButton = page.getByRole("button", { name: /^analyze$/i }).first();
    await expect(analyzeButton).toBeVisible({ timeout: 15_000 });
    await analyzeButton.click();

    // Lands on the workspace at /?date=YYYY-MM-DD&kind=statement.
    await expect(page).toHaveURL(/\/\?date=\d{4}-\d{2}-\d{2}/);
    const finalUrl = new URL(page.url());
    const expectedDate = finalUrl.searchParams.get("date") ?? "";
    expect(expectedDate).toMatch(/^\d{4}-\d{2}-\d{2}$/);

    const dateInput = page.getByLabel(/document date/i);
    await expect(dateInput).toHaveValue(expectedDate);
  });
});
