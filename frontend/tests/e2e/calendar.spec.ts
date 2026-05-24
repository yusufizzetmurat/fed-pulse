import { expect, test } from "@playwright/test";

test.describe("FOMC calendar", () => {
  test("upcoming-meeting link prefills the workspace date input", async ({ page }) => {
    await page.goto("/calendar");
    await expect(page.getByRole("heading", { name: /fomc calendar/i, level: 1 })).toBeVisible();

    // The calendar has a past + upcoming section; pick the first
    // upcoming meeting's Analyze link so the prefill is a real future
    // date the workspace can ingest without yfinance lookups.
    const upcomingLink = page.getByRole("link", { name: /analyze/i }).first();
    await expect(upcomingLink).toBeVisible({ timeout: 15_000 });
    const href = await upcomingLink.getAttribute("href");
    expect(href).toMatch(/\/\?date=\d{4}-\d{2}-\d{2}/);

    await upcomingLink.click();
    await expect(page).toHaveURL(/\/\?date=/);

    // Workspace date input picks up the meeting date.
    const dateInput = page.getByLabel(/document date/i);
    const expected = href?.match(/date=(\d{4}-\d{2}-\d{2})/)?.[1] ?? "";
    await expect(dateInput).toHaveValue(expected);
  });
});
