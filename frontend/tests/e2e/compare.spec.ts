import { expect, test } from "@playwright/test";

test.describe("history → compare flow", () => {
  test("Compare to… link prefills one slot and renders the regime delta", async ({ page }) => {
    test.slow();
    await page.goto("/history");
    await expect(page.getByRole("heading", { name: /^history$/i, level: 1 })).toBeVisible();

    // The compare flow needs at least two runs in history. Skip
    // gracefully when the dev environment has a fresh DB — the rest of
    // the suite still exercises the page render.
    const rowCount = await page.locator("table tbody tr").count();
    test.skip(rowCount < 2, "needs at least two history runs to exercise compare");

    // Each table row carries a "Compare to…" action; the link routes
    // to /compare with the source id in slot A.
    const compareLink = page.getByRole("link", { name: /compare to/i }).first();
    await compareLink.click();

    await expect(page).toHaveURL(/\/compare/);
    await expect(page.getByRole("heading", { name: /compare runs/i, level: 1 })).toBeVisible();

    // Pick the second-most-recent run for slot B from the slot
    // selector. The page-level Δ card only renders when both slots are
    // populated.
    const slotBSelect = page.getByLabel(/Run B/i);
    await slotBSelect.click();
    const optionList = page.getByRole("listbox");
    await expect(optionList).toBeVisible({ timeout: 5_000 });
    await optionList.getByRole("option").nth(1).click();

    await expect(page.getByText(/Δ A − B/i)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/^Regime$/i)).toBeVisible();
  });
});
