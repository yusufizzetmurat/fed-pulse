import { expect, test } from "@playwright/test";

test.describe("sentence strike counterfactual", () => {
  test("striking a sentence renders a regime Δ badge and reset restores baseline", async ({
    page,
  }) => {
    test.slow();
    await page.goto("/");

    // Wait for the initial analyze response so the XAI panel is populated.
    await page.getByRole("button", { name: /analyze/i }).click();
    await expect(page.getByText(/vol-regime prediction set/i)).toBeVisible({
      timeout: 30_000,
    });

    const panelHeading = page.getByRole("heading", { name: /sentence attribution/i });
    await expect(panelHeading).toBeVisible();

    // Scope strike-toggle lookup to the Sentence Attribution card so
    // we don't pick up other aria-pressed buttons on the workspace
    // (WatchlistChips also uses aria-pressed for the active-symbol
    // chip state).
    const sentencePanel = page
      .locator('div')
      .filter({ has: panelHeading })
      .first();
    const sentenceButton = sentencePanel
      .locator('button[aria-pressed="false"]')
      .first();
    await expect(sentenceButton).toBeVisible({ timeout: 5_000 });
    await sentenceButton.click();

    await expect(page.getByText(/1 struck/i)).toBeVisible({ timeout: 30_000 });
    // Δ regime badge appears once the second /analyze returns.
    await expect(page.getByText(/Δ regime/i)).toBeVisible({ timeout: 30_000 });

    const resetButton = page.getByRole("button", { name: /reset/i });
    await resetButton.click();
    await expect(page.getByText(/1 struck/i)).not.toBeVisible();
  });
});
