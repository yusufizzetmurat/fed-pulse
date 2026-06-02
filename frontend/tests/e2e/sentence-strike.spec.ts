import { expect, test } from "@playwright/test";

// Multi-sentence FOMC excerpt mirrors the 2024-09-18 sample so the XAI
// panel renders five sentence chips. With one chip there is nothing to
// misalign after a strike; with five we can verify the struck sentence
// stays visible (line-through) and the rest remain at aria-pressed=false.
const MULTI_SENTENCE_TEXT =
  "Recent indicators suggest that economic activity has continued to expand at a solid pace. " +
  "Job gains have slowed, and the unemployment rate has moved up but remains low. " +
  "Inflation has made further progress toward the Committee's 2 percent objective but remains somewhat " +
  "elevated. The Committee has gained greater confidence that inflation is moving sustainably " +
  "toward 2 percent, and judges that the risks to achieving its employment and inflation goals " +
  "are roughly in balance. In light of the progress on inflation and the balance of risks, the " +
  "Committee decided to lower the target range for the federal funds rate by 1/2 percentage " +
  "point to 4-3/4 to 5 percent.";

test.describe("sentence strike counterfactual", () => {
  test("strikes keep sibling sentences visible and refresh the regime delta", async ({
    page,
  }) => {
    test.slow();
    await page.goto("/");

    // Replace the default single-sentence input with a multi-sentence
    // excerpt so the XAI panel has multiple chips to strike.
    const textarea = page.locator("textarea#text");
    await expect(textarea).toBeVisible();
    await textarea.fill(MULTI_SENTENCE_TEXT);

    await page.getByRole("button", { name: /analyze/i }).click();
    await expect(page.getByText(/vol-regime prediction set/i)).toBeVisible({
      timeout: 30_000,
    });

    const panelHeading = page.getByRole("heading", {
      name: /per-sentence explanation/i,
    });
    await expect(panelHeading).toBeVisible();

    // Scope sentence-chip lookup to the explanation card so other
    // aria-pressed buttons (WatchlistChips, asset toggles) don't leak.
    const sentencePanel = page
      .locator("div")
      .filter({ has: panelHeading })
      .first();
    const sentenceButtons = sentencePanel.locator('button[aria-pressed]');
    await expect(sentenceButtons.first()).toBeVisible({ timeout: 5_000 });
    const sentenceCount = await sentenceButtons.count();
    expect(sentenceCount).toBeGreaterThanOrEqual(3);

    // Strike sentence index 1 (second chip).
    await sentenceButtons.nth(1).click();

    await expect(page.getByText(/1 struck/i)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/2 struck/i)).not.toBeVisible();
    await expect(sentenceButtons.nth(1)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    // At least one sibling chip remains un-struck and visible.
    const unStruckCount = await sentencePanel
      .locator('button[aria-pressed="false"]')
      .count();
    expect(unStruckCount).toBeGreaterThanOrEqual(sentenceCount - 1);
    // Total chip count is unchanged after the strike round-trip.
    await expect(sentenceButtons).toHaveCount(sentenceCount);

    // Δ regime badge appears once the masked /analyze returns.
    await expect(page.getByText(/Δ regime/i)).toBeVisible({ timeout: 30_000 });

    // Strike a second sentence.
    await sentenceButtons.nth(2).click();
    await expect(page.getByText(/2 struck/i)).toBeVisible({ timeout: 30_000 });
    await expect(sentenceButtons.nth(2)).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    await expect(sentenceButtons).toHaveCount(sentenceCount);

    const resetButton = page.getByRole("button", { name: /reset/i });
    await resetButton.click();
    await expect(page.getByText(/2 struck/i)).not.toBeVisible();
    await expect(page.getByText(/1 struck/i)).not.toBeVisible();
    // All chips return to aria-pressed=false.
    const finalUnStruck = await sentencePanel
      .locator('button[aria-pressed="false"]')
      .count();
    expect(finalUnStruck).toBe(sentenceCount);
  });
});
