import { expect, test } from "@playwright/test";

test.describe("command palette", () => {
  test("Cmd+K opens the palette and jumps to a page", async ({ page, browserName }) => {
    await page.goto("/");

    // Use the platform-appropriate modifier. Playwright's keyboard
    // handler accepts ControlOrMeta on all engines.
    await page.keyboard.press("ControlOrMeta+KeyK");
    const dialog = page.getByRole("dialog", { name: /command palette/i });
    await expect(dialog).toBeVisible();

    const input = dialog.getByPlaceholder(/search pages/i);
    await input.fill("history");
    await input.press("Enter");

    await expect(page).toHaveURL(/\/history(\?|$)/);
    await expect(
      page.getByRole("heading", { name: /^history$/i, level: 1 }),
    ).toBeVisible();

    // Sanity: the palette closes after navigation.
    await expect(dialog).not.toBeVisible();

    // Browser-specific guards: webkit sometimes leaves the focus ring on
    // the trigger; not worth blocking the test.
    void browserName;
  });
});
