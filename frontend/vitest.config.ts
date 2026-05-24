import path from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    // Vitest covers unit + component tests; the e2e suite under tests/e2e
    // is Playwright-only and is run via `npm run e2e` against a live dev
    // server.
    include: ["tests/**/*.{test,spec}.{ts,tsx}"],
    exclude: ["**/node_modules/**", "tests/e2e/**"],
    globals: true,
    css: false,
  },
});
