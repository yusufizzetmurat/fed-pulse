import "../styles/globals.css";

import * as React from "react";
import Head from "next/head";
import ReactDOM from "react-dom";
import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";

import { KeyboardShortcuts } from "@/components/shell/keyboard-shortcuts";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SymbolCalendarProvider } from "@/lib/analyze/shared-context";

// axe-core/react runs only in the browser, only in development. The dynamic
// import keeps it out of the production bundle entirely.
if (typeof window !== "undefined" && process.env.NODE_ENV === "development") {
  import("@axe-core/react")
    .then(({ default: axe }) => axe(React, ReactDOM, 1000))
    .catch(() => {
      // axe is a dev convenience; failures to load should not break the app.
    });
}

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <Head>
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="theme-color" content="#0b1220" />
      </Head>
      <TooltipProvider delayDuration={120}>
        <SymbolCalendarProvider>
          <KeyboardShortcuts />
          <Component {...pageProps} />
        </SymbolCalendarProvider>
      </TooltipProvider>
      <Toaster richColors position="top-right" />
    </ThemeProvider>
  );
}
