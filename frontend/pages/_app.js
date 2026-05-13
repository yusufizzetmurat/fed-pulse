import "../styles/globals.css";
import "../styles/legacy-dashboard.css";

import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";

import { TooltipProvider } from "@/components/ui/tooltip";

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <TooltipProvider delayDuration={120}>
        <Component {...pageProps} />
      </TooltipProvider>
      <Toaster richColors position="top-right" />
    </ThemeProvider>
  );
}
