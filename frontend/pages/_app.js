import "../styles/globals.css";
import "../styles/legacy-dashboard.css";

import { ThemeProvider } from "next-themes";
import { Toaster } from "sonner";

export default function App({ Component, pageProps }) {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <Component {...pageProps} />
      <Toaster richColors position="top-right" />
    </ThemeProvider>
  );
}
