import * as React from "react";
import Head from "next/head";
import { useRouter } from "next/router";

// The Terminal view now lives as a tab inside the Research page.
// Preserve the `/console` URL for existing bookmarks by replacing
// the route on the client so the SPA shell stays intact.
export default function ConsoleRedirectPage(): JSX.Element {
  const router = useRouter();
  React.useEffect(() => {
    router.replace("/research?tab=terminal");
  }, [router]);
  return (
    <>
      <Head>
        <title>Redirecting to Research · fed-pulse</title>
      </Head>
      <main className="container py-6">
        <p className="text-sm text-muted-foreground">
          Redirecting to the Research console&hellip;
        </p>
      </main>
    </>
  );
}
