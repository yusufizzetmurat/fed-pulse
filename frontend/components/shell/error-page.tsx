import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { ArrowLeft, Home } from "lucide-react";

import { Header } from "@/components/shell/header";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface ErrorPageProps {
  status: string;
  title: string;
  description: string;
  hint?: React.ReactNode;
}

/**
 * Shared chrome for the 404 / 500 / fallback error routes. Renders inside the
 * same shell as every other page so theme, header, and skip link all stay
 * consistent, and exposes a back-to-home action plus a back-one-step action.
 */
export function ErrorPage({ status, title, description, hint }: ErrorPageProps) {
  const pageTitle = `${status} — ${title} · Fed Pulse`;

  const handleBack = () => {
    if (typeof window !== "undefined" && window.history.length > 1) {
      window.history.back();
    } else if (typeof window !== "undefined") {
      window.location.assign("/analyze");
    }
  };

  return (
    <>
      <Head>
        <title>{pageTitle}</title>
        <meta name="robots" content="noindex" />
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <main
          id="main-content"
          tabIndex={-1}
          className="container flex flex-col items-center justify-center gap-6 py-16 text-center focus:outline-none"
        >
          <Badge variant="outline" className="text-[11px] uppercase tracking-[0.18em]">
            Status {status}
          </Badge>
          <h1 className="max-w-2xl text-balance text-4xl font-semibold tracking-tight sm:text-5xl">
            {title}
          </h1>
          <p className="max-w-xl text-balance text-muted-foreground">{description}</p>
          {hint ? (
            <div className="max-w-xl rounded-md border border-border bg-muted/30 px-4 py-3 text-left text-sm text-muted-foreground">
              {hint}
            </div>
          ) : null}
          <div className="flex flex-wrap items-center justify-center gap-2">
            <Button asChild>
              <Link href="/analyze">
                <Home className="h-4 w-4" aria-hidden="true" />
                Back to home
              </Link>
            </Button>
            <Button type="button" variant="outline" onClick={handleBack}>
              <ArrowLeft className="h-4 w-4" aria-hidden="true" />
              Back one step
            </Button>
          </div>
        </main>
      </div>
    </>
  );
}
