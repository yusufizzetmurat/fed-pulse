import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import { ArrowLeft, ExternalLink, FileText } from "lucide-react";

import { Header } from "@/components/shell/header";
import { StatusBar } from "@/components/shell/status-bar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchDocumentDetail, resolveApiBaseUrl } from "@/lib/analyze/api";
import { errorMessage as toErrorMessage } from "@/lib/analyze/errors";
import type { DocumentDetailResponse } from "@/lib/analyze/types";

// Map the URL path token onto the human-readable label rendered on
// the header chip. Mirrors the backend's _DOCUMENT_DETAIL_SOURCES
// canonical keys — anything outside the set surfaces as a passthrough
// so an old bookmark with a stale token still produces a readable
// label even before the 422 lands.
const TYPE_LABELS: Record<string, string> = {
  statement: "Statement",
  minutes: "Minutes",
  press_conference: "Press conference",
};

function prettyType(type: string): string {
  return TYPE_LABELS[type] ?? type.replace(/_/g, " ");
}

export default function DocumentDetailPage() {
  const router = useRouter();
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [document, setDocument] = React.useState<DocumentDetailResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [notFound, setNotFound] = React.useState(false);
  const [errorState, setErrorState] = React.useState<string | null>(null);

  const type = React.useMemo(() => {
    const value = router.query.type;
    return typeof value === "string" ? value : null;
  }, [router.query.type]);
  const date = React.useMemo(() => {
    const value = router.query.date;
    return typeof value === "string" ? value : null;
  }, [router.query.date]);

  React.useEffect(() => {
    if (!router.isReady) return;
    if (!type || !date) return;
    let cancelled = false;
    setLoading(true);
    setErrorState(null);
    setNotFound(false);
    setDocument(null);
    fetchDocumentDetail(apiBaseUrl, type, date)
      .then((result) => {
        if (cancelled) return;
        if (result === null) {
          setNotFound(true);
          return;
        }
        setDocument(result);
      })
      .catch((err) => {
        if (cancelled) return;
        setErrorState(toErrorMessage(err, "Document fetch failed."));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, router.isReady, type, date]);

  const pageTitle = document
    ? `${prettyType(document.type)} ${document.date} — Fed Pulse`
    : "FOMC document — Fed Pulse";

  return (
    <>
      <Head>
        <title>{pageTitle}</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main
          id="main-content"
          tabIndex={-1}
          className="container space-y-6 py-8 focus:outline-none"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <Button asChild variant="ghost" size="sm">
              <Link href="/calendar">
                <ArrowLeft className="h-4 w-4" />
                Back to calendar
              </Link>
            </Button>
            {document ? (
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant="outline" className="capitalize">
                  <FileText className="h-3 w-3" />
                  {prettyType(document.type)}
                </Badge>
                <Badge variant="outline" className="font-mono">
                  {document.date}
                </Badge>
                {document.source_url ? (
                  <Button asChild variant="outline" size="sm">
                    <a
                      href={document.source_url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="h-4 w-4" />
                      View source
                    </a>
                  </Button>
                ) : null}
              </div>
            ) : null}
          </div>

          {loading ? (
            <Card>
              <CardHeader>
                <Skeleton className="h-6 w-2/3" />
                <Skeleton className="h-4 w-1/3" />
              </CardHeader>
              <CardContent className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-11/12" />
                <Skeleton className="h-4 w-10/12" />
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-9/12" />
              </CardContent>
            </Card>
          ) : errorState ? (
            <Card>
              <CardHeader>
                <CardTitle>Document unavailable</CardTitle>
                <CardDescription>
                  The backend returned an error fetching this document.
                </CardDescription>
              </CardHeader>
              <CardContent className="py-6 text-muted-foreground" data-testid="document-error">
                {errorState}
              </CardContent>
            </Card>
          ) : notFound ? (
            <Card>
              <CardHeader>
                <CardTitle>Document not on file</CardTitle>
                <CardDescription>
                  No {type ? prettyType(type).toLowerCase() : "document"} has
                  been collected for {date ?? "this date"} yet.
                </CardDescription>
              </CardHeader>
              <CardContent className="py-6 text-muted-foreground" data-testid="document-not-found">
                The calendar reflects what the scraper has gathered so far.
                Far-future meetings will surface here once the publication
                window closes and the next ingest run lands.
              </CardContent>
            </Card>
          ) : document ? (
            <Card>
              <CardHeader>
                <CardTitle>{document.title || prettyType(document.type)}</CardTitle>
                {document.scraped_at ? (
                  <CardDescription>
                    Scraped {document.scraped_at}
                  </CardDescription>
                ) : null}
              </CardHeader>
              <CardContent>
                <article
                  data-testid="document-body"
                  className="prose prose-sm max-w-none whitespace-pre-wrap font-serif text-sm leading-relaxed text-foreground dark:prose-invert"
                >
                  {document.cleaned_text}
                </article>
              </CardContent>
            </Card>
          ) : null}
        </main>
      </div>
    </>
  );
}
