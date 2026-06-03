import * as React from "react";
import Head from "next/head";
import Link from "next/link";
import { ArrowUpRight, Clock, Globe } from "lucide-react";
import { toast } from "sonner";

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
import { fetchCrossBankSnapshot, resolveApiBaseUrl } from "@/lib/analyze/api";
import { errorMessage } from "@/lib/analyze/errors";
import type {
  CrossBankCard,
  CrossBankSnapshotResponse,
} from "@/lib/analyze/types";

// Convert an ISO-3166 alpha-2 country code (e.g. "GB") into the matching
// regional indicator flag emoji. The xbank ECB card uses "EU", which is
// not technically a country code but ships a flag glyph in modern fonts.
function flagFromCode(code: string): string {
  if (!code) return "";
  if (code === "EU") return "\u{1F1EA}\u{1F1FA}";
  const upper = code.toUpperCase();
  if (upper.length !== 2) return "";
  const base = 0x1f1e6;
  const a = upper.charCodeAt(0) - 65;
  const b = upper.charCodeAt(1) - 65;
  if (a < 0 || a > 25 || b < 0 || b > 25) return "";
  return String.fromCodePoint(base + a, base + b);
}

function stanceBadgeVariant(label: string | null): "hawkish" | "dovish" | "neutral" | "outline" {
  if (label === "hawkish") return "hawkish";
  if (label === "dovish") return "dovish";
  if (label === "neutral") return "neutral";
  return "outline";
}

function regimeBadgeVariant(label: string | null): "hawkish" | "dovish" | "neutral" | "outline" {
  if (label === "high") return "hawkish";
  if (label === "calm") return "dovish";
  if (label === "normal") return "neutral";
  return "outline";
}

function formatPercent(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatDate(value: string | null | undefined): string {
  if (!value) return "—";
  // gtfintechlab event_dates are year-rounded ("2024-01-01"); show just the
  // year when the day is 01-01 so we don't imply day-level precision.
  if (/-01-01$/.test(value)) return value.slice(0, 4);
  return value;
}

function BankCardSkeleton(): JSX.Element {
  return (
    <Card>
      <CardHeader className="space-y-2">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-3 w-24" />
      </CardHeader>
      <CardContent className="space-y-3">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-8 w-2/3" />
      </CardContent>
    </Card>
  );
}

function BankCardView({ card }: { card: CrossBankCard }): JSX.Element {
  const flag = flagFromCode(card.flag);
  const stanceVariant = stanceBadgeVariant(card.stance_label);
  const regimeVariant = regimeBadgeVariant(card.vol_regime_label);
  const workspaceHref = `/?symbol=${encodeURIComponent(card.symbol)}`;
  const stanceUnavailable =
    card.status === "corpus_missing" || card.status === "classifier_unavailable";
  const stanceUnavailableMessage =
    card.status === "corpus_missing"
      ? "Corpus not ingested yet — coming soon."
      : "Classifier checkpoint unavailable.";

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="space-y-1">
        <div className="flex items-start justify-between gap-2">
          <div className="space-y-0.5">
            <CardTitle className="flex items-center gap-2 text-lg">
              <span aria-hidden="true" className="text-2xl leading-none">
                {flag || <Globe className="h-5 w-5" />}
              </span>
              <span>{card.short_code}</span>
            </CardTitle>
            <CardDescription>{card.display_name}</CardDescription>
          </div>
          <Badge variant="outline" className="font-mono text-[10px] uppercase tracking-wider">
            {card.symbol}
          </Badge>
        </div>
        <p className="flex items-center gap-1 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" aria-hidden="true" />
          Latest sample: {formatDate(card.latest_statement_date)}
        </p>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col justify-between gap-4">
        <section aria-label="Stance" className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <span>Stance</span>
            {card.sample_size > 0 ? <span>{card.sample_size} sentences</span> : null}
          </div>
          {stanceUnavailable ? (
            <p className="text-sm text-muted-foreground">{stanceUnavailableMessage}</p>
          ) : (
            <>
              <div className="flex items-center gap-2">
                <Badge variant={stanceVariant} className="text-sm capitalize">
                  {card.stance_label ?? "—"}
                </Badge>
                <span className="text-sm font-medium text-foreground">
                  {formatPercent(card.stance_confidence)}
                </span>
              </div>
              {card.stance ? (
                <dl className="grid grid-cols-3 gap-1 text-xs text-muted-foreground">
                  <div className="space-y-0.5">
                    <dt className="uppercase tracking-wide">Hawk</dt>
                    <dd className="font-mono text-foreground">{formatPercent(card.stance.hawkish ?? 0)}</dd>
                  </div>
                  <div className="space-y-0.5">
                    <dt className="uppercase tracking-wide">Neutral</dt>
                    <dd className="font-mono text-foreground">{formatPercent(card.stance.neutral ?? 0)}</dd>
                  </div>
                  <div className="space-y-0.5">
                    <dt className="uppercase tracking-wide">Dove</dt>
                    <dd className="font-mono text-foreground">{formatPercent(card.stance.dovish ?? 0)}</dd>
                  </div>
                </dl>
              ) : null}
              {card.time_axis ? (
                <p className="text-xs text-muted-foreground">
                  Time horizon: <span className="text-foreground">{card.time_axis}</span>
                </p>
              ) : null}
            </>
          )}
        </section>

        <section aria-label="Vol regime" className="space-y-2 border-t border-border pt-3">
          <div className="flex items-center justify-between text-xs font-medium uppercase tracking-wide text-muted-foreground">
            <span>Vol regime</span>
            {card.vol_regime_as_of ? <span>as of {card.vol_regime_as_of}</span> : null}
          </div>
          {card.vol_regime_status === "ok" ? (
            <div className="flex items-center gap-2">
              <Badge variant={regimeVariant} className="text-sm capitalize">
                {card.vol_regime_label ?? "—"}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {formatPercent(card.vol_regime_confidence)} conf
              </span>
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">Market data unavailable.</p>
          )}
        </section>

        <Button asChild variant="ghost" size="sm" className="-mx-2 justify-between">
          <Link href={workspaceHref} aria-label={`Open ${card.display_name} in Workspace`}>
            <span>Open {card.symbol} in Workspace</span>
            <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
          </Link>
        </Button>
      </CardContent>
    </Card>
  );
}

export default function CrossBankPage(): JSX.Element {
  const apiBaseUrl = React.useMemo(() => resolveApiBaseUrl(), []);
  const [data, setData] = React.useState<CrossBankSnapshotResponse | null>(null);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const response = await fetchCrossBankSnapshot(apiBaseUrl, controller.signal);
        if (cancelled) return;
        setData(response);
      } catch (err) {
        if (cancelled || controller.signal.aborted) return;
        toast.error(errorMessage(err) || "Failed to load cross-bank snapshot");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [apiBaseUrl]);

  return (
    <>
      <Head>
        <title>Cross-bank — Fed Pulse</title>
      </Head>
      <div className="min-h-screen bg-background text-foreground">
        <Header />
        <StatusBar />
        <main
          id="main-content"
          tabIndex={-1}
          className="container space-y-6 py-8 focus:outline-none"
        >
          <div className="space-y-2">
            <h1 className="flex items-center gap-2 text-2xl font-semibold tracking-tight sm:text-3xl">
              <Globe className="h-6 w-6 text-primary sm:h-7 sm:w-7" />
              Cross-bank stance panel
            </h1>
            <p className="max-w-3xl text-muted-foreground">
              Side-by-side stance and volatility-regime read across six major
              central banks, scored by the xbank-DAPT multi-axis classifier
              against the latest annotated sentences from each bank's
              published communications. Vol regime is a coarse 5-day realised
              volatility band on the bank's flagship equity index. Cards
              refresh hourly.
            </p>
            {data?.generated_at ? (
              <p className="text-xs text-muted-foreground">
                Snapshot generated at {data.generated_at}.
              </p>
            ) : null}
          </div>

          {loading ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <BankCardSkeleton />
              <BankCardSkeleton />
              <BankCardSkeleton />
              <BankCardSkeleton />
              <BankCardSkeleton />
              <BankCardSkeleton />
            </div>
          ) : data ? (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              {data.banks.map((card) => (
                <BankCardView key={card.bank} card={card} />
              ))}
            </div>
          ) : (
            <Card>
              <CardHeader>
                <CardTitle>No snapshot</CardTitle>
                <CardDescription>
                  The cross-bank snapshot could not be loaded. Retry shortly.
                </CardDescription>
              </CardHeader>
            </Card>
          )}
        </main>
      </div>
    </>
  );
}
