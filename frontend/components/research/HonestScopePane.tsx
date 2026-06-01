import * as React from "react";
import { CheckCircle2, MinusCircle, ExternalLink } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const WIKI_NULL = "https://github.com/yusufizzetmurat/fed-pulse/wiki/20_Gated_Fusion_InfoNCE_Comprehensive_Null";
const WIKI_RELATED_WORK = "https://github.com/yusufizzetmurat/fed-pulse/wiki/19_Related_Work_Text_Market_Fusion";

interface ClaimProps {
  title: string;
  detail: string;
  metric?: string;
  source?: string;
}

function Predicts({ title, detail, metric, source }: ClaimProps) {
  return (
    <div className="flex gap-3 rounded-md border border-emerald-700/40 bg-emerald-700/5 p-3">
      <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-700" aria-hidden />
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-foreground">{title}</p>
        <p className="text-sm text-muted-foreground">{detail}</p>
        {metric ? (
          <Badge variant="outline" className="numeric text-[10px]" title={source}>
            {metric}
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

function DoesNotPredict({ title, detail, metric, source }: ClaimProps) {
  return (
    <div className="flex gap-3 rounded-md border border-destructive/40 bg-destructive/5 p-3">
      <MinusCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" aria-hidden />
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-foreground">{title}</p>
        <p className="text-sm text-muted-foreground">{detail}</p>
        {metric ? (
          <Badge variant="outline" className="numeric text-[10px]" title={source}>
            {metric}
          </Badge>
        ) : null}
      </div>
    </div>
  );
}

export function HonestScopePane() {
  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>What this model predicts &mdash; and what it doesn&apos;t</CardTitle>
          <CardDescription>
            Forecast cards on the dashboard are driven by market data only. Text panels are
            descriptive: they show what the Fed said and how it changed, never what the price
            or volatility will do. Every claim below is anchored to a measured macro-F1 or a
            block-bootstrap confidence interval in the wiki.
          </CardDescription>
        </CardHeader>
      </Card>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Predicts (market data only)</CardTitle>
            <CardDescription>
              Cards on the workspace that surface a number with a confidence band.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Predicts
              title="Short-horizon realized volatility"
              detail="Three-class regime (Low / Medium / High) from HAR&apos;s continuous forecast bucketed into terciles. Strongest classifier in the bake-off; the QLIKE-DLq ensemble beats HAR by ~10% in QLIKE loss at every horizon."
              metric="HAR-tercile macro-F1  0.687 (1d) · 0.685 (1w) · 0.654 (1m)"
              source="Wiki §20, Result 2 — 3-class forward-RV-tercile, n=1999, pooled walk-forward"
            />
            <Predicts
              title="Forward trading volume"
              detail="HAR-style baseline applied to log-volume residuals after removing weekly seasonality and FOMC-calendar effects. Lands on the Workspace as the &ldquo;Expected Volume&rdquo; card."
              metric="Market-only R²  ≈ 0.85–0.88 at 1-day"
              source="Wiki §20, Result 3 — abnormal-volume forecast"
            />
            <Predicts
              title="Conformal coverage on the vol bands"
              detail="The 80% and 90% prediction bands on the QLIKE-RV card are calibrated against held-out residuals. Empirical coverage tracks the nominal target on resolved runs."
              metric="Empirical 90% band coverage  ≈ 85–92%"
              source="Wiki §20, Result 1 conformal calibration"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Does NOT predict (text is a description layer)</CardTitle>
            <CardDescription>
              FOMC text adds no measurable forecasting signal over the market baseline. The text
              panels on the workspace (stance, semantic diff, topics, XAI) are honest about what
              the Fed said. They do not feed the forecast cards.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <DoesNotPredict
              title="Price direction or magnitude from FOMC text"
              detail="Across four encoders (finbert_fed_adjacent, bge-large, e5-large, gte-large), text robustly hurts next-day vol forecasting. The fused text+market regime classifier underperforms HAR-tercile by ~0.10 macro-F1 at 1-day."
              metric="Text-vs-market 95% block CI  [−0.022, −0.009] at 1-day"
              source="Wiki §20, Result 2 — text-vs-mkt incremental, daily n=1999"
            />
            <DoesNotPredict
              title="Volatility regime FROM text"
              detail="The late-fusion classifier is still rendered on the Workspace as a second opinion, but only as a transparency disclosure. HAR-tercile is the headline because it&apos;s the better predictor."
              metric="Late-fusion macro-F1  0.592 (1d) — 0.095 below HAR-tercile"
              source="Wiki §20, Result 2 — fused (text+market)"
            />
            <DoesNotPredict
              title="Surprise → drift channel"
              detail="A LoRA fine-tune on the FOMC corpus testing whether text-surprise predicts post-meeting drift returned null. McNemar test against the baseline failed to reject equivalence."
              metric="LoRA McNemar p = 0.92"
              source="Re-confirmation: LoRA fine-tune, FOMC corpus, n=1999 daily + n=167 intraday"
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Why the text panels still matter</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            The dashboard&apos;s text surfaces &mdash; hawkish/dovish stance, sentiment breakdown,
            sentence-level XAI, semantic diff against the previous statement, historical analogs &mdash;
            do not forecast the market. They answer a different, honest question:{" "}
            <span className="text-foreground">
              what did the Fed say, how did the wording change, and how does it compare to past
              statements?
            </span>{" "}
            That&apos;s a legitimate descriptive job and the panels are calibrated against the labelled
            FOMC corpus (gtfintechlab&apos;s Trillion-Dollar-Words derivatives).
          </p>
          <p>
            The discipline is visual on every card: forecast surfaces carry a measured macro-F1 or
            QLIKE-vs-HAR chip; text surfaces carry no forecasting metric and never imply they feed
            the price or vol forecast. XAI explains the stance label, not the price.
          </p>
          <div className="flex flex-wrap items-center gap-2 pt-2">
            <a
              href={WIKI_NULL}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs text-foreground underline decoration-muted-foreground/40 underline-offset-2 hover:decoration-foreground"
            >
              Methodology &amp; measured nulls — wiki §20
              <ExternalLink className="h-3 w-3" aria-hidden />
            </a>
            <span className="text-muted-foreground">·</span>
            <a
              href={WIKI_RELATED_WORK}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs text-foreground underline decoration-muted-foreground/40 underline-offset-2 hover:decoration-foreground"
            >
              Where this sits in the literature — wiki §19
              <ExternalLink className="h-3 w-3" aria-hidden />
            </a>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
