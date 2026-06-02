import * as React from "react";
import { CheckCircle2, MinusCircle } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

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
          <CardTitle>What this model predicts, and what it doesn&apos;t</CardTitle>
          <CardDescription>
            Forecast cards on the dashboard are driven by market data only. Text panels are
            descriptive: they show what the Fed said and how it changed, never what the price
            or volatility will do. Every claim below carries a measured macro-F1 or a
            block-bootstrap confidence interval.
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
              detail="Three-class regime (Low / Medium / High) from HAR&apos;s continuous forecast bucketed into terciles. Strongest classifier on this surface; the QLIKE-DLq ensemble beats HAR by ~10% in QLIKE loss at every horizon."
              metric="HAR-tercile macro-F1  0.687 (1d) · 0.685 (1w) · 0.654 (1m)"
              source="Pooled 5-fold expanding walk-forward eval, n=1999"
            />
            <Predicts
              title="Forward trading volume"
              detail="HAR-style baseline applied to log-volume residuals after removing weekly seasonality and FOMC-calendar effects. Lands on the Workspace as the &ldquo;Expected Volume&rdquo; card."
              metric="Market-only R²  ≈ 0.85–0.88 at 1-day"
              source="Abnormal-volume forecast under the same walk-forward eval"
            />
            <Predicts
              title="Conformal coverage on the vol bands"
              detail="The 80% and 90% prediction bands on the QLIKE-RV card are calibrated against held-out residuals. Empirical coverage tracks the nominal target on resolved runs."
              metric="Empirical 90% band coverage  ≈ 85–92%"
              source="Conformal calibration on the QLIKE-DLq ensemble residuals"
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
              detail="Across four encoders (finbert_fed_adjacent, bge-large, e5-large, gte-large), adding FOMC text to the next-day vol forecaster produced no gain. The un-regularized fusion took a small but consistent loss; correctly gating the fusion (below) removes the drag without adding any signal."
              metric="Text-vs-market 95% block CI  [−0.022, −0.009] at 1-day"
              source="Block-bootstrap incremental CI, four-encoder mean, daily n=1999 (un-regularized fusion)"
            />
            <DoesNotPredict
              title="Volatility regime FROM text"
              detail="The late-fusion classifier is rendered on the Workspace as a second opinion. With output-level residual fusion the learned gate collapses to ≈0.01: the model is free to use FOMC text and learns to ignore it, landing at its own market-only level. HAR-tercile is still the headline because it&apos;s the better predictor."
              metric="Late-fusion macro-F1  0.629 (1d). Text-neutral, ~0.06 below HAR-tercile"
              source="Pooled walk-forward eval; fused 0.629 vs gate-off market-only 0.631 (gate≈0.01)"
            />
            <DoesNotPredict
              title="Surprise → drift channel"
              detail="A LoRA fine-tune on the FOMC corpus testing whether text-surprise predicts post-meeting drift returned null. McNemar test against the market baseline failed to reject equivalence."
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
            The dashboard&apos;s text surfaces (hawkish/dovish stance, sentiment breakdown,
            sentence-level XAI, semantic diff against the previous statement, historical analogs)
            do not forecast the market. They answer a different, honest question:{" "}
            <span className="text-foreground">
              what did the Fed say, how did the wording change, and how does it compare to past
              statements?
            </span>{" "}
            That&apos;s a legitimate descriptive job and the panels are calibrated against the
            labelled FOMC corpus (gtfintechlab&apos;s Trillion-Dollar-Words derivatives).
          </p>
          <p>
            The discipline is visual on every card: forecast surfaces carry a measured macro-F1 or
            QLIKE-vs-HAR chip; text surfaces carry no forecasting metric and never imply they feed
            the price or vol forecast. XAI explains the stance label, not the price.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
