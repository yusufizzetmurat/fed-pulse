import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { errorToneLabel, formatPrice, getErrorTone } from "@/lib/analyze/format";
import type { ErrorBundle } from "@/lib/analyze/derive";
import type { AnalyzeResult } from "@/lib/analyze/types";

interface ErrorBadgesProps {
  result: AnalyzeResult;
  metrics: ErrorBundle;
}

function toneClass(tone: "low" | "medium" | "high" | "neutral"): string {
  if (tone === "low") return "border-hawkish/40 bg-hawkish/10 text-hawkish";
  if (tone === "medium") return "border-amber-500/40 bg-amber-500/10 text-amber-500";
  if (tone === "high") return "border-dovish/40 bg-dovish/10 text-dovish";
  return "border-border bg-muted/50 text-muted-foreground";
}

export function ErrorBadges({ result, metrics }: ErrorBadgesProps) {
  if (!metrics.hasRealized) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Forecast error</CardTitle>
          <CardDescription>
            Enable the realized overlay on a past date to compute MAPE and RMSE.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const closeBaseline = Math.max(
    Math.abs(Number(result.market?.close ?? result.prediction?.close ?? 0)),
    1e-6
  );
  const volBaseline = Math.max(
    Math.abs(Number(result.market?.volatility_5d ?? result.prediction?.volatility ?? 0)),
    1e-6
  );

  const items = [
    {
      label: "Close MAPE",
      value: metrics.close.mape == null ? "N/A" : `${metrics.close.mape.toFixed(2)}%`,
      tone: getErrorTone("mape", metrics.close.mape),
      meta: "Avg absolute % miss vs realized.",
    },
    {
      label: "Close RMSE",
      value: metrics.close.rmse == null ? "N/A" : metrics.close.rmse.toFixed(4),
      tone: getErrorTone("rmse", metrics.close.rmse, closeBaseline),
      meta: `~${((Number(metrics.close.rmse ?? 0) / closeBaseline) * 100).toFixed(2)}% of ${formatPrice(closeBaseline)}.`,
    },
    {
      label: "Vol MAPE",
      value: metrics.vol.mape == null ? "N/A" : `${metrics.vol.mape.toFixed(2)}%`,
      tone: getErrorTone("mape", metrics.vol.mape),
      meta: "Avg absolute % miss vs realized vol.",
    },
    {
      label: "Vol RMSE",
      value: metrics.vol.rmse == null ? "N/A" : metrics.vol.rmse.toFixed(6),
      tone: getErrorTone("rmse", metrics.vol.rmse, volBaseline),
      meta: "Relative to 5d realized-vol baseline.",
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Forecast error</CardTitle>
        <CardDescription>Realized overlay diagnostics for this run.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {items.map((item) => (
            <div
              key={item.label}
              className={`rounded-md border px-3 py-3 ${toneClass(item.tone)}`}
            >
              <div className="flex items-center justify-between text-[11px] font-medium uppercase tracking-wide">
                <span>{item.label}</span>
                <span>{errorToneLabel(item.tone)}</span>
              </div>
              <div className="mt-1 text-xl font-semibold">{item.value}</div>
              <p className="mt-1 text-xs opacity-80">{item.meta}</p>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
