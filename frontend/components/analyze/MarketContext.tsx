import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatPrice } from "@/lib/analyze/format";
import type { AnalyzeResult } from "@/lib/analyze/types";

interface MarketContextProps {
  result: AnalyzeResult;
}

export function MarketContext({ result }: MarketContextProps) {
  const market = result.market || {};
  const model = result.model || {};
  const items: Array<{ label: string; value: string }> = [
    { label: "Symbol", value: market.symbol || "—" },
    { label: "Requested date", value: market.requested_date || "—" },
    { label: "Trading date used", value: market.date_used || "—" },
    { label: "Close", value: formatPrice(market.close) },
    {
      label: "5d volatility proxy",
      value: market.volatility_5d == null ? "—" : Number(market.volatility_5d).toFixed(6),
    },
    {
      label: "Hidden size",
      value: model.hidden_size == null ? "—" : String(model.hidden_size),
    },
    {
      label: "Layers",
      value: model.num_layers == null ? "—" : String(model.num_layers),
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Market & model context</CardTitle>
        <CardDescription>Inputs and runtime diagnostics for this run.</CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {items.map((item) => (
            <div key={item.label} className="rounded-md border border-border bg-muted/30 px-3 py-2">
              <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{item.label}</dt>
              <dd className="mt-0.5 font-medium text-foreground">{item.value}</dd>
            </div>
          ))}
        </dl>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Badge variant={model.checkpoint_loaded ? "hawkish" : "outline"}>
            {model.checkpoint_loaded ? "Checkpoint loaded" : "No checkpoint"}
          </Badge>
          {model.runtime_mode ? <Badge variant="outline">{model.runtime_mode}</Badge> : null}
        </div>
      </CardContent>
    </Card>
  );
}
