import * as React from "react";

import { DocumentIngestionTabs } from "@/components/analyze/DocumentIngestionTabs";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { HORIZON_OPTIONS, SYMBOL_OPTIONS } from "@/lib/analyze/constants";
import type { AnalyzeRequest, ForecastMode, Horizon } from "@/lib/analyze/types";

interface AnalyzeFormProps {
  value: AnalyzeRequest;
  onChange: (next: AnalyzeRequest) => void;
  onSubmit: () => void;
  loading: boolean;
}

const MODE_OPTIONS: Array<{ value: ForecastMode; label: string; description: string }> = [
  { value: "fast", label: "Fast", description: "Checkpoint inference, low latency." },
  { value: "quick_train", label: "Quick Train", description: "Short bounded adaptation before inference." },
  { value: "real_train", label: "Real Train", description: "Async 252-day fine-tune, polled to completion." },
];

export function AnalyzeForm({ value, onChange, onSubmit, loading }: AnalyzeFormProps) {
  const submitLabel = loading
    ? value.forecast_mode === "real_train"
      ? "Running Real Train…"
      : "Running analysis…"
    : "Analyze";

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit();
  };

  const patch = (delta: Partial<AnalyzeRequest>) => onChange({ ...value, ...delta });

  return (
    <Card>
      <CardHeader>
        <CardTitle>Document</CardTitle>
        <CardDescription>
          Paste an FOMC excerpt, choose the asset and horizon, run the forecast.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-6">
          <DocumentIngestionTabs
            text={value.text}
            onChange={(text) => patch({ text })}
          />

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-2">
              <Label htmlFor="date">Document date</Label>
              <Input
                id="date"
                type="date"
                required
                value={value.date}
                onChange={(event) => patch({ date: event.target.value })}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="symbol">Asset</Label>
              <Select value={value.symbol} onValueChange={(next) => patch({ symbol: next })}>
                <SelectTrigger id="symbol">
                  <SelectValue placeholder="Pick a benchmark" />
                </SelectTrigger>
                <SelectContent>
                  {SYMBOL_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label htmlFor="mode">Forecast mode</Label>
              <Select
                value={value.forecast_mode}
                onValueChange={(next) => patch({ forecast_mode: next as ForecastMode })}
              >
                <SelectTrigger id="mode">
                  <SelectValue placeholder="Mode" />
                </SelectTrigger>
                <SelectContent>
                  {MODE_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {MODE_OPTIONS.find((m) => m.value === value.forecast_mode)?.description}
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="horizon">Horizon</Label>
              <Select
                value={value.horizon}
                onValueChange={(next) => patch({ horizon: next as Horizon })}
              >
                <SelectTrigger id="horizon">
                  <SelectValue placeholder="Horizon" />
                </SelectTrigger>
                <SelectContent>
                  {HORIZON_OPTIONS.map((horizon) => (
                    <SelectItem key={horizon} value={horizon}>
                      {horizon}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="flex items-center justify-between gap-4 rounded-md border border-dashed border-border bg-muted/30 px-4 py-3">
            <label htmlFor="realized" className="flex items-center gap-2 text-sm">
              <input
                id="realized"
                type="checkbox"
                checked={value.include_realized}
                onChange={(event) => patch({ include_realized: event.target.checked })}
                className="h-4 w-4 rounded border-border bg-background"
              />
              Overlay realized observations (past dates)
            </label>
            <p className="hidden text-xs text-muted-foreground sm:block">
              Used to compute MAPE / RMSE vs the forecast.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button type="submit" disabled={loading}>
              {submitLabel}
            </Button>
            <p className="text-xs text-muted-foreground">
              Real Train returns a job id and polls every {2}s.
            </p>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
