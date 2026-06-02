import * as React from "react";

import { AssetPicker } from "@/components/analyze/AssetPicker";
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { HORIZON_OPTIONS } from "@/lib/analyze/constants";
import { SAMPLE_STATEMENTS } from "@/lib/analyze/sample-statements";
import type { AnalyzeRequest, Horizon } from "@/lib/analyze/types";

interface AnalyzeFormProps {
  value: AnalyzeRequest;
  onChange: (next: AnalyzeRequest) => void;
  onSubmit: () => void;
  loading: boolean;
  // Fires when the user picks an entry from the sample-loader dropdown.
  // The parent uses this hook to clear stale analysis cards before the
  // new sample's request takes effect; falls back to onChange when the
  // host page does not need that behaviour.
  onSampleLoad?: (next: AnalyzeRequest) => void;
}

// Exported so the picker behavior is unit-testable independent of the
// Radix Select primitive (which is awkward to drive in jsdom).
export function applySampleStatement(
  base: AnalyzeRequest,
  sampleId: string,
): AnalyzeRequest {
  const sample = SAMPLE_STATEMENTS.find((entry) => entry.id === sampleId);
  if (!sample) return base;
  return {
    ...base,
    text: sample.text,
    date: sample.date,
    symbol: sample.symbol ?? "^GSPC",
  };
}

export function AnalyzeForm({ value, onChange, onSubmit, loading, onSampleLoad }: AnalyzeFormProps) {
  const submitLabel = loading ? "Running analysis…" : "Analyze";

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

          <div className="grid gap-4 md:grid-cols-3">
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
              <div className="flex items-center gap-1">
                <Label htmlFor="symbol">Asset</Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      aria-label="What does the asset picker affect?"
                      className="text-muted-foreground hover:text-foreground text-xs leading-none"
                    >
                      ⓘ
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    Drives the market-data panel and the autoregressive price /
                    volatility forecast curve. Statement-level analysis (regime
                    classification, multi-axis sentiment, policy action,
                    credibility) is asset-independent and does not change with
                    this selection.
                  </TooltipContent>
                </Tooltip>
              </div>
              <AssetPicker
                id="symbol"
                value={value.symbol}
                onChange={(next) => patch({ symbol: next })}
              />
            </div>

            <div className="space-y-2">
              <div className="flex items-center gap-1">
                <Label htmlFor="horizon">Horizon</Label>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      aria-label="What does the horizon picker affect?"
                      className="text-muted-foreground hover:text-foreground text-xs leading-none"
                    >
                      ⓘ
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    Drives the price / volatility forecast curve (the
                    autoregressive prediction block). The regime classification
                    always reports the next 10 trading days, independent of
                    this selection.
                  </TooltipContent>
                </Tooltip>
              </div>
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

          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
            <Button
              type="submit"
              disabled={loading}
              className="min-h-[44px] w-full sm:min-h-9 sm:w-auto"
            >
              {submitLabel}
            </Button>
            <Select
              value=""
              onValueChange={(id) => {
                const next = applySampleStatement(value, id);
                (onSampleLoad ?? onChange)(next);
              }}
            >
              <SelectTrigger
                className="h-9 min-h-[44px] w-full sm:min-h-9 sm:w-[16rem]"
                aria-label="Load a sample FOMC statement"
              >
                <SelectValue placeholder="Load sample statement…" />
              </SelectTrigger>
              <SelectContent>
                {SAMPLE_STATEMENTS.map((sample) => (
                  <SelectItem key={sample.id} value={sample.id}>
                    {sample.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Runs the Volatility Regime prediction along with the sentiment breakdown,
              explanation, and credibility checks against the FOMC excerpt.
            </p>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
