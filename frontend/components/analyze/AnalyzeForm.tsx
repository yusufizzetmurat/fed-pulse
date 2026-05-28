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
import { HORIZON_OPTIONS } from "@/lib/analyze/constants";
import { SAMPLE_STATEMENTS } from "@/lib/analyze/sample-statements";
import type { AnalyzeRequest, Horizon } from "@/lib/analyze/types";

interface AnalyzeFormProps {
  value: AnalyzeRequest;
  onChange: (next: AnalyzeRequest) => void;
  onSubmit: () => void;
  loading: boolean;
}

export function AnalyzeForm({ value, onChange, onSubmit, loading }: AnalyzeFormProps) {
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
              <Label htmlFor="symbol">Asset</Label>
              <AssetPicker
                id="symbol"
                value={value.symbol}
                onChange={(next) => patch({ symbol: next })}
              />
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

          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
            <Button type="submit" disabled={loading} className="w-full sm:w-auto">
              {submitLabel}
            </Button>
            <Select
              value=""
              onValueChange={(id) => {
                const sample = SAMPLE_STATEMENTS.find((entry) => entry.id === id);
                if (!sample) return;
                onChange({ ...value, text: sample.text, date: sample.date });
              }}
            >
              <SelectTrigger
                className="h-9 w-full sm:w-[16rem]"
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
