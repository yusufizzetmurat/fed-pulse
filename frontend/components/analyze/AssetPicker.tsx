import * as React from "react";

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSharedSymbols } from "@/lib/analyze/shared-context";
import type { SymbolDescriptor } from "@/lib/analyze/types";

interface AssetPickerProps {
  id?: string;
  value: string;
  onChange: (next: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

function groupByCategory(symbols: SymbolDescriptor[]): Array<[string, SymbolDescriptor[]]> {
  const buckets = new Map<string, SymbolDescriptor[]>();
  for (const entry of symbols) {
    const key = entry.category || "Other";
    const list = buckets.get(key) ?? [];
    list.push(entry);
    buckets.set(key, list);
  }
  return [...buckets.entries()];
}

export function AssetPicker({
  id,
  value,
  onChange,
  disabled,
  placeholder = "Pick an asset",
}: AssetPickerProps) {
  const { symbols } = useSharedSymbols();
  // Inject the current value if /symbols hasn't returned it yet so the
  // controlled Select never renders an empty trigger after a deep-link.
  const expanded = React.useMemo(() => {
    if (!value) return symbols;
    if (symbols.some((entry) => entry.symbol === value)) return symbols;
    return [...symbols, { symbol: value, name: value, category: "Other", default_horizon: "10d" }];
  }, [symbols, value]);
  const groups = React.useMemo(() => groupByCategory(expanded), [expanded]);

  return (
    <Select value={value} onValueChange={onChange} disabled={disabled}>
      <SelectTrigger id={id}>
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent className="max-h-72">
        {groups.map(([category, entries]) => (
          <SelectGroup key={category}>
            <SelectLabel className="text-[10px] uppercase tracking-wide text-muted-foreground">
              {category}
            </SelectLabel>
            {entries.map((entry) => (
              <SelectItem key={entry.symbol} value={entry.symbol}>
                <span className="flex items-center justify-between gap-2">
                  <span className="numeric text-xs">{entry.symbol}</span>
                  <span className="text-xs text-muted-foreground">{entry.name}</span>
                </span>
              </SelectItem>
            ))}
          </SelectGroup>
        ))}
      </SelectContent>
    </Select>
  );
}
