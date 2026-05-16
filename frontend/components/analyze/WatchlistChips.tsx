import * as React from "react";
import { Plus, X } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { addToWatchlist, readWatchlist, removeFromWatchlist } from "@/lib/watchlist";

interface WatchlistChipsProps {
  currentSymbol: string;
  onSelect: (symbol: string) => void;
}

export function WatchlistChips({ currentSymbol, onSelect }: WatchlistChipsProps) {
  const [symbols, setSymbols] = React.useState<string[]>([]);
  const [hydrated, setHydrated] = React.useState(false);

  React.useEffect(() => {
    setSymbols(readWatchlist());
    setHydrated(true);
  }, []);

  if (!hydrated) return null;

  const handleAdd = () => {
    if (!currentSymbol) return;
    setSymbols(addToWatchlist(currentSymbol));
  };
  const handleRemove = (symbol: string) => {
    setSymbols(removeFromWatchlist(symbol));
  };

  const alreadyPinned = symbols.includes(currentSymbol);

  return (
    <div className="flex flex-wrap items-center gap-2">
      {symbols.map((symbol) => (
        <Badge key={symbol} variant={symbol === currentSymbol ? "default" : "outline"} className="gap-1">
          <button
            type="button"
            onClick={() => onSelect(symbol)}
            aria-label={`Select ${symbol}`}
            aria-pressed={symbol === currentSymbol}
            className="rounded-sm font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background"
          >
            {symbol}
          </button>
          <button
            type="button"
            aria-label={`Remove ${symbol} from watchlist`}
            onClick={() => handleRemove(symbol)}
            className="rounded-sm opacity-60 hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background"
          >
            <X className="h-3 w-3" aria-hidden="true" />
          </button>
        </Badge>
      ))}
      <Button
        type="button"
        size="sm"
        variant="ghost"
        onClick={handleAdd}
        disabled={alreadyPinned || !currentSymbol}
        className="h-7 px-2 text-xs"
      >
        <Plus className="h-3 w-3" />
        Pin {currentSymbol}
      </Button>
    </div>
  );
}
