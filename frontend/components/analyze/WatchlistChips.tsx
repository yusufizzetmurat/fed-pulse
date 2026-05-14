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
          <button type="button" onClick={() => onSelect(symbol)} className="font-medium">
            {symbol}
          </button>
          <button
            type="button"
            aria-label={`Remove ${symbol} from watchlist`}
            onClick={() => handleRemove(symbol)}
            className="opacity-60 hover:opacity-100"
          >
            <X className="h-3 w-3" />
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
