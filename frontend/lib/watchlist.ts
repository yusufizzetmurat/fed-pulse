const STORAGE_KEY = "fed-pulse:watchlist:v1";
const MAX_ITEMS = 16;

export interface WatchlistStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function defaultStorage(): WatchlistStorage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function readWatchlist(storage: WatchlistStorage | null = defaultStorage()): string[] {
  if (!storage) return [];
  const raw = storage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter((entry): entry is string => typeof entry === "string");
  } catch {
    return [];
  }
}

export function writeWatchlist(
  symbols: string[],
  storage: WatchlistStorage | null = defaultStorage()
): string[] {
  const unique: string[] = [];
  for (const sym of symbols) {
    if (!sym) continue;
    if (!unique.includes(sym)) unique.push(sym);
    if (unique.length >= MAX_ITEMS) break;
  }
  if (!storage) return unique;
  storage.setItem(STORAGE_KEY, JSON.stringify(unique));
  return unique;
}

export function addToWatchlist(
  symbol: string,
  storage: WatchlistStorage | null = defaultStorage()
): string[] {
  if (!symbol) return readWatchlist(storage);
  const current = readWatchlist(storage);
  if (current.includes(symbol)) return current;
  return writeWatchlist([...current, symbol], storage);
}

export function removeFromWatchlist(
  symbol: string,
  storage: WatchlistStorage | null = defaultStorage()
): string[] {
  const current = readWatchlist(storage);
  const next = current.filter((entry) => entry !== symbol);
  return writeWatchlist(next, storage);
}

export function clearWatchlist(storage: WatchlistStorage | null = defaultStorage()): void {
  if (!storage) return;
  storage.removeItem(STORAGE_KEY);
}
