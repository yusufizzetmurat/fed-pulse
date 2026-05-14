import { describe, expect, it } from "vitest";

import {
  addToWatchlist,
  clearWatchlist,
  readWatchlist,
  removeFromWatchlist,
  writeWatchlist,
  type WatchlistStorage,
} from "@/lib/watchlist";

function makeStorage(initial: Record<string, string> = {}): WatchlistStorage {
  const store = new Map<string, string>(Object.entries(initial));
  return {
    getItem: (key) => (store.has(key) ? store.get(key)! : null),
    setItem: (key, value) => {
      store.set(key, value);
    },
    removeItem: (key) => {
      store.delete(key);
    },
  };
}

describe("watchlist localStorage helper", () => {
  it("reads an empty list when nothing is stored", () => {
    const storage = makeStorage();
    expect(readWatchlist(storage)).toEqual([]);
  });

  it("ignores malformed JSON in storage", () => {
    const storage = makeStorage({ "fed-pulse:watchlist:v1": "not-json" });
    expect(readWatchlist(storage)).toEqual([]);
  });

  it("adds and dedupes symbols", () => {
    const storage = makeStorage();
    expect(addToWatchlist("^GSPC", storage)).toEqual(["^GSPC"]);
    expect(addToWatchlist("^NDX", storage)).toEqual(["^GSPC", "^NDX"]);
    expect(addToWatchlist("^GSPC", storage)).toEqual(["^GSPC", "^NDX"]);
  });

  it("removes symbols", () => {
    const storage = makeStorage();
    writeWatchlist(["^GSPC", "^NDX"], storage);
    expect(removeFromWatchlist("^GSPC", storage)).toEqual(["^NDX"]);
  });

  it("caps the list at 16 items", () => {
    const storage = makeStorage();
    const tickers = Array.from({ length: 20 }, (_, i) => `T${i}`);
    expect(writeWatchlist(tickers, storage)).toHaveLength(16);
  });

  it("clearWatchlist wipes the entry", () => {
    const storage = makeStorage();
    writeWatchlist(["^GSPC"], storage);
    clearWatchlist(storage);
    expect(readWatchlist(storage)).toEqual([]);
  });
});
