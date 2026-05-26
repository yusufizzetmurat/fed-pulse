import type { Horizon } from "@/lib/analyze/types";

const STORAGE_KEY = "fedpulse:workspace-prefs:v1";

export const HORIZON_VALUES: readonly Horizon[] = ["1d", "3d", "5d", "10d"] as const;
export const DEFAULT_SYMBOL = "^GSPC";
export const DEFAULT_HORIZON: Horizon = "10d";

export interface WorkspacePrefs {
  defaultSymbol: string;
  defaultHorizon: Horizon;
}

function isHorizon(value: unknown): value is Horizon {
  return typeof value === "string" && (HORIZON_VALUES as readonly string[]).includes(value);
}

/**
 * Load workspace prefs from localStorage. SSR-safe (returns defaults
 * when `window` is undefined) and tolerant of corrupt JSON / unexpected
 * shape — the workspace falls back to the hardcoded defaults rather
 * than crashing on first render.
 */
export function loadWorkspacePrefs(): WorkspacePrefs {
  const fallback: WorkspacePrefs = {
    defaultSymbol: DEFAULT_SYMBOL,
    defaultHorizon: DEFAULT_HORIZON,
  };
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return fallback;
    const parsed = JSON.parse(raw) as Partial<WorkspacePrefs>;
    const defaultSymbol =
      typeof parsed.defaultSymbol === "string" && parsed.defaultSymbol
        ? parsed.defaultSymbol
        : fallback.defaultSymbol;
    const defaultHorizon = isHorizon(parsed.defaultHorizon)
      ? parsed.defaultHorizon
      : fallback.defaultHorizon;
    return { defaultSymbol, defaultHorizon };
  } catch {
    return fallback;
  }
}

export function saveWorkspacePrefs(prefs: WorkspacePrefs): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // localStorage can throw in private-mode browsers / quota; treat
    // persistence as best-effort and let the next save try again.
  }
}
