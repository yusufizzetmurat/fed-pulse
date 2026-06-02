// Centralised error-to-message mapper used by the workspace pages.
// The product copy spec splits failures into three buckets — model
// unavailable, bad input, network error — and asks each toast to read
// out of that vocabulary rather than the raw axios / fetch error text.
// Keeping the mapping in one place means a new fetch site only has to
// remember to call ``categorizeError`` rather than reinvent the rule.

export type ErrorCategory = "model_unavailable" | "bad_input" | "network" | "not_found";

interface AxiosLikeError {
  response?: {
    status?: number;
    data?: { detail?: string | { error?: string; message?: string } };
  };
  request?: unknown;
  message?: string;
  code?: string;
}

const MODEL_UNAVAILABLE_STATUSES = new Set([409, 500, 502, 503, 504]);
const BAD_INPUT_STATUSES = new Set([400, 422]);
const NOT_FOUND_STATUSES = new Set([404]);

const NETWORK_CODES = new Set([
  "ERR_NETWORK",
  "ECONNABORTED",
  "ECONNREFUSED",
  "ENOTFOUND",
  "ETIMEDOUT",
]);

const MODEL_UNAVAILABLE_COPY = "Model unavailable. Try again later or check the Settings page.";
const NETWORK_COPY = "Network error. Check your connection and try again.";
const NOT_FOUND_COPY = "Not found.";
const FALLBACK_COPY = "Something went wrong. Try again.";

function _extractDetail(err: AxiosLikeError): string | null {
  // Backend handlers return one of:
  //   detail: "human readable string"
  //   detail: { error: "code", message: "human readable string" }
  // Both forms should surface to the user; the dict form is the common
  // shape for the per-symbol unsupported / unavailable rejections.
  const detail = err.response?.data?.detail;
  if (typeof detail === "string" && detail.trim().length > 0) return detail;
  if (detail && typeof detail === "object" && typeof (detail as { message?: string }).message === "string") {
    const msg = (detail as { message: string }).message.trim();
    if (msg.length > 0) return msg;
  }
  return null;
}

export function categorizeError(err: unknown): { category: ErrorCategory; message: string } {
  if (!err) return { category: "network", message: FALLBACK_COPY };
  const axiosErr = err as AxiosLikeError;

  // Network failures: axios sets ``request`` but no ``response`` when
  // the backend was never reached. ``code`` covers the abort/timeout
  // path. ``fetch`` rejects with a generic TypeError ("Failed to
  // fetch") in that same shape.
  if (axiosErr.code && NETWORK_CODES.has(axiosErr.code)) {
    return { category: "network", message: NETWORK_COPY };
  }
  if (axiosErr.request && !axiosErr.response) {
    return { category: "network", message: NETWORK_COPY };
  }
  if (axiosErr.message && /network|failed to fetch|connection|fetch failed/i.test(axiosErr.message)) {
    if (!axiosErr.response) {
      return { category: "network", message: NETWORK_COPY };
    }
  }

  const status = axiosErr.response?.status;
  if (status != null) {
    if (BAD_INPUT_STATUSES.has(status)) {
      return {
        category: "bad_input",
        message: _extractDetail(axiosErr) || "Request rejected. Check the inputs and try again.",
      };
    }
    if (NOT_FOUND_STATUSES.has(status)) {
      return {
        category: "not_found",
        message: _extractDetail(axiosErr) || NOT_FOUND_COPY,
      };
    }
    if (MODEL_UNAVAILABLE_STATUSES.has(status)) {
      // Backend services frequently return a structured 503 with
      // {detail: {error, message}} for "data unavailable for this symbol"
      // (e.g. FX tickers without yfinance volume). Surface the inner
      // message when present so the user sees the actual reason rather
      // than the generic "Model unavailable" fallback.
      return {
        category: "model_unavailable",
        message: _extractDetail(axiosErr) || MODEL_UNAVAILABLE_COPY,
      };
    }
  }

  return { category: "model_unavailable", message: MODEL_UNAVAILABLE_COPY };
}

export function errorMessage(err: unknown, fallback?: string): string {
  const { message } = categorizeError(err);
  if (message === FALLBACK_COPY && fallback) return fallback;
  return message;
}
