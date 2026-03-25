import { useMemo, useState } from "react";
import axios from "axios";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const DEFAULT_TEXT =
  "Recent indicators suggest economic activity has continued to expand at a solid pace.";
const DEFAULT_DATE = new Date().toISOString().slice(0, 10);
const SYMBOL_OPTIONS = [
  { value: "^GSPC", label: "S&P 500 (^GSPC)" },
  { value: "DX-Y.NYB", label: "Dollar Index (DX-Y.NYB)" },
  { value: "^NDX", label: "NASDAQ 100 (^NDX)" },
  { value: "^DJI", label: "Dow Jones (^DJI)" },
  { value: "^VIX", label: "CBOE Volatility Index (^VIX)" },
  { value: "GC=F", label: "Gold Futures (GC=F)" },
  { value: "CL=F", label: "WTI Crude Oil (CL=F)" },
  { value: "EURUSD=X", label: "EUR/USD (EURUSD=X)" },
  { value: "BTC-USD", label: "Bitcoin (BTC-USD)" },
  { value: "^TNX", label: "US 10Y Yield (^TNX)" },
];
const HORIZON_OPTIONS = ["1d", "3d", "5d", "10d"];

function toSignalLabel(label) {
  const value = String(label || "").toUpperCase();
  if (value.includes("POSITIVE") || value.includes("LABEL_1")) {
    return "Hawkish";
  }
  if (value.includes("NEGATIVE") || value.includes("LABEL_0")) {
    return "Dovish";
  }
  return value || "Unknown";
}

function getErrorTone(kind, value, baseline = 0) {
  if (value == null || !Number.isFinite(value)) {
    return "neutral";
  }

  if (kind === "mape") {
    if (value <= 2) {
      return "low";
    }
    if (value <= 5) {
      return "medium";
    }
    return "high";
  }

  const normalized = baseline > 0 ? (value / baseline) * 100 : value;
  if (normalized <= 1) {
    return "low";
  }
  if (normalized <= 2.5) {
    return "medium";
  }
  return "high";
}

function getErrorToneLabel(tone) {
  if (tone === "low") {
    return "Low error";
  }
  if (tone === "medium") {
    return "Medium error";
  }
  if (tone === "high") {
    return "High error";
  }
  return "Awaiting data";
}

function toNumericOrNull(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeTimestamp(value) {
  if (!value) {
    return "";
  }
  // Keep date-like keys stable across forecast and realized series.
  return String(value).split("+")[0];
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

export default function Home() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [date, setDate] = useState(DEFAULT_DATE);
  const [symbol, setSymbol] = useState("^GSPC");
  const [forecastMode, setForecastMode] = useState("fast");
  const [horizon, setHorizon] = useState("3d");
  const [includeRealized, setIncludeRealized] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [trainJob, setTrainJob] = useState(null);

  const apiBaseUrl = useMemo(() => {
    const raw = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    // `backend` is resolvable in Docker network, but not in host browser.
    if (typeof window !== "undefined" && raw.includes("://backend:")) {
      return raw.replace("://backend:", "://localhost:");
    }
    return raw;
  }, []);

  const handleAnalyze = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setTrainJob(null);

    try {
      const response = await axios.post(`${apiBaseUrl}/analyze`, {
        text,
        date,
        symbol,
        forecast_mode: forecastMode,
        horizon,
        include_realized: includeRealized,
      });
      if (forecastMode === "real_train") {
        const jobId = response?.data?.job_id;
        if (!jobId) {
          throw new Error("Real Train did not return a job id.");
        }
        setResult(null);
        setTrainJob({
          job_id: jobId,
          status: response?.data?.status || "queued",
          message: response?.data?.message || "Real Train queued.",
          error: null,
        });

        const maxPollCount = 180;
        for (let pollIndex = 0; pollIndex < maxPollCount; pollIndex += 1) {
          await sleep(2000);
          const statusResponse = await axios.get(`${apiBaseUrl}/train-jobs/${jobId}`);
          const state = statusResponse?.data || {};
          setTrainJob((current) => ({
            ...(current || {}),
            ...state,
            message:
              state.status === "running"
                ? "Real Train is running with 252-day history and writing checkpoint..."
                : state.status === "queued"
                ? "Real Train is queued..."
                : state.status === "succeeded"
                ? "Real Train completed. Rendering results."
                : current?.message || "",
          }));

          if (state.status === "succeeded") {
            setResult(state.result || null);
            setLoading(false);
            return;
          }
          if (state.status === "failed") {
            throw new Error(state.error || "Real Train job failed.");
          }
        }

        throw new Error("Real Train timed out while waiting for completion.");
      }

      setResult(response.data || null);
    } catch (requestError) {
      setResult(null);
      setError(
        requestError?.response?.data?.detail ||
          "Request failed. Ensure the backend container is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const closeSeries = useMemo(() => {
    if (!result?.series) {
      return [];
    }
    const byTimestamp = new Map();
    const ensureRow = (rawTimestamp) => {
      const timestamp = normalizeTimestamp(rawTimestamp);
      if (!timestamp) {
        return null;
      }
      if (!byTimestamp.has(timestamp)) {
        byTimestamp.set(timestamp, {
          timestamp,
          history: null,
          forecast: null,
          forecastLower: null,
          forecastUpper: null,
          forecastBand: null,
          realized: null,
        });
      }
      return byTimestamp.get(timestamp);
    };

    (result.series.timestamps || []).forEach((timestamp, index) => {
      const row = ensureRow(timestamp);
      if (row) {
        row.history = toNumericOrNull(result.series.history_close?.[index]);
      }
    });
    (result.series.forecast_timestamps || []).forEach((timestamp, index) => {
      const row = ensureRow(timestamp);
      const forecast = toNumericOrNull(result.series.forecast_close?.[index]);
      const forecastLower = toNumericOrNull(result.series.forecast_close_lower?.[index]);
      const forecastUpper = toNumericOrNull(result.series.forecast_close_upper?.[index]);
      if (row) {
        row.forecast = forecast;
        row.forecastLower = forecastLower;
        row.forecastUpper = forecastUpper;
        row.forecastBand =
          forecastLower != null && forecastUpper != null
            ? Math.max(forecastUpper - forecastLower, 0)
            : null;
      }
    });
    (result.series.realized_timestamps || []).forEach((timestamp, index) => {
      const row = ensureRow(timestamp);
      if (row) {
        row.realized = toNumericOrNull(result.series.realized_close?.[index]);
      }
    });
    return Array.from(byTimestamp.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }, [result]);

  const volatilitySeries = useMemo(() => {
    if (!result?.series) {
      return [];
    }
    const byTimestamp = new Map();
    const ensureRow = (rawTimestamp) => {
      const timestamp = normalizeTimestamp(rawTimestamp);
      if (!timestamp) {
        return null;
      }
      if (!byTimestamp.has(timestamp)) {
        byTimestamp.set(timestamp, {
          timestamp,
          history: null,
          forecast: null,
          forecastLower: null,
          forecastUpper: null,
          forecastBand: null,
          realized: null,
        });
      }
      return byTimestamp.get(timestamp);
    };

    (result.series.timestamps || []).forEach((timestamp, index) => {
      const row = ensureRow(timestamp);
      if (row) {
        row.history = toNumericOrNull(result.series.history_volatility?.[index]);
      }
    });
    (result.series.forecast_timestamps || []).forEach((timestamp, index) => {
      const row = ensureRow(timestamp);
      const forecast = toNumericOrNull(result.series.forecast_volatility?.[index]);
      const forecastLower = toNumericOrNull(result.series.forecast_volatility_lower?.[index]);
      const forecastUpper = toNumericOrNull(result.series.forecast_volatility_upper?.[index]);
      if (row) {
        row.forecast = forecast;
        row.forecastLower = forecastLower;
        row.forecastUpper = forecastUpper;
        row.forecastBand =
          forecastLower != null && forecastUpper != null
            ? Math.max(forecastUpper - forecastLower, 0)
            : null;
      }
    });
    (result.series.realized_timestamps || []).forEach((timestamp, index) => {
      const row = ensureRow(timestamp);
      if (row) {
        row.realized = toNumericOrNull(result.series.realized_volatility?.[index]);
      }
    });
    return Array.from(byTimestamp.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  }, [result]);

  const closeSummary = useMemo(() => {
    if (!result?.series?.history_close?.length || !result?.series?.forecast_close?.length) {
      return { pct: 0 };
    }
    const lastHistory = Number(result.series.history_close[result.series.history_close.length - 1]);
    const lastForecast = Number(result.series.forecast_close[result.series.forecast_close.length - 1]);
    if (!lastHistory) {
      return { pct: 0 };
    }
    return { pct: ((lastForecast - lastHistory) / lastHistory) * 100 };
  }, [result]);

  const volSummary = useMemo(() => {
    if (!result?.series?.history_volatility?.length || !result?.series?.forecast_volatility?.length) {
      return { diff: 0 };
    }
    const lastHistory = Number(
      result.series.history_volatility[result.series.history_volatility.length - 1]
    );
    const lastForecast = Number(
      result.series.forecast_volatility[result.series.forecast_volatility.length - 1]
    );
    return { diff: lastForecast - lastHistory };
  }, [result]);

  const errorMetrics = useMemo(() => {
    const buildPairs = (forecastTs, forecastValues, realizedTs, realizedValues) => {
      const forecastByTs = new Map();
      (forecastTs || []).forEach((ts, idx) => {
        const key = normalizeTimestamp(ts);
        const value = toNumericOrNull(forecastValues?.[idx]);
        if (key && value != null) {
          forecastByTs.set(key, value);
        }
      });
      const pairs = [];
      (realizedTs || []).forEach((ts, idx) => {
        const key = normalizeTimestamp(ts);
        const realized = toNumericOrNull(realizedValues?.[idx]);
        const forecast = key ? forecastByTs.get(key) : null;
        if (forecast != null && realized != null) {
          pairs.push([forecast, realized]);
        }
      });
      return pairs;
    };

    const closePairs = buildPairs(
      result?.series?.forecast_timestamps,
      result?.series?.forecast_close,
      result?.series?.realized_timestamps,
      result?.series?.realized_close
    );
    const volPairs = buildPairs(
      result?.series?.forecast_timestamps,
      result?.series?.forecast_volatility,
      result?.series?.realized_timestamps,
      result?.series?.realized_volatility
    );

    const calc = (pairs) => {
      if (!pairs.length) {
        return { mape: null, rmse: null };
      }
      const mapeVals = pairs
        .filter(([, actual]) => Math.abs(actual) > 1e-12)
        .map(([predicted, actual]) => Math.abs((actual - predicted) / actual));
      const rmse =
        Math.sqrt(
          pairs.reduce((acc, [predicted, actual]) => {
            const err = actual - predicted;
            return acc + err * err;
          }, 0) / pairs.length
        ) || 0;
      const mape = mapeVals.length
        ? (mapeVals.reduce((acc, value) => acc + value, 0) / mapeVals.length) * 100
        : null;
      return { mape, rmse };
    };

    return {
      close: calc(closePairs),
      vol: calc(volPairs),
      hasRealized: Boolean(closePairs.length || volPairs.length),
    };
  }, [result]);

  const historySplitTimestamp = result?.series?.timestamps?.[result?.series?.timestamps?.length - 1];
  const volScale = result?.series?.volatility_scale || { suggested_ymin: 0.0, suggested_ymax: 1.0 };
  const forecastConfidencePct = Math.round(
    Number(result?.series?.forecast_confidence_level || 0.8) * 100
  );
  const forecastConfidenceLabel = `${forecastConfidencePct}% Confidence Range`;
  const hasCloseConfidence = Boolean(result?.series?.forecast_close_lower?.length);
  const hasVolConfidence = Boolean(result?.series?.forecast_volatility_lower?.length);

  const formatDateTick = (value) => {
    if (!value) {
      return "";
    }
    const clean = String(value).split("+")[0];
    const dateValue = new Date(clean);
    if (Number.isNaN(dateValue.getTime())) {
      return value;
    }
    return dateValue.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  };

  const formatPrice = (value) =>
    `$${Number(value || 0).toLocaleString("en-US", { maximumFractionDigits: 2 })}`;
  const formatPriceDelta = (value) =>
    `${Number(value || 0) >= 0 ? "+" : "-"}$${Math.abs(Number(value || 0)).toLocaleString("en-US", {
      maximumFractionDigits: 2,
    })}`;
  const formatPercentDelta = (value) =>
    `${Number(value || 0) >= 0 ? "+" : ""}${Number(value || 0).toFixed(2)}%`;
  const formatVol = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;
  const formatMetric = (value, digits = 6) =>
    Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "N/A";

  const closeDelta = useMemo(() => {
    const predicted = Number(result?.prediction?.close);
    const current = Number(result?.market?.close);

    if (!Number.isFinite(predicted) || !Number.isFinite(current)) {
      return null;
    }

    const delta = predicted - current;
    return {
      current,
      delta,
      pct: current ? (delta / current) * 100 : 0,
      tone: delta > 0 ? "up" : delta < 0 ? "down" : "flat",
    };
  }, [result]);

  const errorBadges = useMemo(() => {
    const closeBaseline = Math.max(
      Math.abs(Number(result?.market?.close || result?.prediction?.close || 0)),
      1e-6
    );
    const volBaseline = Math.max(
      Math.abs(Number(result?.market?.volatility_5d || result?.prediction?.volatility || 0)),
      1e-6
    );

    const formatRelativeHint = (value, baseline, label) => {
      if (value == null || !Number.isFinite(value)) {
        return "Awaiting realized observations";
      }
      return `~${((value / baseline) * 100).toFixed(2)}% of ${label}`;
    };

    return [
      {
        label: "Close MAPE",
        value:
          errorMetrics.close.mape == null ? "N/A" : `${errorMetrics.close.mape.toFixed(2)}%`,
        tone: getErrorTone("mape", errorMetrics.close.mape),
        toneLabel: getErrorToneLabel(getErrorTone("mape", errorMetrics.close.mape)),
        meta: "Percentage miss vs realized close",
      },
      {
        label: "Close RMSE",
        value: errorMetrics.close.rmse == null ? "N/A" : errorMetrics.close.rmse.toFixed(4),
        tone: getErrorTone("rmse", errorMetrics.close.rmse, closeBaseline),
        toneLabel: getErrorToneLabel(
          getErrorTone("rmse", errorMetrics.close.rmse, closeBaseline)
        ),
        meta: formatRelativeHint(errorMetrics.close.rmse, closeBaseline, "current spot"),
      },
      {
        label: "Volatility MAPE",
        value: errorMetrics.vol.mape == null ? "N/A" : `${errorMetrics.vol.mape.toFixed(2)}%`,
        tone: getErrorTone("mape", errorMetrics.vol.mape),
        toneLabel: getErrorToneLabel(getErrorTone("mape", errorMetrics.vol.mape)),
        meta: "Percentage miss vs realized volatility",
      },
      {
        label: "Volatility RMSE",
        value: errorMetrics.vol.rmse == null ? "N/A" : errorMetrics.vol.rmse.toFixed(6),
        tone: getErrorTone("rmse", errorMetrics.vol.rmse, volBaseline),
        toneLabel: getErrorToneLabel(getErrorTone("rmse", errorMetrics.vol.rmse, volBaseline)),
        meta: formatRelativeHint(errorMetrics.vol.rmse, volBaseline, "5d vol proxy"),
      },
    ];
  }, [errorMetrics, result]);

  const currentSpotBandCheck = useMemo(() => {
    const lower = toNumericOrNull(
      result?.series?.forecast_close_lower?.[result?.series?.forecast_close_lower?.length - 1]
    );
    const upper = toNumericOrNull(
      result?.series?.forecast_close_upper?.[result?.series?.forecast_close_upper?.length - 1]
    );
    const current = toNumericOrNull(result?.market?.close);
    if (lower == null || upper == null || current == null) {
      return null;
    }
    const withinBand = current >= lower && current <= upper;
    const distance = withinBand ? 0 : current < lower ? lower - current : current - upper;
    const reference = Math.max(Math.abs(current), 1e-6);
    return {
      withinBand,
      lower,
      upper,
      current,
      distance,
      distancePct: (distance / reference) * 100,
      tone: withinBand ? "good" : distance / reference <= 0.015 ? "caution" : "danger",
    };
  }, [result]);

  const realizedBandCheck = useMemo(() => {
    const forecastTs = result?.series?.forecast_timestamps || [];
    const lowerBand = result?.series?.forecast_close_lower || [];
    const upperBand = result?.series?.forecast_close_upper || [];
    const realizedTs = result?.series?.realized_timestamps || [];
    const realizedClose = result?.series?.realized_close || [];

    const forecastByTs = new Map();
    forecastTs.forEach((ts, idx) => {
      const key = normalizeTimestamp(ts);
      const lower = toNumericOrNull(lowerBand[idx]);
      const upper = toNumericOrNull(upperBand[idx]);
      if (key && lower != null && upper != null) {
        forecastByTs.set(key, { lower, upper });
      }
    });

    const overlaps = [];
    realizedTs.forEach((ts, idx) => {
      const key = normalizeTimestamp(ts);
      const realizedValue = toNumericOrNull(realizedClose[idx]);
      const band = key ? forecastByTs.get(key) : null;
      if (key && band && realizedValue != null) {
        overlaps.push({ ts: key, realizedValue, lower: band.lower, upper: band.upper });
      }
    });

    if (!overlaps.length) {
      return null;
    }
    overlaps.sort((a, b) => a.ts.localeCompare(b.ts));
    const latest = overlaps[overlaps.length - 1];
    const { realizedValue, lower, upper } = latest;

    const withinBand = realizedValue >= lower && realizedValue <= upper;
    const distance = withinBand ? 0 : realizedValue < lower ? lower - realizedValue : realizedValue - upper;
    const reference = Math.max(Math.abs(realizedValue), 1e-6);
    return {
      withinBand,
      realizedValue,
      lower,
      upper,
      distance,
      distancePct: (distance / reference) * 100,
      tone: withinBand ? "good" : distance / reference <= 0.02 ? "caution" : "danger",
    };
  }, [result]);

  const evaluationCards = useMemo(() => {
    const diagnostics = result?.model;
    if (!diagnostics) {
      return [];
    }

    const runtimeLabel =
      diagnostics.runtime_mode === "quick_train" ? "Quick-train adaptation active" : "Best checkpoint inference";
    const checkpointLabel = diagnostics.checkpoint_loaded
      ? "Saved checkpoint is loaded for analysis"
      : "No saved checkpoint found, using runtime initialization";

    return [
      {
        label: "Runtime State",
        tone: diagnostics.checkpoint_loaded ? "good" : "neutral",
        title: runtimeLabel,
        value: checkpointLabel,
        meta: `LSTM ${diagnostics.hidden_size} hidden x ${diagnostics.num_layers} layer(s), dropout ${Number(
          diagnostics.dropout || 0
        ).toFixed(2)}${
          Number.isFinite(Number(diagnostics.combined_rmse))
            ? `, best RMSE ${formatMetric(diagnostics.combined_rmse, 6)}`
            : ""
        }`,
      },
      {
        label: "Current Spot Check",
        tone: currentSpotBandCheck?.tone || "neutral",
        title: currentSpotBandCheck
          ? currentSpotBandCheck.withinBand
            ? "Current market close sits inside the projected band"
            : "Current market close sits outside the projected band"
          : "Projected band check unavailable",
        value: currentSpotBandCheck
          ? `${formatPrice(currentSpotBandCheck.lower)} to ${formatPrice(currentSpotBandCheck.upper)}`
          : "Run a forecast with confidence bands enabled",
        meta: currentSpotBandCheck
          ? currentSpotBandCheck.withinBand
            ? `Spot close ${formatPrice(currentSpotBandCheck.current)} remains within the ${forecastConfidenceLabel.toLowerCase()}.`
            : `Spot close is ${formatPercentDelta(currentSpotBandCheck.distancePct)} beyond the outer band.`
          : "Confidence bands are required to evaluate the live spot.",
      },
      {
        label: "Realized Validation",
        tone: realizedBandCheck?.tone || "neutral",
        title: realizedBandCheck
          ? realizedBandCheck.withinBand
            ? "Latest realized close landed inside the forecast band"
            : "Latest realized close broke outside the forecast band"
          : "Realized overlay not available",
        value: realizedBandCheck
          ? `${formatPrice(realizedBandCheck.lower)} to ${formatPrice(realizedBandCheck.upper)}`
          : errorMetrics.hasRealized
          ? "Realized close mismatch"
          : "Enable realized overlay for past dates",
        meta: realizedBandCheck
          ? `Realized close ${formatPrice(realizedBandCheck.realizedValue)} vs close MAPE ${
              errorMetrics.close.mape == null ? "N/A" : `${errorMetrics.close.mape.toFixed(2)}%`
            }.`
          : "Choose a historical date and enable the realized overlay to validate this checkpoint.",
      },
    ];
  }, [currentSpotBandCheck, errorMetrics, forecastConfidenceLabel, realizedBandCheck, result]);

  const renderTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) {
      return null;
    }
    const row = payload[0]?.payload || {};
    const isVolatilityChart = payload.some((item) =>
      String(item?.name || "").toLowerCase().includes("volatility")
    );
    const formatValue = isVolatilityChart ? formatVol : formatPrice;
    const historyLabel = isVolatilityChart ? "History Volatility" : "History Close";
    const realizedLabel = isVolatilityChart ? "Realized Volatility" : "Realized Close";
    const tooltipRows = [];

    if (row.history != null) {
      tooltipRows.push({ label: historyLabel, value: formatValue(row.history) });
    }
    if (row.forecast != null) {
      tooltipRows.push({ label: "Forecast Estimate", value: formatValue(row.forecast) });
      if (row.forecastLower != null && row.forecastUpper != null) {
        tooltipRows.push({
          label: forecastConfidenceLabel,
          value: `${formatValue(row.forecastLower)} - ${formatValue(row.forecastUpper)}`,
        });
      }
    }
    if (row.realized != null) {
      tooltipRows.push({ label: realizedLabel, value: formatValue(row.realized) });
    }
    return (
      <div className="chartTooltip">
        <p>{formatDateTick(label)}</p>
        {tooltipRows.map((item) => (
          <p key={`${label}-${item.label}`}>
            {item.label}: {item.value}
          </p>
        ))}
      </div>
    );
  };

  return (
    <main className="container">
      <header className="pageHeader">
        <h1>FOMC Quant Forecast Terminal</h1>
        <p className="subtitle">
          Multi-asset market outlook from central bank language, sentiment, and quantitative signals.
        </p>
      </header>

      <form onSubmit={handleAnalyze} className="card inputCard">
        <div className="rowBetween">
          <label htmlFor="statement">FOMC Statement Text</label>
          <button
            type="button"
            className="ghostBtn"
            onClick={() => setIsExpanded((current) => !current)}
          >
            {isExpanded ? "Collapse text" : "Expand text"}
          </button>
        </div>
        <textarea
          className="terminalInput"
          id="statement"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={isExpanded ? 10 : 3}
          placeholder="Paste an FOMC statement excerpt..."
          required
        />
        <div className="controlRow">
          <div className="controlGroup">
            <label htmlFor="date">Document Date</label>
            <input
              className="controlInput"
              id="date"
              type="date"
              value={date}
              onChange={(event) => setDate(event.target.value)}
              required
            />
            <p className="controlHint">Statement release date for aligning market context.</p>
          </div>
          <div className="controlGroup">
            <label htmlFor="symbol">Market Symbol</label>
            <select
              className="controlInput"
              id="symbol"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value)}
            >
              {SYMBOL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            <p className="controlHint">Choose the primary risk asset for this scenario.</p>
          </div>
          <div className="controlGroup">
            <label htmlFor="forecastMode">Forecast Mode</label>
            <select
              className="controlInput"
              id="forecastMode"
              value={forecastMode}
              onChange={(event) => setForecastMode(event.target.value)}
            >
              <option value="fast">Fast</option>
              <option value="quick_train">Quick Train</option>
              <option value="real_train">Real Train</option>
            </select>
            <p className="controlHint">
              Fast is low-latency. Quick Train adapts briefly. Real Train runs async on 252-day history.
            </p>
          </div>
          <div className="controlGroup">
            <label htmlFor="horizon">Prediction Horizon</label>
            <select
              className="controlInput"
              id="horizon"
              value={horizon}
              onChange={(event) => setHorizon(event.target.value)}
            >
              {HORIZON_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
            <p className="controlHint">Forward trading days to forecast.</p>
          </div>
        </div>
        <div className="toggleRow">
          <label htmlFor="realizedOverlay" className="toggleLabel">
            <input
              id="realizedOverlay"
              type="checkbox"
              checked={includeRealized}
              onChange={(event) => setIncludeRealized(event.target.checked)}
            />
            Show realized forward overlay (past dates only)
          </label>
          <p className="controlHint">
            Overlays forecast with realized observations when available.
          </p>
        </div>
        <div className="actionRow">
          <button type="submit" disabled={loading}>
            {loading
              ? forecastMode === "real_train"
                ? "Running Real Train..."
                : "Running Quant Analysis..."
              : "Analyze"}
          </button>
          <span className="helperText">All metadata fields stay visible after each run.</span>
        </div>
      </form>

      {error ? <p className="error">{error}</p> : null}
      {trainJob ? (
        <section className="card">
          <h2>Real Train Job</h2>
          <p>
            <strong>Job ID:</strong> {trainJob.job_id}
          </p>
          <p>
            <strong>Status:</strong> {trainJob.status || "queued"}
          </p>
          <p>{trainJob.message || "Waiting for job updates..."}</p>
          {trainJob.error ? (
            <p className="error">
              <strong>Error:</strong> {trainJob.error}
            </p>
          ) : null}
        </section>
      ) : null}

      {result ? (
        <>
          <section className="resultGrid">
            <article className="card metricCard">
              <h2>Sentiment Signal</h2>
              <p className="metricValue">{toSignalLabel(result?.sentiment?.label)}</p>
              <p className="metricMeta">
                Confidence: {Number(result?.sentiment?.score || 0).toFixed(4)}
              </p>
            </article>

            <article className="card metricCard">
              <h2>Predicted Volatility</h2>
              <p className="metricValue">
                {Number(result?.prediction?.volatility || 0).toFixed(4)}
              </p>
              <p className="metricMeta">Horizon: {result?.prediction?.horizon || "3d"}</p>
              <p className="metricSub">
                Delta vs last history: {volSummary.diff >= 0 ? "+" : ""}
                {volSummary.diff.toFixed(4)}
              </p>
            </article>

            <article className="card metricCard">
              <h2>Predicted Close</h2>
              <p className="metricValue">{formatPrice(result?.prediction?.close || 0)}</p>
              <p className="metricMeta">Symbol: {result?.market?.symbol || symbol}</p>
              <div className={`deltaHighlight deltaHighlight--${closeDelta?.tone || "flat"}`}>
                <span className="deltaLabel">Spread vs Current Market Close</span>
                <div className="deltaValueRow">
                  <strong>{closeDelta ? formatPriceDelta(closeDelta.delta) : "N/A"}</strong>
                  <span>{closeDelta ? formatPercentDelta(closeDelta.pct) : "No spot close available"}</span>
                </div>
                <p className="deltaMeta">
                  Current market close: {closeDelta ? formatPrice(closeDelta.current) : "N/A"}
                </p>
              </div>
              <p className="metricSub">
                Forecast change: {closeSummary.pct >= 0 ? "+" : ""}
                {closeSummary.pct.toFixed(2)}%
              </p>
            </article>
          </section>

          <section className="card metricBadgeCard">
            <h2>Forecast Error Snapshot</h2>
            {errorMetrics.hasRealized ? (
              <div className="badgeGrid">
                {errorBadges.map((badge) => (
                  <div key={badge.label} className={`badgeItem badgeItem--${badge.tone}`}>
                    <div className="badgeHeader">
                      <span>{badge.label}</span>
                      <small className={`badgeTone badgeTone--${badge.tone}`}>{badge.toneLabel}</small>
                    </div>
                    <strong>{badge.value}</strong>
                    <p className="badgeMeta">{badge.meta}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="metricMeta">
                Enable <strong>realized overlay</strong> and choose a past date to calculate MAPE/RMSE.
              </p>
            )}
          </section>

          <section className="card evaluationCard">
            <h2>Model Evaluation Snapshot</h2>
            <p className="chartNote">
              Uses the active checkpoint metadata plus live and realized band checks to judge whether the
              forecast is behaving credibly.
            </p>
            <div className="evaluationGrid">
              {evaluationCards.map((card) => (
                <article key={card.label} className={`evaluationItem evaluationItem--${card.tone}`}>
                  <span className="evaluationLabel">{card.label}</span>
                  <strong>{card.title}</strong>
                  <p className="evaluationValue">{card.value}</p>
                  <p className="evaluationMeta">{card.meta}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="card chartCard">
            <h2>Close Price Forecast</h2>
            <p className="chartNote">
              Forecast estimate is shown as a solid line. The shaded band marks the{" "}
              {forecastConfidenceLabel.toLowerCase()}.
            </p>
            <div className="chartWrap">
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={closeSeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2f4264" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={formatDateTick}
                    tick={{ fill: "#b8c9ea", fontSize: 12 }}
                  />
                  <YAxis tickFormatter={formatPrice} tick={{ fill: "#b8c9ea", fontSize: 12 }} />
                  <Tooltip content={renderTooltip} />
                  <Legend />
                  <ReferenceLine x={historySplitTimestamp} stroke="#93c5fd" strokeDasharray="4 4" />
                  <Area
                    type="monotone"
                    dataKey="history"
                    name="History Close"
                    stroke="#38bdf8"
                    fill="url(#closeHistoryGradient)"
                    strokeWidth={2}
                  />
                  {hasCloseConfidence ? (
                    <>
                      <Area
                        type="monotone"
                        dataKey="forecastLower"
                        stackId="closeConfidence"
                        stroke="none"
                        fill="transparent"
                        legendType="none"
                        isAnimationActive={false}
                      />
                      <Area
                        type="monotone"
                        dataKey="forecastBand"
                        name={forecastConfidenceLabel}
                        stackId="closeConfidence"
                        stroke="none"
                        fill="url(#closeConfidenceGradient)"
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="forecastUpper"
                        stroke="#fbbf24"
                        strokeOpacity={0.38}
                        strokeDasharray="6 4"
                        strokeWidth={1.25}
                        dot={false}
                        legendType="none"
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="forecastLower"
                        stroke="#fbbf24"
                        strokeOpacity={0.38}
                        strokeDasharray="6 4"
                        strokeWidth={1.25}
                        dot={false}
                        legendType="none"
                        isAnimationActive={false}
                      />
                    </>
                  ) : null}
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast Estimate"
                    stroke="#f59e0b"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 4, stroke: "#f59e0b", fill: "#fff7ed" }}
                  />
                  {includeRealized ? (
                    <Area
                      type="monotone"
                      dataKey="realized"
                      name="Realized Close"
                      stroke="#34d399"
                      fill="url(#closeRealizedGradient)"
                      strokeWidth={2}
                    />
                  ) : null}
                  <defs>
                    <linearGradient id="closeHistoryGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.42} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.03} />
                    </linearGradient>
                    <linearGradient id="closeForecastGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.36} />
                      <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="closeConfidenceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.28} />
                      <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.06} />
                    </linearGradient>
                    <linearGradient id="closeRealizedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#34d399" stopOpacity={0.30} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card chartCard">
            <h2>Volatility Forecast</h2>
            <p className="chartNote">
              Forecast estimate is shown as a solid line. The shaded band marks the{" "}
              {forecastConfidenceLabel.toLowerCase()}.
            </p>
            <div className="chartWrap">
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={volatilitySeries}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2f4264" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={formatDateTick}
                    tick={{ fill: "#b8c9ea", fontSize: 12 }}
                  />
                  <YAxis
                    domain={[volScale.suggested_ymin, volScale.suggested_ymax]}
                    tickFormatter={formatVol}
                    tick={{ fill: "#b8c9ea", fontSize: 12 }}
                  />
                  <Tooltip content={renderTooltip} />
                  <Legend />
                  <ReferenceLine x={historySplitTimestamp} stroke="#93c5fd" strokeDasharray="4 4" />
                  <Area
                    type="monotone"
                    dataKey="history"
                    name="History Volatility"
                    stroke="#22d3ee"
                    fill="url(#volHistoryGradient)"
                    strokeWidth={2}
                  />
                  {hasVolConfidence ? (
                    <>
                      <Area
                        type="monotone"
                        dataKey="forecastLower"
                        stackId="volConfidence"
                        stroke="none"
                        fill="transparent"
                        legendType="none"
                        isAnimationActive={false}
                      />
                      <Area
                        type="monotone"
                        dataKey="forecastBand"
                        name={forecastConfidenceLabel}
                        stackId="volConfidence"
                        stroke="none"
                        fill="url(#volConfidenceGradient)"
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="forecastUpper"
                        stroke="#fb7185"
                        strokeOpacity={0.42}
                        strokeDasharray="6 4"
                        strokeWidth={1.25}
                        dot={false}
                        legendType="none"
                        isAnimationActive={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="forecastLower"
                        stroke="#fb7185"
                        strokeOpacity={0.42}
                        strokeDasharray="6 4"
                        strokeWidth={1.25}
                        dot={false}
                        legendType="none"
                        isAnimationActive={false}
                      />
                    </>
                  ) : null}
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast Estimate"
                    stroke="#f43f5e"
                    strokeWidth={3}
                    dot={false}
                    activeDot={{ r: 4, stroke: "#f43f5e", fill: "#ffe4e6" }}
                  />
                  {includeRealized ? (
                    <Area
                      type="monotone"
                      dataKey="realized"
                      name="Realized Volatility"
                      stroke="#34d399"
                      fill="url(#volRealizedGradient)"
                      strokeWidth={2}
                    />
                  ) : null}
                  <defs>
                    <linearGradient id="volHistoryGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="volForecastGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.34} />
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.02} />
                    </linearGradient>
                    <linearGradient id="volConfidenceGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.24} />
                      <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.05} />
                    </linearGradient>
                    <linearGradient id="volRealizedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#34d399" stopOpacity={0.30} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card">
            <h2>Market Context</h2>
            <p>
              <strong>Symbol:</strong> {result?.market?.symbol}
            </p>
            <p>
              <strong>Requested Date:</strong> {result?.market?.requested_date}
            </p>
            <p>
              <strong>Trading Date Used:</strong> {result?.market?.date_used}
            </p>
            <p>
              <strong>Close:</strong> {Number(result?.market?.close || 0).toFixed(2)}
            </p>
            <p>
              <strong>5d Volatility Proxy:</strong>{" "}
              {Number(result?.market?.volatility_5d || 0).toFixed(6)}
            </p>
            <p>
              <strong>Forecast Mode:</strong> {forecastMode}
            </p>
            <p>
              <strong>Model Hidden Size:</strong> {result?.model?.hidden_size ?? "N/A"}
            </p>
            <p>
              <strong>Checkpoint Loaded:</strong> {result?.model?.checkpoint_loaded ? "Yes" : "No"}
            </p>
            <p>
              <strong>Prediction Horizon:</strong> {horizon}
            </p>
          </section>
        </>
      ) : null}
    </main>
  );
}
