import { useMemo, useState } from "react";
import axios from "axios";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
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

    try {
      const response = await axios.post(`${apiBaseUrl}/analyze`, {
        text,
        date,
        symbol,
        forecast_mode: forecastMode,
        horizon,
        include_realized: includeRealized,
      });
      setResult(response.data);
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
    const rows = [];
    (result.series.timestamps || []).forEach((timestamp, index) => {
      rows.push({
        timestamp,
        history: Number(result.series.history_close?.[index] || 0),
        forecast: null,
        realized: null,
      });
    });
    (result.series.forecast_timestamps || []).forEach((timestamp, index) => {
      rows.push({
        timestamp,
        history: null,
        forecast: Number(result.series.forecast_close?.[index] || 0),
        realized: null,
      });
    });
    (result.series.realized_timestamps || []).forEach((timestamp, index) => {
      const existing = rows.find((item) => item.timestamp === timestamp);
      const value = Number(result.series.realized_close?.[index] || 0);
      if (existing) {
        existing.realized = value;
      } else {
        rows.push({ timestamp, history: null, forecast: null, realized: value });
      }
    });
    return rows;
  }, [result]);

  const volatilitySeries = useMemo(() => {
    if (!result?.series) {
      return [];
    }
    const rows = [];
    (result.series.timestamps || []).forEach((timestamp, index) => {
      rows.push({
        timestamp,
        history: Number(result.series.history_volatility?.[index] || 0),
        forecast: null,
        realized: null,
      });
    });
    (result.series.forecast_timestamps || []).forEach((timestamp, index) => {
      rows.push({
        timestamp,
        history: null,
        forecast: Number(result.series.forecast_volatility?.[index] || 0),
        realized: null,
      });
    });
    (result.series.realized_timestamps || []).forEach((timestamp, index) => {
      const existing = rows.find((item) => item.timestamp === timestamp);
      const value = Number(result.series.realized_volatility?.[index] || 0);
      if (existing) {
        existing.realized = value;
      } else {
        rows.push({ timestamp, history: null, forecast: null, realized: value });
      }
    });
    return rows;
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
    const forecastClose = result?.series?.forecast_close || [];
    const realizedClose = result?.series?.realized_close || [];
    const forecastVol = result?.series?.forecast_volatility || [];
    const realizedVol = result?.series?.realized_volatility || [];

    const calc = (pred, actual) => {
      const pairs = pred
        .map((value, idx) => [Number(value), Number(actual[idx])])
        .filter(([p, a]) => Number.isFinite(p) && Number.isFinite(a));
      if (!pairs.length) {
        return { mape: null, rmse: null };
      }
      const mapeVals = pairs
        .filter(([, a]) => Math.abs(a) > 1e-12)
        .map(([p, a]) => Math.abs((a - p) / a));
      const rmse =
        Math.sqrt(
          pairs.reduce((acc, [p, a]) => {
            const err = a - p;
            return acc + err * err;
          }, 0) / pairs.length
        ) || 0;
      const mape = mapeVals.length
        ? (mapeVals.reduce((acc, value) => acc + value, 0) / mapeVals.length) * 100
        : null;
      return { mape, rmse };
    };

    return {
      close: calc(forecastClose, realizedClose),
      vol: calc(forecastVol, realizedVol),
      hasRealized: Boolean(realizedClose.length && realizedVol.length),
    };
  }, [result]);

  const historySplitTimestamp = result?.series?.timestamps?.[result?.series?.timestamps?.length - 1];
  const volScale = result?.series?.volatility_scale || { suggested_ymin: 0.0, suggested_ymax: 1.0 };

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
  const formatVol = (value) => `${(Number(value || 0) * 100).toFixed(2)}%`;

  const renderTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) {
      return null;
    }
    return (
      <div className="chartTooltip">
        <p>{formatDateTick(label)}</p>
        {payload
          .filter((item) => item?.value != null)
          .map((item) => (
            <p key={`${label}-${item.name}`}>
              {item.name}: {Number(item.value).toFixed(4)}
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
            </select>
            <p className="controlHint">Fast is low-latency. Quick Train adapts to recent history.</p>
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
            {loading ? "Running Quant Analysis..." : "Analyze"}
          </button>
          <span className="helperText">All metadata fields stay visible after each run.</span>
        </div>
      </form>

      {error ? <p className="error">{error}</p> : null}

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
              <p className="metricValue">
                {Number(result?.prediction?.close || 0).toFixed(2)}
              </p>
              <p className="metricMeta">Symbol: {result?.market?.symbol || symbol}</p>
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
                <div className="badgeItem">
                  <span>Close MAPE</span>
                  <strong>
                    {errorMetrics.close.mape == null ? "N/A" : `${errorMetrics.close.mape.toFixed(2)}%`}
                  </strong>
                </div>
                <div className="badgeItem">
                  <span>Close RMSE</span>
                  <strong>
                    {errorMetrics.close.rmse == null ? "N/A" : errorMetrics.close.rmse.toFixed(4)}
                  </strong>
                </div>
                <div className="badgeItem">
                  <span>Volatility MAPE</span>
                  <strong>
                    {errorMetrics.vol.mape == null ? "N/A" : `${errorMetrics.vol.mape.toFixed(2)}%`}
                  </strong>
                </div>
                <div className="badgeItem">
                  <span>Volatility RMSE</span>
                  <strong>
                    {errorMetrics.vol.rmse == null ? "N/A" : errorMetrics.vol.rmse.toFixed(6)}
                  </strong>
                </div>
              </div>
            ) : (
              <p className="metricMeta">
                Enable <strong>realized overlay</strong> and choose a past date to calculate MAPE/RMSE.
              </p>
            )}
          </section>

          <section className="card chartCard">
            <h2>Close Price Forecast</h2>
            <div className="chartWrap">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={closeSeries}>
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
                  <Area
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast Close"
                    stroke="#f59e0b"
                    fill="url(#closeForecastGradient)"
                    strokeWidth={2}
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
                    <linearGradient id="closeRealizedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#34d399" stopOpacity={0.30} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card chartCard">
            <h2>Volatility Forecast</h2>
            <div className="chartWrap">
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={volatilitySeries}>
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
                  <Area
                    type="monotone"
                    dataKey="forecast"
                    name="Forecast Volatility"
                    stroke="#f43f5e"
                    fill="url(#volForecastGradient)"
                    strokeWidth={2}
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
                    <linearGradient id="volRealizedGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#34d399" stopOpacity={0.30} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                </AreaChart>
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
              <strong>Prediction Horizon:</strong> {horizon}
            </p>
          </section>
        </>
      ) : null}
    </main>
  );
}
