import { useMemo, useState } from "react";
import axios from "axios";

const DEFAULT_TEXT =
  "Recent indicators suggest economic activity has continued to expand at a solid pace.";
const DEFAULT_DATE = new Date().toISOString().slice(0, 10);

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
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const apiBaseUrl = useMemo(() => {
    return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
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

  return (
    <main className="container">
      <header className="pageHeader">
        <p className="kicker">SWE599 Multi-Modal Demo</p>
        <h1>FOMC Sentiment & Volatility Forecaster</h1>
      </header>

      <form onSubmit={handleAnalyze} className="card inputCard">
        <label htmlFor="statement">FOMC Statement Text</label>
        <textarea
          id="statement"
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={10}
          placeholder="Paste an FOMC statement excerpt..."
          required
        />
        <div className="controlRow">
          <div className="controlGroup">
            <label htmlFor="date">Document Date</label>
            <input
              id="date"
              type="date"
              value={date}
              onChange={(event) => setDate(event.target.value)}
              required
            />
          </div>
          <div className="controlGroup">
            <label htmlFor="symbol">Market Symbol</label>
            <select
              id="symbol"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value)}
            >
              <option value="^GSPC">S&P 500 (^GSPC)</option>
              <option value="DX-Y.NYB">Dollar Index (DX-Y.NYB)</option>
            </select>
          </div>
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Running Multi-Modal Analysis..." : "Analyze"}
        </button>
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
            </article>
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
          </section>
        </>
      ) : null}
    </main>
  );
}
